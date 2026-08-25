#!/usr/bin/env python3
"""Serialize report generation and publish it after both experiments finish."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path

try:
    from . import build_results
    from .experiment_status import experiment_issues
    from .stage_status import RUN_NAMES
except ImportError:  # Direct script execution.
    import build_results
    from experiment_status import experiment_issues
    from stage_status import RUN_NAMES


def _write_finalize_marker(output_root: Path, variant: str, status: str) -> None:
    marker = output_root / f".finalize_passed-{variant}"
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(f"{status}\n")
    temporary.replace(marker)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-variant", choices=sorted(RUN_NAMES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--gpu-profile", required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    with (output_root / ".finalize.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active_issues = experiment_issues(
            args.active_variant, output_root, asset_root
        )
        if active_issues:
            print(f"ACTIVE_EXPERIMENT_INCOMPLETE={args.active_variant}")
            for issue in active_issues:
                print(f"- {issue}")
            return 1

        incomplete = {
            variant: experiment_issues(variant, output_root, asset_root)
            for variant in RUN_NAMES
            if variant != args.active_variant
        }
        conv_history = (
            asset_root
            / "huggingface"
            / "BlueSourceJY"
            / "SiT-Complementary"
            / "experiments"
            / "bs256_lr1e-4"
            / "conv-layer"
            / "fid_cfg1_50k.tsv"
        )
        build_args = [
            "--output-dir",
            str(output_root / "training_results"),
            "--conv-history",
            str(conv_history),
            "--gpu-profile",
            args.gpu_profile,
        ]
        if any(incomplete.values()):
            result = build_results.main(build_args)
            if result != 0:
                return result
            waiting = ",".join(sorted(incomplete))
            _write_finalize_marker(output_root, args.active_variant, "pending")
            print(f"JOINT_RESULTS_PENDING={waiting}")
            return 0

        result = build_results.main([*build_args, "--strict"])
        if result == 0:
            _write_finalize_marker(output_root, args.active_variant, "complete")
            print(
                "JOINT_RESULTS_COMPLETE="
                f"{output_root / 'training_results' / 'TRAINING_RESULTS.md'}"
            )
        return result


if __name__ == "__main__":
    raise SystemExit(main())
