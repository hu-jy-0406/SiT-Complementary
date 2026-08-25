#!/usr/bin/env python3
"""Validate all checkpoints and evaluations required by one experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from .stage_status import FINAL_STEP, RUN_NAMES, completion_issues
except ImportError:  # Direct script execution.
    from stage_status import FINAL_STEP, RUN_NAMES, completion_issues


TARGETS = {
    "conv": (*range(2_000_000, 4_000_001, 250_000), FINAL_STEP),
    "rotation-head": (*range(250_000, 4_000_001, 250_000), FINAL_STEP),
}


def experiment_issues(
    variant: str, output_root: Path, asset_root: Path
) -> list[str]:
    issues: list[str] = []
    for target in TARGETS[variant]:
        issues.extend(
            f"target {target}: {issue}"
            for issue in completion_issues(variant, target, output_root, asset_root)
        )
    return issues


def write_complete_marker(variant: str, output_root: Path) -> Path:
    marker = output_root / f".experiment_complete-{variant}"
    temporary = marker.with_name(f".{marker.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(
            {
                "schema": "sit-experiment-complete-v1",
                "variant": variant,
                "final_step": FINAL_STEP,
            },
            sort_keys=True,
        )
        + "\n"
    )
    temporary.replace(marker)
    return marker


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(RUN_NAMES), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--write-marker", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    asset_root = args.asset_root.resolve()
    issues = experiment_issues(args.variant, output_root, asset_root)
    if issues:
        if not args.quiet:
            print(f"EXPERIMENT_INCOMPLETE variant={args.variant}")
            for issue in issues:
                print(f"- {issue}")
        return 1

    marker = None
    if args.write_marker:
        marker = write_complete_marker(args.variant, output_root)
    if not args.quiet:
        print(f"EXPERIMENT_COMPLETE variant={args.variant}")
        if marker is not None:
            print(f"EXPERIMENT_COMPLETE_MARKER={marker}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
