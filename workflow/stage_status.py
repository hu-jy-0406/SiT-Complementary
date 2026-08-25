#!/usr/bin/env python3
"""Cheaply determine whether one restart-safe H20 stage is already complete."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


FINAL_STEP = 4_003_200
CONV_FIRST_STEP = 2_000_000
RECORD_SCHEMA = "sit-fid-evaluation-v1"
PROTOCOL_ID = "sit-imagenet256-pytorch-fid-euler250-seed0-padded50176-v1"
REFERENCE_SHA256 = "b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758"
EXACT_PROTOCOL = {
    "record_schema": RECORD_SCHEMA,
    "protocol_id": PROTOCOL_ID,
    "num_requested": 50_000,
    "num_png": 50_176,
    "seed": 0,
    "sampler": "euler",
    "sampling_method": "euler",
    "sampling_steps": 250,
    "world_size": 8,
    "per_proc_batch_size": 64,
    "fid_impl": "pytorch-fid 0.3.0",
    "fid_batch_size": 128,
    "fid_num_workers": 8,
    "reference_sha256": REFERENCE_SHA256,
}
RUN_NAMES = {
    "conv": "SiT-S-2-ConvLayer-bs256-lr1e-4-800ep",
    "rotation-head": "SiT-S-2-RotationHead-bs256-lr1e-4-800ep",
}
FINAL_ARTIFACTS = (
    "TRAINING_RESULTS.md",
    "training_results.json",
    "fid_results.tsv",
    "fid_cfg1_training_curves.png",
    "conv_fid_cfg1_curve.png",
    "rotation_head_fid_cfg1_curve.png",
)


def _required_evaluations() -> list[tuple[str, int, float]]:
    evaluations = [("conv", 1_950_000, 1.0)]
    evaluations.extend(
        ("conv", step, 1.0) for step in range(2_000_000, 4_000_001, 250_000)
    )
    evaluations.extend(
        ("rotation-head", step, 1.0)
        for step in range(250_000, 4_000_001, 250_000)
    )
    evaluations.extend(
        (variant, FINAL_STEP, cfg)
        for variant in RUN_NAMES
        for cfg in (1.0, 4.0)
    )
    return evaluations


def _same_path(recorded: object, expected: Path) -> bool:
    if not isinstance(recorded, str) or not recorded:
        return False
    return Path(recorded).expanduser().resolve() == expected.resolve()


def _evaluation_issues(
    path: Path,
    variant: str,
    step: int,
    cfg: float,
    *,
    checkpoint: Path,
    reference: Path,
) -> list[str]:
    if not path.is_file():
        return [f"missing evaluation shard: {path}"]
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid evaluation shard {path}: {exc}"]

    issues = []
    if record.get("status") != "ok":
        issues.append(f"evaluation is not successful: {path}")
    if record.get("variant") != variant:
        issues.append(f"evaluation variant mismatch: {path}")
    try:
        record_step = int(record.get("step"))
    except (TypeError, ValueError):
        record_step = None
    if record_step != step:
        issues.append(f"evaluation step mismatch: {path}")
    try:
        record_cfg = float(record.get("cfg"))
    except (TypeError, ValueError):
        record_cfg = None
    if record_cfg != cfg:
        issues.append(f"evaluation CFG mismatch: {path}")
    for field, expected in EXACT_PROTOCOL.items():
        if record.get(field) != expected:
            issues.append(
                f"evaluation protocol mismatch for {field}: {path}"
            )
    expected_phase = (
        "final"
        if step == FINAL_STEP
        else "resume-baseline"
        if variant == "conv" and step == 1_950_000
        else "periodic"
    )
    if record.get("phase") != expected_phase:
        issues.append(f"evaluation phase mismatch: {path}")
    try:
        fid = float(record.get("fid"))
    except (TypeError, ValueError):
        fid = math.nan
    if not math.isfinite(fid) or fid < 0:
        issues.append(f"evaluation FID is missing or invalid: {path}")
    if record.get("checkpoint_step") != step:
        issues.append(f"evaluation checkpoint step mismatch: {path}")
    if not _same_path(record.get("checkpoint"), checkpoint):
        issues.append(f"evaluation checkpoint path mismatch: {path}")
    if not _same_path(record.get("reference"), reference):
        issues.append(f"evaluation reference path mismatch: {path}")
    checkpoint_sha = record.get("checkpoint_sha256")
    if not isinstance(checkpoint_sha, str) or not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha
    ):
        issues.append(f"evaluation checkpoint hash is missing or invalid: {path}")
    return issues


def completion_issues(
    variant: str, target: int, output_root: Path, asset_root: Path
) -> list[str]:
    run_name = RUN_NAMES[variant]
    checkpoint = (
        output_root
        / "training"
        / run_name
        / "checkpoints"
        / f"{target:07d}.pt"
    )
    issues = []
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        issues.append(f"missing target checkpoint: {checkpoint}")

    raw_root = output_root / "training_results" / "raw" / variant
    reference = asset_root / "fid" / "VIRTUAL_imagenet256_labeled.npz"
    cfgs = (1.0, 4.0) if target == FINAL_STEP else (1.0,)
    for cfg in cfgs:
        label = f"{cfg:g}"
        path = raw_root / f"step-{target:07d}-cfg-{label}.json"
        issues.extend(
            _evaluation_issues(
                path,
                variant,
                target,
                cfg,
                checkpoint=checkpoint,
                reference=reference,
            )
        )

    if variant == "conv" and target == CONV_FIRST_STEP:
        conv_source = (
            asset_root
            / "huggingface"
            / "BlueSourceJY"
            / "SiT-Complementary"
            / "checkpoints"
            / "bs256_lr1e-4"
            / "conv-layer"
            / "1950000.pt"
        )
        if not conv_source.is_file():
            issues.append(f"missing Conv resume checkpoint: {conv_source}")
        baseline = raw_root / "step-1950000-cfg-1.json"
        issues.extend(
            _evaluation_issues(
                baseline,
                variant,
                1_950_000,
                1.0,
                checkpoint=conv_source,
                reference=reference,
            )
        )

    if variant == "rotation-head" and target == FINAL_STEP:
        result_root = output_root / "training_results"
        for required_variant, required_step, required_cfg in _required_evaluations():
            label = f"{required_cfg:g}"
            shard = (
                result_root
                / "raw"
                / required_variant
                / f"step-{required_step:07d}-cfg-{label}.json"
            )
            if required_variant == "conv" and required_step == 1_950_000:
                required_checkpoint = (
                    asset_root
                    / "huggingface"
                    / "BlueSourceJY"
                    / "SiT-Complementary"
                    / "checkpoints"
                    / "bs256_lr1e-4"
                    / "conv-layer"
                    / "1950000.pt"
                )
            else:
                required_checkpoint = (
                    output_root
                    / "training"
                    / RUN_NAMES[required_variant]
                    / "checkpoints"
                    / f"{required_step:07d}.pt"
                )
            issues.extend(
                _evaluation_issues(
                    shard,
                    required_variant,
                    required_step,
                    required_cfg,
                    checkpoint=required_checkpoint,
                    reference=reference,
                )
            )
        for filename in FINAL_ARTIFACTS:
            if not (result_root / filename).is_file():
                issues.append(f"missing final result artifact: {result_root / filename}")
        summary_path = result_root / "training_results.json"
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                issues.append(f"invalid final result summary: {exc}")
            else:
                if summary.get("status") != "COMPLETE":
                    issues.append("final result summary is not COMPLETE")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(RUN_NAMES), required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    issues = completion_issues(
        args.variant,
        args.target,
        args.output_root.resolve(),
        args.asset_root.resolve(),
    )
    if issues:
        if not args.quiet:
            print("STAGE_INCOMPLETE")
            for issue in issues:
                print(f"- {issue}")
        return 1
    if not args.quiet:
        print(
            f"STAGE_COMPLETE variant={args.variant} target={args.target}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
