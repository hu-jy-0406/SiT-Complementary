#!/usr/bin/env python3
"""Export and merge verified experiment results across isolated servers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

try:
    from . import build_results
    from .experiment_status import TARGETS, experiment_issues
    from .stage_status import FINAL_STEP, RUN_NAMES
except ImportError:  # Direct script execution.
    import build_results
    from experiment_status import TARGETS, experiment_issues
    from stage_status import FINAL_STEP, RUN_NAMES


BUNDLE_SCHEMA = "sit-portable-experiment-bundle-v1"
SUPPORTED_PROFILES = {"a100-40gb", "h20"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_raw_names(variant: str) -> tuple[str, ...]:
    names = [f"step-{step:07d}-cfg-1.json" for step in TARGETS[variant][:-1]]
    if variant == "conv":
        names.insert(0, "step-1950000-cfg-1.json")
    names.extend(
        f"step-{FINAL_STEP:07d}-cfg-{cfg}.json" for cfg in (1, 4)
    )
    return tuple(names)


def _read_profile(output_root: Path) -> str:
    marker = output_root / ".gpu_profile"
    try:
        profile = marker.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read GPU profile marker {marker}: {exc}") from exc
    if profile not in SUPPORTED_PROFILES:
        raise RuntimeError(f"unsupported GPU profile in {marker}: {profile!r}")
    return profile


def _copy_and_hash(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return sha256_file(destination)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def validate_bundle(bundle_dir: Path, expected_variant: str | None = None) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest = _load_json(bundle_dir / "manifest.json")
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise RuntimeError(f"unsupported bundle schema: {bundle_dir}")
    if manifest.get("final_step") != FINAL_STEP:
        raise RuntimeError(f"bundle final step is invalid: {bundle_dir}")
    variant = manifest.get("variant")
    if variant not in RUN_NAMES or (
        expected_variant is not None and variant != expected_variant
    ):
        raise RuntimeError(
            f"bundle variant mismatch: expected {expected_variant!r}, got {variant!r}"
        )
    if manifest.get("gpu_profile") not in SUPPORTED_PROFILES:
        raise RuntimeError(f"invalid bundle GPU profile: {bundle_dir}")

    required = {
        "checkpoint/final.pt",
        *(f"raw/{name}" for name in required_raw_names(variant)),
    }
    if variant == "conv":
        required.add("conv_history.tsv")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != required:
        raise RuntimeError(f"bundle file manifest is incomplete or unexpected: {bundle_dir}")
    for relative, expected_hash in files.items():
        path = bundle_dir / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"bundle file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"bundle SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}"
            )

    checkpoint_hash = files["checkpoint/final.pt"]
    for cfg in (1, 4):
        record = _load_json(bundle_dir / "raw" / f"step-{FINAL_STEP:07d}-cfg-{cfg}.json")
        if record.get("variant") != variant or record.get("step") != FINAL_STEP:
            raise RuntimeError(f"invalid final record identity in {bundle_dir}")
        if record.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError(f"final record/checkpoint hash mismatch in {bundle_dir}")
    return manifest


def export_bundle(
    variant: str, output_root: Path, asset_root: Path, bundle_dir: Path
) -> int:
    output_root = output_root.resolve()
    asset_root = asset_root.resolve()
    bundle_dir = bundle_dir.resolve()
    issues = experiment_issues(variant, output_root, asset_root)
    if issues:
        print(f"EXPERIMENT_INCOMPLETE variant={variant}")
        for issue in issues:
            print(f"- {issue}")
        return 1

    profile = _read_profile(output_root)
    final_checkpoint = (
        output_root
        / "training"
        / RUN_NAMES[variant]
        / "checkpoints"
        / f"{FINAL_STEP:07d}.pt"
    )
    if bundle_dir.exists():
        existing = validate_bundle(bundle_dir, variant)
        if existing["files"]["checkpoint/final.pt"] == sha256_file(final_checkpoint):
            print(f"PORTABLE_BUNDLE_ALREADY_COMPLETE={bundle_dir}")
            return 0
        raise RuntimeError(
            f"bundle destination contains a different completed run: {bundle_dir}"
        )

    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{bundle_dir.name}.tmp-", dir=bundle_dir.parent)
    )
    try:
        files: dict[str, str] = {}
        files["checkpoint/final.pt"] = _copy_and_hash(
            final_checkpoint, temporary / "checkpoint" / "final.pt"
        )
        raw_root = output_root / "training_results" / "raw" / variant
        for name in required_raw_names(variant):
            relative = f"raw/{name}"
            files[relative] = _copy_and_hash(raw_root / name, temporary / relative)
        if variant == "conv":
            history = (
                asset_root
                / "huggingface"
                / "BlueSourceJY"
                / "SiT-Complementary"
                / "experiments"
                / "bs256_lr1e-4"
                / "conv-layer"
                / "fid_cfg1_50k.tsv"
            )
            files["conv_history.tsv"] = _copy_and_hash(
                history, temporary / "conv_history.tsv"
            )
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "variant": variant,
            "gpu_profile": profile,
            "final_step": FINAL_STEP,
            "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "files": files,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        validate_bundle(temporary, variant)
        os.replace(temporary, bundle_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(f"PORTABLE_BUNDLE_COMPLETE={bundle_dir}")
    return 0


def _write_if_compatible(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError(f"refusing to overwrite a conflicting import: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _copy_if_compatible(source: Path, destination: Path, expected_hash: str) -> None:
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_hash:
            raise RuntimeError(f"refusing to overwrite a conflicting import: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"copied checkpoint failed verification: {source}")
    os.replace(temporary, destination)


def merge_bundles(
    conv_bundle: Path, rotation_bundle: Path, output_root: Path
) -> int:
    bundles = {
        "conv": conv_bundle.resolve(),
        "rotation-head": rotation_bundle.resolve(),
    }
    manifests = {
        variant: validate_bundle(bundle, variant)
        for variant, bundle in bundles.items()
    }
    profiles = {manifest["gpu_profile"] for manifest in manifests.values()}
    if len(profiles) != 1:
        raise RuntimeError(f"bundle GPU profiles do not match: {sorted(profiles)}")
    profile = profiles.pop()

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / ".portable_merge.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        profile_marker = output_root / ".gpu_profile"
        _write_if_compatible(profile_marker, f"{profile}\n".encode())
        result_root = output_root / "training_results"
        import_root = output_root / "portable_import"

        for variant, bundle in bundles.items():
            manifest = manifests[variant]
            checkpoint = import_root / variant / f"{FINAL_STEP:07d}.pt"
            _copy_if_compatible(
                bundle / "checkpoint" / "final.pt",
                checkpoint,
                manifest["files"]["checkpoint/final.pt"],
            )
            for name in required_raw_names(variant):
                source = bundle / "raw" / name
                record = _load_json(source)
                if record.get("variant") != variant:
                    raise RuntimeError(f"record variant mismatch: {source}")
                if int(record.get("step", -1)) == FINAL_STEP:
                    record["portable_original_checkpoint"] = record.get("checkpoint")
                    record["checkpoint"] = str(checkpoint)
                payload = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
                _write_if_compatible(result_root / "raw" / variant / name, payload)

        conv_history = import_root / "conv_history.tsv"
        _copy_if_compatible(
            bundles["conv"] / "conv_history.tsv",
            conv_history,
            manifests["conv"]["files"]["conv_history.tsv"],
        )
        result = build_results.main(
            [
                "--output-dir",
                str(result_root),
                "--conv-history",
                str(conv_history),
                "--gpu-profile",
                profile,
                "--strict",
            ]
        )
        if result != 0:
            return result
        _write_if_compatible(
            output_root / ".portable_results_complete",
            b"sit-portable-results-complete-v1\n",
        )
    print(f"PORTABLE_RESULTS_COMPLETE={result_root / 'TRAINING_RESULTS.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--variant", choices=sorted(RUN_NAMES), required=True)
    export_parser.add_argument("--output-root", type=Path, required=True)
    export_parser.add_argument("--asset-root", type=Path, required=True)
    export_parser.add_argument("--bundle-dir", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--conv-bundle", type=Path, required=True)
    merge_parser.add_argument("--rotation-head-bundle", type=Path, required=True)
    merge_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "export":
        return export_bundle(
            args.variant, args.output_root, args.asset_root, args.bundle_dir
        )
    return merge_bundles(
        args.conv_bundle, args.rotation_head_bundle, args.output_root
    )


if __name__ == "__main__":
    raise SystemExit(main())
