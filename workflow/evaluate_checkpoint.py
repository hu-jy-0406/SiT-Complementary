"""Run one strict, restart-safe FID evaluation and write an atomic JSON shard."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

try:
    from .checkpoint_tool import inspect as inspect_checkpoint
except ImportError:  # Support ``python workflow/evaluate_checkpoint.py``.
    from checkpoint_tool import inspect as inspect_checkpoint


REPO_ROOT = Path(__file__).resolve().parents[1]
RECORD_SCHEMA = "sit-fid-evaluation-v1"
PROTOCOL_ID = "sit-imagenet256-pytorch-fid-euler250-seed0-padded50176-v1"
REFERENCE_SHA256 = "b32732719497e42660a9affb4a966068cba0855ac449b82015e34ec376d20758"
REQUIRED_PYTORCH_FID_VERSION = "0.3.0"
NUM_REQUESTED = 50_000
EXPECTED_NUM_PNG = 50_176
EXPECTED_NPROC = 8
EXPECTED_PER_PROC_BATCH_SIZE = 64
EXPECTED_FID_BATCH_SIZE = 128
EXPECTED_NUM_WORKERS = 8
VARIANT_TO_SAMPLER = {"conv": "conv", "rotation-head": "rot-head"}

EVALUATION_IDENTITY_FIELDS = (
    "record_schema",
    "variant",
    "step",
    "phase",
    "cfg",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_step",
    "num_requested",
    "num_png",
    "seed",
    "sampler",
    "sampling_method",
    "sampling_steps",
    "world_size",
    "per_proc_batch_size",
    "fid_impl",
    "fid_batch_size",
    "fid_num_workers",
    "reference",
    "reference_sha256",
    "protocol_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_reference(reference: Path) -> str:
    """Hash the exact NPZ consumed by PyTorch-FID and reject drift."""

    actual = sha256(reference)
    if actual != REFERENCE_SHA256:
        raise RuntimeError(
            f"FID reference SHA-256 mismatch for {reference}: "
            f"expected {REFERENCE_SHA256}, got {actual}"
        )
    return actual


def require_exact_protocol(args: argparse.Namespace, fid_version: str) -> None:
    expected = {
        "nproc": EXPECTED_NPROC,
        "per_proc_batch_size": EXPECTED_PER_PROC_BATCH_SIZE,
        "fid_batch_size": EXPECTED_FID_BATCH_SIZE,
        "num_workers": EXPECTED_NUM_WORKERS,
    }
    mismatches = [
        f"--{name.replace('_', '-')} must be {wanted}, got {getattr(args, name)}"
        for name, wanted in expected.items()
        if getattr(args, name) != wanted
    ]
    if fid_version != REQUIRED_PYTORCH_FID_VERSION:
        mismatches.append(
            "pytorch-fid must be exactly "
            f"{REQUIRED_PYTORCH_FID_VERSION}, got {fid_version}"
        )
    if mismatches:
        raise RuntimeError("Evaluation protocol mismatch: " + "; ".join(mismatches))


def evaluation_already_complete(
    existing: object, expected_identity: dict[str, object]
) -> bool:
    """Only reuse a shard that describes this exact evaluation."""

    if not isinstance(existing, dict) or existing.get("status") != "ok":
        return False
    try:
        fid = float(existing["fid"])
    except (KeyError, TypeError, ValueError):
        return False
    if not math.isfinite(fid):
        return False
    return all(
        existing.get(field) == expected_identity[field]
        for field in EVALUATION_IDENTITY_FIELDS
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content)
    os.replace(temporary, path)


def run_logged(command: list[str], log_path: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    with log_path.open("w") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
            lines.append(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return "".join(lines)


def safe_reset_sample_dir(sample_dir: Path, sample_root: Path) -> None:
    resolved_dir = sample_dir.resolve()
    resolved_root = sample_root.resolve()
    if resolved_dir.parent != resolved_root:
        raise RuntimeError(f"Refusing to clean unexpected sample path: {resolved_dir}")
    if resolved_dir.exists():
        shutil.rmtree(resolved_dir)


def cfg_label(cfg: float) -> str:
    return f"{cfg:g}"


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANT_TO_SAMPLER), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cfg", type=float, choices=[1.0, 4.0], required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--fid-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--keep-samples", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    reference = args.reference.resolve()
    result_root = args.result_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not reference.is_file():
        raise FileNotFoundError(reference)

    installed_fid_version = version("pytorch-fid")
    require_exact_protocol(args, installed_fid_version)
    reference_sha = verify_reference(reference)
    checkpoint_step = inspect_checkpoint(checkpoint)
    if checkpoint_step != args.step:
        raise RuntimeError(
            f"Checkpoint state step {checkpoint_step} does not match --step {args.step}"
        )
    checkpoint_sha = sha256(checkpoint)
    label = cfg_label(args.cfg)
    raw_dir = result_root / "raw" / args.variant
    json_path = raw_dir / f"step-{args.step:07d}-cfg-{label}.json"
    fid_log = raw_dir / f"step-{args.step:07d}-cfg-{label}.txt"
    sample_log = raw_dir / f"step-{args.step:07d}-cfg-{label}-sampling.txt"

    if args.step == 4_003_200:
        phase = "final"
    elif args.variant == "conv" and args.step == 1_950_000:
        phase = "resume-baseline"
    else:
        phase = "periodic"
    fid_impl = f"pytorch-fid {installed_fid_version}"
    identity: dict[str, object] = {
        "record_schema": RECORD_SCHEMA,
        "variant": args.variant,
        "step": args.step,
        "phase": phase,
        "cfg": args.cfg,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": checkpoint_step,
        "num_requested": NUM_REQUESTED,
        "num_png": EXPECTED_NUM_PNG,
        "seed": 0,
        "sampler": "euler",
        "sampling_method": "euler",
        "sampling_steps": 250,
        "world_size": args.nproc,
        "per_proc_batch_size": args.per_proc_batch_size,
        "fid_impl": fid_impl,
        "fid_batch_size": args.fid_batch_size,
        "fid_num_workers": args.num_workers,
        "reference": str(reference),
        "reference_sha256": reference_sha,
        "protocol_id": PROTOCOL_ID,
    }

    if json_path.is_file():
        existing = json.loads(json_path.read_text())
        if evaluation_already_complete(existing, identity):
            print(f"EVALUATION_ALREADY_COMPLETE={json_path}")
            return
        raise RuntimeError(f"Conflicting evaluation shard already exists: {json_path}")

    sample_root = result_root / "work" / "samples"
    sample_root.mkdir(parents=True, exist_ok=True)
    sampler_variant = VARIANT_TO_SAMPLER[args.variant]
    cfg_string = str(float(args.cfg))
    folder_name = (
        f"SiT-S-2-{sampler_variant}-{checkpoint.stem}-cfg-{cfg_string}-"
        f"{args.per_proc_batch_size}-ODE-250-euler"
    )
    sample_dir = sample_root / folder_name

    # A partial folder cannot be resumed statistically safely because the old
    # sampler did not advance each rank's RNG. Always restart this one eval.
    safe_reset_sample_dir(sample_dir, sample_root)
    sampling_command = [
        "torchrun",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.nproc}",
        "sample_ddp.py",
        "ODE",
        "--variant",
        sampler_variant,
        "--model",
        "SiT-S/2",
        "--num-fid-samples",
        "50000",
        "--keep-padded-samples",
        "--skip-npz",
        "--cfg-scale",
        cfg_string,
        "--per-proc-batch-size",
        str(args.per_proc_batch_size),
        "--num-sampling-steps",
        "250",
        "--sampling-method",
        "euler",
        "--global-seed",
        "0",
        "--sample-dir",
        str(sample_root),
        "--ckpt",
        str(checkpoint),
    ]
    run_logged(sampling_command, sample_log)

    expected_png = math.ceil(
        NUM_REQUESTED / (args.nproc * args.per_proc_batch_size)
    ) * (args.nproc * args.per_proc_batch_size)
    if expected_png != EXPECTED_NUM_PNG:
        raise AssertionError(
            f"Protocol constants produced {expected_png} samples, expected "
            f"{EXPECTED_NUM_PNG}"
        )
    png_files = sorted(sample_dir.glob("*.png"))
    if len(png_files) != expected_png:
        raise RuntimeError(
            f"Expected {expected_png} PNGs in {sample_dir}, found {len(png_files)}"
        )
    if png_files[0].stem != "000000" or int(png_files[-1].stem) != expected_png - 1:
        raise RuntimeError("Sample indices are not contiguous")

    fid_command = [
        "pytorch-fid",
        str(sample_dir),
        str(reference),
        "--device",
        "cuda:0",
        "--batch-size",
        str(args.fid_batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    fid_output = run_logged(fid_command, fid_log)
    matches = re.findall(r"FID:\s*([0-9]+(?:\.[0-9]+)?)", fid_output)
    if not matches:
        raise RuntimeError(f"Could not parse FID from {fid_log}")
    fid = float(matches[-1])

    record = {
        **identity,
        "epoch": args.step / 5004,
        "fid": fid,
        "status": "ok",
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(json_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    if not args.keep_samples:
        safe_reset_sample_dir(sample_dir, sample_root)
    print(f"EVALUATION_COMPLETE={json_path}")
    print(f"FID={fid}")


if __name__ == "__main__":
    main()
