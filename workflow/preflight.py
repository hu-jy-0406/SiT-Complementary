"""Fail fast before committing an 8-GPU H20 allocation to a long run."""

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder


EXPECTED_IMAGENET_CLASSES = 1_000
EXPECTED_IMAGENET_TRAIN_IMAGES = 1_281_167


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-train", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--min-free-gib", type=int, default=100)
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--require-clean-git", action="store_true")
    args = parser.parse_args()

    for command in ("torchrun", "pytorch-fid"):
        if shutil.which(command) is None:
            raise RuntimeError(f"Required command is missing: {command}")

    train_dir = args.imagenet_train.resolve()
    if not train_dir.is_dir():
        raise RuntimeError(f"ImageNet train directory does not exist: {train_dir}")
    dataset = ImageFolder(train_dir)
    if len(dataset.classes) != EXPECTED_IMAGENET_CLASSES:
        raise RuntimeError(
            f"ImageNet has {len(dataset.classes)} classes, expected "
            f"{EXPECTED_IMAGENET_CLASSES}"
        )
    if len(dataset) != EXPECTED_IMAGENET_TRAIN_IMAGES:
        raise RuntimeError(
            f"ImageNet has {len(dataset)} train images, expected "
            f"{EXPECTED_IMAGENET_TRAIN_IMAGES}"
        )

    divisor = args.nproc * args.gradient_accumulation_steps
    if args.global_batch_size % divisor:
        raise RuntimeError("Global batch is not divisible by world size × accumulation")
    micro_batch = args.global_batch_size // divisor

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_root).free / 2**30
    if free_gib < args.min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free at {output_root}; "
            f"at least {args.min_free_gib} GiB is required"
        )

    if not args.skip_gpu_check:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        visible = torch.cuda.device_count()
        if visible != args.nproc:
            raise RuntimeError(
                f"Expected exactly {args.nproc} visible GPUs, found {visible}"
            )
        names = [torch.cuda.get_device_name(index) for index in range(visible)]
        non_h20 = [name for name in names if "H20" not in name.upper()]
        if non_h20:
            raise RuntimeError(f"Expected H20 GPUs, found: {names}")
        if not torch.distributed.is_nccl_available():
            raise RuntimeError("PyTorch NCCL support is unavailable")
        print(f"GPUS={names}")

    if args.require_clean_git:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(
                "Repository is dirty. Commit or discard site edits before training."
            )

    print(f"IMAGENET_IMAGES={len(dataset)}")
    print(f"MICRO_BATCH_PER_GPU={micro_batch}")
    print(f"OUTPUT_FREE_GIB={free_gib:.1f}")
    print("PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
