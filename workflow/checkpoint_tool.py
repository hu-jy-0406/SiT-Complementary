"""Inspect checkpoints and select the latest valid resumable state."""

import argparse
from pathlib import Path
import sys

import torch


def infer_step(checkpoint: dict, path: Path) -> int:
    if checkpoint.get("train_steps") is not None:
        return int(checkpoint["train_steps"])
    steps = {
        int(state["step"])
        for state in checkpoint["opt"].get("state", {}).values()
        if "step" in state
    }
    if len(steps) == 1:
        return steps.pop()
    if path.stem.isdigit():
        return int(path.stem)
    raise RuntimeError(f"Cannot infer optimizer step from {path}")


def inspect(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model", "ema", "opt"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"{path} is missing keys: {sorted(missing)}")
    step = infer_step(checkpoint, path)
    if path.stem.isdigit() and int(path.stem) != step:
        raise RuntimeError(
            f"Checkpoint filename step {path.stem} disagrees with state step {step}"
        )
    return step


def latest(directory: Path, max_step: int, fallback: Path | None) -> Path | None:
    candidates = []
    if directory.is_dir():
        candidates.extend(
            path
            for path in directory.glob("*.pt")
            if path.stem.isdigit() and int(path.stem) < max_step
        )
    if fallback is not None and fallback.is_file():
        candidates.append(fallback)
    candidates.sort(
        key=lambda path: int(path.stem) if path.stem.isdigit() else -1,
        reverse=True,
    )
    for path in candidates:
        try:
            step = inspect(path)
        except Exception as exc:  # continue past a corrupt preempted checkpoint
            print(f"Ignoring invalid checkpoint {path}: {exc}", file=sys.stderr)
            continue
        if step < max_step:
            return path.resolve()
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("path", type=Path)
    latest_parser = subparsers.add_parser("latest")
    latest_parser.add_argument("--directory", type=Path, required=True)
    latest_parser.add_argument("--max-step", type=int, required=True)
    latest_parser.add_argument("--fallback", type=Path)
    args = parser.parse_args()

    if args.command == "inspect":
        print(inspect(args.path))
    else:
        path = latest(args.directory, args.max_step, args.fallback)
        if path is None:
            raise SystemExit(2)
        print(path)


if __name__ == "__main__":
    main()
