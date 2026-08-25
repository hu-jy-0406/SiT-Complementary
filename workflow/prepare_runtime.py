#!/usr/bin/env python3
"""Prepare pinned assets once, safely, when two nodes share ASSET_ROOT."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()

    asset_root = args.asset_root.expanduser().resolve()
    asset_root.mkdir(parents=True, exist_ok=True)
    lock_path = asset_root / ".prepare.lock"
    environment = os.environ.copy()
    environment.update(
        {
            "SIT_VAE_ROOT": str(asset_root / "vae"),
            "TORCH_HOME": str(asset_root / "cache" / "torch"),
            "HF_HOME": str(asset_root / "cache" / "huggingface"),
        }
    )

    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        local_only = ["--local-only"] if args.local_only else []
        for script in ("prepare_assets.py", "prewarm.py"):
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "workflow" / script),
                    "--asset-root",
                    str(asset_root),
                    *local_only,
                ],
                check=True,
                cwd=REPO_ROOT,
                env=environment,
            )
    print(f"ASSET_PREPARATION_COMPLETE={asset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
