"""Log an already-computed FID value to the matching W&B training run."""

import argparse
import os
from pathlib import Path
import sys

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wandb_utils import generate_run_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--project", default="SiT-Complementary")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--cfg", type=float, required=True)
    parser.add_argument("--fid", type=float, required=True)
    args = parser.parse_args()

    if "WANDB_API_KEY" not in os.environ:
        raise RuntimeError("WANDB_API_KEY must be set in the process environment")

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        id=generate_run_id(args.run_name),
        resume="allow",
    )
    run.log({f"fid/cfg_{args.cfg:g}": args.fid}, step=args.step)
    run.finish()


if __name__ == "__main__":
    main()
