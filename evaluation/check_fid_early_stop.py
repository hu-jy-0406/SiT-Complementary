"""Apply the historical three-significant-increases FID early-stop rule."""

import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--consecutive", type=int, default=3)
    parser.add_argument("--min-absolute-rise", type=float, default=0.25)
    parser.add_argument("--min-relative-rise", type=float, default=0.005)
    args = parser.parse_args()

    with args.history.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if row["status"] == "ok" and float(row["cfg"]) == 1.0
        ]

    increases = 0
    for previous, current in zip(rows[-(args.consecutive + 1) :], rows[-args.consecutive :]):
        previous_fid = float(previous["fid"])
        rise = float(current["fid"]) - previous_fid
        threshold = max(
            args.min_absolute_rise,
            args.min_relative_rise * abs(previous_fid),
        )
        if rise >= threshold:
            increases += 1
        else:
            increases = 0

    if increases >= args.consecutive:
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.write_text(
            "FID increased significantly for "
            f"{args.consecutive} consecutive evaluations.\n"
        )
        print(f"FID_EARLY_STOP marker={args.marker}")
    else:
        print(f"FID_CONTINUE consecutive_significant_increases={increases}")


if __name__ == "__main__":
    main()
