#!/usr/bin/env python3
"""Convert the Hugging Face ImageNet-1K parquet export to ImageFolder."""

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def load_classes(path: Path):
    namespace = {}
    exec(path.read_text(), namespace)
    return namespace["IMAGENET2012_CLASSES"]


def convert_split(src_dir: Path, dst_dir: Path, split: str, classes, wnids):
    files = sorted((src_dir / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {split}")

    counts = [0] * len(classes)
    for parquet_path in files:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(batch_size=256, columns=["image", "label"]):
            rows = batch.to_pylist()
            for row in rows:
                label = int(row["label"])
                if not 0 <= label < len(classes):
                    raise ValueError(f"Invalid {split} label {label} in {parquet_path}")
                image = row["image"]
                image_bytes = image["bytes"]
                if image_bytes is None:
                    raise ValueError(f"Missing image bytes in {parquet_path}")
                original_name = os.path.basename(image.get("path") or "image.JPEG")
                # The source paths are unique, but prefixing the label keeps the
                # generated names deterministic even if a future export changes.
                filename = f"{counts[label]:06d}_{original_name}"
                output = dst_dir / split / wnids[label] / filename
                output.parent.mkdir(parents=True, exist_ok=True)
                if not output.exists():
                    output.write_bytes(image_bytes)
                counts[label] += 1
        print(f"{split}: processed {parquet_path.name}", flush=True)
    return sum(counts), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("classes", type=Path)
    args = parser.parse_args()

    classes = load_classes(args.classes)
    wnids = list(classes.keys())
    args.dst.mkdir(parents=True, exist_ok=True)
    class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
    (args.dst / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2) + "\n"
    )
    (args.dst / "classes.json").write_text(
        json.dumps({wnid: name for wnid, name in classes.items()}, indent=2) + "\n"
    )

    summary = {}
    for split in ("train", "validation"):
        total, counts = convert_split(args.src, args.dst, split, classes, wnids)
        summary[split] = {"total": total, "per_class": counts}
    (args.dst / "conversion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v["total"] for k, v in summary.items()}))


if __name__ == "__main__":
    main()
