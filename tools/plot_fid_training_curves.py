#!/usr/bin/env python3
"""Plot aligned CFG=1 50k PyTorch-FID histories for the SiT-S/2 runs."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


STEPS_PER_EPOCH = 5004
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = REPOSITORY_ROOT / "experiments" / "bs256_lr1e-4"
DEFAULT_BASE_HISTORY = DEFAULT_EXPERIMENT_DIR / "base" / "fid_cfg1_50k.tsv"
DEFAULT_ROT_HISTORY = DEFAULT_EXPERIMENT_DIR / "rotation-layer" / "fid_cfg1_50k.tsv"
DEFAULT_CONV_HISTORY = DEFAULT_EXPERIMENT_DIR / "conv-layer" / "fid_cfg1_50k.tsv"

SERIES = (
    ("Base", "#2563eb"),
    ("Rotation-layer", "#dc2626"),
    ("Conv-layer", "#059669"),
)


def _read_history(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    try:
        source = str(path.resolve().relative_to(REPOSITORY_ROOT))
    except ValueError:
        source = str(path)
    by_step: dict[int, dict[str, object]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("status") != "ok" or float(row.get("cfg", "nan")) != 1.0:
                continue
            step = int(row["step"])
            by_step[step] = {
                "step": step,
                "epoch": step / STEPS_PER_EPOCH,
                "fid": float(row["fid"]),
                "cfg": float(row["cfg"]),
                "num_png": int(row["num_png"]),
                "checkpoint": row["checkpoint"],
                "timestamp_utc": row["timestamp_utc"],
                "source": source,
            }
    return [by_step[step] for step in sorted(by_step)]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def _bounds(histories: dict[str, list[dict[str, object]]]) -> tuple[float, float]:
    values = [float(row["fid"]) for rows in histories.values() for row in rows]
    if not values:
        return 0.0, 100.0
    low = 5.0 * math.floor((min(values) - 2.0) / 5.0)
    high = 5.0 * math.ceil((max(values) + 2.0) / 5.0)
    return max(0.0, low), max(low + 5.0, high)


def _write_csv(path: Path, histories: dict[str, list[dict[str, object]]]) -> None:
    fields = (
        "model", "step", "epoch", "fid", "cfg", "num_png",
        "checkpoint", "timestamp_utc", "source",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for model, _ in SERIES:
            for row in histories.get(model, []):
                writer.writerow({"model": model, **row})


def _write_png(path: Path, histories: dict[str, list[dict[str, object]]]) -> None:
    width, height = 1600, 900
    left, top, right, bottom = 125, 150, 1540, 785
    y_min, y_max = _bounds(histories)
    image = Image.new("RGB", (width, height), "#f8fafc")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((55, 45, width - 55, height - 45), radius=22,
                           fill="white", outline="#dbe3ee", width=2)
    draw.text((left, 67), "SiT-S/2 ImageNet 256: FID vs Training Progress",
              font=_font(34, bold=True), fill="#0f172a")
    draw.text((left, 113),
              "CFG=1 · PyTorch-FID · 50,176 images · global batch 256 · lr 1e-4 · lower is better",
              font=_font(19), fill="#475569")

    def px(epoch: float) -> float:
        return left + (epoch / 800.0) * (right - left)

    def py(fid: float) -> float:
        return bottom - ((fid - y_min) / (y_max - y_min)) * (bottom - top)

    draw.rectangle((left, top, right, bottom), fill="#ffffff", outline="#94a3b8", width=2)
    for epoch in range(0, 801, 100):
        x = px(epoch)
        draw.line((x, top, x, bottom), fill="#e2e8f0", width=1)
        label = str(epoch)
        box = draw.textbbox((0, 0), label, font=_font(16))
        draw.text((x - (box[2] - box[0]) / 2, bottom + 12), label,
                  font=_font(16), fill="#475569")
    y_tick = y_min
    while y_tick <= y_max + 1e-9:
        y = py(y_tick)
        draw.line((left, y, right, y), fill="#e2e8f0", width=1)
        label = f"{y_tick:g}"
        box = draw.textbbox((0, 0), label, font=_font(16))
        draw.text((left - 18 - (box[2] - box[0]), y - 10), label,
                  font=_font(16), fill="#475569")
        y_tick += 5.0

    draw.text(((left + right) / 2 - 95, bottom + 54),
              "Training epoch (5,004 steps/epoch)", font=_font(18), fill="#334155")
    draw.text((28, (top + bottom) / 2 + 50), "PyTorch FID", font=_font(18),
              fill="#334155")

    for model, color in SERIES:
        rows = histories.get(model, [])
        if not rows:
            continue
        points = [(px(float(row["epoch"])), py(float(row["fid"]))) for row in rows]
        if len(points) > 1:
            draw.line(points, fill=color, width=5, joint="curve")
        radius = 4 if len(rows) <= 20 else 3
        for x, y in points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         fill="white", outline=color, width=3)

    legend_x, legend_y = right - 445, top + 22
    present = [(model, color, histories[model]) for model, color in SERIES if histories.get(model)]
    legend_h = 26 + 44 * len(present)
    draw.rounded_rectangle((legend_x, legend_y, right - 20, legend_y + legend_h), radius=12,
                           fill="#ffffff", outline="#cbd5e1", width=2)
    for index, (model, color, rows) in enumerate(present):
        y = legend_y + 21 + index * 44
        draw.line((legend_x + 20, y, legend_x + 68, y), fill=color, width=5)
        draw.ellipse((legend_x + 40, y - 4, legend_x + 48, y + 4), fill="white",
                     outline=color, width=2)
        latest = rows[-1]
        label = f"{model}  ·  latest {float(latest['fid']):.3f} @ {float(latest['epoch']):.1f} ep"
        draw.text((legend_x + 82, y - 13), label, font=_font(17, bold=True), fill="#1e293b")

    image.save(path, optimize=True)


def _write_svg(path: Path, histories: dict[str, list[dict[str, object]]]) -> None:
    width, height = 1600, 900
    left, top, right, bottom = 125, 150, 1540, 785
    y_min, y_max = _bounds(histories)

    def px(epoch: float) -> float:
        return left + (epoch / 800.0) * (right - left)

    def py(fid: float) -> float:
        return bottom - ((fid - y_min) / (y_max - y_min)) * (bottom - top)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<rect x="55" y="45" width="1490" height="810" rx="22" fill="white" stroke="#dbe3ee" stroke-width="2"/>',
        '<text x="125" y="95" font-family="DejaVu Sans" font-size="34" font-weight="700" fill="#0f172a">SiT-S/2 ImageNet 256: FID vs Training Progress</text>',
        '<text x="125" y="130" font-family="DejaVu Sans" font-size="19" fill="#475569">CFG=1 · PyTorch-FID · 50,176 images · global batch 256 · lr 1e-4 · lower is better</text>',
        f'<rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="white" stroke="#94a3b8" stroke-width="2"/>',
    ]
    for epoch in range(0, 801, 100):
        x = px(epoch)
        lines.extend([
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" stroke="#e2e8f0"/>',
            f'<text x="{x:.2f}" y="{bottom+32}" text-anchor="middle" font-family="DejaVu Sans" font-size="16" fill="#475569">{epoch}</text>',
        ])
    y_tick = y_min
    while y_tick <= y_max + 1e-9:
        y = py(y_tick)
        lines.extend([
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="#e2e8f0"/>',
            f'<text x="{left-18}" y="{y+6:.2f}" text-anchor="end" font-family="DejaVu Sans" font-size="16" fill="#475569">{y_tick:g}</text>',
        ])
        y_tick += 5.0
    lines.extend([
        f'<text x="{(left+right)/2:.2f}" y="{bottom+74}" text-anchor="middle" font-family="DejaVu Sans" font-size="18" fill="#334155">Training epoch (5,004 steps/epoch)</text>',
        f'<text x="35" y="{(top+bottom)/2:.2f}" text-anchor="middle" transform="rotate(-90 35 {(top+bottom)/2:.2f})" font-family="DejaVu Sans" font-size="18" fill="#334155">PyTorch FID</text>',
    ])
    for model, color in SERIES:
        rows = histories.get(model, [])
        if not rows:
            continue
        points = " ".join(f"{px(float(row['epoch'])):.2f},{py(float(row['fid'])):.2f}" for row in rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="5" stroke-linejoin="round" stroke-linecap="round"/>')
        radius = 4 if len(rows) <= 20 else 3
        for row in rows:
            lines.append(f'<circle cx="{px(float(row["epoch"])):.2f}" cy="{py(float(row["fid"])):.2f}" r="{radius}" fill="white" stroke="{color}" stroke-width="3"/>')
    present = [(model, color, histories[model]) for model, color in SERIES if histories.get(model)]
    legend_x, legend_y = right - 445, top + 22
    legend_h = 26 + 44 * len(present)
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="425" height="{legend_h}" rx="12" fill="white" stroke="#cbd5e1" stroke-width="2"/>')
    for index, (model, color, rows) in enumerate(present):
        y = legend_y + 21 + index * 44
        latest = rows[-1]
        label = html.escape(f"{model} · latest {float(latest['fid']):.3f} @ {float(latest['epoch']):.1f} ep")
        lines.extend([
            f'<line x1="{legend_x+20}" y1="{y}" x2="{legend_x+68}" y2="{y}" stroke="{color}" stroke-width="5"/>',
            f'<circle cx="{legend_x+44}" cy="{y}" r="4" fill="white" stroke="{color}" stroke-width="2"/>',
            f'<text x="{legend_x+82}" y="{y+6}" font-family="DejaVu Sans" font-size="17" font-weight="700" fill="#1e293b">{label}</text>',
        ])
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def generate_plot(
    output_dir: str | os.PathLike[str],
    base_history: str | os.PathLike[str] = DEFAULT_BASE_HISTORY,
    rot_history: str | os.PathLike[str] = DEFAULT_ROT_HISTORY,
    conv_history: str | os.PathLike[str] = DEFAULT_CONV_HISTORY,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    histories = {
        "Base": _read_history(Path(base_history)),
        "Rotation-layer": _read_history(Path(rot_history)),
        "Conv-layer": _read_history(Path(conv_history)),
    }
    if not any(histories.values()):
        raise RuntimeError("No completed CFG=1 FID records were found")
    paths = {
        "csv": output / "fid_cfg1_50k_training_curves.csv",
        "png": output / "fid_cfg1_50k_training_curves.png",
        "svg": output / "fid_cfg1_50k_training_curves.svg",
    }
    _write_csv(paths["csv"], histories)
    _write_png(paths["png"], histories)
    _write_svg(paths["svg"], histories)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-history", default=str(DEFAULT_BASE_HISTORY))
    parser.add_argument("--rot-history", default=str(DEFAULT_ROT_HISTORY))
    parser.add_argument("--conv-history", default=str(DEFAULT_CONV_HISTORY))
    args = parser.parse_args()
    paths = generate_plot(
        args.output_dir, args.base_history, args.rot_history, args.conv_history
    )
    for kind, path in paths.items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    main()
