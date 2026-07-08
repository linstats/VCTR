"""Plot GRAPE A(t) point estimates without bootstrap confidence bands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

GRAPE_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(GRAPE_ROOT / "runs" / ".matplotlib"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.diagnostics.bootstrap_coefficients import (  # noqa: E402
    DEFAULT_CONFIG,
    RUN_ROOT,
    load_config,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--age-min", type=float, default=None)
    parser.add_argument("--age-max", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must be positive.")
    config = load_config(resolve_path(args.config))
    run_name = args.run_name or str(config["name"])
    run_dir = resolve_path(args.run_root) / run_name
    summary_path = run_dir / "coefficient_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Aggregate bootstrap output first: {summary_path}")
    summary = pd.read_csv(summary_path)
    summary = summary[summary["rank"] == 1].copy()
    blocks = sorted(int(value) for value in summary["block"].unique())
    if len(blocks) > args.rows * args.cols:
        raise ValueError(f"Layout {args.rows}x{args.cols} cannot hold {len(blocks)} blocks.")

    age_min = float(summary["age"].min()) if args.age_min is None else float(args.age_min)
    age_max = float(summary["age"].max()) if args.age_max is None else float(args.age_max)
    if age_min >= age_max:
        raise ValueError("age-min must be smaller than age-max.")
    visible = summary[summary["age"].between(age_min, age_max)]
    if visible.empty:
        raise ValueError("Selected age range contains no coefficient estimates.")
    max_abs = float(np.max(np.abs(visible["A_hat"].to_numpy(dtype=float))))
    y_limit = max(0.5, float(np.ceil((1.12 * max_abs) * 10.0) / 10.0))

    figure, axes = plt.subplots(
        args.rows,
        args.cols,
        figsize=(12.0, 10.5),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.asarray(axes, dtype=object).reshape(-1)
    for panel_index, block_index in enumerate(blocks):
        axis = axes_flat[panel_index]
        block = summary[summary["block"] == block_index].sort_values("age")
        axis.plot(block["age"], block["A_hat"], color="#1F4E79", linewidth=2.35)
        axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.75)
        axis.set_xlim(age_min, age_max)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_title(f"Block {block_index}", fontsize=10)
        axis.grid(axis="y", alpha=0.14, linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        row = panel_index // args.cols
        col = panel_index % args.cols
        if row == args.rows - 1:
            axis.set_xlabel("Age (years)")
        if col == 0:
            axis.set_ylabel(r"$\hat{a}_{1s}(t)$")
    for axis in axes_flat[len(blocks) :]:
        axis.set_visible(False)

    image_label = str(config["image_type"]).upper()
    model_label = str(config.get("model_label", "paired VCTR"))
    figure.suptitle(f"{image_label} {model_label}: coefficient functions", fontsize=15, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975), h_pad=1.2, w_pad=0.8)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{str(config['image_type']).lower()}_at_line_only_{args.rows}x{args.cols}"
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    figure.savefig(png_path, dpi=240, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "source": str(summary_path),
        "layout": [int(args.rows), int(args.cols)],
        "block_order": blocks,
        "age_range": [age_min, age_max],
        "y_range": [-y_limit, y_limit],
        "confidence_band": False,
        "png": png_path.name,
        "pdf": pdf_path.name,
    }
    metadata_path = figure_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
