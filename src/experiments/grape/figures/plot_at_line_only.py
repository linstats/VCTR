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
    FEATURE_ROOT,
    RUN_ROOT,
    build_model,
    load_config,
    resolve_path,
    select_z_columns,
)
from src.experiments.grape.evaluation.compare_models import (  # noqa: E402
    feature_dir,
    load_feature_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--age-min", type=float, default=None)
    parser.add_argument("--age-max", type=float, default=None)
    return parser.parse_args()


def fit_independence_curves(
    config: dict[str, object],
    feature_root: Path,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Estimate stage-1 curves after treating the two eyes as independent."""

    package_dir = feature_dir(
        resolve_path(feature_root),
        str(config["image_type"]),
        str(config["S"]),
        int(config["R"]),
    )
    dataset_full, manifest, meta = load_feature_dataset(package_dir)
    all_z_names = [str(value) for value in meta["transforms"]["Z"]["columns"]]
    dataset, _ = select_z_columns(dataset_full, all_z_names, config, manifest)
    model = build_model(config)
    flat = dataset.to_iid_observations()
    A_independence, _ = model._estimate_stage1_A(  # noqa: SLF001
        flat_Z=flat.Z,
        flat_X=model._flatten_X(flat.X),  # noqa: SLF001
        flat_y=flat.y,
        flat_t=flat.t,
        t_eval=t_grid,
        p0=dataset.Z.shape[1],
        bandwidth=float(config["signal_h"]),
    )
    return A_independence.reshape((t_grid.size,) + dataset.X.shape[2:])


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

    t_grid = np.sort(summary["t"].unique().astype(float))
    A_independence = fit_independence_curves(config, args.feature_root, t_grid)
    if A_independence.shape != (t_grid.size, 1, len(blocks)):
        raise ValueError(
            "Unexpected independence-curve shape: "
            f"{A_independence.shape}; expected {(t_grid.size, 1, len(blocks))}."
        )

    age_min = float(summary["age"].min()) if args.age_min is None else float(args.age_min)
    age_max = float(summary["age"].max()) if args.age_max is None else float(args.age_max)
    if age_min >= age_max:
        raise ValueError("age-min must be smaller than age-max.")
    visible = summary[summary["age"].between(age_min, age_max)]
    if visible.empty:
        raise ValueError("Selected age range contains no coefficient estimates.")
    visible_t = t_grid[
        np.asarray(
            [
                age_min
                <= float(summary.loc[summary["t"] == value, "age"].iloc[0])
                <= age_max
                for value in t_grid
            ],
            dtype=bool,
        )
    ]
    t_mask = np.isin(t_grid, visible_t)
    max_abs = max(
        float(np.max(np.abs(visible["A_hat"].to_numpy(dtype=float)))),
        float(np.max(np.abs(A_independence[t_mask]))),
    )
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
        axis.plot(
            block["age"],
            A_independence[:, 0, block_index - 1],
            color="#6B7280",
            linewidth=1.5,
            linestyle="--",
            label="independence",
        )
        axis.plot(
            block["age"],
            block["A_hat"],
            color="#C2410C",
            linewidth=2.0,
            label="paired",
        )
        axis.axhline(0.0, color="0.4", linestyle="--", linewidth=0.75)
        axis.set_xlim(age_min, age_max)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_title(
            rf"$\hat{{a}}_{{1,{block_index}}}(t)$",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.14, linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        row = panel_index // args.cols
        col = panel_index % args.cols
        if row == args.rows - 1:
            axis.set_xlabel("Age (years)")
        if col == 0:
            axis.set_ylabel("Coefficient")
    for axis in axes_flat[len(blocks) :]:
        axis.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96), h_pad=1.2, w_pad=0.8)
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
        "curves": ["independence", "paired"],
        "png": png_path.name,
        "pdf": pdf_path.name,
    }
    metadata_path = figure_dir / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
