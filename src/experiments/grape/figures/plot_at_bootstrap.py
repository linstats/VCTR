"""Plot spatially arranged GRAPE A(t) curves with bootstrap intervals."""

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
    load_config,
    resolve_path,
)
from src.experiments.grape.evaluation.compare_models import feature_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Plot the saved full-sample A(t) curve without bootstrap intervals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    run_name = args.run_name or str(config["name"])
    run_dir = resolve_path(args.run_root) / run_name
    if args.original_only:
        original_path = run_dir / "original_fit.npz"
        if not original_path.exists():
            raise FileNotFoundError(f"Fit the full-sample curve first: {original_path}")
        with np.load(original_path) as original:
            age_grid = np.asarray(original["age_grid"], dtype=float)
            A_hat = np.asarray(original["A_hat"], dtype=float)
        rows: list[dict[str, float | int]] = []
        for grid_idx, age in enumerate(age_grid):
            for rank_idx in range(A_hat.shape[1]):
                for block_idx in range(A_hat.shape[2]):
                    rows.append(
                        {
                            "age": float(age),
                            "rank": rank_idx + 1,
                            "block": block_idx + 1,
                            "A_hat": float(A_hat[grid_idx, rank_idx, block_idx]),
                        }
                    )
        summary = pd.DataFrame(rows)
    else:
        summary_path = run_dir / "coefficient_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Aggregate bootstrap checkpoints first: {summary_path}")
        summary = pd.read_csv(summary_path)
    dims = tuple(int(value) for value in str(config["S"]).split("x"))
    if len(dims) != 3 or dims[2] != 1:
        raise ValueError("Current figure layout expects S1xS2x1.")
    n_rows, n_cols = dims[:2]

    package_dir = feature_dir(
        resolve_path(args.feature_root),
        str(config["image_type"]),
        str(config["S"]),
        int(config["R"]),
    )
    t_observed = np.load(package_dir / "t.npy")
    meta = json.loads((package_dir / "meta.json").read_text(encoding="utf-8"))
    age_meta = meta["transforms"]["t"]
    age_observed = float(age_meta["age_min"]) + t_observed * (
        float(age_meta["age_max"]) - float(age_meta["age_min"])
    )

    if args.original_only:
        y_min = min(float(summary["A_hat"].min()), 0.0)
        y_max = max(float(summary["A_hat"].max()), 0.0)
    else:
        y_min = float(summary["ci_lower_pointwise"].min())
        y_max = float(summary["ci_upper_pointwise"].max())
    y_pad = 0.05 * max(y_max - y_min, 1e-8)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.8 * n_cols, 2.25 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for block_idx in range(n_rows * n_cols):
        row = block_idx % n_rows
        col = block_idx // n_rows
        axis = axes[row, col]
        block = summary[(summary["block"] == block_idx + 1) & (summary["rank"] == 1)].sort_values("age")
        if not args.original_only:
            axis.fill_between(
                block["age"].to_numpy(),
                block["ci_lower_pointwise"].to_numpy(),
                block["ci_upper_pointwise"].to_numpy(),
                color="#4C78A8",
                alpha=0.24,
                linewidth=0,
            )
        axis.plot(block["age"], block["A_hat"], color="#1F4E79", linewidth=1.8)
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.8)
        axis.plot(
            age_observed,
            np.full_like(age_observed, y_min - 0.55 * y_pad),
            "|",
            color="0.5",
            alpha=0.12,
            markersize=3,
        )
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.set_title(f"Block {block_idx + 1}  (row {row + 1}, col {col + 1})", fontsize=9)
        axis.grid(alpha=0.18, linewidth=0.6)
        if row == n_rows - 1:
            axis.set_xlabel("Age (years)")
        if col == 0:
            axis.set_ylabel(r"$\hat{a}_{1s}(t)$")

    image_label = str(config["image_type"]).upper()
    model_label = str(config.get("model_label", "X-only paired VCTR"))
    if args.original_only:
        title = (
            f"{image_label} {model_label}: full-sample coefficient functions\n"
            f"Sensitivity fit; fixed S={config['S']}, R={config['R']}, "
            f"h={config['signal_h']}, hbar={config['variance_hbar']}"
        )
        stem = f"{str(config['image_type']).lower()}_at_full_sample"
    else:
        n_success = int(summary["n_success"].iloc[0])
        run_label = "Pilot" if "pilot" in str(config["name"]).lower() else "Final"
        title = (
            f"{image_label} {model_label}: coefficient functions with "
            f"{int(100 * float(config['confidence_level']))}% pointwise bootstrap intervals\n"
            f"{run_label} B={n_success}; fixed S={config['S']}, R={config['R']}, "
            f"h={config['signal_h']}, hbar={config['variance_hbar']}"
        )
        stem = f"{str(config['image_type']).lower()}_at_pointwise_ci"
    figure.suptitle(title, fontsize=12, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965), h_pad=1.0, w_pad=0.8)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{stem}.png", dpi=220)
    figure.savefig(figure_dir / f"{stem}.pdf")
    plt.close(figure)
    print(json.dumps({"figure_dir": str(figure_dir), "png": f"{stem}.png", "pdf": f"{stem}.pdf"}, indent=2))


if __name__ == "__main__":
    main()
