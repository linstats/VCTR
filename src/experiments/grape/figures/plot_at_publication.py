"""Create publication-style GRAPE A(t) main and full-range supplement figures."""

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
    return parser.parse_args()


def observed_ages(config: dict[str, object], feature_root: Path) -> np.ndarray:
    package_dir = feature_dir(
        feature_root,
        str(config["image_type"]),
        str(config["S"]),
        int(config["R"]),
    )
    t = np.load(package_dir / "t.npy")
    meta = json.loads((package_dir / "meta.json").read_text(encoding="utf-8"))
    age_meta = meta["transforms"]["t"]
    return float(age_meta["age_min"]) + t * (
        float(age_meta["age_max"]) - float(age_meta["age_min"])
    )


def publication_settings(config: dict[str, object]) -> dict[str, float]:
    settings = {
        "main_age_min": 20.0,
        "main_age_max": 68.0,
        "main_y_abs": 2.5,
        **dict(config.get("publication_figure", {})),
    }
    if float(settings["main_age_min"]) >= float(settings["main_age_max"]):
        raise ValueError("main_age_min must be smaller than main_age_max.")
    if float(settings["main_y_abs"]) <= 0:
        raise ValueError("main_y_abs must be positive.")
    return {name: float(value) for name, value in settings.items()}


def draw_figure(
    summary: pd.DataFrame,
    ages: np.ndarray,
    config: dict[str, object],
    *,
    age_min: float,
    age_max: float,
    y_abs: float,
    title: str,
    output_stem: Path,
) -> None:
    dims = tuple(int(value) for value in str(config["S"]).split("x"))
    if len(dims) != 3 or dims[2] != 1:
        raise ValueError("Publication layout expects S1xS2x1.")
    n_rows, n_cols = dims[:2]
    figure = plt.figure(figsize=(11.2, 15.5), constrained_layout=False)
    grid = figure.add_gridspec(
        n_rows + 1,
        n_cols,
        height_ratios=[1.0] * n_rows + [0.28],
        hspace=0.34,
        wspace=0.14,
        left=0.075,
        right=0.985,
        bottom=0.055,
        top=0.925,
    )
    axes = np.empty((n_rows, n_cols), dtype=object)
    for row in range(n_rows):
        for col in range(n_cols):
            shared = axes[0, 0] if row != 0 or col != 0 else None
            axes[row, col] = figure.add_subplot(grid[row, col], sharex=shared, sharey=shared)

    for block_index in range(n_rows * n_cols):
        row = block_index % n_rows
        col = block_index // n_rows
        axis = axes[row, col]
        block = summary[(summary["block"] == block_index + 1) & (summary["rank"] == 1)].sort_values("age")
        support = (
            block["support_ok"].astype(bool).to_numpy()
            if "support_ok" in block
            else np.ones(len(block), dtype=bool)
        )
        axis.fill_between(
            block["age"].to_numpy(),
            block["ci_lower_pointwise"].to_numpy(),
            block["ci_upper_pointwise"].to_numpy(),
            where=support,
            color="#4C78A8",
            alpha=0.14,
            linewidth=0,
        )
        axis.plot(block["age"], block["A_hat"], color="#1F4E79", linewidth=2.2)
        axis.axhline(0.0, color="0.35", linestyle="--", linewidth=0.75)
        axis.set_xlim(age_min, age_max)
        axis.set_ylim(-y_abs, y_abs)
        axis.set_title(f"Block {block_index + 1}", fontsize=10, pad=4)
        axis.grid(axis="y", alpha=0.13, linewidth=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(labelsize=8)
        if row < n_rows - 1:
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("Age (years)", fontsize=9)
        if col == 0:
            axis.set_ylabel(r"$\hat{a}_{1s}(t)$", fontsize=10)
        else:
            axis.tick_params(labelleft=False)

    density_axis = figure.add_subplot(grid[n_rows, :], sharex=axes[0, 0])
    shown_ages = ages[(ages >= age_min) & (ages <= age_max)]
    bins = np.linspace(age_min, age_max, num=25)
    density_axis.hist(
        shown_ages,
        bins=bins,
        color="#7A8DA6",
        alpha=0.72,
        edgecolor="white",
        linewidth=0.35,
    )
    density_axis.set_xlim(age_min, age_max)
    density_axis.set_ylabel("Pairs", fontsize=8)
    density_axis.set_xlabel("Observed age distribution", fontsize=9)
    density_axis.grid(axis="y", alpha=0.12, linewidth=0.5)
    density_axis.spines["top"].set_visible(False)
    density_axis.spines["right"].set_visible(False)
    density_axis.tick_params(labelsize=8)
    density_axis.text(
        0.995,
        0.88,
        f"n={shown_ages.size} paired visits",
        transform=density_axis.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="0.35",
    )

    figure.suptitle(title, fontsize=15, y=0.982)
    figure.savefig(output_stem.with_suffix(".png"), dpi=240, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    settings = publication_settings(config)
    run_name = args.run_name or str(config["name"])
    run_dir = resolve_path(args.run_root) / run_name
    summary_path = run_dir / "coefficient_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Aggregate bootstrap checkpoints first: {summary_path}")
    summary = pd.read_csv(summary_path)
    ages = observed_ages(config, resolve_path(args.feature_root))
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    title = "ROI coefficient functions: X + VF-PC1 + gender"
    main_stem = figure_dir / "roi_at_publication_main_age20_68"
    draw_figure(
        summary,
        ages,
        config,
        age_min=settings["main_age_min"],
        age_max=settings["main_age_max"],
        y_abs=settings["main_y_abs"],
        title=title,
        output_stem=main_stem,
    )

    full_age_min = float(np.min(ages))
    full_age_max = float(np.max(ages))
    full_ci_abs = float(
        max(
            abs(float(summary["ci_lower_pointwise"].min())),
            abs(float(summary["ci_upper_pointwise"].max())),
        )
    )
    full_y_abs = float(np.ceil((full_ci_abs + 0.1) * 2.0) / 2.0)
    supplement_stem = figure_dir / "roi_at_publication_supplement_full_age"
    draw_figure(
        summary,
        ages,
        config,
        age_min=full_age_min,
        age_max=full_age_max,
        y_abs=full_y_abs,
        title=f"{title}\nFull observed age range",
        output_stem=supplement_stem,
    )

    metadata = {
        "run_dir": str(run_dir),
        "figure_dir": str(figure_dir),
        "main": {
            "age_range": [settings["main_age_min"], settings["main_age_max"]],
            "y_range": [-settings["main_y_abs"], settings["main_y_abs"]],
            "png": main_stem.with_suffix(".png").name,
            "pdf": main_stem.with_suffix(".pdf").name,
        },
        "supplement": {
            "age_range": [full_age_min, full_age_max],
            "y_range": [-full_y_abs, full_y_abs],
            "png": supplement_stem.with_suffix(".png").name,
            "pdf": supplement_stem.with_suffix(".pdf").name,
        },
    }
    (figure_dir / "publication_figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
