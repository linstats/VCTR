"""Plot a GRAPE sigma(t) curve with optional pointwise bootstrap intervals."""

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
    parser.add_argument("--original-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    run_name = args.run_name or str(config["name"])
    run_dir = resolve_path(args.run_root) / run_name

    if args.original_only:
        path = run_dir / "original_fit.npz"
        if not path.exists():
            raise FileNotFoundError(f"Fit the full-sample curve first: {path}")
        with np.load(path) as original:
            age = np.asarray(original["age_grid"], dtype=float)
            sigma = np.sqrt(np.asarray(original["sigma2_hat_grid"], dtype=float))
            y_sd = float(original["y_sd"])
            support = np.asarray(original["local_support_pairs"], dtype=int)
            variance_support = (
                np.asarray(original["variance_support_pairs"], dtype=int)
                if "variance_support_pairs" in original.files
                else support
            )
        frame = pd.DataFrame(
            {
                "age": age,
                "sigma_hat_iop": sigma * y_sd,
                "local_support_pairs": support,
                "variance_support_pairs": variance_support,
            }
        )
    else:
        path = run_dir / "variance_summary.csv"
        if not path.exists():
            raise FileNotFoundError(f"Aggregate bootstrap checkpoints first: {path}")
        frame = pd.read_csv(path)

    min_support = int(config.get("min_variance_support_pairs", config.get("min_local_support_pairs", 0)))
    support_column = "variance_support_pairs" if "variance_support_pairs" in frame else "local_support_pairs"
    support_ok = frame[support_column].to_numpy(dtype=int) >= min_support
    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    if np.any(~support_ok):
        axis.fill_between(
            frame["age"].to_numpy(),
            0.0,
            1.0,
            where=~support_ok,
            color="0.9",
            alpha=0.8,
            transform=axis.get_xaxis_transform(),
        )
    if not args.original_only:
        axis.fill_between(
            frame["age"],
            frame["sigma_ci_lower_pointwise_iop"],
            frame["sigma_ci_upper_pointwise_iop"],
            where=support_ok,
            color="#F58518",
            alpha=0.25,
            linewidth=0,
        )
    axis.plot(frame["age"], frame["sigma_hat_iop"], color="#B45309", linewidth=2.0)

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
    axis.plot(age_observed, np.zeros_like(age_observed), "|", color="0.45", alpha=0.15, markersize=5)
    axis.set_ylim(bottom=0.0)
    axis.set_xlabel("Age (years)")
    axis.set_ylabel(r"Estimated $\sigma(t)$ (IOP units)")
    axis.grid(alpha=0.2, linewidth=0.6)
    image_label = str(config["image_type"]).upper()
    if args.original_only:
        title = f"{image_label} full-sample sigma curve; h={config['signal_h']}, hbar={config['variance_hbar']}"
        stem = f"{str(config['image_type']).lower()}_sigma_full_sample"
    else:
        n_success = int(frame["n_success"].iloc[0])
        title = f"{image_label} sigma(t) with 95% pointwise bootstrap intervals; pilot B={n_success}"
        stem = f"{str(config['image_type']).lower()}_sigma_pointwise_ci"
    axis.set_title(title)
    figure.tight_layout()
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_dir / f"{stem}.png", dpi=220)
    figure.savefig(figure_dir / f"{stem}.pdf")
    plt.close(figure)
    print(json.dumps({"figure_dir": str(figure_dir), "png": f"{stem}.png", "pdf": f"{stem}.pdf"}, indent=2))


if __name__ == "__main__":
    main()
