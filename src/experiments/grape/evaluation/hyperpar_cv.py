"""Tune GRAPE X-only VCTR hyperparameters by full held-out prediction CV."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.evaluation.compare_models import (  # noqa: E402
    grouped_kfold_indices,
    load_feature_dataset,
    predict_paired_vctr,
    subset_dataset,
)


GRAPE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = GRAPE_ROOT / "data" / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "hyperpar_cv"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "hyperpar_cv" / "x_only_grid_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--num-tasks", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def resolve_path(path: str | Path, *, base: Path = GRAPE_ROOT) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (base / path).resolve()


def rel_to_repo(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_feature_name(name: str) -> tuple[str, int]:
    if not name.startswith("S") or "_R" not in name:
        raise ValueError(f"Invalid feature package name: {name!r}")
    s_part, r_part = name[1:].split("_R", maxsplit=1)
    return s_part, int(r_part)


def positive_float_list(values: Any, name: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    parsed = [float(value) for value in values]
    if any(value <= 0 for value in parsed):
        raise ValueError(f"All {name} values must be positive.")
    return parsed


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if int(config["folds"]) < 2:
        raise ValueError("folds must be at least 2.")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative.")
    if str(config.get("split_group", "subject_id")) != "subject_id":
        raise ValueError("hyperpar CV requires split_group='subject_id'.")
    if str(config.get("z_mode", "none")) != "none":
        raise ValueError("hyperpar CV currently supports only z_mode='none'.")
    if float(config.get("ridge", 0.0)) < 0:
        raise ValueError("ridge must be non-negative.")
    positive_float_list(config["signal_h_candidates"], "signal_h_candidates")
    positive_float_list(config["variance_h_candidates"], "variance_h_candidates")
    if float(config.get("mape_eps_std", 1e-6)) < 0:
        raise ValueError("mape_eps_std must be non-negative.")
    if float(config.get("mape_eps_iop", 1e-6)) < 0:
        raise ValueError("mape_eps_iop must be non-negative.")
    return config


def discover_feature_packages(feature_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("feature_packages"):
        packages = []
        for item in config["feature_packages"]:
            image_type = str(item["image_type"])
            s_value = str(item["S"])
            r_value = int(item["R"])
            path = feature_root / f"{image_type}_192_iop_le35" / f"S{s_value}_R{r_value}"
            packages.append({"image_type": image_type, "S": s_value, "R": r_value, "feature_dir": path})
        return sorted(packages, key=lambda row: (row["image_type"], row["S"], row["R"]))

    image_types = [str(value) for value in config.get("image_types", ["cfp", "roi"])]
    packages: list[dict[str, Any]] = []
    for image_type in image_types:
        root = feature_root / f"{image_type}_192_iop_le35"
        for path in sorted(root.glob("S*_R*")):
            if not path.is_dir():
                continue
            s_value, r_value = parse_feature_name(path.name)
            packages.append({"image_type": image_type, "S": s_value, "R": r_value, "feature_dir": path})
    return sorted(packages, key=lambda row: (row["image_type"], row["S"], row["R"]))


def shard_packages(
    packages: list[dict[str, Any]],
    *,
    task_index: int | None,
    num_tasks: int,
) -> list[dict[str, Any]]:
    if num_tasks < 1:
        raise ValueError("--num-tasks must be positive.")
    if task_index is None:
        if num_tasks != 1:
            raise ValueError("--task-index is required when --num-tasks is greater than 1.")
        task_index = 1
    if task_index < 1 or task_index > num_tasks:
        raise ValueError("--task-index must be between 1 and --num-tasks.")
    return [row for idx, row in enumerate(packages) if idx % num_tasks == task_index - 1]


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_mean: float,
    y_sd: float,
    eps_std: float,
    eps_iop: float,
) -> dict[str, float]:
    resid_std = y_true - y_pred
    y_true_iop = y_true * y_sd + y_mean
    y_pred_iop = y_pred * y_sd + y_mean
    resid_iop = y_true_iop - y_pred_iop
    mape_std = np.abs(resid_std) / np.maximum(np.abs(y_true), eps_std)
    mape_iop = np.abs(resid_iop) / np.maximum(np.abs(y_true_iop), eps_iop)
    return {
        "rmse_std": float(np.sqrt(np.mean(np.square(resid_std)))),
        "mape_std_pct": float(100.0 * np.mean(mape_std)),
        "rmse_iop": float(np.sqrt(np.mean(np.square(resid_iop)))),
        "mape_iop_pct": float(100.0 * np.mean(mape_iop)),
    }


def validate_grouped_folds(manifest: pd.DataFrame, folds: list[np.ndarray], split_group: str) -> None:
    groups = manifest[split_group].to_numpy()
    n_rows = len(groups)
    seen = np.zeros(n_rows, dtype=int)
    all_indices = np.arange(n_rows)
    for fold_id, holdout_indices in enumerate(folds, start=1):
        seen[holdout_indices] += 1
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[holdout_indices] = False
        train_groups = set(groups[all_indices[train_mask]])
        holdout_groups = set(groups[holdout_indices])
        overlap = train_groups.intersection(holdout_groups)
        if overlap:
            example = sorted(str(value) for value in overlap)[:5]
            raise ValueError(f"Fold {fold_id} has split_group leakage for {split_group}: {example}")
    if not np.all(seen == 1):
        raise ValueError("Grouped folds must cover each row exactly once.")


def evaluate_candidate(
    *,
    dataset: Any,
    manifest: pd.DataFrame,
    folds: list[np.ndarray],
    image_type: str,
    s_value: str,
    r_value: int,
    signal_h: float,
    variance_hbar: float,
    config: dict[str, Any],
    y_mean: float,
    y_sd: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    image_config = {
        "image_type": image_type,
        "S": s_value,
        "R": int(r_value),
        "signal_h": float(signal_h),
        "variance_hbar": float(variance_hbar),
    }
    eps_std = float(config.get("mape_eps_std", 1e-6))
    eps_iop = float(config.get("mape_eps_iop", 1e-6))

    for fold_id, holdout_indices in enumerate(folds, start=1):
        train_mask = np.ones(dataset.n_subject, dtype=bool)
        train_mask[holdout_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train = subset_dataset(dataset, train_indices, z_mode="none")
        holdout = subset_dataset(dataset, holdout_indices, z_mode="none")
        pred = predict_paired_vctr(
            train,
            holdout,
            config=config,
            image_config=image_config,
            z_mode="none",
        )
        predictions.append(pred)
        truths.append(holdout.y)
        fold_metrics = metric_values(
            holdout.y,
            pred,
            y_mean=y_mean,
            y_sd=y_sd,
            eps_std=eps_std,
            eps_iop=eps_iop,
        )
        fold_rows.append(
            {
                "image_type": image_type,
                "S": s_value,
                "R": int(r_value),
                "signal_h": float(signal_h),
                "variance_hbar": float(variance_hbar),
                "fold": int(fold_id),
                "n_pairs": int(holdout.n_subject),
                **fold_metrics,
            }
        )

    truth_all = np.concatenate(truths, axis=0)
    pred_all = np.concatenate(predictions, axis=0)
    summary_metrics = metric_values(
        truth_all,
        pred_all,
        y_mean=y_mean,
        y_sd=y_sd,
        eps_std=eps_std,
        eps_iop=eps_iop,
    )
    summary = {
        "image_type": image_type,
        "S": s_value,
        "R": int(r_value),
        "signal_h": float(signal_h),
        "variance_hbar": float(variance_hbar),
        "split_group": str(config["split_group"]),
        "z_mode": str(config["z_mode"]),
        "n_pairs": int(dataset.n_subject),
        "n_split_groups": int(pd.Series(manifest[str(config["split_group"])]).nunique()),
        "ridge": float(config.get("ridge", 0.0)),
        **summary_metrics,
    }
    return summary, fold_rows


def write_frame(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def feature_output_paths(shard_dir: Path, package: dict[str, Any]) -> tuple[Path, Path]:
    stem = f"{package['image_type']}_S{package['S']}_R{package['R']}"
    return shard_dir / "features" / f"{stem}_summary.csv", shard_dir / "features" / f"{stem}_fold_metrics.csv"


def rebuild_shard_outputs(shard_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts = [pd.read_csv(path) for path in sorted((shard_dir / "features").glob("*_summary.csv"))]
    fold_parts = [pd.read_csv(path) for path in sorted((shard_dir / "features").glob("*_fold_metrics.csv"))]
    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    folds = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    if not summary.empty:
        summary = summary.sort_values(
            ["image_type", "rmse_std", "rmse_iop", "mape_std_pct", "mape_iop_pct", "S", "R", "signal_h", "variance_hbar"],
            kind="mergesort",
        )
    if not folds.empty:
        folds = folds.sort_values(
            ["image_type", "S", "R", "signal_h", "variance_hbar", "fold"],
            kind="mergesort",
        )
    summary.to_csv(shard_dir / "summary_all.csv", index=False)
    folds.to_csv(shard_dir / "fold_metrics.csv", index=False)
    return summary, folds


def write_run_readme(run_dir: Path, config: dict[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        f"# {config['name']}",
        "",
        "## Purpose",
        "",
        "Select X-only VCTR hyperparameters by full three-stage held-out prediction CV.",
        "",
        "## Selection Metric",
        "",
        "- Primary: `rmse_std` on subject-level grouped held-out folds.",
        "- Secondary: `rmse_iop`, `mape_std_pct`, `mape_iop_pct`.",
        f"- Ridge: `{float(config.get('ridge', 0.0)):.1e}` for numerical stabilization.",
        "",
        "## Best by Image",
        "",
        "| image_type | S | R | h | hbar | rmse_std | rmse_iop | mape_std_pct | mape_iop_pct |",
        "| :-- | :-- | --: | --: | --: | --: | --: | --: | --: |",
    ]
    if not summary.empty:
        best = summary.groupby("image_type", as_index=False).head(1)
        for _, row in best.iterrows():
            lines.append(
                "| "
                f"{row['image_type']} | `{row['S']}` | {int(row['R'])} | "
                f"{row['signal_h']:.6g} | {row['variance_hbar']:.6g} | "
                f"{row['rmse_std']:.6f} | {row['rmse_iop']:.6f} | "
                f"{row['mape_std_pct']:.6f} | {row['mape_iop_pct']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_all.csv`: all candidate-level metrics.",
            "- `fold_metrics.csv`: fold-level metrics for all candidates.",
            "- `summary_best_by_image.csv`: best candidate per image type.",
            "- `shard_*/`: shard-level outputs.",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_package(
    *,
    package: dict[str, Any],
    config: dict[str, Any],
    shard_dir: Path,
) -> tuple[int, int]:
    summary_path, fold_path = feature_output_paths(shard_dir, package)
    expected = len(config["signal_h_candidates"]) * len(config["variance_h_candidates"])
    if summary_path.exists() and fold_path.exists():
        existing = pd.read_csv(summary_path)
        if len(existing) == expected:
            return expected, 0

    dataset, manifest, meta = load_feature_dataset(package["feature_dir"])
    y_mean = float(meta["transforms"]["y"]["mean"])
    y_sd = float(meta["transforms"]["y"]["sd"])
    folds = grouped_kfold_indices(
        manifest[str(config["split_group"])].to_numpy(),
        int(config["seed"]),
        int(config["folds"]),
    )
    validate_grouped_folds(manifest, folds, str(config["split_group"]))
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    futures = []
    with ThreadPoolExecutor(max_workers=int(config.get("_max_workers", 1))) as executor:
        for h in config["signal_h_candidates"]:
            for hbar in config["variance_h_candidates"]:
                futures.append(
                    executor.submit(
                        evaluate_candidate,
                        dataset=dataset,
                        manifest=manifest,
                        folds=folds,
                        image_type=str(package["image_type"]),
                        s_value=str(package["S"]),
                        r_value=int(package["R"]),
                        signal_h=float(h),
                        variance_hbar=float(hbar),
                        config=config,
                        y_mean=y_mean,
                        y_sd=y_sd,
                    )
                )
        for future in as_completed(futures):
            summary, fold_metrics = future.result()
            summary_rows.append(summary)
            fold_rows.extend(fold_metrics)

    summary_rows.sort(
        key=lambda row: (
            row["image_type"],
            row["rmse_std"],
            row["rmse_iop"],
            row["mape_std_pct"],
            row["mape_iop_pct"],
            row["S"],
            row["R"],
            row["signal_h"],
            row["variance_hbar"],
        )
    )
    fold_rows.sort(
        key=lambda row: (
            row["image_type"],
            row["S"],
            row["R"],
            row["signal_h"],
            row["variance_hbar"],
            row["fold"],
        )
    )
    write_frame(summary_path, summary_rows)
    write_frame(fold_path, fold_rows)
    return len(summary_rows), len(fold_rows)


def aggregate_run(run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    summary_parts = [pd.read_csv(path) for path in sorted(run_dir.glob("shard_*/summary_all.csv"))]
    fold_parts = [pd.read_csv(path) for path in sorted(run_dir.glob("shard_*/fold_metrics.csv"))]
    if not summary_parts:
        raise FileNotFoundError(f"No shard summary files found under {run_dir}.")
    summary = pd.concat(summary_parts, ignore_index=True)
    folds = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    summary = summary.sort_values(
        ["image_type", "rmse_std", "rmse_iop", "mape_std_pct", "mape_iop_pct", "S", "R", "signal_h", "variance_hbar"],
        kind="mergesort",
    )
    if not folds.empty:
        folds = folds.sort_values(
            ["image_type", "S", "R", "signal_h", "variance_hbar", "fold"],
            kind="mergesort",
        )
    best = summary.groupby("image_type", as_index=False).head(1)
    summary.to_csv(run_dir / "summary_all.csv", index=False)
    folds.to_csv(run_dir / "fold_metrics.csv", index=False)
    best.to_csv(run_dir / "summary_best_by_image.csv", index=False)
    write_run_readme(run_dir, config, summary)
    return {
        "run_dir": rel_to_repo(run_dir),
        "n_candidates": int(len(summary)),
        "n_fold_rows": int(len(folds)),
        "summary_all": rel_to_repo(run_dir / "summary_all.csv"),
        "fold_metrics": rel_to_repo(run_dir / "fold_metrics.csv"),
        "summary_best_by_image": rel_to_repo(run_dir / "summary_best_by_image.csv"),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    run_name = args.run_name or str(config["name"])
    run_root = resolve_path(args.run_root)
    feature_root = resolve_path(args.feature_root)
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.json")

    if args.aggregate:
        payload = aggregate_run(run_dir, config)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    config["_max_workers"] = int(args.max_workers)
    packages_all = discover_feature_packages(feature_root, config)
    packages = shard_packages(packages_all, task_index=args.task_index, num_tasks=int(args.num_tasks))
    task_index = args.task_index or 1
    shard_dir = run_dir / f"shard_{int(task_index):02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / "packages.json").write_text(
        json.dumps([{**row, "feature_dir": rel_to_repo(row["feature_dir"])} for row in packages], indent=2) + "\n",
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    n_summary = 0
    n_fold = 0
    for idx, package in enumerate(packages, start=1):
        package_summary, package_folds = run_package(package=package, config=config, shard_dir=shard_dir)
        n_summary += package_summary
        n_fold += package_folds
        rebuild_shard_outputs(shard_dir)
        print(
            f"feature_done shard={task_index}/{args.num_tasks} "
            f"feature={idx}/{len(packages)} image_type={package['image_type']} "
            f"S={package['S']} R={package['R']} candidates={package_summary}",
            flush=True,
        )

    summary, folds = rebuild_shard_outputs(shard_dir)
    elapsed = time.perf_counter() - t0
    payload = {
        "run_dir": rel_to_repo(run_dir),
        "shard_dir": rel_to_repo(shard_dir),
        "task_index": int(task_index),
        "num_tasks": int(args.num_tasks),
        "n_features": int(len(packages)),
        "n_candidates": int(len(summary)),
        "n_fold_rows": int(len(folds)),
        "elapsed_seconds": float(elapsed),
        "summary_all": rel_to_repo(shard_dir / "summary_all.csv"),
        "fold_metrics": rel_to_repo(shard_dir / "fold_metrics.csv"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
