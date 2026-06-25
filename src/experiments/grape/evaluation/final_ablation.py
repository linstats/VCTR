"""Run final local GRAPE prediction ablation with fixed hyperparameters."""

from __future__ import annotations

import argparse
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
    feature_dir,
    fit_linear,
    grouped_kfold_indices,
    load_feature_dataset,
    predict_iid_vctr,
    predict_linear,
    predict_paired_vctr,
    subset_dataset,
)


GRAPE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = GRAPE_ROOT / "data" / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "final_ablation"
OUTPUT_ROOT = GRAPE_ROOT / "outputs" / "final_ablation"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "final_ablation" / "v1_full_cv_selected.json"

MODEL_NAMES = (
    "z_only_linear",
    "x_only_linear",
    "xz_linear",
    "x_only_iid_vctr",
    "x_only_paired_vctr",
    "xz_iid_vctr",
    "xz_paired_vctr",
)

CONTRASTS = (
    ("x_only_paired_vctr", "z_only_linear", "image_vctr_vs_z_only"),
    ("x_only_paired_vctr", "x_only_linear", "varying_coef_vs_x_linear"),
    ("x_only_paired_vctr", "x_only_iid_vctr", "paired_refit_vs_x_iid"),
    ("xz_paired_vctr", "xz_iid_vctr", "paired_refit_vs_xz_iid"),
    ("xz_paired_vctr", "x_only_paired_vctr", "adding_z_to_paired_vctr"),
    ("xz_linear", "x_only_linear", "adding_z_to_linear"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    if int(config["folds"]) < 2:
        raise ValueError("folds must be at least 2.")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative.")
    if float(config.get("ridge", 0.0)) < 0:
        raise ValueError("ridge must be non-negative.")
    if str(config.get("split_group", "subject_id")) != "subject_id":
        raise ValueError("final ablation requires split_group='subject_id'.")
    if float(config.get("mape_eps_std", 1e-6)) < 0:
        raise ValueError("mape_eps_std must be non-negative.")
    if float(config.get("mape_eps_iop", 1e-6)) < 0:
        raise ValueError("mape_eps_iop must be non-negative.")
    models = list(config.get("models", []))
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if not models:
        raise ValueError("config.models must be non-empty.")
    if not config.get("image_configs"):
        raise ValueError("config.image_configs must be non-empty.")
    return config


def validate_grouped_folds(manifest: pd.DataFrame, folds: list[np.ndarray], split_group: str) -> None:
    groups = manifest[split_group].to_numpy()
    seen = np.zeros(len(groups), dtype=int)
    all_indices = np.arange(len(groups))
    for fold_id, holdout_indices in enumerate(folds, start=1):
        seen[holdout_indices] += 1
        train_mask = np.ones(len(groups), dtype=bool)
        train_mask[holdout_indices] = False
        train_groups = set(groups[all_indices[train_mask]])
        holdout_groups = set(groups[holdout_indices])
        overlap = train_groups.intersection(holdout_groups)
        if overlap:
            example = sorted(str(value) for value in overlap)[:5]
            raise ValueError(f"Fold {fold_id} has split_group leakage for {split_group}: {example}")
    if not np.all(seen == 1):
        raise ValueError("Grouped folds must cover each row exactly once.")


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
    return {
        "rmse_std": float(np.sqrt(np.mean(np.square(resid_std)))),
        "mae_std": float(np.mean(np.abs(resid_std))),
        "mape_std_pct": float(100.0 * np.mean(np.abs(resid_std) / np.maximum(np.abs(y_true), eps_std))),
        "rmse_iop": float(np.sqrt(np.mean(np.square(resid_iop)))),
        "mae_iop": float(np.mean(np.abs(resid_iop))),
        "mape_iop_pct": float(100.0 * np.mean(np.abs(resid_iop) / np.maximum(np.abs(y_true_iop), eps_iop))),
    }


def prediction_rows(
    *,
    image_type: str,
    model_name: str,
    fold_id: int,
    manifest: pd.DataFrame,
    holdout_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_mean: float,
    y_sd: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    holdout_manifest = manifest.iloc[holdout_indices].reset_index(drop=True)
    for row_idx in range(y_true.shape[0]):
        for eye_idx, eye in enumerate(("OD", "OS")):
            true_std = float(y_true[row_idx, eye_idx])
            pred_std = float(y_pred[row_idx, eye_idx])
            resid_std = true_std - pred_std
            true_iop = float(true_std * y_sd + y_mean)
            pred_iop = float(pred_std * y_sd + y_mean)
            rows.append(
                {
                    "image_type": image_type,
                    "model": model_name,
                    "fold": int(fold_id),
                    "pair_id": holdout_manifest.loc[row_idx, "pair_id"],
                    "subject_id": holdout_manifest.loc[row_idx, "subject_id"],
                    "eye": eye,
                    "y_true_std": true_std,
                    "y_pred_std": pred_std,
                    "resid_std": resid_std,
                    "y_true_iop": true_iop,
                    "y_pred_iop": pred_iop,
                    "resid_iop": true_iop - pred_iop,
                }
            )
    return rows


def predict_model(
    model_name: str,
    *,
    train: Any,
    holdout: Any,
    config: dict[str, Any],
    image_config: dict[str, Any],
) -> np.ndarray:
    if model_name == "z_only_linear":
        coef = fit_linear(train, use_x=False, use_z=True)
        return predict_linear(holdout, coef, use_x=False, use_z=True)
    if model_name == "x_only_linear":
        coef = fit_linear(train, use_x=True, use_z=False)
        return predict_linear(holdout, coef, use_x=True, use_z=False)
    if model_name == "xz_linear":
        coef = fit_linear(train, use_x=True, use_z=True)
        return predict_linear(holdout, coef, use_x=True, use_z=True)
    if model_name == "x_only_iid_vctr":
        return predict_iid_vctr(train, holdout, config=config, image_config=image_config, z_mode="none")
    if model_name == "x_only_paired_vctr":
        return predict_paired_vctr(train, holdout, config=config, image_config=image_config, z_mode="none")
    if model_name == "xz_iid_vctr":
        return predict_iid_vctr(train, holdout, config=config, image_config=image_config, z_mode="full")
    if model_name == "xz_paired_vctr":
        return predict_paired_vctr(train, holdout, config=config, image_config=image_config, z_mode="full")
    raise ValueError(f"Unknown model: {model_name}")


def evaluate_one_image(
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    feature_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image_type = str(image_config["image_type"])
    s_value = str(image_config["S"])
    r_value = int(image_config["R"])
    package_dir = feature_dir(feature_root, image_type, s_value, r_value)
    dataset, manifest, meta = load_feature_dataset(package_dir)
    y_mean = float(meta["transforms"]["y"]["mean"])
    y_sd = float(meta["transforms"]["y"]["sd"])
    split_group = str(config["split_group"])
    folds = grouped_kfold_indices(manifest[split_group].to_numpy(), int(config["seed"]), int(config["folds"]))
    validate_grouped_folds(manifest, folds, split_group)
    eps_std = float(config.get("mape_eps_std", 1e-6))
    eps_iop = float(config.get("mape_eps_iop", 1e-6))

    model_names = list(config["models"])
    fold_metric_rows: list[dict[str, Any]] = []
    prediction_row_list: list[dict[str, Any]] = []
    all_predictions: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    all_truth: list[np.ndarray] = []

    for fold_id, holdout_indices in enumerate(folds, start=1):
        train_mask = np.ones(dataset.n_subject, dtype=bool)
        train_mask[holdout_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train = subset_dataset(dataset, train_indices, z_mode="full")
        holdout = subset_dataset(dataset, holdout_indices, z_mode="full")
        all_truth.append(holdout.y)

        for model_name in model_names:
            pred = predict_model(
                model_name,
                train=train,
                holdout=holdout,
                config=config,
                image_config=image_config,
            )
            all_predictions[model_name].append(pred)
            fold_metrics = metric_values(
                holdout.y,
                pred,
                y_mean=y_mean,
                y_sd=y_sd,
                eps_std=eps_std,
                eps_iop=eps_iop,
            )
            fold_metric_rows.append(
                {
                    "image_type": image_type,
                    "S": s_value,
                    "R": r_value,
                    "signal_h": float(image_config["signal_h"]),
                    "variance_hbar": float(image_config["variance_hbar"]),
                    "split_group": split_group,
                    "model": model_name,
                    "fold": int(fold_id),
                    "n_pairs": int(holdout.n_subject),
                    **fold_metrics,
                }
            )
            prediction_row_list.extend(
                prediction_rows(
                    image_type=image_type,
                    model_name=model_name,
                    fold_id=fold_id,
                    manifest=manifest,
                    holdout_indices=holdout_indices,
                    y_true=holdout.y,
                    y_pred=pred,
                    y_mean=y_mean,
                    y_sd=y_sd,
                )
            )

    truth = np.concatenate(all_truth, axis=0)
    summary_rows: list[dict[str, Any]] = []
    for model_name in model_names:
        pred = np.concatenate(all_predictions[model_name], axis=0)
        summary_metrics = metric_values(
            truth,
            pred,
            y_mean=y_mean,
            y_sd=y_sd,
            eps_std=eps_std,
            eps_iop=eps_iop,
        )
        summary_rows.append(
            {
                "image_type": image_type,
                "S": s_value,
                "R": r_value,
                "signal_h": float(image_config["signal_h"]),
                "variance_hbar": float(image_config["variance_hbar"]),
                "split_group": split_group,
                "model": model_name,
                "n_pairs": int(dataset.n_subject),
                "n_split_groups": int(pd.Series(manifest[split_group]).nunique()),
                "ridge": float(config.get("ridge", 0.0)),
                **summary_metrics,
            }
        )
    return summary_rows, fold_metric_rows, prediction_row_list


def build_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_type, image_df in summary.groupby("image_type", sort=True):
        by_model = image_df.set_index("model")
        for model, reference, contrast in CONTRASTS:
            if model not in by_model.index or reference not in by_model.index:
                continue
            row = by_model.loc[model]
            ref = by_model.loc[reference]
            delta_rmse_iop = float(row["rmse_iop"] - ref["rmse_iop"])
            delta_rmse_std = float(row["rmse_std"] - ref["rmse_std"])
            rows.append(
                {
                    "image_type": image_type,
                    "contrast": contrast,
                    "model": model,
                    "reference_model": reference,
                    "model_rmse_iop": float(row["rmse_iop"]),
                    "reference_rmse_iop": float(ref["rmse_iop"]),
                    "delta_rmse_iop": delta_rmse_iop,
                    "delta_rmse_std": delta_rmse_std,
                    "pct_delta_rmse_iop": float(100.0 * delta_rmse_iop / ref["rmse_iop"]),
                    "model_mae_iop": float(row["mae_iop"]),
                    "reference_mae_iop": float(ref["mae_iop"]),
                }
            )
    return pd.DataFrame(rows)


def eye_residual_corr(predictions: pd.DataFrame, value_col: str) -> float:
    pivot = predictions.pivot_table(
        index=["fold", "pair_id"],
        columns="eye",
        values=value_col,
        aggfunc="first",
    )
    if "OD" not in pivot.columns or "OS" not in pivot.columns:
        return float("nan")
    paired = pivot[["OD", "OS"]].dropna()
    if len(paired) < 2:
        return float("nan")
    return float(paired["OD"].corr(paired["OS"]))


def build_residual_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (image_type, model), group in predictions.groupby(["image_type", "model"], sort=True):
        abs_std = group["resid_std"].abs()
        abs_iop = group["resid_iop"].abs()
        rows.append(
            {
                "image_type": image_type,
                "model": model,
                "n_predictions": int(len(group)),
                "resid_std_mean": float(group["resid_std"].mean()),
                "resid_std_sd": float(group["resid_std"].std(ddof=1)),
                "abs_resid_std_mean": float(abs_std.mean()),
                "abs_resid_std_median": float(abs_std.median()),
                "abs_resid_std_q90": float(abs_std.quantile(0.9)),
                "resid_iop_mean": float(group["resid_iop"].mean()),
                "resid_iop_sd": float(group["resid_iop"].std(ddof=1)),
                "abs_resid_iop_mean": float(abs_iop.mean()),
                "abs_resid_iop_median": float(abs_iop.median()),
                "abs_resid_iop_q90": float(abs_iop.quantile(0.9)),
                "od_os_resid_std_corr": eye_residual_corr(group, "resid_std"),
                "od_os_resid_iop_corr": eye_residual_corr(group, "resid_iop"),
            }
        )
    return pd.DataFrame(rows)


def write_run_readme(run_dir: Path, config: dict[str, Any], summary: pd.DataFrame, ablation: pd.DataFrame) -> None:
    best = summary.sort_values(["image_type", "rmse_iop", "rmse_std", "model"], kind="mergesort").groupby(
        "image_type",
        as_index=False,
    ).head(1)
    lines = [
        f"# {config['name']}",
        "",
        "## Purpose",
        "",
        "Final local GRAPE ablation using full three-stage hyperparameter CV selected X-only configurations.",
        "",
        "## Fixed Configurations",
        "",
        "| image_type | S | R | h | hbar |",
        "| :-- | :-- | --: | --: | --: |",
    ]
    for image_config in config["image_configs"]:
        lines.append(
            f"| {str(image_config['image_type']).upper()} | `{image_config['S']}` | {int(image_config['R'])} | "
            f"{float(image_config['signal_h']):.6g} | {float(image_config['variance_hbar']):.6g} |"
        )
    lines.extend(
        [
            "",
            "## Settings",
            "",
            f"- Split: `{config['split_group']}` grouped {int(config['folds'])}-fold CV",
            f"- `a_eval_mode`: `{config.get('a_eval_mode', 'full')}`",
            f"- `ridge`: `{float(config.get('ridge', 0.0)):.1e}` for numerical stabilization",
            "",
            "## Best Model By Image",
            "",
            "| image_type | best_model | rmse_iop | mae_iop | rmse_std |",
            "| :-- | :-- | --: | --: | --: |",
        ]
    )
    for _, row in best.iterrows():
        lines.append(
            f"| {row['image_type']} | `{row['model']}` | {row['rmse_iop']:.6f} | "
            f"{row['mae_iop']:.6f} | {row['rmse_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Model Ordering",
            "",
            "| image_type | model | rmse_iop | mae_iop | mape_iop_pct |",
            "| :-- | :-- | --: | --: | --: |",
        ]
    )
    ordered = summary.sort_values(["image_type", "rmse_iop", "rmse_std", "model"], kind="mergesort")
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['image_type']} | `{row['model']}` | {row['rmse_iop']:.6f} | "
            f"{row['mae_iop']:.6f} | {row['mape_iop_pct']:.6f} |"
        )
    if not ablation.empty:
        lines.extend(
            [
                "",
                "## Key Contrasts",
                "",
                "`delta_rmse_iop < 0` means the model improves over the reference.",
                "",
                "| image_type | contrast | model | reference | delta_rmse_iop | pct_delta_rmse_iop |",
                "| :-- | :-- | :-- | :-- | --: | --: |",
            ]
        )
        for _, row in ablation.iterrows():
            lines.append(
                f"| {row['image_type']} | `{row['contrast']}` | `{row['model']}` | "
                f"`{row['reference_model']}` | {row['delta_rmse_iop']:.6f} | "
                f"{row['pct_delta_rmse_iop']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `summary_metrics.csv`: model-level held-out metrics",
            "- `fold_metrics.csv`: fold-level held-out metrics",
            "- `predictions.csv`: held-out predictions and residuals",
            "- `ablation_table.csv`: manuscript-facing model contrasts",
            "- `residual_summary.csv`: residual diagnostics summary",
            "- `config.json`: copied experiment config",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_output_readme(output_dir: Path, run_name: str, summary: pd.DataFrame) -> None:
    best = summary.sort_values(["image_type", "rmse_iop", "rmse_std", "model"], kind="mergesort").groupby(
        "image_type",
        as_index=False,
    ).head(1)
    lines = [
        "# Final ablation outputs",
        "",
        "Curated outputs for the final local GRAPE prediction ablation.",
        "",
        f"Source run: `src/experiments/grape/runs/final_ablation/{run_name}/`",
        "",
        "## Best Model By Image",
        "",
        "| image_type | best_model | rmse_iop | mae_iop | rmse_std |",
        "| :-- | :-- | --: | --: | --: |",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['image_type']} | `{row['model']}` | {row['rmse_iop']:.6f} | "
            f"{row['mae_iop']:.6f} | {row['rmse_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{run_name}_summary_metrics.csv`: model-level held-out metrics.",
            f"- `{run_name}_ablation_table.csv`: manuscript-facing model contrasts.",
            f"- `{run_name}_residual_summary.csv`: residual diagnostics summary.",
            "",
            "`predictions.csv` and fold-level details remain in the run directory.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_outputs(
    *,
    output_dir: Path,
    run_name: str,
    summary: pd.DataFrame,
    ablation: pd.DataFrame,
    residual_summary: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / f"{run_name}_summary_metrics.csv", index=False)
    ablation.to_csv(output_dir / f"{run_name}_ablation_table.csv", index=False)
    residual_summary.to_csv(output_dir / f"{run_name}_residual_summary.csv", index=False)
    write_output_readme(output_dir, run_name, summary)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    run_name = args.run_name or str(config["name"])
    feature_root = resolve_path(args.feature_root)
    run_root = resolve_path(args.run_root)
    output_root = resolve_path(args.output_root)
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.json")

    t0 = time.perf_counter()
    summary_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []
    for image_config in config["image_configs"]:
        rows, folds, preds = evaluate_one_image(
            config=config,
            image_config=image_config,
            feature_root=feature_root,
        )
        summary_rows.extend(rows)
        fold_rows.extend(folds)
        pred_rows.extend(preds)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["image_type", "rmse_iop", "rmse_std", "model"],
        kind="mergesort",
    )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["image_type", "model", "fold"],
        kind="mergesort",
    )
    predictions = pd.DataFrame(pred_rows).sort_values(
        ["image_type", "model", "fold", "pair_id", "eye"],
        kind="mergesort",
    )
    ablation = build_ablation_table(summary).sort_values(["image_type", "contrast"], kind="mergesort")
    residual_summary = build_residual_summary(predictions).sort_values(["image_type", "model"], kind="mergesort")

    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    ablation.to_csv(run_dir / "ablation_table.csv", index=False)
    residual_summary.to_csv(run_dir / "residual_summary.csv", index=False)
    write_run_readme(run_dir, config, summary, ablation)
    export_outputs(
        output_dir=output_root,
        run_name=run_name,
        summary=summary,
        ablation=ablation,
        residual_summary=residual_summary,
    )

    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "run_dir": rel_to_repo(run_dir),
                "output_dir": rel_to_repo(output_root),
                "summary_metrics": rel_to_repo(run_dir / "summary_metrics.csv"),
                "fold_metrics": rel_to_repo(run_dir / "fold_metrics.csv"),
                "predictions": rel_to_repo(run_dir / "predictions.csv"),
                "ablation_table": rel_to_repo(run_dir / "ablation_table.csv"),
                "residual_summary": rel_to_repo(run_dir / "residual_summary.csv"),
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
