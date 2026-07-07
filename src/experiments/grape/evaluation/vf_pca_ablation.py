"""Evaluate incremental prediction from fold-local PCA of GRAPE VF covariates."""

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

from src.data import PairedEyeDataset  # noqa: E402
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
from src.experiments.grape.evaluation.final_ablation import (  # noqa: E402
    metric_values,
    prediction_rows,
    validate_grouped_folds,
)
from src.experiments.grape.evaluation.vf_pca import (  # noqa: E402
    FoldVFPCATransformer,
    compose_pca_covariates,
    split_sex_vf,
)


GRAPE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = GRAPE_ROOT / "data" / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "vf_pca"
OUTPUT_ROOT = GRAPE_ROOT / "outputs" / "vf_pca"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "vf_pca" / "v1_fixed_x_tuning.json"

MODEL_NAMES = (
    "y_bar",
    "x_only_paired_vctr",
    "x_sex_paired_vctr",
    "vf_pca_only_linear",
    "sex_vf_pca_only_linear",
    "x_vf_pca_linear",
    "x_sex_vf_pca_linear",
    "x_vf_pca_iid_vctr",
    "x_sex_vf_pca_iid_vctr",
    "x_vf_pca_paired_vctr",
    "x_sex_vf_pca_paired_vctr",
    "x_full60_paired_vctr",
)

CONTRASTS = (
    ("x_only_paired_vctr", "y_bar", "x_only_vs_y_bar"),
    ("x_vf_pca_paired_vctr", "y_bar", "x_vf_pca_vs_y_bar"),
    ("x_sex_vf_pca_paired_vctr", "y_bar", "x_sex_vf_pca_vs_y_bar"),
    ("x_full60_paired_vctr", "y_bar", "x_full60_vs_y_bar"),
    ("x_sex_paired_vctr", "x_only_paired_vctr", "adding_sex_to_x"),
    ("x_vf_pca_paired_vctr", "x_only_paired_vctr", "adding_vf_pca_to_x"),
    ("x_sex_vf_pca_paired_vctr", "x_sex_paired_vctr", "adding_vf_pca_to_x_sex"),
    ("x_sex_vf_pca_paired_vctr", "x_vf_pca_paired_vctr", "adding_sex_to_x_vf_pca"),
    ("x_vf_pca_paired_vctr", "x_vf_pca_iid_vctr", "paired_refit_for_x_vf_pca"),
    ("x_sex_vf_pca_paired_vctr", "x_sex_vf_pca_iid_vctr", "paired_refit_for_x_sex_vf_pca"),
    ("x_vf_pca_paired_vctr", "x_vf_pca_linear", "varying_paired_vs_linear_x_vf_pca"),
    ("x_sex_vf_pca_paired_vctr", "x_sex_vf_pca_linear", "varying_paired_vs_linear_x_sex_vf_pca"),
    ("x_full60_paired_vctr", "x_only_paired_vctr", "adding_full60_to_x"),
    ("x_vf_pca_paired_vctr", "x_full60_paired_vctr", "vf_pca_vs_full60"),
    ("x_sex_vf_pca_paired_vctr", "x_full60_paired_vctr", "sex_vf_pca_vs_full60"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the first image with 2 outer folds, 2 inner folds, and the first two K candidates.",
    )
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
    if int(config["outer_folds"]) < 2 or int(config["inner_folds"]) < 2:
        raise ValueError("outer_folds and inner_folds must both be at least 2.")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be non-negative.")
    if str(config.get("split_group", "subject_id")) != "subject_id":
        raise ValueError("VF PCA evaluation requires split_group='subject_id'.")
    if str(config.get("pca_weighting", "subject_equal")) not in {"subject_equal", "row_equal"}:
        raise ValueError("pca_weighting must be 'subject_equal' or 'row_equal'.")
    candidates = [int(value) for value in config.get("pca_components", [])]
    if not candidates or any(value < 1 for value in candidates) or len(candidates) != len(set(candidates)):
        raise ValueError("pca_components must contain unique positive integers.")
    models = list(config.get("models", []))
    unknown = sorted(set(models).difference(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    if not models:
        raise ValueError("config.models must be non-empty.")
    selection_model = str(config.get("selection_model", "x_sex_vf_pca_paired_vctr"))
    if selection_model not in {"x_vf_pca_paired_vctr", "x_sex_vf_pca_paired_vctr"}:
        raise ValueError("selection_model must be an X+VF-PCA paired VCTR model.")
    if not config.get("image_configs"):
        raise ValueError("config.image_configs must be non-empty.")
    if float(config.get("ridge", 0.0)) < 0:
        raise ValueError("ridge must be non-negative.")
    return config


def smoke_config(config: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    result["name"] = f"{config['name']}_smoke"
    result["outer_folds"] = 2
    result["inner_folds"] = 2
    result["pca_components"] = list(config["pca_components"])[:2]
    result["image_configs"] = list(config["image_configs"])[:1]
    result["notes"] = list(config.get("notes", [])) + ["Smoke override; not for scientific interpretation."]
    return result


def replace_z(dataset: PairedEyeDataset, Z: np.ndarray) -> PairedEyeDataset:
    """Return a dataset sharing X/y/t but using a supplied covariate matrix."""

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] != dataset.n_subject:
        raise ValueError("Replacement Z must have shape (n_subject, p0).")
    return PairedEyeDataset(
        subject_ids=np.asarray(dataset.subject_ids).copy(),
        eye_ids=np.asarray(dataset.eye_ids).copy(),
        t=np.asarray(dataset.t).copy(),
        X=np.asarray(dataset.X).copy(),
        Z=Z,
        y=np.asarray(dataset.y).copy(),
        meta=dict(dataset.meta),
    )


def pca_dataset_variants(
    train: PairedEyeDataset,
    holdout: PairedEyeDataset,
    *,
    train_groups: np.ndarray,
    n_components: int,
    weighting: str,
) -> tuple[dict[str, PairedEyeDataset], dict[str, PairedEyeDataset], FoldVFPCATransformer]:
    train_sex, train_vf = split_sex_vf(train.Z)
    holdout_sex, holdout_vf = split_sex_vf(holdout.Z)
    transformer = FoldVFPCATransformer.fit(
        train_vf,
        train_groups,
        n_components=n_components,
        weighting=weighting,
    )
    train_scores = transformer.transform(train_vf)
    holdout_scores = transformer.transform(holdout_vf)
    train_vf_standardized = transformer.standardize(train_vf)
    holdout_vf_standardized = transformer.standardize(holdout_vf)
    train_variants = {
        "none": replace_z(train, np.empty((train.n_subject, 0), dtype=float)),
        "sex": replace_z(train, train_sex),
        "vf_pca": replace_z(
            train,
            compose_pca_covariates(sex=train_sex, scores=train_scores, include_sex=False),
        ),
        "sex_vf_pca": replace_z(
            train,
            compose_pca_covariates(sex=train_sex, scores=train_scores, include_sex=True),
        ),
        "full60": replace_z(train, np.column_stack([train_sex, train_vf_standardized])),
    }
    holdout_variants = {
        "none": replace_z(holdout, np.empty((holdout.n_subject, 0), dtype=float)),
        "sex": replace_z(holdout, holdout_sex),
        "vf_pca": replace_z(
            holdout,
            compose_pca_covariates(sex=holdout_sex, scores=holdout_scores, include_sex=False),
        ),
        "sex_vf_pca": replace_z(
            holdout,
            compose_pca_covariates(sex=holdout_sex, scores=holdout_scores, include_sex=True),
        ),
        "full60": replace_z(holdout, np.column_stack([holdout_sex, holdout_vf_standardized])),
    }
    return train_variants, holdout_variants, transformer


def predict_model(
    model_name: str,
    *,
    train: dict[str, PairedEyeDataset],
    holdout: dict[str, PairedEyeDataset],
    config: dict[str, Any],
    image_config: dict[str, Any],
) -> np.ndarray:
    if model_name == "y_bar":
        return np.full_like(holdout["none"].y, float(np.mean(train["none"].y)), dtype=float)
    if model_name == "x_only_paired_vctr":
        return predict_paired_vctr(train["none"], holdout["none"], config=config, image_config=image_config, z_mode="none")
    if model_name == "x_sex_paired_vctr":
        return predict_paired_vctr(train["sex"], holdout["sex"], config=config, image_config=image_config, z_mode="full")
    if model_name in {"vf_pca_only_linear", "sex_vf_pca_only_linear"}:
        key = "sex_vf_pca" if model_name.startswith("sex_") else "vf_pca"
        coef = fit_linear(train[key], use_x=False, use_z=True)
        return predict_linear(holdout[key], coef, use_x=False, use_z=True)
    if model_name in {"x_vf_pca_linear", "x_sex_vf_pca_linear"}:
        key = "sex_vf_pca" if model_name.startswith("x_sex_") else "vf_pca"
        coef = fit_linear(train[key], use_x=True, use_z=True)
        return predict_linear(holdout[key], coef, use_x=True, use_z=True)
    if model_name in {"x_vf_pca_iid_vctr", "x_sex_vf_pca_iid_vctr"}:
        key = "sex_vf_pca" if model_name.startswith("x_sex_") else "vf_pca"
        return predict_iid_vctr(train[key], holdout[key], config=config, image_config=image_config, z_mode="full")
    if model_name in {"x_vf_pca_paired_vctr", "x_sex_vf_pca_paired_vctr"}:
        key = "sex_vf_pca" if model_name.startswith("x_sex_") else "vf_pca"
        return predict_paired_vctr(train[key], holdout[key], config=config, image_config=image_config, z_mode="full")
    if model_name == "x_full60_paired_vctr":
        return predict_paired_vctr(
            train["full60"],
            holdout["full60"],
            config=config,
            image_config=image_config,
            z_mode="full",
        )
    raise ValueError(f"Unknown model: {model_name}")


def select_components(
    outer_train: PairedEyeDataset,
    outer_manifest: pd.DataFrame,
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    outer_fold: int,
    y_mean: float,
    y_sd: float,
) -> tuple[int, list[dict[str, Any]]]:
    split_group = str(config["split_group"])
    groups = outer_manifest[split_group].to_numpy()
    inner_seed = int(config["seed"]) + int(config.get("inner_seed_offset", 1000)) + outer_fold
    inner_folds = grouped_kfold_indices(groups, inner_seed, int(config["inner_folds"]))
    validate_grouped_folds(outer_manifest, inner_folds, split_group)
    rows: list[dict[str, Any]] = []
    candidate_scores: list[tuple[float, int]] = []

    for n_components in sorted(int(value) for value in config["pca_components"]):
        fold_predictions: list[np.ndarray] = []
        fold_truth: list[np.ndarray] = []
        for inner_fold, holdout_indices in enumerate(inner_folds, start=1):
            mask = np.ones(outer_train.n_subject, dtype=bool)
            mask[holdout_indices] = False
            train_indices = np.flatnonzero(mask)
            inner_train = subset_dataset(outer_train, train_indices, z_mode="full")
            inner_holdout = subset_dataset(outer_train, holdout_indices, z_mode="full")
            inner_train_groups = outer_manifest.iloc[train_indices][split_group].to_numpy()
            train_variants, holdout_variants, _ = pca_dataset_variants(
                inner_train,
                inner_holdout,
                train_groups=inner_train_groups,
                n_components=n_components,
                weighting=str(config["pca_weighting"]),
            )
            pred = predict_model(
                str(config["selection_model"]),
                train=train_variants,
                holdout=holdout_variants,
                config=config,
                image_config=image_config,
            )
            metrics = metric_values(
                inner_holdout.y,
                pred,
                y_mean=y_mean,
                y_sd=y_sd,
                eps_std=float(config.get("mape_eps_std", 1e-6)),
                eps_iop=float(config.get("mape_eps_iop", 1e-6)),
            )
            rows.append(
                {
                    "image_type": str(image_config["image_type"]),
                    "outer_fold": int(outer_fold),
                    "inner_fold": int(inner_fold),
                    "n_components": int(n_components),
                    "selection_model": str(config["selection_model"]),
                    "n_train_pairs": int(inner_train.n_subject),
                    "n_holdout_pairs": int(inner_holdout.n_subject),
                    **metrics,
                }
            )
            fold_predictions.append(pred)
            fold_truth.append(inner_holdout.y)
        aggregate = metric_values(
            np.concatenate(fold_truth, axis=0),
            np.concatenate(fold_predictions, axis=0),
            y_mean=y_mean,
            y_sd=y_sd,
            eps_std=float(config.get("mape_eps_std", 1e-6)),
            eps_iop=float(config.get("mape_eps_iop", 1e-6)),
        )
        candidate_scores.append((float(aggregate["rmse_std"]), int(n_components)))
        rows.append(
            {
                "image_type": str(image_config["image_type"]),
                "outer_fold": int(outer_fold),
                "inner_fold": 0,
                "n_components": int(n_components),
                "selection_model": str(config["selection_model"]),
                "n_train_pairs": int(outer_train.n_subject),
                "n_holdout_pairs": int(outer_train.n_subject),
                **aggregate,
            }
        )
    selected = min(candidate_scores, key=lambda item: (item[0], item[1]))[1]
    return selected, rows


def pca_diagnostic_rows(
    transformer: FoldVFPCATransformer,
    *,
    image_type: str,
    outer_fold: int,
    vf_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variance_rows: list[dict[str, Any]] = []
    loading_rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for pc_index, ratio in enumerate(transformer.explained_variance_ratio_, start=1):
        cumulative += float(ratio)
        variance_rows.append(
            {
                "image_type": image_type,
                "outer_fold": int(outer_fold),
                "pc": int(pc_index),
                "explained_variance_ratio": float(ratio),
                "cumulative_explained_variance_ratio": cumulative,
                "n_training_rows": transformer.n_training_rows_,
                "n_training_subjects": transformer.n_training_groups_,
            }
        )
        for variable, loading in zip(vf_names, transformer.components_[pc_index - 1], strict=True):
            loading_rows.append(
                {
                    "image_type": image_type,
                    "outer_fold": int(outer_fold),
                    "pc": int(pc_index),
                    "variable": variable,
                    "loading": float(loading),
                }
            )
    return variance_rows, loading_rows


def evaluate_one_image(
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    feature_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    image_type = str(image_config["image_type"])
    package_dir = feature_dir(feature_root, image_type, str(image_config["S"]), int(image_config["R"]))
    dataset, manifest, meta = load_feature_dataset(package_dir)
    split_group = str(config["split_group"])
    folds = grouped_kfold_indices(manifest[split_group].to_numpy(), int(config["seed"]), int(config["outer_folds"]))
    validate_grouped_folds(manifest, folds, split_group)
    z_names = list(meta["transforms"]["Z"]["columns"])
    if not z_names or z_names[0] != "is_female" or len(z_names) != dataset.Z.shape[1]:
        raise ValueError("Feature metadata must identify is_female followed by every VF column.")
    vf_names = z_names[1:]
    y_mean = float(meta["transforms"]["y"]["mean"])
    y_sd = float(meta["transforms"]["y"]["sd"])

    result: dict[str, list[dict[str, Any]]] = {
        "fold_metrics": [],
        "predictions": [],
        "inner_cv_metrics": [],
        "selected_components": [],
        "explained_variance": [],
        "loadings": [],
    }
    all_truth: list[np.ndarray] = []
    all_predictions: dict[str, list[np.ndarray]] = {name: [] for name in config["models"]}
    for outer_fold, holdout_indices in enumerate(folds, start=1):
        mask = np.ones(dataset.n_subject, dtype=bool)
        mask[holdout_indices] = False
        train_indices = np.flatnonzero(mask)
        outer_train = subset_dataset(dataset, train_indices, z_mode="full")
        outer_holdout = subset_dataset(dataset, holdout_indices, z_mode="full")
        outer_train_manifest = manifest.iloc[train_indices].reset_index(drop=True)
        selected_k, inner_rows = select_components(
            outer_train,
            outer_train_manifest,
            config=config,
            image_config=image_config,
            outer_fold=outer_fold,
            y_mean=y_mean,
            y_sd=y_sd,
        )
        result["inner_cv_metrics"].extend(inner_rows)
        result["selected_components"].append(
            {
                "image_type": image_type,
                "outer_fold": int(outer_fold),
                "selected_k": int(selected_k),
                "selection_model": str(config["selection_model"]),
                "n_outer_train_pairs": int(outer_train.n_subject),
                "n_outer_holdout_pairs": int(outer_holdout.n_subject),
                "n_outer_train_subjects": int(outer_train_manifest[split_group].nunique()),
            }
        )
        train_variants, holdout_variants, transformer = pca_dataset_variants(
            outer_train,
            outer_holdout,
            train_groups=outer_train_manifest[split_group].to_numpy(),
            n_components=selected_k,
            weighting=str(config["pca_weighting"]),
        )
        variance_rows, loading_rows = pca_diagnostic_rows(
            transformer,
            image_type=image_type,
            outer_fold=outer_fold,
            vf_names=vf_names,
        )
        result["explained_variance"].extend(variance_rows)
        result["loadings"].extend(loading_rows)
        all_truth.append(outer_holdout.y)

        for model_name in config["models"]:
            pred = predict_model(
                model_name,
                train=train_variants,
                holdout=holdout_variants,
                config=config,
                image_config=image_config,
            )
            all_predictions[model_name].append(pred)
            metrics = metric_values(
                outer_holdout.y,
                pred,
                y_mean=y_mean,
                y_sd=y_sd,
                eps_std=float(config.get("mape_eps_std", 1e-6)),
                eps_iop=float(config.get("mape_eps_iop", 1e-6)),
            )
            result["fold_metrics"].append(
                {
                    "image_type": image_type,
                    "S": str(image_config["S"]),
                    "R": int(image_config["R"]),
                    "signal_h": float(image_config["signal_h"]),
                    "variance_hbar": float(image_config["variance_hbar"]),
                    "model": model_name,
                    "outer_fold": int(outer_fold),
                    "selected_k": int(selected_k),
                    "n_pairs": int(outer_holdout.n_subject),
                    **metrics,
                }
            )
            result["predictions"].extend(
                prediction_rows(
                    image_type=image_type,
                    model_name=model_name,
                    fold_id=outer_fold,
                    manifest=manifest,
                    holdout_indices=holdout_indices,
                    y_true=outer_holdout.y,
                    y_pred=pred,
                    y_mean=y_mean,
                    y_sd=y_sd,
                )
            )

    truth = np.concatenate(all_truth, axis=0)
    result["summary_metrics"] = []
    for model_name in config["models"]:
        pred = np.concatenate(all_predictions[model_name], axis=0)
        metrics = metric_values(
            truth,
            pred,
            y_mean=y_mean,
            y_sd=y_sd,
            eps_std=float(config.get("mape_eps_std", 1e-6)),
            eps_iop=float(config.get("mape_eps_iop", 1e-6)),
        )
        result["summary_metrics"].append(
            {
                "image_type": image_type,
                "S": str(image_config["S"]),
                "R": int(image_config["R"]),
                "signal_h": float(image_config["signal_h"]),
                "variance_hbar": float(image_config["variance_hbar"]),
                "model": model_name,
                "n_pairs": int(dataset.n_subject),
                "n_subjects": int(manifest[split_group].nunique()),
                "pca_selection": "nested_patient_grouped_cv",
                "pca_weighting": str(config["pca_weighting"]),
                **metrics,
            }
        )
    return result


def build_ablation_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for image_type, image_df in summary.groupby("image_type", sort=True):
        by_model = image_df.set_index("model")
        for model, reference, contrast in CONTRASTS:
            if model not in by_model.index or reference not in by_model.index:
                continue
            current = by_model.loc[model]
            baseline = by_model.loc[reference]
            delta = float(current["rmse_iop"] - baseline["rmse_iop"])
            rows.append(
                {
                    "image_type": image_type,
                    "contrast": contrast,
                    "model": model,
                    "reference_model": reference,
                    "model_rmse_iop": float(current["rmse_iop"]),
                    "reference_rmse_iop": float(baseline["rmse_iop"]),
                    "delta_rmse_iop": delta,
                    "delta_rmse_std": float(current["rmse_std"] - baseline["rmse_std"]),
                    "pct_delta_rmse_iop": float(100.0 * delta / baseline["rmse_iop"]),
                }
            )
    return pd.DataFrame(rows)


def build_loading_stability(loadings: pd.DataFrame) -> pd.DataFrame:
    if loadings.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in loadings.groupby(["image_type", "pc", "variable"], sort=True):
        values = group["loading"].to_numpy(dtype=float)
        nonzero = values[np.abs(values) > np.finfo(float).eps]
        sign_agreement = float(max(np.mean(nonzero > 0), np.mean(nonzero < 0))) if nonzero.size else float("nan")
        rows.append(
            {
                "image_type": keys[0],
                "pc": int(keys[1]),
                "variable": keys[2],
                "n_outer_folds": int(values.size),
                "loading_mean": float(values.mean()),
                "loading_sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
                "mean_abs_loading": float(np.abs(values).mean()),
                "sign_agreement": sign_agreement,
            }
        )
    return pd.DataFrame(rows)


def build_prediction_contrast_uncertainty(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Patient-cluster bootstrap CIs for differences between OOF RMSE values.

    These intervals condition on the fitted outer-CV models. They quantify
    patient-level variation in the saved OOF prediction contrast, not the
    additional uncertainty from rerunning PCA selection and model fitting.
    """

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive.")
    rows: list[dict[str, Any]] = []
    for image_index, (image_type, image_df) in enumerate(predictions.groupby("image_type", sort=True)):
        available = set(image_df["model"])
        for contrast_index, (model, reference, contrast) in enumerate(CONTRASTS):
            if model not in available or reference not in available:
                continue
            pair = image_df[image_df["model"].isin([model, reference])].copy()
            pair["squared_error"] = np.square(pair["resid_iop"].to_numpy(dtype=float))
            aggregate = (
                pair.groupby(["subject_id", "model"], sort=True)["squared_error"]
                .agg(["sum", "count"])
                .reset_index()
            )
            sse = aggregate.pivot(index="subject_id", columns="model", values="sum").dropna()
            counts = aggregate.pivot(index="subject_id", columns="model", values="count").loc[sse.index]
            if not np.array_equal(counts[model].to_numpy(), counts[reference].to_numpy()):
                raise ValueError(f"Prediction counts differ for contrast {contrast!r}.")
            model_sse = sse[model].to_numpy(dtype=float)
            reference_sse = sse[reference].to_numpy(dtype=float)
            observation_counts = counts[model].to_numpy(dtype=float)
            point_model = float(np.sqrt(model_sse.sum() / observation_counts.sum()))
            point_reference = float(np.sqrt(reference_sse.sum() / observation_counts.sum()))

            rng = np.random.default_rng(seed + 1000 * image_index + contrast_index)
            draws = rng.multinomial(
                len(sse),
                np.full(len(sse), 1.0 / len(sse)),
                size=n_bootstrap,
            ).astype(float)
            denominators = draws @ observation_counts
            model_rmse = np.sqrt((draws @ model_sse) / denominators)
            reference_rmse = np.sqrt((draws @ reference_sse) / denominators)
            delta = model_rmse - reference_rmse
            rows.append(
                {
                    "image_type": image_type,
                    "contrast": contrast,
                    "model": model,
                    "reference_model": reference,
                    "n_subjects": int(len(sse)),
                    "n_bootstrap": int(n_bootstrap),
                    "delta_rmse_iop": point_model - point_reference,
                    "ci_lower_delta_rmse_iop": float(np.quantile(delta, 0.025)),
                    "ci_upper_delta_rmse_iop": float(np.quantile(delta, 0.975)),
                    "bootstrap_probability_improvement": float(np.mean(delta < 0)),
                    "inference_scope": "patient_cluster_bootstrap_of_fixed_oof_predictions",
                }
            )
    return pd.DataFrame(rows)


def write_readme(run_dir: Path, config: dict[str, Any]) -> None:
    lines = [
        f"# {config['name']}",
        "",
        "Nested patient-grouped CV evaluation of bilateral-mean VF PCA covariates.",
        "",
        "PCA standardization, covariance estimation, and transformation are fit only on training patients.",
        "Repeated visits receive inverse visit-count weights so every training patient has equal total PCA weight.",
        "Sex is excluded from PCA and evaluated as a separate scalar covariate.",
        "",
        "This experiment tests the incremental prediction value of bilateral OD/OS-mean VF covariates; it does not test eye-specific VF effects.",
        "",
        "## Outputs",
        "",
        "- `summary_metrics.csv` and `ablation_table.csv`: outer-CV prediction results",
        "- `prediction_contrast_uncertainty.csv`: patient-cluster bootstrap CIs for fixed OOF prediction contrasts",
        "- `fold_metrics.csv` and `predictions.csv`: outer-fold details",
        "- `inner_cv_metrics.csv` and `selected_k_by_fold.csv`: nested K selection audit",
        "- `explained_variance_by_fold.csv`: fold-local PCA variance summary",
        "- `pca_loadings_by_fold.csv` and `loading_stability.csv`: fold-local loadings audit",
        "- `run_metadata.json`: provenance and design limitations",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    if args.smoke:
        config = smoke_config(config)
    run_name = args.run_name or str(config["name"])
    feature_root = resolve_path(args.feature_root)
    run_dir = resolve_path(args.run_root) / run_name
    output_dir = (run_dir / "curated_smoke") if args.smoke else (resolve_path(args.output_root) / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    else:
        shutil.copy2(config_path, run_dir / "config.json")

    started = time.perf_counter()
    collected: dict[str, list[dict[str, Any]]] = {}
    for image_config in config["image_configs"]:
        result = evaluate_one_image(config=config, image_config=image_config, feature_root=feature_root)
        for key, rows in result.items():
            collected.setdefault(key, []).extend(rows)

    frames = {key: pd.DataFrame(rows) for key, rows in collected.items()}
    summary = frames["summary_metrics"].sort_values(["image_type", "rmse_iop", "model"], kind="mergesort")
    ablation = build_ablation_table(summary)
    if not ablation.empty:
        ablation = ablation.sort_values(["image_type", "contrast"], kind="mergesort")
    loading_stability = build_loading_stability(frames["loadings"])
    contrast_uncertainty = build_prediction_contrast_uncertainty(
        frames["predictions"],
        n_bootstrap=int(config.get("contrast_bootstrap_B", 2000)),
        seed=int(config["seed"]) + int(config.get("contrast_bootstrap_seed_offset", 2000)),
    )
    if not contrast_uncertainty.empty:
        contrast_uncertainty = contrast_uncertainty.sort_values(["image_type", "contrast"])
    output_frames = {
        "summary_metrics.csv": summary,
        "ablation_table.csv": ablation,
        "prediction_contrast_uncertainty.csv": contrast_uncertainty,
        "fold_metrics.csv": frames["fold_metrics"].sort_values(["image_type", "model", "outer_fold"]),
        "predictions.csv": frames["predictions"].sort_values(["image_type", "model", "fold", "pair_id", "eye"]),
        "inner_cv_metrics.csv": frames["inner_cv_metrics"].sort_values(
            ["image_type", "outer_fold", "n_components", "inner_fold"]
        ),
        "selected_k_by_fold.csv": frames["selected_components"].sort_values(["image_type", "outer_fold"]),
        "explained_variance_by_fold.csv": frames["explained_variance"].sort_values(
            ["image_type", "outer_fold", "pc"]
        ),
        "pca_loadings_by_fold.csv": frames["loadings"].sort_values(
            ["image_type", "outer_fold", "pc", "variable"]
        ),
        "loading_stability.csv": loading_stability.sort_values(["image_type", "pc", "variable"]),
    }
    for filename, frame in output_frames.items():
        frame.to_csv(run_dir / filename, index=False)

    # Curated outputs exclude row-level predictions and fold-specific raw loadings.
    for filename in (
        "summary_metrics.csv",
        "ablation_table.csv",
        "prediction_contrast_uncertainty.csv",
        "selected_k_by_fold.csv",
        "explained_variance_by_fold.csv",
        "loading_stability.csv",
    ):
        if not output_frames[filename].empty:
            output_frames[filename].to_csv(output_dir / filename, index=False)

    metadata = {
        "experiment": str(config["name"]),
        "config": rel_to_repo(config_path),
        "feature_root": rel_to_repo(feature_root),
        "outer_split": f"{config['split_group']}-grouped",
        "outer_folds": int(config["outer_folds"]),
        "inner_folds": int(config["inner_folds"]),
        "selection_model": str(config["selection_model"]),
        "pca_components": [int(value) for value in config["pca_components"]],
        "pca_weighting": str(config["pca_weighting"]),
        "contrast_bootstrap_B": int(config.get("contrast_bootstrap_B", 2000)),
        "contrast_bootstrap_scope": "patient-cluster bootstrap of fixed outer-CV predictions",
        "vf_representation": "59 bilateral OD/OS means shared by both eye outcomes",
        "sex_in_pca": False,
        "smoke": bool(args.smoke),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_readme(run_dir, config)
    write_readme(output_dir, config)
    print(json.dumps({"run_dir": rel_to_repo(run_dir), "output_dir": rel_to_repo(output_dir), **metadata}, indent=2))


if __name__ == "__main__":
    main()
