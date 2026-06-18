"""Compare final GRAPE prediction models by grouped held-out CV."""

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

from src.data import PairedEyeDataset
from src.models import PairedEyeVCTRModel
from src.models.covariance import invert_blocks


GRAPE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = GRAPE_ROOT / "data" / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "model_comparison"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "model_comparison" / "v1_best_models.json"
EYE_ORDER = np.array(["OD", "OS"], dtype=object)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args()


def parse_s(value: str) -> tuple[int, ...]:
    parts = tuple(int(part) for part in value.lower().split("x"))
    if not parts or any(part <= 0 for part in parts):
        raise ValueError(f"Invalid S value: {value!r}")
    return parts


def s_label(blocks: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in blocks)


def resolve_path(path: Path, *, base: Path = GRAPE_ROOT) -> Path:
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
    split_group = str(config.get("split_group", "pair_id"))
    if split_group not in {"pair_id", "subject_id"}:
        raise ValueError("split_group must be either 'pair_id' or 'subject_id'.")
    return config


def feature_dir(feature_root: Path, image_type: str, S: str, R: int) -> Path:
    return feature_root / f"{image_type}_192_iop_le35" / f"S{S}_R{int(R)}"


def load_feature_dataset(package_dir: Path) -> tuple[PairedEyeDataset, pd.DataFrame, dict[str, Any]]:
    required = ["X_star.npy", "y.npy", "Z.npy", "t.npy", "manifest.csv", "meta.json"]
    missing = [name for name in required if not (package_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature package files under {package_dir}: {missing}")

    X = np.load(package_dir / "X_star.npy")
    y = np.load(package_dir / "y.npy")
    Z = np.load(package_dir / "Z.npy")
    t = np.load(package_dir / "t.npy")
    manifest = pd.read_csv(package_dir / "manifest.csv")
    meta = json.loads((package_dir / "meta.json").read_text(encoding="utf-8"))

    order = np.argsort(t, kind="mergesort")
    X = X[order]
    y = y[order]
    Z = Z[order]
    t = t[order]
    manifest = manifest.iloc[order].reset_index(drop=True)
    subject_ids = manifest["pair_id"].to_numpy() if "pair_id" in manifest.columns else np.arange(t.shape[0])

    dataset = PairedEyeDataset(
        subject_ids=subject_ids,
        eye_ids=EYE_ORDER.copy(),
        t=t,
        X=X,
        Z=Z,
        y=y,
        meta={"feature_package": rel_to_repo(package_dir), "sorted_by_t": True},
    )
    return dataset, manifest, meta


def subset_dataset(dataset: PairedEyeDataset, indices: np.ndarray, *, z_mode: str = "full") -> PairedEyeDataset:
    indices = np.asarray(indices, dtype=int)
    if z_mode == "none":
        Z = np.empty((indices.shape[0], 0), dtype=float)
    elif z_mode == "full":
        Z = dataset.Z[indices]
    else:
        raise ValueError(f"Unknown z_mode={z_mode!r}")
    return PairedEyeDataset(
        subject_ids=dataset.subject_ids[indices],
        eye_ids=np.asarray(dataset.eye_ids).copy(),
        t=dataset.t[indices],
        X=dataset.X[indices],
        Z=Z,
        y=dataset.y[indices],
        meta=dict(dataset.meta),
    )


def grouped_kfold_indices(groups: np.ndarray, seed: int, folds: int) -> list[np.ndarray]:
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    n_folds = min(int(folds), int(unique_groups.shape[0]))
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    fold_group_sets = np.array_split(shuffled_groups, n_folds)
    return [
        np.flatnonzero(np.isin(groups, fold_groups)).astype(int)
        for fold_groups in fold_group_sets
        if len(fold_groups) > 0
    ]


def flatten_X(dataset: PairedEyeDataset) -> np.ndarray:
    return dataset.X.reshape(dataset.n_subject, 2, -1)


def linear_design(dataset: PairedEyeDataset, *, use_x: bool, use_z: bool) -> np.ndarray:
    pieces: list[np.ndarray] = []
    if use_z:
        pieces.append(np.repeat(dataset.Z, 2, axis=0))
    if use_x:
        pieces.append(flatten_X(dataset).reshape(dataset.n_subject * 2, -1))
    if not pieces:
        raise ValueError("At least one of use_x/use_z must be true.")
    return np.concatenate(pieces, axis=1)


def fit_linear(train: PairedEyeDataset, *, use_x: bool, use_z: bool) -> np.ndarray:
    design = linear_design(train, use_x=use_x, use_z=use_z)
    target = train.y.reshape(-1)
    coef, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coef


def predict_linear(holdout: PairedEyeDataset, coef: np.ndarray, *, use_x: bool, use_z: bool) -> np.ndarray:
    design = linear_design(holdout, use_x=use_x, use_z=use_z)
    return (design @ coef).reshape(holdout.n_subject, 2)


def build_model(config: dict[str, Any], image_config: dict[str, Any], *, z_mode: str) -> PairedEyeVCTRModel:
    return PairedEyeVCTRModel(
        covariance_mode="exchangeable_varying_sigma",
        a_eval_mode=str(config.get("a_eval_mode", "anchor_grid")),
        a_eval_num_points=int(config.get("a_eval_num_points", 80)),
        signal_bandwidth=float(image_config["signal_h"]),
        variance_bandwidth=float(image_config["variance_hbar"]),
        ridge=float(config.get("ridge", 0.0)),
    )


def predict_iid_vctr(
    train: PairedEyeDataset,
    holdout: PairedEyeDataset,
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    z_mode: str,
) -> np.ndarray:
    train_ds = train if z_mode == "full" else subset_dataset(train, np.arange(train.n_subject), z_mode="none")
    holdout_ds = holdout if z_mode == "full" else subset_dataset(holdout, np.arange(holdout.n_subject), z_mode="none")
    model = build_model(config, image_config, z_mode=z_mode)
    initial = model.initial_fit_iid(train_ds)
    train_flat = train_ds.to_iid_observations()
    A_holdout, _ = model._estimate_stage1_A(  # noqa: SLF001 - experiment-level prediction helper.
        flat_Z=train_flat.Z,
        flat_X=model._flatten_X(train_flat.X),  # noqa: SLF001
        flat_y=train_flat.y,
        flat_t=train_flat.t,
        t_eval=holdout_ds.t,
        p0=train_ds.Z.shape[1],
        bandwidth=float(image_config["signal_h"]),
    )
    x_holdout = holdout_ds.X.reshape(holdout_ds.n_subject, 2, -1)
    signal = np.sum(x_holdout * A_holdout[:, None, :], axis=2)
    if holdout_ds.Z.shape[1] == 0:
        return signal
    return signal + holdout_ds.Z @ initial.beta_hat[:, None]


def predict_paired_vctr(
    train: PairedEyeDataset,
    holdout: PairedEyeDataset,
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    z_mode: str,
) -> np.ndarray:
    train_ds = train if z_mode == "full" else subset_dataset(train, np.arange(train.n_subject), z_mode="none")
    holdout_ds = holdout if z_mode == "full" else subset_dataset(holdout, np.arange(holdout.n_subject), z_mode="none")
    model = build_model(config, image_config, z_mode=z_mode)
    initial = model.initial_fit_iid(train_ds)
    covariance = model.estimate_covariance(train_ds, initial)
    paired = model.refit_with_covariance(train_ds, covariance, initial)
    Sigma_inv = invert_blocks(covariance.Sigma_hat_blocks)

    train_x = train_ds.X.reshape(train_ds.n_subject, 2, -1)
    n_features = train_x.shape[2]
    p0 = train_ds.Z.shape[1]
    A_holdout = np.zeros((holdout_ds.n_subject, n_features), dtype=float)
    h = float(image_config["signal_h"])

    for row, t0 in enumerate(holdout_ds.t):
        lhs = np.zeros((p0 + 2 * n_features, p0 + 2 * n_features), dtype=float)
        rhs = np.zeros(p0 + 2 * n_features, dtype=float)
        for subj in range(train_ds.n_subject):
            kh = model._kernel_scalar_weight(train_ds.t[subj], float(t0), h)  # noqa: SLF001
            if kh <= 0:
                continue
            sst = (train_ds.t[subj] - float(t0)) / h
            Vi = np.zeros((2, p0 + 2 * n_features), dtype=float)
            if p0:
                Vi[:, :p0] = train_ds.Z[subj]
            Vi[:, p0 : p0 + n_features] = train_x[subj]
            Vi[:, p0 + n_features :] = train_x[subj] * sst
            Wi = kh * Sigma_inv[subj]
            lhs += Vi.T @ Wi @ Vi
            rhs += Vi.T @ Wi @ train_ds.y[subj]
        coef = np.linalg.solve(lhs + float(config.get("ridge", 0.0)) * np.eye(lhs.shape[0]), rhs)
        A_holdout[row] = coef[p0 : p0 + n_features]

    x_holdout = holdout_ds.X.reshape(holdout_ds.n_subject, 2, -1)
    signal = np.sum(x_holdout * A_holdout[:, None, :], axis=2)
    if holdout_ds.Z.shape[1] == 0:
        return signal
    return signal + holdout_ds.Z @ paired.beta_hat[:, None]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, y_mean: float, y_sd: float) -> dict[str, float]:
    resid = y_true - y_pred
    resid_raw = (y_true * y_sd + y_mean) - (y_pred * y_sd + y_mean)
    return {
        "rmse_std": float(np.sqrt(np.mean(resid**2))),
        "mae_std": float(np.mean(np.abs(resid))),
        "rmse_iop": float(np.sqrt(np.mean(resid_raw**2))),
        "mae_iop": float(np.mean(np.abs(resid_raw))),
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
        for eye_idx, eye in enumerate(("od", "os")):
            rows.append(
                {
                    "image_type": image_type,
                    "model": model_name,
                    "fold": fold_id,
                    "pair_id": holdout_manifest.loc[row_idx, "pair_id"],
                    "subject_id": holdout_manifest.loc[row_idx, "subject_id"],
                    "eye": eye.upper(),
                    "y_true_std": float(y_true[row_idx, eye_idx]),
                    "y_pred_std": float(y_pred[row_idx, eye_idx]),
                    "resid_std": float(y_true[row_idx, eye_idx] - y_pred[row_idx, eye_idx]),
                    "y_true_iop": float(y_true[row_idx, eye_idx] * y_sd + y_mean),
                    "y_pred_iop": float(y_pred[row_idx, eye_idx] * y_sd + y_mean),
                    "resid_iop": float((y_true[row_idx, eye_idx] - y_pred[row_idx, eye_idx]) * y_sd),
                }
            )
    return rows


def evaluate_one_image(
    *,
    config: dict[str, Any],
    image_config: dict[str, Any],
    feature_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image_type = str(image_config["image_type"])
    S = str(image_config["S"])
    R = int(image_config["R"])
    package_dir = feature_dir(feature_root, image_type, S, R)
    dataset, manifest, meta = load_feature_dataset(package_dir)
    y_mean = float(meta["transforms"]["y"]["mean"])
    y_sd = float(meta["transforms"]["y"]["sd"])
    split_group = str(config.get("split_group", "pair_id"))
    fold_groups = manifest[split_group].to_numpy()
    folds = grouped_kfold_indices(fold_groups, int(config["seed"]), int(config["folds"]))

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

        fold_predictions: dict[str, np.ndarray] = {}
        z_coef = fit_linear(train, use_x=False, use_z=True)
        fold_predictions["z_only_linear"] = predict_linear(holdout, z_coef, use_x=False, use_z=True)

        xz_coef = fit_linear(train, use_x=True, use_z=True)
        fold_predictions["xz_linear"] = predict_linear(holdout, xz_coef, use_x=True, use_z=True)

        fold_predictions["x_only_vctr"] = predict_paired_vctr(
            train,
            holdout,
            config=config,
            image_config=image_config,
            z_mode="none",
        )
        fold_predictions["xz_iid_vctr"] = predict_iid_vctr(
            train,
            holdout,
            config=config,
            image_config=image_config,
            z_mode="full",
        )
        fold_predictions["xz_paired_vctr"] = predict_paired_vctr(
            train,
            holdout,
            config=config,
            image_config=image_config,
            z_mode="full",
        )

        for model_name in model_names:
            pred = fold_predictions[model_name]
            all_predictions[model_name].append(pred)
            fold_metrics = metrics(holdout.y, pred, y_mean, y_sd)
            fold_metric_rows.append(
                {
                    "image_type": image_type,
                    "S": S,
                    "R": R,
                    "split_group": split_group,
                    "model": model_name,
                    "fold": fold_id,
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
        summary_rows.append(
            {
                "image_type": image_type,
                "S": S,
                "R": R,
                "signal_h": float(image_config["signal_h"]),
                "variance_hbar": float(image_config["variance_hbar"]),
                "split_group": split_group,
                "model": model_name,
                "n_pairs": int(dataset.n_subject),
                **metrics(truth, pred, y_mean, y_sd),
            }
        )
    return summary_rows, fold_metric_rows, prediction_row_list


def write_readme(run_dir: Path, config: dict[str, Any], summary: pd.DataFrame) -> None:
    best = summary.sort_values(["image_type", "rmse_iop"], kind="mergesort").groupby("image_type").head(1)
    lines = [
        f"# {config['name']}",
        "",
        "## 目的",
        "",
        f"固定 v3 选出的 CFP/ROI 低复杂度配置，使用 {config.get('split_group', 'pair_id')}-level "
        "5-fold held-out CV 比较线性基线和 VCTR 模型。",
        "",
        "## 模型",
        "",
        "- `z_only_linear`: only vector covariates Z",
        "- `xz_linear`: reduced image features X_star plus Z",
        "- `x_only_vctr`: VCTR using X_star only",
        "- `xz_iid_vctr`: X+Z VCTR using iid stage-1 prediction",
        "- `xz_paired_vctr`: full paired-eye VCTR",
        "",
        "## 最佳 held-out RMSE",
        "",
        "| image_type | best_model | rmse_iop | rmse_std |",
        "| :-- | :-- | --: | --: |",
    ]
    for _, row in best.iterrows():
        lines.append(
            f"| {row['image_type']} | {row['model']} | {row['rmse_iop']:.6f} | {row['rmse_std']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 完整排序",
            "",
            "| image_type | model | rmse_iop | mae_iop |",
            "| :-- | :-- | --: | --: |",
        ]
    )
    ordered = summary.sort_values(["image_type", "rmse_iop", "model"], kind="mergesort")
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['image_type']} | {row['model']} | {row['rmse_iop']:.6f} | {row['mae_iop']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 当前表是 held-out prediction ablation，不是最终 inferential result。",
            f"- 当前 split 是 {config.get('split_group', 'pair_id')}-level。",
            "- 如果 `x_only_vctr` 最优，说明当前固定配置下 `Z` 没有带来泛化预测收益。",
            "- `xz_paired_vctr` 与 `xz_iid_vctr` 的差异应和 residual/covariance diagnostics 一起解释。",
        ]
    )
    lines.extend(
        [
            "",
            "## 输出",
            "",
            "- `summary_metrics.csv`: model-level held-out metrics",
            "- `fold_metrics.csv`: fold-level held-out metrics",
            "- `predictions.csv`: held-out predictions on standardized and original IOP scales",
            "- `config.json`: copied experiment config",
        ]
    )
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    run_name = args.run_name or str(config["name"])
    feature_root = resolve_path(args.feature_root)
    run_root = resolve_path(args.run_root)
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

    summary = pd.DataFrame(summary_rows).sort_values(["image_type", "rmse_iop", "model"], kind="mergesort")
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["image_type", "model", "fold"], kind="mergesort")
    predictions = pd.DataFrame(pred_rows).sort_values(
        ["image_type", "model", "fold", "pair_id", "eye"],
        kind="mergesort",
    )
    summary.to_csv(run_dir / "summary_metrics.csv", index=False)
    fold_metrics.to_csv(run_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    write_readme(run_dir, config, summary)
    elapsed = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "run_dir": rel_to_repo(run_dir),
                "summary_metrics": rel_to_repo(run_dir / "summary_metrics.csv"),
                "fold_metrics": rel_to_repo(run_dir / "fold_metrics.csv"),
                "predictions": rel_to_repo(run_dir / "predictions.csv"),
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
