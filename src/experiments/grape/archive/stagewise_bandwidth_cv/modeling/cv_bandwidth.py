"""Run bandwidth CV for one GRAPE feature package."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PairedEyeDataset
from src.models import PairedEyeVCTRModel


GRAPE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = GRAPE_ROOT / "data"
FEATURE_ROOT = DATA_ROOT / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "cv_bandwidth"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "bandwidth_grids" / "v1_adaptive_support.json"
DEFAULT_IMAGE_TYPES = ("cfp", "roi")


def parse_s(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("S must have format like 3x3x1.") from exc
    if not parts or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("S must contain positive integers.")
    return parts


def s_label(blocks: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in blocks)


def parse_float_list(values: Any, name: str) -> tuple[float, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list.")
    parsed = tuple(float(value) for value in values)
    if any(value <= 0 for value in parsed):
        raise ValueError(f"All {name} values must be positive.")
    return parsed


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


def read_task_csv(path: Path, task_index: int) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if int(row["task_id"]) == int(task_index):
            return row
    raise ValueError(f"task_id={task_index} not found in {path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-type", choices=DEFAULT_IMAGE_TYPES)
    parser.add_argument("--S", type=parse_s)
    parser.add_argument("--R", type=int)
    parser.add_argument("--bandwidth-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--task-csv", type=Path, default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--a-eval-mode", choices=["full", "anchor_grid"], default="full")
    parser.add_argument("--a-eval-num-points", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return a non-zero exit code after writing failure outputs.",
    )
    args = parser.parse_args()

    if args.task_csv is not None:
        if args.task_index is None:
            raise ValueError("--task-index is required when --task-csv is provided.")
        row = read_task_csv(resolve_path(args.task_csv), args.task_index)
        args.image_type = row["image_type"]
        args.S = parse_s(row["S"])
        args.R = int(row["R"])
        args.bandwidth_config = Path(row["bandwidth_config"])
        args.run_name = row["run_name"]
        args.task_id = int(row["task_id"])

    missing = [
        name
        for name in ("image_type", "S", "R", "run_name", "task_id")
        if getattr(args, name) is None
    ]
    if missing:
        raise ValueError(f"Missing required arguments: {', '.join('--' + name.replace('_', '-') for name in missing)}.")
    if args.R <= 0:
        raise ValueError("--R must be positive.")
    if args.ridge < 0:
        raise ValueError("--ridge must be non-negative.")
    return args


def load_config(path: Path) -> dict[str, Any]:
    path = resolve_path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    parse_float_list(config.get("signal_h_candidates"), "signal_h_candidates")
    parse_float_list(config.get("variance_h_candidates"), "variance_h_candidates")
    if int(config.get("folds", 0)) < 2:
        raise ValueError("config.folds must be at least 2.")
    int(config["seed"])
    eligibility = config.get("eligibility")
    if not isinstance(eligibility, dict):
        raise ValueError("config.eligibility must be an object.")
    float(eligibility["support_quantile"])
    float(eligibility["support_safety_factor"])
    float(eligibility["edge_warning_factor"])
    z_mode = str(config.get("z_mode", "full"))
    if z_mode not in {"full", "none"}:
        raise ValueError("config.z_mode must be either 'full' or 'none'.")
    split_group = str(config.get("split_group", "pair_id"))
    if split_group not in {"pair_id", "subject_id"}:
        raise ValueError("config.split_group must be either 'pair_id' or 'subject_id'.")
    return config


def feature_dir(feature_root: Path, image_type: str, blocks_per_mode: tuple[int, ...], rank: int) -> Path:
    return feature_root / f"{image_type}_192_iop_le35" / f"S{s_label(blocks_per_mode)}_R{rank}"


def load_dataset(package_dir: Path) -> tuple[PairedEyeDataset, pd.DataFrame, dict[str, Any]]:
    required = ["X_star.npy", "y.npy", "Z.npy", "t.npy", "manifest.csv", "meta.json"]
    missing = [name for name in required if not (package_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing feature package files under {package_dir}: {missing}")

    X = np.load(package_dir / "X_star.npy")
    y = np.load(package_dir / "y.npy")
    Z = np.load(package_dir / "Z.npy")
    t = np.load(package_dir / "t.npy")
    manifest = pd.read_csv(package_dir / "manifest.csv")
    feature_meta = json.loads((package_dir / "meta.json").read_text(encoding="utf-8"))

    order = np.argsort(t, kind="mergesort")
    X = X[order]
    y = y[order]
    Z = Z[order]
    t = t[order]
    manifest = manifest.iloc[order].reset_index(drop=True)
    subject_ids = manifest["pair_id"].to_numpy() if "pair_id" in manifest.columns else np.arange(t.shape[0])

    dataset = PairedEyeDataset(
        subject_ids=subject_ids,
        eye_ids=np.array(["OD", "OS"], dtype=object),
        t=t,
        X=X,
        Z=Z,
        y=y,
        meta={
            "feature_package": rel_to_repo(package_dir),
            "sorted_by_t": True,
        },
    )
    return dataset, manifest, feature_meta


def apply_z_mode(dataset: PairedEyeDataset, z_mode: str) -> PairedEyeDataset:
    if z_mode == "full":
        return dataset
    if z_mode != "none":
        raise ValueError(f"Unknown z_mode={z_mode!r}.")
    return PairedEyeDataset(
        subject_ids=np.asarray(dataset.subject_ids).copy(),
        eye_ids=np.asarray(dataset.eye_ids).copy(),
        t=np.asarray(dataset.t).copy(),
        X=np.asarray(dataset.X).copy(),
        Z=np.empty((dataset.n_subject, 0), dtype=float),
        y=np.asarray(dataset.y).copy(),
        A_true=None if dataset.A_true is None else np.asarray(dataset.A_true).copy(),
        beta_true=None,
        Sigma_true=None if dataset.Sigma_true is None else np.asarray(dataset.Sigma_true).copy(),
        meta={**dict(dataset.meta), "z_mode": "none"},
    )


def grouped_kfold_indices(groups: np.ndarray, seed: int, folds: int) -> list[np.ndarray]:
    groups = np.asarray(groups)
    unique_groups = pd.unique(groups)
    if unique_groups.shape[0] < 2:
        raise ValueError("At least two groups are required for CV.")
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    n_folds = min(int(folds), int(unique_groups.shape[0]))
    rng = np.random.default_rng(seed)
    shuffled_groups = rng.permutation(unique_groups)
    fold_group_sets = np.array_split(shuffled_groups, n_folds)
    return [
        np.flatnonzero(np.isin(groups, fold_groups)).astype(int)
        for fold_groups in fold_group_sets
        if len(fold_groups) > 0
    ]


def support_stats_for_h(t: np.ndarray, h: float, fold_indices: list[np.ndarray]) -> dict[str, float]:
    t = np.asarray(t, dtype=float).reshape(-1)
    train_support_obs: list[int] = []
    all_support_obs: list[int] = []
    for holdout_indices in fold_indices:
        train_mask = np.ones(t.shape[0], dtype=bool)
        train_mask[holdout_indices] = False
        train_t = t[train_mask]
        holdout_t = t[holdout_indices]
        for t0 in holdout_t:
            train_support_obs.append(int(2 * np.sum(np.abs(train_t - t0) <= h)))
    for t0 in t:
        all_support_obs.append(int(2 * np.sum(np.abs(t - t0) <= h)))

    train = np.asarray(train_support_obs, dtype=float)
    all_data = np.asarray(all_support_obs, dtype=float)
    return {
        "min_train_support_obs": float(np.min(train)),
        "q05_train_support_obs": float(np.quantile(train, 0.05)),
        "median_train_support_obs": float(np.median(train)),
        "min_all_support_obs": float(np.min(all_data)),
        "q05_all_support_obs": float(np.quantile(all_data, 0.05)),
        "median_all_support_obs": float(np.median(all_data)),
    }


def evaluate_signal_bandwidth_eligibility(
    *,
    t: np.ndarray,
    candidates: tuple[float, ...],
    fold_indices: list[np.ndarray],
    p0: int,
    rank: int,
    blocks_per_mode: tuple[int, ...],
    support_quantile: float,
    support_safety_factor: float,
    edge_warning_factor: float,
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    n_blocks = int(np.prod(blocks_per_mode))
    n_features = int(rank * n_blocks)
    local_parameter_count = int(p0 + 2 * n_features)
    rows: list[dict[str, Any]] = []
    eligible: list[float] = []

    for h in candidates:
        stats = support_stats_for_h(t, h, fold_indices)
        support_value = float(stats["q05_train_support_obs"])
        threshold = float(support_safety_factor * local_parameter_count)
        edge_threshold = float(edge_warning_factor * local_parameter_count)
        is_eligible = support_value >= threshold
        edge_unstable = float(stats["min_train_support_obs"]) < edge_threshold
        row = {
            "bandwidth": float(h),
            "eligible": bool(is_eligible),
            "edge_unstable": bool(edge_unstable),
            "support_quantile": float(support_quantile),
            "support_safety_factor": float(support_safety_factor),
            "edge_warning_factor": float(edge_warning_factor),
            "support_threshold_obs": threshold,
            "edge_warning_threshold_obs": edge_threshold,
            "local_parameter_count": local_parameter_count,
            "n_features": n_features,
            "n_blocks": n_blocks,
            **stats,
        }
        if is_eligible:
            eligible.append(float(h))
        rows.append(row)
    return rows, tuple(eligible)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_score_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    else:
        fieldnames = ["bandwidth", "cv_score"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_eligibility_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["bandwidth", "eligible"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_score(rows: list[dict[str, Any]], selected: float | None) -> float | None:
    if selected is None:
        return None
    for row in rows:
        if abs(float(row["bandwidth"]) - float(selected)) < 1e-12:
            return float(row["cv_score"])
    return None


def finite_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finite: list[dict[str, Any]] = []
    for row in rows:
        score = row.get("cv_score")
        if score is None:
            continue
        score_float = float(score)
        if np.isfinite(score_float):
            finite.append(row)
    return finite


def select_best_bandwidth(rows: list[dict[str, Any]]) -> float:
    finite = finite_score_rows(rows)
    if not finite:
        raise np.linalg.LinAlgError("All candidate bandwidths failed during grouped CV.")
    return float(min(finite, key=lambda row: (float(row["cv_score"]), float(row["bandwidth"])))["bandwidth"])


def signal_kfold_cv_scores(
    model: PairedEyeVCTRModel,
    dataset: PairedEyeDataset,
    candidates: tuple[float, ...],
    fold_indices: list[np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bandwidth in candidates:
        try:
            fold_scores = [
                model._subject_fold_mse(dataset, float(bandwidth), holdout_indices)  # noqa: SLF001
                for holdout_indices in fold_indices
            ]
            rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": float(np.mean(fold_scores)),
                    "fold_scores": [float(value) for value in fold_scores],
                    "status": "success",
                }
            )
        except Exception as exc:  # noqa: BLE001 - persist candidate-level failure.
            rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": None,
                    "fold_scores": [],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows


def variance_kfold_cv_scores(
    model: PairedEyeVCTRModel,
    t: np.ndarray,
    residual_pairs: np.ndarray,
    candidates: tuple[float, ...],
    fold_indices: list[np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    squared_pairs = np.square(residual_pairs)
    for bandwidth in candidates:
        try:
            fold_scores: list[float] = []
            for holdout_indices in fold_indices:
                train_mask = np.ones(len(t), dtype=bool)
                train_mask[holdout_indices] = False
                train_t = t[train_mask]
                train_sq = squared_pairs[train_mask]
                holdout_t = t[holdout_indices]
                holdout_sq = squared_pairs[holdout_indices]
                sigma_hat = model._smooth_variance_curve(  # noqa: SLF001
                    train_t,
                    train_sq,
                    holdout_t,
                    float(bandwidth),
                )
                holdout_target = np.mean(holdout_sq, axis=1)
                fold_scores.append(float(np.mean(np.square(holdout_target - sigma_hat))))
            rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": float(np.mean(fold_scores)),
                    "fold_scores": [float(value) for value in fold_scores],
                    "status": "success",
                }
            )
        except Exception as exc:  # noqa: BLE001 - persist candidate-level failure.
            rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": None,
                    "fold_scores": [],
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return rows


def ensure_run_config(run_dir: Path, config_path: Path, config: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "config.json"
    if target.exists():
        return
    try:
        target.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except FileExistsError:
        return
    source_target = run_dir / "source_bandwidth_config.txt"
    if not source_target.exists():
        source_target.write_text(rel_to_repo(config_path) + "\n", encoding="utf-8")


def update_run_manifest(run_dir: Path, row: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.csv"
    lock_path = run_dir / ".manifest.lock"
    fieldnames = [
        "task_id",
        "image_type",
        "S",
        "R",
        "status",
        "feature_dir",
        "output_dir",
        "bandwidth_config",
        "started_at_utc",
        "finished_at_utc",
        "elapsed_seconds",
    ]
    try:
        import fcntl

        with lock_path.open("w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            _rewrite_manifest(manifest_path, fieldnames, row)
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
    except ImportError:
        _rewrite_manifest(manifest_path, fieldnames, row)


def _rewrite_manifest(manifest_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    if manifest_path.exists():
        with manifest_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    row_str = {key: "" if row.get(key) is None else str(row.get(key)) for key in fieldnames}
    rows = [existing for existing in rows if existing.get("task_id") != row_str["task_id"]]
    rows.append(row_str)
    rows.sort(key=lambda item: int(item["task_id"]))
    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(manifest_path)


def build_result_base(
    *,
    args: argparse.Namespace,
    config_path: Path,
    package_dir: Path,
    task_dir: Path,
    started_at: datetime,
) -> dict[str, Any]:
    return {
        "task_id": int(args.task_id),
        "run_name": str(args.run_name),
        "image_type": str(args.image_type),
        "S": s_label(args.S),
        "R": int(args.R),
        "feature_dir": rel_to_repo(package_dir),
        "output_dir": rel_to_repo(task_dir),
        "bandwidth_config": rel_to_repo(config_path),
        "a_eval_mode": args.a_eval_mode,
        "a_eval_num_points": int(args.a_eval_num_points),
        "ridge": float(args.ridge),
        "started_at_utc": started_at.isoformat(),
    }


def run_task(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    config_path = resolve_path(args.bandwidth_config)
    config = load_config(config_path)
    feature_root = resolve_path(args.feature_root)
    run_root = resolve_path(args.run_root)
    run_dir = run_root / str(args.run_name)
    task_dir = run_dir / f"task_{int(args.task_id):04d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    ensure_run_config(run_dir, config_path, config)

    package_dir = feature_dir(feature_root, args.image_type, args.S, args.R)
    result_base = build_result_base(
        args=args,
        config_path=config_path,
        package_dir=package_dir,
        task_dir=task_dir,
        started_at=started_at,
    )

    dataset_full, manifest, feature_meta = load_dataset(package_dir)
    signal_candidates = parse_float_list(config["signal_h_candidates"], "signal_h_candidates")
    variance_candidates = parse_float_list(config["variance_h_candidates"], "variance_h_candidates")
    folds = int(config["folds"])
    seed = int(config["seed"])
    z_mode = str(config.get("z_mode", "full"))
    split_group = str(config.get("split_group", "pair_id"))
    dataset = apply_z_mode(dataset_full, z_mode)
    fold_indices = grouped_kfold_indices(manifest[split_group].to_numpy(), seed, folds)
    eligibility = config["eligibility"]
    support_quantile = float(eligibility["support_quantile"])
    if abs(support_quantile - 0.05) > 1e-12:
        raise ValueError("Only support_quantile=0.05 is currently implemented.")

    eligibility_rows, eligible_signal_h = evaluate_signal_bandwidth_eligibility(
        t=dataset.t,
        candidates=signal_candidates,
        fold_indices=fold_indices,
        p0=dataset.Z.shape[1],
        rank=int(args.R),
        blocks_per_mode=args.S,
        support_quantile=support_quantile,
        support_safety_factor=float(eligibility["support_safety_factor"]),
        edge_warning_factor=float(eligibility["edge_warning_factor"]),
    )
    write_eligibility_csv(task_dir / "signal_bandwidth_eligibility.csv", eligibility_rows)
    write_json(
        task_dir / "eligible_bandwidths.json",
        {
            "signal_h_candidates": list(signal_candidates),
            "eligible_signal_h": list(eligible_signal_h),
            "rejected_signal_h": [
                row["bandwidth"] for row in eligibility_rows if not bool(row["eligible"])
            ],
            "variance_h_candidates": list(variance_candidates),
            "eligibility_rows": eligibility_rows,
        },
    )

    if not eligible_signal_h:
        finished_at = datetime.now(timezone.utc)
        result = {
            **result_base,
            "status": "no_eligible_signal_bandwidth",
            "finished_at_utc": finished_at.isoformat(),
            "elapsed_seconds": time.perf_counter() - t0,
            "n_pairs": int(dataset.n_subject),
            "p0": int(dataset.Z.shape[1]),
            "z_mode": z_mode,
            "split_group": split_group,
            "X_shape": list(dataset.X.shape),
            "y_shape": list(dataset.y.shape),
            "Z_shape": list(dataset.Z.shape),
            "t_shape": list(dataset.t.shape),
            "eligible_signal_h": [],
            "variance_h_candidates": list(variance_candidates),
        }
        write_score_csv(task_dir / "signal_cv_scores.csv", [])
        write_score_csv(task_dir / "variance_cv_scores.csv", [])
        write_json(task_dir / "result.json", result)
        write_json(
            task_dir / "meta.json",
            {
                "config": config,
                "feature_meta": feature_meta,
                "manifest_columns": list(manifest.columns),
            },
        )
        return result, run_dir, package_dir, config_path

    model = PairedEyeVCTRModel(
        covariance_mode="exchangeable_varying_sigma",
        a_eval_mode=args.a_eval_mode,
        a_eval_num_points=int(args.a_eval_num_points),
        ridge=float(args.ridge),
    )

    use_grouped_runner_cv = z_mode != "full" or split_group != "pair_id"
    if use_grouped_runner_cv:
        signal_scores = signal_kfold_cv_scores(model, dataset, tuple(eligible_signal_h), fold_indices)
        best_signal_h = select_best_bandwidth(signal_scores)
        initial_for_variance = model._fit_initial_iid_with_bandwidth(dataset, best_signal_h)  # noqa: SLF001
        residual_pairs = initial_for_variance.residuals.reshape(dataset.n_subject, 2)
        variance_scores = variance_kfold_cv_scores(
            model,
            dataset.t,
            residual_pairs,
            variance_candidates,
            fold_indices,
        )
        best_variance_h = select_best_bandwidth(variance_scores)
        final_model = PairedEyeVCTRModel(
            covariance_mode="exchangeable_varying_sigma",
            a_eval_mode=args.a_eval_mode,
            a_eval_num_points=int(args.a_eval_num_points),
            signal_bandwidth=best_signal_h,
            variance_bandwidth=best_variance_h,
            ridge=float(args.ridge),
        )
        fit_result = final_model.fit(dataset)
        fit_result.initial.meta.update(
            {
                "signal_bandwidth_selected": best_signal_h,
                "signal_bandwidth_method": "runner_grouped_kfold_cv",
                "signal_bandwidth_grid": list(eligible_signal_h),
                "signal_bandwidth_cv_scores": signal_scores,
                "signal_bandwidth_cv_metric": f"{split_group}_grouped_kfold_mse",
                "signal_bandwidth_cv_folds": len(fold_indices),
                "signal_bandwidth_cv_seed": seed,
            }
        )
        fit_result.covariance.meta.update(
            {
                "variance_bandwidth_selected": best_variance_h,
                "variance_bandwidth_method": "runner_grouped_kfold_cv",
                "variance_bandwidth_grid": list(variance_candidates),
                "variance_bandwidth_cv_scores": variance_scores,
                "variance_bandwidth_cv_metric": f"{split_group}_grouped_kfold_squared_residual_mse",
                "variance_bandwidth_cv_folds": len(fold_indices),
                "variance_bandwidth_cv_seed": seed,
            }
        )
        fit_result.meta.update(
            {
                "signal_bandwidth_selected": best_signal_h,
                "signal_bandwidth_method": "runner_grouped_kfold_cv",
                "signal_bandwidth_cv_scores": signal_scores,
                "signal_bandwidth_cv_metric": f"{split_group}_grouped_kfold_mse",
                "variance_bandwidth_selected": best_variance_h,
                "variance_bandwidth_method": "runner_grouped_kfold_cv",
                "variance_bandwidth_cv_scores": variance_scores,
                "variance_bandwidth_cv_metric": f"{split_group}_grouped_kfold_squared_residual_mse",
            }
        )
    else:
        model.signal_bandwidth = None
        model.signal_bandwidth_method = "stage1_kfold_cv"
        model.signal_bandwidth_grid = tuple(eligible_signal_h)
        model.signal_bandwidth_cv_folds = folds
        model.signal_bandwidth_cv_seed = seed
        model.variance_bandwidth = None
        model.variance_bandwidth_method = "stage2_kfold_cv"
        model.variance_bandwidth_grid = tuple(variance_candidates)
        model.variance_bandwidth_cv_folds = folds
        model.variance_bandwidth_cv_seed = seed
        fit_result = model.fit(dataset)
        signal_scores = fit_result.initial.meta.get("signal_bandwidth_cv_scores", [])
        variance_scores = fit_result.covariance.meta.get("variance_bandwidth_cv_scores", [])
        best_signal_h = float(fit_result.initial.meta["signal_bandwidth_selected"])
        best_variance_h = fit_result.covariance.meta.get("variance_bandwidth_selected")
        best_variance_h = None if best_variance_h is None else float(best_variance_h)

    write_score_csv(task_dir / "signal_cv_scores.csv", signal_scores)
    write_score_csv(task_dir / "variance_cv_scores.csv", variance_scores)
    np.save(task_dir / "beta_hat.npy", fit_result.beta_hat)

    finished_at = datetime.now(timezone.utc)
    result = {
        **result_base,
        "status": "success",
        "finished_at_utc": finished_at.isoformat(),
        "elapsed_seconds": time.perf_counter() - t0,
        "n_pairs": int(dataset.n_subject),
        "p0": int(dataset.Z.shape[1]),
        "z_mode": z_mode,
        "split_group": split_group,
        "n_split_groups": int(pd.Series(manifest[split_group]).nunique()),
        "X_shape": list(dataset.X.shape),
        "y_shape": list(dataset.y.shape),
        "Z_shape": list(dataset.Z.shape),
        "t_shape": list(dataset.t.shape),
        "eligible_signal_h": list(eligible_signal_h),
        "variance_h_candidates": list(variance_candidates),
        "best_signal_h": best_signal_h,
        "best_variance_h": best_variance_h,
        "signal_cv_score": selected_score(signal_scores, best_signal_h),
        "variance_cv_score": selected_score(variance_scores, best_variance_h),
        "rho_hat": float(fit_result.covariance.rho_hat),
        "sigma2_hat_mean": float(np.mean(fit_result.covariance.sigma2_hat_t)),
        "beta_hat_shape": list(np.asarray(fit_result.beta_hat).shape),
        "runner_grouped_cv": bool(use_grouped_runner_cv),
    }
    write_json(task_dir / "result.json", result)
    write_json(
        task_dir / "meta.json",
        {
            "config": config,
            "feature_meta": feature_meta,
            "manifest_columns": list(manifest.columns),
            "model_meta": fit_result.meta,
            "initial_meta": fit_result.initial.meta,
            "covariance_meta": fit_result.covariance.meta,
        },
    )
    return result, run_dir, package_dir, config_path


def main() -> None:
    args = parse_args()
    result: dict[str, Any] | None = None
    run_dir: Path | None = None
    package_dir: Path | None = None
    config_path: Path | None = None
    try:
        result, run_dir, package_dir, config_path = run_task(args)
    except Exception as exc:  # noqa: BLE001 - runner must persist task-level failures.
        started_at = datetime.now(timezone.utc)
        config_path = resolve_path(args.bandwidth_config)
        feature_root = resolve_path(args.feature_root)
        run_root = resolve_path(args.run_root)
        run_dir = run_root / str(args.run_name)
        task_dir = run_dir / f"task_{int(args.task_id):04d}"
        task_dir.mkdir(parents=True, exist_ok=True)
        package_dir = feature_dir(feature_root, args.image_type, args.S, args.R)
        result = {
            **build_result_base(
                args=args,
                config_path=config_path,
                package_dir=package_dir,
                task_dir=task_dir,
                started_at=started_at,
            ),
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        write_json(task_dir / "result.json", result)
        (task_dir / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        if args.fail_on_error:
            raise
    finally:
        if result is not None and run_dir is not None and package_dir is not None and config_path is not None:
            update_run_manifest(
                run_dir,
                {
                    "task_id": result.get("task_id"),
                    "image_type": result.get("image_type"),
                    "S": result.get("S"),
                    "R": result.get("R"),
                    "status": result.get("status"),
                    "feature_dir": rel_to_repo(package_dir),
                    "output_dir": result.get("output_dir"),
                    "bandwidth_config": rel_to_repo(config_path),
                    "started_at_utc": result.get("started_at_utc"),
                    "finished_at_utc": result.get("finished_at_utc"),
                    "elapsed_seconds": result.get("elapsed_seconds"),
                },
            )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
