"""Run configurable row- or patient-bootstrap inference for GRAPE VCTR coefficients."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

# Each process performs dense linear algebra. Prevent BLAS oversubscription
# when bootstrap replicates are distributed across process workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import PairedEyeDataset  # noqa: E402
from src.experiments.grape.evaluation.compare_models import (  # noqa: E402
    feature_dir,
    load_feature_dataset,
    subset_dataset,
)
from src.experiments.grape.evaluation.vf_pca import FoldVFPCATransformer, split_sex_vf  # noqa: E402
from src.models import PairedEyeVCTRModel  # noqa: E402
from src.models.covariance import invert_blocks, smooth_variance_curve  # noqa: E402


GRAPE_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = GRAPE_ROOT / "data" / "features"
RUN_ROOT = GRAPE_ROOT / "runs" / "coefficient_bootstrap"
DEFAULT_CONFIG = GRAPE_ROOT / "configs" / "coefficient_bootstrap" / "roi_x_only_at_pilot_b100.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replicates", type=int, default=None, help="Override B for smoke testing.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--max-workers", type=int, default=None, help="Override config max_workers.")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--original-only",
        action="store_true",
        help="Fit and save only the full-sample A(t) curve; do not run bootstrap replicates.",
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
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    required = (
        "name",
        "image_type",
        "S",
        "R",
        "signal_h",
        "variance_hbar",
        "bootstrap_replicates",
        "bootstrap_seed",
        "resample_unit",
        "t_grid",
    )
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    if str(config["z_mode"]) not in {"none", "full", "selected", "pca_gender"}:
        raise ValueError("z_mode must be 'none', 'full', 'selected', or 'pca_gender'.")
    if str(config["z_mode"]) == "selected":
        z_columns = config.get("z_columns")
        if not isinstance(z_columns, list) or not z_columns:
            raise ValueError("z_mode='selected' requires a non-empty z_columns list.")
        if len(z_columns) != len(set(str(value) for value in z_columns)):
            raise ValueError("z_columns must not contain duplicates.")
    if str(config["z_mode"]) == "pca_gender":
        if int(config.get("pca_components", 0)) <= 0:
            raise ValueError("z_mode='pca_gender' requires positive pca_components.")
        if str(config.get("pca_weighting", "subject_equal")) not in {"subject_equal", "row_equal"}:
            raise ValueError("pca_weighting must be 'subject_equal' or 'row_equal'.")
    if str(config["resample_unit"]) not in {"subject_id", "pair_id"}:
        raise ValueError("resample_unit must be 'subject_id' or 'pair_id'.")
    if not bool(config.get("keep_cp_features_fixed", False)):
        raise ValueError("This workflow requires keep_cp_features_fixed=true.")
    if not bool(config.get("keep_transforms_fixed", False)):
        raise ValueError("This workflow requires keep_transforms_fixed=true.")
    if int(config["bootstrap_replicates"]) <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if int(config["R"]) <= 0:
        raise ValueError("R must be positive.")
    if float(config["signal_h"]) <= 0 or float(config["variance_hbar"]) <= 0:
        raise ValueError("signal_h and variance_hbar must be positive.")
    if int(config.get("max_workers", 1)) <= 0:
        raise ValueError("max_workers must be positive.")
    return config


def build_t_grid(config: dict[str, Any]) -> np.ndarray:
    spec = dict(config["t_grid"])
    if str(spec.get("method")) != "linear":
        raise ValueError("Only t_grid.method='linear' is currently supported.")
    n_points = int(spec["num_points"])
    if n_points < 2:
        raise ValueError("t_grid.num_points must be at least 2.")
    lower = float(spec["min"])
    upper = float(spec["max"])
    if not lower < upper:
        raise ValueError("t_grid.min must be smaller than t_grid.max.")
    return np.linspace(lower, upper, num=n_points, dtype=float)


def build_model(config: dict[str, Any]) -> PairedEyeVCTRModel:
    return PairedEyeVCTRModel(
        covariance_mode=str(config.get("covariance_mode", "exchangeable_varying_sigma")),
        a_eval_mode="full",
        signal_bandwidth=float(config["signal_h"]),
        variance_bandwidth=float(config["variance_hbar"]),
        ridge=float(config.get("ridge", 0.0)),
    )


def fit_coefficients_on_grid(
    dataset: PairedEyeDataset,
    config: dict[str, Any],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the paired model and return final A(t), beta, and diagnostics."""

    model = build_model(config)
    initial = model.initial_fit_iid(dataset)
    covariance = model.estimate_covariance(dataset, initial)
    if str(config["z_mode"]) != "none":
        final = model.refit_with_covariance(dataset, covariance, initial)
        beta_hat = np.asarray(final.beta_hat, dtype=float)
    else:
        beta_hat = np.empty(0, dtype=float)
    A_hat, beta_local = model.estimate_stage3_A_at(dataset, covariance, initial, t_grid)
    if covariance.residual_pairs is None:
        raise ValueError("Covariance estimate must retain paired residuals.")
    sigma2_hat_grid = smooth_variance_curve(
        residual_pairs=covariance.residual_pairs,
        t=dataset.t,
        t_eval=t_grid,
        bandwidth=float(config["variance_hbar"]),
    )
    local_support_pairs = np.asarray(
        [np.count_nonzero(np.abs(dataset.t - t0) < float(config["signal_h"])) for t0 in t_grid],
        dtype=int,
    )
    variance_support_pairs = np.asarray(
        [np.count_nonzero(np.abs(dataset.t - t0) < float(config["variance_hbar"])) for t0 in t_grid],
        dtype=int,
    )
    sigma_min_eigenvalue = float(
        np.min(np.linalg.eigvalsh(np.asarray(covariance.Sigma_hat_blocks, dtype=float)))
    )
    local_design_condition_numbers = stage3_local_design_condition_numbers(
        dataset,
        model,
        covariance.Sigma_hat_blocks,
        t_grid,
        bandwidth=float(config["signal_h"]),
    )
    diagnostics = {
        "rho_hat": float(covariance.rho_hat),
        "sigma2_hat_mean": float(np.mean(covariance.sigma2_hat_t)),
        "sigma2_hat_min": float(np.min(covariance.sigma2_hat_t)),
        "sigma2_hat_max": float(np.max(covariance.sigma2_hat_t)),
        "sigma_min_eigenvalue": sigma_min_eigenvalue,
        "beta_local_max_abs": float(np.max(np.abs(beta_local))) if beta_local.size else 0.0,
        "beta_hat_max_abs": float(np.max(np.abs(beta_hat))) if beta_hat.size else 0.0,
        "sigma2_hat_grid": sigma2_hat_grid,
        "local_support_pairs": local_support_pairs,
        "variance_support_pairs": variance_support_pairs,
        "local_design_condition_numbers": local_design_condition_numbers,
    }
    return A_hat, beta_hat, diagnostics


def stage3_local_design_condition_numbers(
    dataset: PairedEyeDataset,
    model: PairedEyeVCTRModel,
    sigma_blocks: np.ndarray,
    t_grid: np.ndarray,
    *,
    bandwidth: float,
) -> np.ndarray:
    """Condition numbers of the ridge-stabilized stage-3 GLS systems."""

    x_mat = dataset.X.reshape(dataset.n_subject, 2, -1)
    n_features = x_mat.shape[2]
    p0 = dataset.Z.shape[1]
    dimension = p0 + 2 * n_features
    sigma_inverse = invert_blocks(sigma_blocks)
    result = np.empty(len(t_grid), dtype=float)
    for grid_index, t0 in enumerate(t_grid):
        lhs = np.zeros((dimension, dimension), dtype=float)
        for subject in range(dataset.n_subject):
            kernel_weight = model._kernel_scalar_weight(  # noqa: SLF001 - mirrors public stage-3 calculation.
                dataset.t[subject],
                float(t0),
                bandwidth,
            )
            if kernel_weight <= 0:
                continue
            scaled_time = (dataset.t[subject] - float(t0)) / bandwidth
            design = np.zeros((2, dimension), dtype=float)
            if p0:
                design[:, :p0] = dataset.Z[subject]
            design[:, p0 : p0 + n_features] = x_mat[subject]
            design[:, p0 + n_features :] = x_mat[subject] * scaled_time
            weight = kernel_weight * sigma_inverse[subject]
            lhs += design.T @ weight @ design
        stabilized = lhs + float(model.ridge) * np.eye(dimension)
        result[grid_index] = float(np.linalg.cond(stabilized))
    return result


def fit_A_on_grid(
    dataset: PairedEyeDataset,
    config: dict[str, Any],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Backward-compatible X-only coefficient-grid helper."""

    A_hat, beta_hat, diagnostics = fit_coefficients_on_grid(dataset, config, t_grid)
    if beta_hat.size:
        raise ValueError("fit_A_on_grid is only valid when z_mode='none'.")
    return A_hat, diagnostics


def patient_cluster_resample(
    dataset: PairedEyeDataset,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
    *,
    replicate: int,
    z_mode: str = "none",
) -> tuple[PairedEyeDataset, np.ndarray, int]:
    """Resample patients and retain every visit and both eyes for each draw."""

    if len(manifest) != dataset.n_subject:
        raise ValueError("manifest and dataset must have the same row count.")
    if "subject_id" not in manifest or "pair_id" not in manifest:
        raise ValueError("manifest must contain subject_id and pair_id.")

    patient_ids = pd.unique(manifest["subject_id"])
    sampled_patient_ids = rng.choice(patient_ids, size=len(patient_ids), replace=True)
    index_chunks: list[np.ndarray] = []
    pair_id_chunks: list[np.ndarray] = []
    manifest_subjects = manifest["subject_id"].to_numpy()
    manifest_pairs = manifest["pair_id"].astype(str).to_numpy()

    for draw_idx, patient_id in enumerate(sampled_patient_ids):
        indices = np.flatnonzero(manifest_subjects == patient_id)
        if indices.size == 0:
            raise RuntimeError(f"Sampled patient {patient_id!r} has no visits.")
        index_chunks.append(indices)
        pair_id_chunks.append(
            np.asarray(
                [f"boot{replicate:04d}_draw{draw_idx:04d}_{pair_id}" for pair_id in manifest_pairs[indices]],
                dtype=str,
            )
        )

    indices = np.concatenate(index_chunks)
    bootstrap_dataset = subset_dataset(dataset, indices, z_mode="none" if z_mode == "none" else "full")
    bootstrap_dataset.subject_ids = np.concatenate(pair_id_chunks)
    bootstrap_dataset.meta.update(
        {
            "bootstrap_replicate": int(replicate),
            "bootstrap_resample_unit": "subject_id",
            "bootstrap_rows": int(indices.size),
        }
    )
    return bootstrap_dataset, np.asarray(sampled_patient_ids), int(np.unique(sampled_patient_ids).size)


def pair_row_resample(
    dataset: PairedEyeDataset,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
    *,
    replicate: int,
    z_mode: str = "none",
) -> tuple[PairedEyeDataset, np.ndarray, int]:
    """Resample paired-visit rows while retaining the OD/OS pair within each row."""

    if len(manifest) != dataset.n_subject:
        raise ValueError("manifest and dataset must have the same row count.")
    if "pair_id" not in manifest:
        raise ValueError("manifest must contain pair_id.")

    indices = rng.choice(np.arange(dataset.n_subject), size=dataset.n_subject, replace=True)
    pair_ids = manifest["pair_id"].astype(str).to_numpy()
    sampled_pair_ids = pair_ids[indices]
    bootstrap_dataset = subset_dataset(dataset, indices, z_mode="none" if z_mode == "none" else "full")
    bootstrap_dataset.subject_ids = np.asarray(
        [
            f"boot{replicate:04d}_draw{draw_idx:04d}_{pair_id}"
            for draw_idx, pair_id in enumerate(sampled_pair_ids)
        ],
        dtype=str,
    )
    bootstrap_dataset.meta.update(
        {
            "bootstrap_replicate": int(replicate),
            "bootstrap_resample_unit": "pair_id",
            "bootstrap_rows": int(indices.size),
        }
    )
    return bootstrap_dataset, sampled_pair_ids, int(np.unique(sampled_pair_ids).size)


def select_z_columns(
    dataset: PairedEyeDataset,
    z_names: list[str],
    config: dict[str, Any],
    manifest: pd.DataFrame | None = None,
) -> tuple[PairedEyeDataset, list[str]]:
    """Apply the configured covariate subset or fixed PCA transform."""

    mode = str(config["z_mode"])
    indices = np.arange(dataset.n_subject)
    if mode == "none":
        return subset_dataset(dataset, indices, z_mode="none"), []
    if dataset.Z.shape[1] != len(z_names):
        raise ValueError("Feature-package Z columns do not match Z.npy width.")
    if mode == "full":
        return subset_dataset(dataset, indices, z_mode="full"), list(z_names)

    if mode == "pca_gender":
        if manifest is None or "subject_id" not in manifest:
            raise ValueError("z_mode='pca_gender' requires a manifest with subject_id.")
        if len(manifest) != dataset.n_subject:
            raise ValueError("manifest and dataset must have the same row count.")
        if not z_names or z_names[0] != "is_female":
            raise ValueError("PCA mode expects is_female followed by VF columns.")
        sex, vf = split_sex_vf(dataset.Z)
        transformer = FoldVFPCATransformer.fit(
            vf,
            manifest["subject_id"].to_numpy(),
            n_components=int(config["pca_components"]),
            weighting=str(config.get("pca_weighting", "subject_equal")),
        )
        selected = subset_dataset(dataset, indices, z_mode="full")
        selected.Z = np.column_stack([sex, transformer.transform(vf)])
        selected_names = ["is_female"] + [
            f"vf_pc_{component:02d}" for component in range(1, transformer.n_components + 1)
        ]
        selected.meta["vf_pca_transform"] = {
            "mean": transformer.mean_.copy(),
            "scale": transformer.scale_.copy(),
            "components": transformer.components_.copy(),
            "explained_variance_ratio": transformer.explained_variance_ratio_.copy(),
            "singular_values": transformer.singular_values_.copy(),
            "vf_names": np.asarray(z_names[1:], dtype=str),
            "weighting": str(config.get("pca_weighting", "subject_equal")),
            "n_training_rows": transformer.n_training_rows_,
            "n_training_groups": transformer.n_training_groups_,
        }
        return selected, selected_names

    requested = [str(value) for value in config["z_columns"]]
    missing = [name for name in requested if name not in z_names]
    if missing:
        raise ValueError(f"Requested z_columns are absent from the feature package: {missing}")
    column_indices = np.asarray([z_names.index(name) for name in requested], dtype=int)
    selected = subset_dataset(dataset, indices, z_mode="full")
    selected.Z = selected.Z[:, column_indices]
    selected.meta["selected_z_columns"] = requested
    return selected, requested


def replicate_rng(base_seed: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(replicate)]))


def atomic_savez(path: Path, **arrays: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_pca_transform(dataset: PairedEyeDataset, run_dir: Path) -> None:
    """Persist the fixed full-sample PCA basis used by every bootstrap draw."""

    transform = dataset.meta.get("vf_pca_transform")
    if transform is None:
        return
    np.savez_compressed(
        run_dir / "pca_transform.npz",
        mean=np.asarray(transform["mean"], dtype=float),
        scale=np.asarray(transform["scale"], dtype=float),
        components=np.asarray(transform["components"], dtype=float),
        explained_variance_ratio=np.asarray(transform["explained_variance_ratio"], dtype=float),
        singular_values=np.asarray(transform["singular_values"], dtype=float),
        vf_names=np.asarray(transform["vf_names"], dtype=str),
        weighting=np.asarray(str(transform["weighting"])),
        n_training_rows=np.asarray(int(transform["n_training_rows"])),
        n_training_groups=np.asarray(int(transform["n_training_groups"])),
    )
    loading_rows: list[dict[str, Any]] = []
    components = np.asarray(transform["components"], dtype=float)
    vf_names = np.asarray(transform["vf_names"], dtype=str)
    explained = np.asarray(transform["explained_variance_ratio"], dtype=float)
    for pc_index in range(components.shape[0]):
        for variable, loading in zip(vf_names, components[pc_index], strict=True):
            loading_rows.append(
                {
                    "pc": pc_index + 1,
                    "variable": str(variable),
                    "loading": float(loading),
                    "explained_variance_ratio": float(explained[pc_index]),
                }
            )
    pd.DataFrame(loading_rows).to_csv(run_dir / "pca_loadings.csv", index=False)


def original_fit(
    dataset: PairedEyeDataset,
    config: dict[str, Any],
    t_grid: np.ndarray,
    meta: dict[str, Any],
    z_names: list[str],
    output: Path,
) -> None:
    if output.exists():
        with np.load(output) as existing:
            required = {"sigma2_hat_grid", "local_support_pairs", "variance_support_pairs"}
            if required.issubset(existing.files):
                return
    start = time.perf_counter()
    A_hat, beta_hat, diagnostics = fit_coefficients_on_grid(dataset, config, t_grid)
    age_meta = meta["transforms"]["t"]
    age_grid = float(age_meta["age_min"]) + t_grid * (
        float(age_meta["age_max"]) - float(age_meta["age_min"])
    )
    atomic_savez(
        output,
        t_grid=t_grid,
        age_grid=age_grid,
        A_hat=A_hat,
        beta_hat=beta_hat,
        Z_names=np.asarray(z_names, dtype=str),
        y_sd=np.array(float(meta["transforms"]["y"]["sd"])),
        elapsed_seconds=np.array(time.perf_counter() - start),
        **{key: np.array(value) for key, value in diagnostics.items()},
    )


def run_replicate(
    replicate: int,
    *,
    dataset: PairedEyeDataset,
    manifest: pd.DataFrame,
    config: dict[str, Any],
    t_grid: np.ndarray,
    output: Path,
    failure_output: Path,
) -> dict[str, Any]:
    start = time.perf_counter()
    rng = replicate_rng(int(config["bootstrap_seed"]), replicate)
    resample_unit = str(config["resample_unit"])
    resampler = patient_cluster_resample if resample_unit == "subject_id" else pair_row_resample
    bootstrap_dataset, sampled_ids, n_unique = resampler(
        dataset, manifest, rng, replicate=replicate, z_mode=str(config["z_mode"])
    )
    try:
        A_hat, beta_hat, diagnostics = fit_coefficients_on_grid(bootstrap_dataset, config, t_grid)
        elapsed = time.perf_counter() - start
        resample_arrays: dict[str, Any] = {
            "sampled_original_ids": np.asarray(sampled_ids, dtype=str),
            "n_unique_original_units": np.array(n_unique),
            "resample_unit": np.array(resample_unit),
        }
        if resample_unit == "subject_id":
            resample_arrays.update(
                sampled_original_subject_ids=np.asarray(sampled_ids, dtype=str),
                n_unique_original_subjects=np.array(n_unique),
            )
        else:
            resample_arrays.update(
                sampled_original_pair_ids=np.asarray(sampled_ids, dtype=str),
                n_unique_original_pairs=np.array(n_unique),
            )
        atomic_savez(
            output,
            replicate=np.array(replicate),
            A_hat=A_hat,
            beta_hat=beta_hat,
            n_bootstrap_rows=np.array(bootstrap_dataset.n_subject),
            elapsed_seconds=np.array(elapsed),
            **resample_arrays,
            **{key: np.array(value) for key, value in diagnostics.items()},
        )
        if failure_output.exists():
            failure_output.unlink()
        status_diagnostics = {
            key: value for key, value in diagnostics.items() if np.asarray(value).ndim == 0
        }
        return {
            "replicate": replicate,
            "status": "success",
            "n_bootstrap_rows": bootstrap_dataset.n_subject,
            "resample_unit": resample_unit,
            "n_unique_original_units": n_unique,
            "elapsed_seconds": elapsed,
            **status_diagnostics,
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:  # noqa: BLE001 - persist failures so long bootstrap runs can resume.
        elapsed = time.perf_counter() - start
        failure = {
            "replicate": replicate,
            "status": "failure",
            "n_bootstrap_rows": bootstrap_dataset.n_subject,
            "resample_unit": resample_unit,
            "n_unique_original_units": n_unique,
            "elapsed_seconds": elapsed,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        write_json(failure_output, failure)
        return failure


def scan_status(replicate_dir: Path, failure_dir: Path, total: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for replicate in range(total):
        output = replicate_dir / f"bootstrap_{replicate:04d}.npz"
        failure = failure_dir / f"bootstrap_{replicate:04d}.json"
        if output.exists():
            with np.load(output) as data:
                rows.append(
                    {
                        "replicate": replicate,
                        "status": "success",
                        "n_bootstrap_rows": int(data["n_bootstrap_rows"]),
                        "resample_unit": str(data["resample_unit"]) if "resample_unit" in data else "subject_id",
                        "n_unique_original_units": int(
                            data["n_unique_original_units"]
                            if "n_unique_original_units" in data
                            else data["n_unique_original_subjects"]
                        ),
                        "rho_hat": float(data["rho_hat"]),
                        "sigma2_hat_mean": float(data["sigma2_hat_mean"]),
                        "sigma2_hat_min": float(data["sigma2_hat_min"]),
                        "sigma2_hat_max": float(data["sigma2_hat_max"]),
                        "sigma_min_eigenvalue": float(data["sigma_min_eigenvalue"]),
                        "elapsed_seconds": float(data["elapsed_seconds"]),
                        "error_type": "",
                        "error_message": "",
                    }
                )
        elif failure.exists():
            rows.append(json.loads(failure.read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def write_readme(run_dir: Path, config: dict[str, Any], total: int) -> None:
    is_pilot = "pilot" in str(config["name"]).lower()
    lines = [
        f"# {config['name']}",
        "",
        "Bootstrap pilot for fixed-hyperparameter GRAPE coefficient functions.",
        "",
        "## Fixed model",
        "",
        f"- Image: `{config['image_type']}`",
        f"- `S={config['S']}`, `R={int(config['R'])}`",
        f"- `h={float(config['signal_h'])}`, `hbar={float(config['variance_hbar'])}`",
        f"- `z_mode={config['z_mode']}`",
        f"- `ridge={float(config.get('ridge', 0.0)):.1e}`",
        f"- Requested bootstrap replicates: `{total}`",
        f"- Resampling unit: `{config['resample_unit']}`",
        (
            "- Each sampled paired-visit row retains its OD/OS outcomes"
            if str(config["resample_unit"]) == "pair_id"
            else "- Each sampled patient retains all visits and both eyes"
        ),
        "- CP features and preprocessing transforms are fixed",
        "",
        "## Outputs",
        "",
        "- `original_fit.npz`: full-sample coefficient estimate on the common grid",
        "- `replicates/`: one restartable checkpoint per successful replicate",
        "- `failures/`: persisted exception details",
        "- `replicate_status.csv`: replicate-level numerical diagnostics",
        "- `bootstrap_draws.npz` and `coefficient_summary.csv`: generated by aggregation",
        "- `beta_summary_all.csv`: regression-coefficient audit table when `z_mode` is not `none`",
        "- `pca_transform.npz` and `pca_loadings.csv`: fixed PCA definition when `z_mode=pca_gender`",
        "- `variance_summary.csv`: fixed-grid sigma-squared/sigma estimates and pointwise intervals",
        "- `figures/`: generated from the aggregated results",
        "",
        (
            "This run is a workflow and numerical-stability pilot, not final manuscript inference."
            if is_pilot
            else "This is the final configured pointwise-percentile bootstrap run; inspect diagnostics before export."
        ),
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)
    total = int(args.replicates if args.replicates is not None else config["bootstrap_replicates"])
    if total <= 0:
        raise ValueError("replicates must be positive.")
    max_workers = int(args.max_workers if args.max_workers is not None else config.get("max_workers", 1))
    if max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    feature_root = resolve_path(args.feature_root)
    run_root = resolve_path(args.run_root)
    run_name = args.run_name or str(config["name"])
    run_dir = run_root / run_name
    replicate_dir = run_dir / "replicates"
    failure_dir = run_dir / "failures"
    replicate_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    copied_config = run_dir / "config.json"
    if copied_config.exists():
        existing = json.loads(copied_config.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(f"Existing run config differs from {config_path}.")
    else:
        shutil.copy2(config_path, copied_config)

    package_dir = feature_dir(
        feature_root,
        str(config["image_type"]),
        str(config["S"]),
        int(config["R"]),
    )
    dataset_full, manifest, meta = load_feature_dataset(package_dir)
    all_z_names = [str(value) for value in meta["transforms"]["Z"]["columns"]]
    dataset, z_names = select_z_columns(
        dataset_full,
        all_z_names,
        config,
        manifest,
    )
    t_grid = build_t_grid(config)
    original_fit(dataset, config, t_grid, meta, z_names, run_dir / "original_fit.npz")
    save_pca_transform(dataset, run_dir)
    write_readme(run_dir, config, total)
    write_json(
        run_dir / "run_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": rel_to_repo(config_path),
            "feature_package": rel_to_repo(package_dir),
            "run_dir": rel_to_repo(run_dir),
            "requested_replicates_this_invocation": total,
            "max_workers": max_workers,
            "n_pairs": dataset.n_subject,
            "n_patients": int(manifest["subject_id"].nunique()),
            "A_shape_per_fit": [int(t_grid.size), int(config["R"]), int(np.prod([int(x) for x in str(config["S"]).split("x")]))],
            "beta_shape_per_fit": [int(dataset.Z.shape[1])],
            "Z_names": z_names,
            "pca_transform_fixed_across_bootstrap": str(config["z_mode"]) == "pca_gender",
            "resample_unit": str(config["resample_unit"]),
        },
    )

    if args.original_only:
        print(
            json.dumps(
                {
                    "run_dir": rel_to_repo(run_dir),
                    "original_fit": rel_to_repo(run_dir / "original_fit.npz"),
                    "bootstrap_started": False,
                },
                indent=2,
            )
        )
        return

    pending: list[int] = []
    for replicate in range(total):
        output = replicate_dir / f"bootstrap_{replicate:04d}.npz"
        if output.exists() and args.resume:
            print(f"[{replicate + 1}/{total}] skip completed replicate={replicate}", flush=True)
        else:
            pending.append(replicate)

    def print_row(row: dict[str, Any], completed: int) -> None:
        print(
            f"[{completed}/{total}] {row['status']} replicate={row['replicate']} "
            f"rows={row['n_bootstrap_rows']} unique_units={row['n_unique_original_units']} "
            f"elapsed={row['elapsed_seconds']:.2f}s",
            flush=True,
        )

    already_complete = total - len(pending)
    if max_workers == 1:
        for completed_offset, replicate in enumerate(pending, start=1):
            row = run_replicate(
                replicate,
                dataset=dataset,
                manifest=manifest,
                config=config,
                t_grid=t_grid,
                output=replicate_dir / f"bootstrap_{replicate:04d}.npz",
                failure_output=failure_dir / f"bootstrap_{replicate:04d}.json",
            )
            print_row(row, already_complete + completed_offset)
    elif pending:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    run_replicate,
                    replicate,
                    dataset=dataset,
                    manifest=manifest,
                    config=config,
                    t_grid=t_grid,
                    output=replicate_dir / f"bootstrap_{replicate:04d}.npz",
                    failure_output=failure_dir / f"bootstrap_{replicate:04d}.json",
                ): replicate
                for replicate in pending
            }
            for completed_offset, future in enumerate(as_completed(futures), start=1):
                print_row(future.result(), already_complete + completed_offset)

    status = scan_status(replicate_dir, failure_dir, total)
    status.to_csv(run_dir / "replicate_status.csv", index=False)
    n_success = int((status["status"] == "success").sum()) if not status.empty else 0
    n_failure = int((status["status"] == "failure").sum()) if not status.empty else 0
    print(
        json.dumps(
            {
                "run_dir": rel_to_repo(run_dir),
                "requested_replicates": total,
                "successful_replicates": n_success,
                "failed_replicates": n_failure,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
