"""Run patient-cluster bootstrap inference for GRAPE VCTR coefficients."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
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
    load_feature_dataset,
    subset_dataset,
)
from src.models import PairedEyeVCTRModel  # noqa: E402


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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
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
    if str(config["z_mode"]) != "none":
        raise ValueError("This pilot implementation currently requires z_mode='none'.")
    if str(config["resample_unit"]) != "subject_id":
        raise ValueError("Patient-cluster bootstrap requires resample_unit='subject_id'.")
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


def fit_A_on_grid(
    dataset: PairedEyeDataset,
    config: dict[str, Any],
    t_grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit stages 1-2 and evaluate the covariance-aware stage-3 A on a fixed grid."""

    model = build_model(config)
    initial = model.initial_fit_iid(dataset)
    covariance = model.estimate_covariance(dataset, initial)
    A_hat, beta_local = model.estimate_stage3_A_at(dataset, covariance, initial, t_grid)
    sigma_min_eigenvalue = float(
        np.min(np.linalg.eigvalsh(np.asarray(covariance.Sigma_hat_blocks, dtype=float)))
    )
    diagnostics = {
        "rho_hat": float(covariance.rho_hat),
        "sigma2_hat_mean": float(np.mean(covariance.sigma2_hat_t)),
        "sigma2_hat_min": float(np.min(covariance.sigma2_hat_t)),
        "sigma2_hat_max": float(np.max(covariance.sigma2_hat_t)),
        "sigma_min_eigenvalue": sigma_min_eigenvalue,
        "beta_local_max_abs": float(np.max(np.abs(beta_local))) if beta_local.size else 0.0,
    }
    return A_hat, diagnostics


def patient_cluster_resample(
    dataset: PairedEyeDataset,
    manifest: pd.DataFrame,
    rng: np.random.Generator,
    *,
    replicate: int,
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
    bootstrap_dataset = subset_dataset(dataset, indices, z_mode="none")
    bootstrap_dataset.subject_ids = np.concatenate(pair_id_chunks)
    bootstrap_dataset.meta.update(
        {
            "bootstrap_replicate": int(replicate),
            "bootstrap_resample_unit": "subject_id",
            "bootstrap_rows": int(indices.size),
        }
    )
    return bootstrap_dataset, np.asarray(sampled_patient_ids), int(np.unique(sampled_patient_ids).size)


def replicate_rng(base_seed: int, replicate: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(replicate)]))


def atomic_savez(path: Path, **arrays: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def original_fit(
    dataset: PairedEyeDataset,
    config: dict[str, Any],
    t_grid: np.ndarray,
    meta: dict[str, Any],
    output: Path,
) -> None:
    if output.exists():
        return
    start = time.perf_counter()
    A_hat, diagnostics = fit_A_on_grid(dataset, config, t_grid)
    age_meta = meta["transforms"]["t"]
    age_grid = float(age_meta["age_min"]) + t_grid * (
        float(age_meta["age_max"]) - float(age_meta["age_min"])
    )
    atomic_savez(
        output,
        t_grid=t_grid,
        age_grid=age_grid,
        A_hat=A_hat,
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
    bootstrap_dataset, sampled_ids, n_unique = patient_cluster_resample(
        dataset,
        manifest,
        rng,
        replicate=replicate,
    )
    try:
        A_hat, diagnostics = fit_A_on_grid(bootstrap_dataset, config, t_grid)
        elapsed = time.perf_counter() - start
        atomic_savez(
            output,
            replicate=np.array(replicate),
            A_hat=A_hat,
            sampled_original_subject_ids=np.asarray(sampled_ids, dtype=str),
            n_unique_original_subjects=np.array(n_unique),
            n_bootstrap_rows=np.array(bootstrap_dataset.n_subject),
            elapsed_seconds=np.array(elapsed),
            **{key: np.array(value) for key, value in diagnostics.items()},
        )
        if failure_output.exists():
            failure_output.unlink()
        return {
            "replicate": replicate,
            "status": "success",
            "n_bootstrap_rows": bootstrap_dataset.n_subject,
            "n_unique_original_subjects": n_unique,
            "elapsed_seconds": elapsed,
            **diagnostics,
            "error_type": "",
            "error_message": "",
        }
    except Exception as exc:  # noqa: BLE001 - persist failures so long bootstrap runs can resume.
        elapsed = time.perf_counter() - start
        failure = {
            "replicate": replicate,
            "status": "failure",
            "n_bootstrap_rows": bootstrap_dataset.n_subject,
            "n_unique_original_subjects": n_unique,
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
                        "n_unique_original_subjects": int(data["n_unique_original_subjects"]),
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
        "Patient-cluster bootstrap pilot for fixed-hyperparameter GRAPE coefficient functions.",
        "",
        "## Fixed model",
        "",
        f"- Image: `{config['image_type']}`",
        f"- `S={config['S']}`, `R={int(config['R'])}`",
        f"- `h={float(config['signal_h'])}`, `hbar={float(config['variance_hbar'])}`",
        f"- `z_mode={config['z_mode']}`",
        f"- `ridge={float(config.get('ridge', 0.0)):.1e}`",
        f"- Requested bootstrap replicates: `{total}`",
        "- Resampling unit: real patient `subject_id`; all visits and both eyes are retained",
        "- CP features and preprocessing transforms are fixed",
        "",
        "## Outputs",
        "",
        "- `original_fit.npz`: full-sample coefficient estimate on the common grid",
        "- `replicates/`: one restartable checkpoint per successful replicate",
        "- `failures/`: persisted exception details",
        "- `replicate_status.csv`: replicate-level numerical diagnostics",
        "- `bootstrap_draws.npz` and `coefficient_summary.csv`: generated by aggregation",
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
    dataset = subset_dataset(dataset_full, np.arange(dataset_full.n_subject), z_mode="none")
    t_grid = build_t_grid(config)
    original_fit(dataset, config, t_grid, meta, run_dir / "original_fit.npz")
    write_readme(run_dir, config, total)
    write_json(
        run_dir / "run_metadata.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config": rel_to_repo(config_path),
            "feature_package": rel_to_repo(package_dir),
            "run_dir": rel_to_repo(run_dir),
            "requested_replicates_this_invocation": total,
            "n_pairs": dataset.n_subject,
            "n_patients": int(manifest["subject_id"].nunique()),
            "A_shape_per_fit": [int(t_grid.size), int(config["R"]), int(np.prod([int(x) for x in str(config["S"]).split("x")]))],
        },
    )

    for replicate in range(total):
        output = replicate_dir / f"bootstrap_{replicate:04d}.npz"
        failure = failure_dir / f"bootstrap_{replicate:04d}.json"
        if output.exists() and args.resume:
            print(f"[{replicate + 1}/{total}] skip completed replicate={replicate}", flush=True)
            continue
        row = run_replicate(
            replicate,
            dataset=dataset,
            manifest=manifest,
            config=config,
            t_grid=t_grid,
            output=output,
            failure_output=failure,
        )
        print(
            f"[{replicate + 1}/{total}] {row['status']} replicate={replicate} "
            f"rows={row['n_bootstrap_rows']} unique_patients={row['n_unique_original_subjects']} "
            f"elapsed={row['elapsed_seconds']:.2f}s",
            flush=True,
        )

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
