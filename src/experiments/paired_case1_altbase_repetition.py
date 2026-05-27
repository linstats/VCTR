"""Repeated paired Case 1 altbase simulation driver."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dgps import PairedCase1AltbaseDGP
from src.metrics import (
    beta_mae,
    beta_rmse,
    miae,
    rho_abs_error,
    rmise,
    sigma2_miae,
    sigma2_rmise,
    sigma_frobenius_error,
)
from src.models import PairedEyeVCTRModel


DEFAULT_COEF_TYPES = ("base1", "base2", "base3", "base4")


@dataclass(slots=True)
class PairedCase1AltbaseRecord:
    n_subject: int
    coef_type: str
    rep: int
    seed: int
    success: int
    error_message: str
    elapsed_seconds: float
    covariance_mode: str
    signal_bandwidth_input: str
    signal_bandwidth_method: str
    best_signal_bandwidth: float | None
    variance_bandwidth_input: str
    variance_bandwidth_method: str
    best_variance_bandwidth: float | None
    sigma2_true: float
    rho_true: float
    miae_iid: float | None
    rmise_iid: float | None
    beta_mae_iid: float | None
    beta_rmse_iid: float | None
    miae_final: float | None
    rmise_final: float | None
    beta_mae_final: float | None
    beta_rmse_final: float | None
    sigma2_miae: float | None
    sigma2_rmise: float | None
    rho_abs_error: float | None
    Sigma_fro_error: float | None


def format_duration(seconds: float) -> str:
    """Format a duration in seconds for progress logging."""

    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subject-values", type=int, nargs="+", default=[1000, 1400])
    parser.add_argument("--coef-types", type=str, nargs="+", default=list(DEFAULT_COEF_TYPES))
    parser.add_argument("--n-rep", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=123)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--S", type=int, default=25)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument(
        "--beta",
        type=str,
        default="2.0,1.0,-1.0,0.5",
        help="Comma-separated beta vector. Default matches the altbase design.",
    )
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional list of rho values. If provided, overrides --rho for batch runs.",
    )
    parser.add_argument("--covariance-mode", type=str, default="exchangeable_varying_sigma")
    parser.add_argument("--signal-bandwidth", type=float, default=0.18)
    parser.add_argument("--signal-bandwidth-method", type=str, default="stage1_kfold_cv")
    parser.add_argument(
        "--signal-bandwidth-grid",
        type=str,
        default=None,
        help="Comma-separated signal-bandwidth candidates. If provided while --signal-bandwidth is omitted, auto CV is used.",
    )
    parser.add_argument("--variance-bandwidth", type=float, default=0.18)
    parser.add_argument("--variance-bandwidth-method", type=str, default="stage2_kfold_cv")
    parser.add_argument(
        "--variance-bandwidth-grid",
        type=str,
        default=None,
        help="Comma-separated variance-bandwidth candidates. If provided while --variance-bandwidth is omitted, auto CV is used.",
    )
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Defaults to run_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--save-estimates", action="store_true")
    return parser.parse_args()


def parse_beta(beta_arg: str, p0: int) -> tuple[float, ...]:
    parts = [part.strip() for part in beta_arg.split(",") if part.strip()]
    beta = tuple(float(part) for part in parts)
    if len(beta) != p0:
        raise ValueError("--beta must contain exactly p0 comma-separated values.")
    return beta


def parse_bandwidth_grid(grid_arg: str | None) -> tuple[float, ...] | None:
    if grid_arg is None:
        return None
    parts = [part.strip() for part in grid_arg.split(",") if part.strip()]
    if not parts:
        raise ValueError("bandwidth grid must not be empty.")
    return tuple(float(part) for part in parts)


def to_json_safe(obj):
    if isinstance(obj, dict):
        return {key: to_json_safe(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(value) for value in obj]
    if isinstance(obj, np.generic):
        return to_json_safe(obj.item())
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def default_run_name() -> str:
    """Return the default run directory name."""

    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolved_rho_values(args: argparse.Namespace) -> list[float]:
    """Return the rho values to iterate over for this run."""

    return [float(value) for value in (args.rho_values or [args.rho])]


def prepare_run_root(base_dir: Path, run_name: str | None) -> Path:
    """Create and return the run-specific output directory."""

    resolved_name = run_name or default_run_name()
    run_root = base_dir / resolved_name
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def output_paths(run_root: Path) -> dict[str, Path]:
    """Return canonical output paths for one repetition run."""

    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "raw": results_dir / "raw_results.csv",
        "summary": results_dir / "summary_results.csv",
        "progress": run_root / "progress.json",
        "config": run_root / "run_config.json",
    }


def maybe_save_dataset(output_root: Path, seed: int, dataset) -> None:
    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / f"seed_{seed:04d}_dataset.npz",
        subject_ids=dataset.subject_ids,
        eye_ids=dataset.eye_ids,
        t=dataset.t,
        Z=dataset.Z,
        X=dataset.X,
        y=dataset.y,
        A_true=dataset.A_true,
        beta_true=dataset.beta_true,
        Sigma_true=dataset.Sigma_true,
    )


def maybe_save_estimate(output_root: Path, seed: int, result) -> None:
    estimates_dir = output_root / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        estimates_dir / f"seed_{seed:04d}_estimate.npz",
        A_hat_iid=result.initial.A_hat,
        beta_hat_iid=result.initial.beta_hat,
        residuals_iid=result.initial.residuals,
        covariance_mode=result.covariance.covariance_mode,
        best_signal_bandwidth=result.initial.meta["signal_bandwidth_selected"],
        signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
        signal_bandwidth_grid=np.asarray(result.initial.meta["signal_bandwidth_grid"], dtype=float),
        signal_bandwidth_cv_scores=np.asarray(result.initial.meta["signal_bandwidth_cv_scores"], dtype=object),
        best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
        variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
        variance_bandwidth_grid=np.asarray(result.covariance.meta.get("variance_bandwidth_grid", []), dtype=float),
        variance_bandwidth_cv_scores=np.asarray(result.covariance.meta.get("variance_bandwidth_cv_scores", []), dtype=object),
        sigma2_hat_t=result.covariance.sigma2_hat_t,
        sigma2_hat_mean=result.covariance.sigma2_hat,
        rho_hat=result.covariance.rho_hat,
        Sigma_hat=result.covariance.Sigma_hat,
        Sigma_hat_blocks=result.covariance.Sigma_hat_blocks,
        A_hat_final=result.A_hat,
        beta_hat_final=result.beta_hat,
        fitted_values=result.fitted_values,
    )


def rho_label(rho: float) -> str:
    """Return a filesystem-friendly rho label."""

    return f"{rho:.3f}".replace("-", "m").replace(".", "p")


def artifact_stem(n_subject: int, coef_type: str, rho: float, rep: int, seed: int) -> str:
    """Return a unique artifact stem for one repetition."""

    return f"n{n_subject}_{coef_type}_rho{rho_label(rho)}_rep{rep:03d}_seed{seed:04d}"


def run_one(
    *,
    n_subject: int,
    coef_type: str,
    rho_true: float,
    rep: int,
    seed: int,
    beta_true: tuple[float, ...],
    args: argparse.Namespace,
    signal_bandwidth_grid: tuple[float, ...] | None,
    variance_bandwidth_grid: tuple[float, ...] | None,
    output_root: Path,
) -> PairedCase1AltbaseRecord:
    start = time.perf_counter()
    if args.signal_bandwidth is not None:
        signal_bandwidth_input = f"{args.signal_bandwidth:.12g}"
        failure_signal_bandwidth_method = "fixed"
    elif signal_bandwidth_grid is None:
        signal_bandwidth_input = f"{PairedEyeVCTRModel.DEFAULT_SIGNAL_BANDWIDTH:.12g}"
        failure_signal_bandwidth_method = "default_fixed"
    else:
        signal_bandwidth_input = "auto"
        failure_signal_bandwidth_method = args.signal_bandwidth_method
    if args.covariance_mode == "exchangeable_constant":
        variance_bandwidth_input = "not_used"
        failure_variance_bandwidth_method = "not_used"
    elif args.variance_bandwidth is not None:
        variance_bandwidth_input = f"{args.variance_bandwidth:.12g}"
        failure_variance_bandwidth_method = "fixed"
    elif variance_bandwidth_grid is None:
        variance_bandwidth_input = f"{PairedEyeVCTRModel.DEFAULT_VARIANCE_BANDWIDTH:.12g}"
        failure_variance_bandwidth_method = "default_fixed"
    else:
        variance_bandwidth_input = "auto"
        failure_variance_bandwidth_method = args.variance_bandwidth_method
    try:
        dataset = PairedCase1AltbaseDGP(
            n_subject=n_subject,
            R=args.R,
            S=args.S,
            p0=args.p0,
            coef_type=coef_type,
            beta_true=beta_true,
            sigma2=args.sigma2,
            rho=rho_true,
        ).sample(seed=seed)

        model = PairedEyeVCTRModel(
            covariance_mode=args.covariance_mode,
            signal_bandwidth=args.signal_bandwidth,
            signal_bandwidth_method=args.signal_bandwidth_method,
            signal_bandwidth_grid=signal_bandwidth_grid,
            variance_bandwidth=args.variance_bandwidth,
            variance_bandwidth_method=args.variance_bandwidth_method,
            variance_bandwidth_grid=variance_bandwidth_grid,
            ridge=args.ridge,
        )
        result = model.fit(dataset)

        if args.save_data:
            maybe_save_dataset_with_stem(
                output_root=output_root,
                stem=artifact_stem(n_subject, coef_type, rho_true, rep, seed),
                dataset=dataset,
            )
        if args.save_estimates:
            maybe_save_estimate_with_stem(
                output_root=output_root,
                stem=artifact_stem(n_subject, coef_type, rho_true, rep, seed),
                result=result,
            )

        elapsed = time.perf_counter() - start
        best_signal_bandwidth = float(result.initial.meta["signal_bandwidth_selected"])
        return PairedCase1AltbaseRecord(
            n_subject=n_subject,
            coef_type=coef_type,
            rep=rep,
            seed=seed,
            success=1,
            error_message="",
            elapsed_seconds=elapsed,
            covariance_mode=result.covariance.covariance_mode,
            signal_bandwidth_input=signal_bandwidth_input,
            signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
            best_signal_bandwidth=best_signal_bandwidth,
            variance_bandwidth_input=variance_bandwidth_input,
            variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
            best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
            sigma2_true=args.sigma2,
            rho_true=rho_true,
            miae_iid=miae(dataset.A_true, result.initial.A_hat),
            rmise_iid=rmise(dataset.A_true, result.initial.A_hat),
            beta_mae_iid=beta_mae(dataset.beta_true, result.initial.beta_hat),
            beta_rmse_iid=beta_rmse(dataset.beta_true, result.initial.beta_hat),
            miae_final=miae(dataset.A_true, result.A_hat),
            rmise_final=rmise(dataset.A_true, result.A_hat),
            beta_mae_final=beta_mae(dataset.beta_true, result.beta_hat),
            beta_rmse_final=beta_rmse(dataset.beta_true, result.beta_hat),
            sigma2_miae=sigma2_miae(args.sigma2, result.covariance.sigma2_hat_t),
            sigma2_rmise=sigma2_rmise(args.sigma2, result.covariance.sigma2_hat_t),
            rho_abs_error=rho_abs_error(rho_true, result.covariance.rho_hat),
            Sigma_fro_error=sigma_frobenius_error(dataset.Sigma_true, result.covariance.Sigma_hat),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return PairedCase1AltbaseRecord(
            n_subject=n_subject,
            coef_type=coef_type,
            rep=rep,
            seed=seed,
            success=0,
            error_message=f"{type(exc).__name__}: {exc}",
            elapsed_seconds=elapsed,
            covariance_mode=args.covariance_mode,
            signal_bandwidth_input=signal_bandwidth_input,
            signal_bandwidth_method=failure_signal_bandwidth_method,
            best_signal_bandwidth=None,
            variance_bandwidth_input=variance_bandwidth_input,
            variance_bandwidth_method=failure_variance_bandwidth_method,
            best_variance_bandwidth=None,
            sigma2_true=args.sigma2,
            rho_true=rho_true,
            miae_iid=None,
            rmise_iid=None,
            beta_mae_iid=None,
            beta_rmse_iid=None,
            miae_final=None,
            rmise_final=None,
            beta_mae_final=None,
            beta_rmse_final=None,
            sigma2_miae=None,
            sigma2_rmise=None,
            rho_abs_error=None,
            Sigma_fro_error=None,
        )


def maybe_save_dataset_with_stem(output_root: Path, stem: str, dataset) -> None:
    """Save one dataset snapshot under a unique stem."""

    data_dir = output_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        data_dir / f"{stem}_dataset.npz",
        subject_ids=dataset.subject_ids,
        eye_ids=dataset.eye_ids,
        t=dataset.t,
        Z=dataset.Z,
        X=dataset.X,
        y=dataset.y,
        A_true=dataset.A_true,
        beta_true=dataset.beta_true,
        Sigma_true=dataset.Sigma_true,
    )


def maybe_save_estimate_with_stem(output_root: Path, stem: str, result) -> None:
    """Save one estimate snapshot under a unique stem."""

    estimates_dir = output_root / "estimates"
    estimates_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        estimates_dir / f"{stem}_estimate.npz",
        A_hat_iid=result.initial.A_hat,
        beta_hat_iid=result.initial.beta_hat,
        residuals_iid=result.initial.residuals,
        covariance_mode=result.covariance.covariance_mode,
        best_signal_bandwidth=result.initial.meta["signal_bandwidth_selected"],
        signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
        signal_bandwidth_grid=np.asarray(result.initial.meta["signal_bandwidth_grid"], dtype=float),
        signal_bandwidth_cv_scores=np.asarray(result.initial.meta["signal_bandwidth_cv_scores"], dtype=object),
        best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
        variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
        variance_bandwidth_grid=np.asarray(result.covariance.meta.get("variance_bandwidth_grid", []), dtype=float),
        variance_bandwidth_cv_scores=np.asarray(result.covariance.meta.get("variance_bandwidth_cv_scores", []), dtype=object),
        sigma2_hat_t=result.covariance.sigma2_hat_t,
        sigma2_hat_mean=result.covariance.sigma2_hat,
        rho_hat=result.covariance.rho_hat,
        Sigma_hat=result.covariance.Sigma_hat,
        Sigma_hat_blocks=result.covariance.Sigma_hat_blocks,
        A_hat_final=result.A_hat,
        beta_hat_final=result.beta_hat,
        fitted_values=result.fitted_values,
    )


def write_run_config(run_root: Path, args: argparse.Namespace, total_jobs: int) -> None:
    """Write the run configuration at the root of this run directory."""

    run_config = {
        "script": "src/experiments/paired_case1_altbase_repetition.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "total_jobs": total_jobs,
        "n_subject_values": args.n_subject_values,
        "coef_types": args.coef_types,
        "n_rep": args.n_rep,
        "seed_base": args.seed_base,
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "beta": args.beta,
        "sigma2": args.sigma2,
        "rho": args.rho,
        "rho_values": args.rho_values,
        "covariance_mode": args.covariance_mode,
        "signal_bandwidth": args.signal_bandwidth,
        "signal_bandwidth_method": args.signal_bandwidth_method,
        "signal_bandwidth_grid": args.signal_bandwidth_grid,
        "variance_bandwidth": args.variance_bandwidth,
        "variance_bandwidth_method": args.variance_bandwidth_method,
        "variance_bandwidth_grid": args.variance_bandwidth_grid,
        "ridge": args.ridge,
        "n_jobs": args.n_jobs,
        "save_data": args.save_data,
        "save_estimates": args.save_estimates,
    }
    with output_paths(run_root)["config"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(run_config), f, indent=2)


def initialize_raw_csv(run_root: Path) -> None:
    """Create the raw results CSV with header."""

    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(PairedCase1AltbaseRecord.__dataclass_fields__.keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_raw_record(run_root: Path, record: PairedCase1AltbaseRecord) -> None:
    """Append one completed record to the raw results CSV."""

    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(PairedCase1AltbaseRecord.__dataclass_fields__.keys())
    with raw_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(asdict(record))


def rewrite_summary_csv(run_root: Path, summary: list[dict[str, float | int | str]]) -> None:
    """Rewrite the summary CSV from all currently completed records."""

    if not summary:
        return
    summary_path = output_paths(run_root)["summary"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_progress_snapshot(
    run_root: Path,
    *,
    completed_jobs: int,
    total_jobs: int,
    records: list[PairedCase1AltbaseRecord],
    global_start: float,
) -> None:
    """Persist incremental progress for long runs."""

    elapsed_total = time.perf_counter() - global_start
    avg_elapsed = elapsed_total / completed_jobs if completed_jobs else 0.0
    eta_seconds = avg_elapsed * (total_jobs - completed_jobs) if completed_jobs else None
    progress = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "completed_jobs": completed_jobs,
        "total_jobs": total_jobs,
        "n_success": int(sum(record.success for record in records)),
        "n_fail": int(sum(1 - record.success for record in records)),
        "elapsed_seconds": elapsed_total,
        "eta_seconds": eta_seconds,
        "latest_record": None if not records else to_json_safe(asdict(records[-1])),
    }
    with output_paths(run_root)["progress"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(progress), f, indent=2)


def build_tasks(
    args: argparse.Namespace,
    beta_true: tuple[float, ...],
    signal_bandwidth_grid: tuple[float, ...] | None,
    variance_bandwidth_grid: tuple[float, ...] | None,
    output_root: Path,
) -> list[dict]:
    """Build the full repetition task list."""

    tasks: list[dict] = []
    for n_subject in args.n_subject_values:
        for coef_type in args.coef_types:
            for rho_true in resolved_rho_values(args):
                for rep in range(args.n_rep):
                    seed = args.seed_base + rep
                    tasks.append(
                        {
                            "n_subject": n_subject,
                            "coef_type": coef_type,
                            "rho_true": rho_true,
                            "rep": rep,
                            "seed": seed,
                            "beta_true": beta_true,
                            "args": args,
                            "signal_bandwidth_grid": signal_bandwidth_grid,
                            "variance_bandwidth_grid": variance_bandwidth_grid,
                            "output_root": output_root,
                        }
                    )
    return tasks


def print_progress_line(
    *,
    record: PairedCase1AltbaseRecord,
    completed_jobs: int,
    total_jobs: int,
    global_start: float,
    n_rep: int,
) -> None:
    """Print one repetition progress line with group-local progress."""

    status = "done" if record.success else "fail"
    elapsed_total = time.perf_counter() - global_start
    avg_elapsed = elapsed_total / completed_jobs
    eta_seconds = avg_elapsed * (total_jobs - completed_jobs)
    print(
        f"[{completed_jobs}/{total_jobs}] {status} "
        f"n_subject={record.n_subject} coef={record.coef_type:10s} rho={record.rho_true:.3f} "
        f"rep={record.rep + 1}/{n_rep} seed={record.seed} "
        f"best_h={record.best_signal_bandwidth if record.best_signal_bandwidth is not None else 'NA'} "
        f"best_hbar={record.best_variance_bandwidth if record.best_variance_bandwidth is not None else 'NA'} "
        f"elapsed={format_duration(record.elapsed_seconds)} "
        f"eta={format_duration(eta_seconds)}"
    )


def run(args: argparse.Namespace) -> tuple[list[PairedCase1AltbaseRecord], Path]:
    base_output_root = Path(__file__).with_suffix("")
    beta_true = parse_beta(args.beta, args.p0)
    signal_bandwidth_grid = parse_bandwidth_grid(args.signal_bandwidth_grid)
    variance_bandwidth_grid = parse_bandwidth_grid(args.variance_bandwidth_grid)
    run_root = prepare_run_root(base_output_root, args.run_name)
    tasks = build_tasks(args, beta_true, signal_bandwidth_grid, variance_bandwidth_grid, run_root)
    records: list[PairedCase1AltbaseRecord] = []
    total_jobs = len(tasks)
    completed_jobs = 0
    global_start = time.perf_counter()
    write_run_config(run_root, args, total_jobs)
    initialize_raw_csv(run_root)

    print(f"[run] total_jobs={total_jobs} n_jobs={args.n_jobs} run_dir={run_root}")

    if args.n_jobs == 1:
        for task in tasks:
            record = run_one(**task)
            records.append(record)
            completed_jobs += 1
            append_raw_record(run_root, record)
            rewrite_summary_csv(run_root, summarize(records))
            write_progress_snapshot(
                run_root,
                completed_jobs=completed_jobs,
                total_jobs=total_jobs,
                records=records,
                global_start=global_start,
            )
            print_progress_line(
                record=record,
                completed_jobs=completed_jobs,
                total_jobs=total_jobs,
                global_start=global_start,
                n_rep=args.n_rep,
            )
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, **task) for task in tasks]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed_jobs += 1
                append_raw_record(run_root, record)
                rewrite_summary_csv(run_root, summarize(records))
                write_progress_snapshot(
                    run_root,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    records=records,
                    global_start=global_start,
                )
                print_progress_line(
                    record=record,
                    completed_jobs=completed_jobs,
                    total_jobs=total_jobs,
                    global_start=global_start,
                    n_rep=args.n_rep,
                )

    for n_subject in args.n_subject_values:
        for coef_type in args.coef_types:
            for rho_true in resolved_rho_values(args):
                group_records = [
                    record
                    for record in records
                    if record.n_subject == n_subject
                    and record.coef_type == coef_type
                    and record.rho_true == rho_true
                ]
                group_summary = summarize(group_records)[0]
                print(
                    f"[group done] n_subject={n_subject} coef={coef_type:10s} rho={rho_true:.3f} "
                    f"success={group_summary['n_success']}/{group_summary['n_rep']} "
                    f"MIAE_final={group_summary['miae_final_mean'] if group_summary['miae_final_mean'] is not None else 'NA'} "
                    f"({group_summary['miae_final_std'] if group_summary['miae_final_std'] is not None else 'NA'}) "
                    f"best_h={group_summary['best_signal_bandwidth_mean'] if group_summary['best_signal_bandwidth_mean'] is not None else 'NA'} "
                    f"({group_summary['best_signal_bandwidth_std'] if group_summary['best_signal_bandwidth_std'] is not None else 'NA'}) "
                    f"best_hbar={group_summary['best_variance_bandwidth_mean'] if group_summary['best_variance_bandwidth_mean'] is not None else 'NA'} "
                    f"({group_summary['best_variance_bandwidth_std'] if group_summary['best_variance_bandwidth_std'] is not None else 'NA'})"
                )
    return records, run_root


def summarize(records: Iterable[PairedCase1AltbaseRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str, float, str, str], list[PairedCase1AltbaseRecord]] = {}
    for rec in records:
        grouped.setdefault(
            (rec.n_subject, rec.coef_type, rec.rho_true, rec.covariance_mode, rec.signal_bandwidth_method),
            [],
        ).append(rec)

    metric_fields = (
        "miae_iid",
        "rmise_iid",
        "beta_mae_iid",
        "beta_rmse_iid",
        "miae_final",
        "rmise_final",
        "beta_mae_final",
        "beta_rmse_final",
        "sigma2_miae",
        "sigma2_rmise",
        "rho_abs_error",
        "Sigma_fro_error",
        "best_signal_bandwidth",
        "best_variance_bandwidth",
        "elapsed_seconds",
    )
    rows: list[dict[str, float | int | str]] = []
    for (n_subject, coef_type, rho_true, covariance_mode, signal_bandwidth_method), vals in sorted(grouped.items()):
        row: dict[str, float | int | str] = {
            "n_subject": n_subject,
            "coef_type": coef_type,
            "rho_true": rho_true,
            "covariance_mode": covariance_mode,
            "signal_bandwidth_method": signal_bandwidth_method,
            "n_rep": len(vals),
            "n_success": int(sum(v.success for v in vals)),
            "n_fail": int(sum(1 - v.success for v in vals)),
        }
        for field in metric_fields:
            arr = np.array(
                [getattr(v, field) for v in vals if getattr(v, field) is not None],
                dtype=float,
            )
            row[f"{field}_mean"] = float(np.mean(arr)) if arr.size else None
            row[f"{field}_std"] = float(np.std(arr, ddof=0)) if arr.size else None
        rows.append(row)
    return rows


def maybe_write_outputs(run_root: Path, summary: list[dict[str, float | int | str]]) -> None:
    """Finalize summary output paths for this run."""

    rewrite_summary_csv(run_root, summary)
    paths = output_paths(run_root)
    print(f"Wrote raw results to {paths['raw']}")
    print(f"Wrote summary results to {paths['summary']}")
    print(f"Wrote run config to {paths['config']}")


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    def fmt(value) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.4f}"

    print("\nPaired Case 1 altbase repetition summary")
    for row in summary:
        print(
            f"n_subject={row['n_subject']}, coef={row['coef_type']}, rho={float(row['rho_true']):.3f}, "
            f"mode={row['covariance_mode']}, signal_method={row['signal_bandwidth_method']}: "
            f"MIAE_final={fmt(row['miae_final_mean'])} ({fmt(row['miae_final_std'])}), "
            f"RMISE_final={fmt(row['rmise_final_mean'])} ({fmt(row['rmise_final_std'])}), "
            f"beta_MAE_final={fmt(row['beta_mae_final_mean'])} ({fmt(row['beta_mae_final_std'])}), "
            f"beta_RMSE_final={fmt(row['beta_rmse_final_mean'])} ({fmt(row['beta_rmse_final_std'])}), "
            f"best_h={fmt(row['best_signal_bandwidth_mean'])} ({fmt(row['best_signal_bandwidth_std'])}), "
            f"best_hbar={fmt(row['best_variance_bandwidth_mean'])} ({fmt(row['best_variance_bandwidth_std'])}), "
            f"success={row['n_success']}/{row['n_rep']}"
        )


def main() -> None:
    args = parse_args()
    records, run_root = run(args)
    summary = summarize(records)
    maybe_write_outputs(run_root, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
