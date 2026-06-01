"""Exact backfill driver for missing paired Case 2 altbase repetition tasks."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.experiments.paired_case2_altbase_smoke import (
    Case2AltbaseSmokeConfig,
    parse_beta,
    parse_bandwidth_grid,
    run_case2_altbase_once,
    to_json_safe,
)
from src.metrics import beta_mae, rmise


@dataclass(slots=True)
class BackfillTask:
    source_part: int
    n_subject: int
    coef_type: str
    rho_true: float
    rep: int
    seed: int


@dataclass(slots=True)
class PairedCase2AltbaseBackfillRecord:
    source_part: int
    source_manifest: str
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


@dataclass(slots=True)
class SummaryRecord:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--S", type=int, default=27)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument("--beta", type=str, default="2.0,1.0,-1.0,0.5")
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--covariance-mode", type=str, default="exchangeable_varying_sigma")
    parser.add_argument("--signal-bandwidth", type=float, default=0.20)
    parser.add_argument("--signal-bandwidth-method", type=str, default="stage1_kfold_cv")
    parser.add_argument("--signal-bandwidth-grid", type=str, default=None)
    parser.add_argument("--variance-bandwidth", type=float, default=0.20)
    parser.add_argument("--variance-bandwidth-method", type=str, default="stage2_kfold_cv")
    parser.add_argument("--variance-bandwidth-grid", type=str, default=None)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=12)
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes > 0:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def output_paths(run_root: Path) -> dict[str, Path]:
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "results_dir": results_dir,
        "raw": results_dir / "raw_results.csv",
        "summary": results_dir / "summary_results.csv",
        "progress": run_root / "progress.json",
        "config": run_root / "run_config.json",
    }


def prepare_run_root(run_name: str) -> Path:
    run_root = Path(__file__).resolve().with_suffix("").parent / "backfill_runs" / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def load_manifest(path: Path) -> list[BackfillTask]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tasks: list[BackfillTask] = []
    for row in rows:
        tasks.append(
            BackfillTask(
                source_part=int(row["part"]),
                n_subject=int(row["n_subject"]),
                coef_type=str(row["coef_type"]),
                rho_true=float(row["rho_true"]),
                rep=int(row["rep"]),
                seed=int(row["seed"]),
            )
        )
    return tasks


def write_run_config(run_root: Path, args: argparse.Namespace, tasks: list[BackfillTask]) -> None:
    run_config = {
        "script": "src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_backfill.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "manifest": str(args.manifest),
        "task_count": len(tasks),
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "beta": args.beta,
        "sigma2": args.sigma2,
        "covariance_mode": args.covariance_mode,
        "signal_bandwidth": args.signal_bandwidth,
        "signal_bandwidth_method": args.signal_bandwidth_method,
        "signal_bandwidth_grid": args.signal_bandwidth_grid,
        "variance_bandwidth": args.variance_bandwidth,
        "variance_bandwidth_method": args.variance_bandwidth_method,
        "variance_bandwidth_grid": args.variance_bandwidth_grid,
        "ridge": args.ridge,
        "n_jobs": args.n_jobs,
        "parts": sorted({task.source_part for task in tasks}),
    }
    with output_paths(run_root)["config"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(run_config), f, indent=2)


def initialize_raw_csv(run_root: Path) -> None:
    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(PairedCase2AltbaseBackfillRecord.__dataclass_fields__.keys())
    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_raw_record(run_root: Path, record: PairedCase2AltbaseBackfillRecord) -> None:
    raw_path = output_paths(run_root)["raw"]
    fieldnames = list(PairedCase2AltbaseBackfillRecord.__dataclass_fields__.keys())
    with raw_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(asdict(record))


def write_progress_snapshot(
    run_root: Path,
    *,
    completed_jobs: int,
    total_jobs: int,
    records: list[PairedCase2AltbaseBackfillRecord],
    global_start: float,
) -> None:
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


def print_progress_line(
    *,
    record: PairedCase2AltbaseBackfillRecord,
    completed_jobs: int,
    total_jobs: int,
    global_start: float,
) -> None:
    status = "done" if record.success else "fail"
    elapsed_total = time.perf_counter() - global_start
    avg_elapsed = elapsed_total / completed_jobs
    eta_seconds = avg_elapsed * (total_jobs - completed_jobs)
    print(
        f"[{completed_jobs}/{total_jobs}] {status} "
        f"part={record.source_part} n_subject={record.n_subject} coef={record.coef_type:10s} "
        f"rho={record.rho_true:.3f} rep={record.rep + 1} seed={record.seed} "
        f"elapsed={format_duration(record.elapsed_seconds)} eta={format_duration(eta_seconds)}"
    )


def summary_records(records: Iterable[PairedCase2AltbaseBackfillRecord]) -> list[SummaryRecord]:
    converted: list[SummaryRecord] = []
    for record in records:
        converted.append(
            SummaryRecord(
                n_subject=record.n_subject,
                coef_type=record.coef_type,
                rep=record.rep,
                seed=record.seed,
                success=record.success,
                error_message=record.error_message,
                elapsed_seconds=record.elapsed_seconds,
                covariance_mode=record.covariance_mode,
                signal_bandwidth_input=record.signal_bandwidth_input,
                signal_bandwidth_method=record.signal_bandwidth_method,
                best_signal_bandwidth=record.best_signal_bandwidth,
                variance_bandwidth_input=record.variance_bandwidth_input,
                variance_bandwidth_method=record.variance_bandwidth_method,
                best_variance_bandwidth=record.best_variance_bandwidth,
                sigma2_true=record.sigma2_true,
                rho_true=record.rho_true,
                miae_iid=record.miae_iid,
                rmise_iid=record.rmise_iid,
                beta_mae_iid=record.beta_mae_iid,
                beta_rmse_iid=record.beta_rmse_iid,
                miae_final=record.miae_final,
                rmise_final=record.rmise_final,
                beta_mae_final=record.beta_mae_final,
                beta_rmse_final=record.beta_rmse_final,
                sigma2_miae=record.sigma2_miae,
                sigma2_rmise=record.sigma2_rmise,
                rho_abs_error=record.rho_abs_error,
                Sigma_fro_error=record.Sigma_fro_error,
            )
        )
    return converted


def summarize(records: Iterable[SummaryRecord]) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[int, str, float, str, str], list[SummaryRecord]] = {}
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
            arr = np.array([getattr(v, field) for v in vals if getattr(v, field) is not None], dtype=float)
            row[f"{field}_mean"] = float(np.mean(arr)) if arr.size else None
            row[f"{field}_std"] = float(np.std(arr, ddof=0)) if arr.size else None
        rows.append(row)
    return rows


def print_summary(summary: list[dict[str, float | int | str]]) -> None:
    def fmt(value) -> str:
        if value is None:
            return "NA"
        return f"{float(value):.4f}"

    print("\nPaired Case 2 altbase backfill summary")
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
            f"sigma2_MIAE={fmt(row['sigma2_miae_mean'])} ({fmt(row['sigma2_miae_std'])}), "
            f"rho_abs_err={fmt(row['rho_abs_error_mean'])} ({fmt(row['rho_abs_error_std'])}), "
            f"Sigma_fro_err={fmt(row['Sigma_fro_error_mean'])} ({fmt(row['Sigma_fro_error_std'])}), "
            f"elapsed={fmt(row['elapsed_seconds_mean'])} ({fmt(row['elapsed_seconds_std'])})"
        )


def maybe_write_outputs(run_root: Path, summary: list[dict[str, float | int | str]]) -> None:
    if summary:
        with output_paths(run_root)["summary"].open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            for row in summary:
                writer.writerow(row)
    paths = output_paths(run_root)
    print(f"Wrote raw results to {paths['raw']}")
    print(f"Wrote summary results to {paths['summary']}")
    print(f"Wrote run config to {paths['config']}")


def run_one(
    *,
    task: BackfillTask,
    args: argparse.Namespace,
    beta_true: tuple[float, ...],
    signal_bandwidth_grid: tuple[float, ...] | None,
    variance_bandwidth_grid: tuple[float, ...] | None,
    source_manifest: str,
) -> PairedCase2AltbaseBackfillRecord:
    start = time.perf_counter()
    if args.signal_bandwidth is not None:
        signal_bandwidth_input = f"{args.signal_bandwidth:.12g}"
        failure_signal_bandwidth_method = "fixed"
    elif signal_bandwidth_grid is None:
        signal_bandwidth_input = "default_fixed"
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
        variance_bandwidth_input = "default_fixed"
        failure_variance_bandwidth_method = "default_fixed"
    else:
        variance_bandwidth_input = "auto"
        failure_variance_bandwidth_method = args.variance_bandwidth_method

    try:
        config = Case2AltbaseSmokeConfig(
            seed=task.seed,
            n_subject=task.n_subject,
            R=args.R,
            S=args.S,
            p0=args.p0,
            coef_type=task.coef_type,
            beta_true=beta_true,
            sigma2=args.sigma2,
            rho=task.rho_true,
            covariance_mode=args.covariance_mode,
            signal_bandwidth=args.signal_bandwidth,
            signal_bandwidth_method=args.signal_bandwidth_method,
            signal_bandwidth_grid=signal_bandwidth_grid,
            variance_bandwidth=args.variance_bandwidth,
            variance_bandwidth_method=args.variance_bandwidth_method,
            variance_bandwidth_grid=variance_bandwidth_grid,
            ridge=args.ridge,
        )
        dataset, result, metrics = run_case2_altbase_once(config)
        elapsed = time.perf_counter() - start
        return PairedCase2AltbaseBackfillRecord(
            source_part=task.source_part,
            source_manifest=source_manifest,
            n_subject=task.n_subject,
            coef_type=task.coef_type,
            rep=task.rep,
            seed=task.seed,
            success=1,
            error_message="",
            elapsed_seconds=elapsed,
            covariance_mode=result.covariance.covariance_mode,
            signal_bandwidth_input=signal_bandwidth_input,
            signal_bandwidth_method=result.initial.meta["signal_bandwidth_method"],
            best_signal_bandwidth=float(result.initial.meta["signal_bandwidth_selected"]),
            variance_bandwidth_input=variance_bandwidth_input,
            variance_bandwidth_method=result.covariance.meta.get("variance_bandwidth_method"),
            best_variance_bandwidth=result.covariance.meta.get("variance_bandwidth_selected"),
            sigma2_true=args.sigma2,
            rho_true=task.rho_true,
            miae_iid=metrics["miae_iid"],
            rmise_iid=rmise(dataset.A_true, result.initial.A_hat),
            beta_mae_iid=beta_mae(dataset.beta_true, result.initial.beta_hat),
            beta_rmse_iid=metrics["beta_rmse_iid"],
            miae_final=metrics["miae_final"],
            rmise_final=rmise(dataset.A_true, result.A_hat),
            beta_mae_final=beta_mae(dataset.beta_true, result.beta_hat),
            beta_rmse_final=metrics["beta_rmse_final"],
            sigma2_miae=metrics["sigma2_miae"],
            sigma2_rmise=metrics["sigma2_rmise"],
            rho_abs_error=metrics["rho_abs_error"],
            Sigma_fro_error=metrics["Sigma_fro_error"],
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return PairedCase2AltbaseBackfillRecord(
            source_part=task.source_part,
            source_manifest=source_manifest,
            n_subject=task.n_subject,
            coef_type=task.coef_type,
            rep=task.rep,
            seed=task.seed,
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
            rho_true=task.rho_true,
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


def main() -> None:
    args = parse_args()
    tasks = load_manifest(args.manifest)
    if not tasks:
        raise SystemExit("manifest contains no tasks")

    beta_true = parse_beta(args.beta, args.p0)
    if beta_true is None:
        raise ValueError("parsed beta must not be None for backfill runs.")
    signal_bandwidth_grid = parse_bandwidth_grid(args.signal_bandwidth_grid)
    variance_bandwidth_grid = parse_bandwidth_grid(args.variance_bandwidth_grid)

    run_root = prepare_run_root(args.run_name)
    write_run_config(run_root, args, tasks)
    initialize_raw_csv(run_root)

    total_jobs = len(tasks)
    completed_jobs = 0
    records: list[PairedCase2AltbaseBackfillRecord] = []
    global_start = time.perf_counter()
    source_manifest = str(args.manifest)

    print(f"[run] total_jobs={total_jobs} n_jobs={args.n_jobs} run_dir={run_root}")

    if args.n_jobs == 1:
        for task in tasks:
            record = run_one(
                task=task,
                args=args,
                beta_true=beta_true,
                signal_bandwidth_grid=signal_bandwidth_grid,
                variance_bandwidth_grid=variance_bandwidth_grid,
                source_manifest=source_manifest,
            )
            records.append(record)
            completed_jobs += 1
            append_raw_record(run_root, record)
            summary = summarize(summary_records(records))
            if summary:
                with output_paths(run_root)["summary"].open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                    writer.writeheader()
                    for row in summary:
                        writer.writerow(row)
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
            )
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    task=task,
                    args=args,
                    beta_true=beta_true,
                    signal_bandwidth_grid=signal_bandwidth_grid,
                    variance_bandwidth_grid=variance_bandwidth_grid,
                    source_manifest=source_manifest,
                )
                for task in tasks
            ]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                completed_jobs += 1
                append_raw_record(run_root, record)
                summary = summarize(summary_records(records))
                if summary:
                    with output_paths(run_root)["summary"].open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                        writer.writeheader()
                        for row in summary:
                            writer.writerow(row)
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
                )

    legacy_records = summary_records(records)
    summary = summarize(legacy_records)
    maybe_write_outputs(run_root, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
