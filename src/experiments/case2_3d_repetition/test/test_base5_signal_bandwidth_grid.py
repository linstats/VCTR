"""Run a local Case 2 Base 5 signal-bandwidth grid with parallel workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

# Each fit is already dominated by dense linear algebra. Keep one BLAS thread
# per worker so six concurrent fits do not oversubscribe the local machine.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.case2_3d_repetition import (
    Case23DRecord,
    append_raw_record,
    initialize_raw_csv,
    run_one as run_case2_one,
    to_json_safe,
    write_progress_snapshot,
)


SIGNAL_BANDWIDTHS = {
    2000: (0.10, 0.12, 0.14, 0.16, 0.18),
    5000: (0.06, 0.08, 0.10, 0.12, 0.18),
}

COMPARISON_FIELDS = (
    "n_subject",
    "signal_bandwidth",
    "rank_by_miae_final",
    "success",
    "error_message",
    "miae_iid",
    "miae_final",
    "star_miae_improvement_pct",
    "rmise_iid",
    "rmise_final",
    "star_rmise_improvement_pct",
    "beta_mae_iid",
    "beta_mae_final",
    "beta_rmse_iid",
    "beta_rmse_final",
    "sigma2_miae",
    "sigma2_rmise",
    "rho_true",
    "rho_hat",
    "rho_error",
    "rho_abs_error",
    "variance_bandwidth",
    "a_eval_mode",
    "a_eval_selected_points",
    "elapsed_seconds",
    "seed",
    "rep",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel process workers. Default: 6.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output directory name under the test directory.",
    )
    return parser.parse_args()


def output_root(run_name: str | None) -> Path:
    test_dir = Path(__file__).resolve().parent
    return test_dir / (run_name or Path(__file__).stem)


def build_repetition_args(n_subject: int, signal_bandwidth: float) -> argparse.Namespace:
    """Build the fixed arguments expected by repetition.run_one()."""

    return argparse.Namespace(
        n_subject_values=[n_subject],
        coef_types=["base5"],
        n_rep=1,
        seed_base=123,
        R=6,
        S=27,
        p0=4,
        a_eval_mode="anchor_grid",
        a_eval_num_points=500,
        a_eval_grid="quantile",
        a_interp="linear",
        beta="2.0,1.0,-1.0,0.5",
        sigma2=1.0,
        sigma2_function="sin",
        sigma2_functions=["sin"],
        rho=0.6,
        rho_values=[0.6],
        covariance_mode="exchangeable_varying_sigma",
        signal_bandwidth=signal_bandwidth,
        signal_bandwidth_method="stage1_kfold_cv",
        signal_bandwidth_grid=None,
        variance_bandwidth=0.18,
        variance_bandwidth_method="stage2_kfold_cv",
        variance_bandwidth_grid=None,
        ridge=1e-4,
        large_n_threshold=2000,
        prompt_accelerate_large_n=False,
        n_jobs=1,
        run_name=None,
        save_data=False,
        save_estimates=False,
        plot_functions=False,
        plot_a_indices="all",
        plot_max_a_panels=16,
    )


def build_tasks(run_root: Path) -> list[dict[str, object]]:
    beta_true = (2.0, 1.0, -1.0, 0.5)
    tasks: list[dict[str, object]] = []
    for n_subject, bandwidths in SIGNAL_BANDWIDTHS.items():
        for signal_bandwidth in bandwidths:
            tasks.append(
                {
                    "n_subject": n_subject,
                    "coef_type": "base5",
                    "sigma2_function": "sin",
                    "rho_true": 0.6,
                    "rep": 0,
                    "seed": 123,
                    "beta_true": beta_true,
                    "args": build_repetition_args(n_subject, signal_bandwidth),
                    "signal_bandwidth_grid": None,
                    "variance_bandwidth_grid": None,
                    "output_root": run_root,
                }
            )
    return tasks


def write_run_config(run_root: Path, workers: int) -> None:
    config = {
        "script": "src/experiments/case2_3d_repetition/test/test_base5_signal_bandwidth_grid.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "workers": workers,
        "parallel_backend": "ProcessPoolExecutor",
        "n_rep": 1,
        "seed": 123,
        "coef_type": "base5",
        "sigma2": 1.0,
        "sigma2_function": "sin",
        "rho": 0.6,
        "R": 6,
        "S": 27,
        "p0": 4,
        "beta": [2.0, 1.0, -1.0, 0.5],
        "covariance_mode": "exchangeable_varying_sigma",
        "variance_bandwidth": 0.18,
        "a_eval_mode": "anchor_grid",
        "a_eval_num_points": 500,
        "a_eval_grid": "quantile",
        "a_interp": "linear",
        "ridge": 1e-4,
        "save_data": False,
        "save_estimates": False,
        "plot_functions": False,
        "signal_bandwidths": {str(n): list(values) for n, values in SIGNAL_BANDWIDTHS.items()},
    }
    with (run_root / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def percentage_improvement(initial: float | None, final: float | None) -> float | None:
    if initial is None or final is None or initial == 0:
        return None
    return 100.0 * (initial - final) / initial


def record_signal_bandwidth(record: Case23DRecord) -> float:
    if record.best_signal_bandwidth is not None:
        return float(record.best_signal_bandwidth)
    return float(record.signal_bandwidth_input)


def comparison_rows(records: list[Case23DRecord]) -> list[dict[str, object]]:
    ranks: dict[tuple[int, float], int] = {}
    for n_subject in SIGNAL_BANDWIDTHS:
        successful = sorted(
            (
                record
                for record in records
                if record.n_subject == n_subject and record.success == 1 and record.miae_final is not None
            ),
            key=lambda record: float(record.miae_final),
        )
        for rank, record in enumerate(successful, start=1):
            ranks[(record.n_subject, record_signal_bandwidth(record))] = rank

    rows: list[dict[str, object]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item.n_subject,
            record_signal_bandwidth(item),
        ),
    ):
        signal_bandwidth = record_signal_bandwidth(record)
        rows.append(
            {
                "n_subject": record.n_subject,
                "signal_bandwidth": signal_bandwidth,
                "rank_by_miae_final": ranks.get((record.n_subject, signal_bandwidth)),
                "success": record.success,
                "error_message": record.error_message,
                "miae_iid": record.miae_iid,
                "miae_final": record.miae_final,
                "star_miae_improvement_pct": percentage_improvement(record.miae_iid, record.miae_final),
                "rmise_iid": record.rmise_iid,
                "rmise_final": record.rmise_final,
                "star_rmise_improvement_pct": percentage_improvement(record.rmise_iid, record.rmise_final),
                "beta_mae_iid": record.beta_mae_iid,
                "beta_mae_final": record.beta_mae_final,
                "beta_rmse_iid": record.beta_rmse_iid,
                "beta_rmse_final": record.beta_rmse_final,
                "sigma2_miae": record.sigma2_miae,
                "sigma2_rmise": record.sigma2_rmise,
                "rho_true": record.rho_true,
                "rho_hat": record.rho_true + record.rho_error if record.rho_error is not None else None,
                "rho_error": record.rho_error,
                "rho_abs_error": record.rho_abs_error,
                "variance_bandwidth": record.best_variance_bandwidth,
                "a_eval_mode": record.a_eval_mode,
                "a_eval_selected_points": record.a_eval_selected_points,
                "elapsed_seconds": record.elapsed_seconds,
                "seed": record.seed,
                "rep": record.rep,
            }
        )
    return rows


def write_comparison_tables(run_root: Path, records: list[Case23DRecord]) -> None:
    rows = comparison_rows(records)
    results_dir = run_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("summary_results.csv", "bandwidth_comparison.csv"):
        with (results_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerows(rows)


def print_record(record: Case23DRecord, completed: int, total: int) -> None:
    status = "done" if record.success else "fail"
    print(
        f"[{completed}/{total}] {status} n={record.n_subject} "
        f"h_A={record.best_signal_bandwidth if record.best_signal_bandwidth is not None else record.signal_bandwidth_input} "
        f"MIAE*={record.miae_final if record.miae_final is not None else 'NA'} "
        f"sigma2_MIAE={record.sigma2_miae if record.sigma2_miae is not None else 'NA'} "
        f"rho_error={record.rho_error if record.rho_error is not None else 'NA'} "
        f"elapsed={record.elapsed_seconds:.2f}s"
    )


def print_best_by_n(records: list[Case23DRecord]) -> None:
    print("\nBest bandwidth by final MIAE")
    for n_subject in SIGNAL_BANDWIDTHS:
        successful = [
            record
            for record in records
            if record.n_subject == n_subject and record.success == 1 and record.miae_final is not None
        ]
        if not successful:
            print(f"n={n_subject}: no successful fits")
            continue
        best = min(successful, key=lambda record: float(record.miae_final))
        print(
            f"n={n_subject}: h_A={best.best_signal_bandwidth:.2f}, "
            f"MIAE*={best.miae_final:.6f}, sigma2_MIAE={best.sigma2_miae:.6f}, "
            f"rho_abs_error={best.rho_abs_error:.6f}"
        )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    run_root = output_root(args.run_name)
    run_root.mkdir(parents=True, exist_ok=False)
    initialize_raw_csv(run_root)
    write_run_config(run_root, args.workers)

    tasks = build_tasks(run_root)
    records: list[Case23DRecord] = []
    global_start = time.perf_counter()
    total_jobs = len(tasks)
    workers = min(args.workers, total_jobs)

    print(f"[run] total_jobs={total_jobs} workers={workers} run_dir={run_root}")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_case2_one, **task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            append_raw_record(run_root, record)
            write_comparison_tables(run_root, records)
            write_progress_snapshot(
                run_root,
                completed_jobs=completed,
                total_jobs=total_jobs,
                records=records,
                global_start=global_start,
            )
            print_record(record, completed, total_jobs)

    write_comparison_tables(run_root, records)
    with (run_root / "records.json").open("w", encoding="utf-8") as file:
        json.dump(to_json_safe([asdict(record) for record in records]), file, indent=2)
    print_best_by_n(records)

    if any(record.success == 0 for record in records):
        raise RuntimeError(f"At least one grid fit failed; inspect {run_root / 'records.json'}.")


if __name__ == "__main__":
    main()
