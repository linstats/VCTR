"""Manifest-driven paired Case 2 altbase repetition runner."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.paired_case2_altbase_repetition import (  # noqa: E402
    PairedCase2AltbaseRecord,
    append_raw_record,
    initialize_raw_csv,
    output_paths,
    print_progress_line,
    rewrite_summary_csv,
    run_one,
    summarize,
    to_json_safe,
    write_progress_snapshot,
)
from src.experiments.paired_case2_altbase_smoke import parse_beta, parse_bandwidth_grid  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--S", type=int, default=27)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument("--a-eval-mode", type=str, default="anchor_grid", choices=["full", "anchor_grid"])
    parser.add_argument("--a-eval-num-points", type=int, default=500)
    parser.add_argument("--a-eval-grid", type=str, default="quantile", choices=["quantile", "uniform"])
    parser.add_argument("--a-interp", type=str, default="linear", choices=["linear"])
    parser.add_argument("--beta", type=str, default="2.0,1.0,-1.0,0.5")
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--covariance-mode", type=str, default="exchangeable_varying_sigma")
    parser.add_argument("--signal-bandwidth", type=float, default=0.18)
    parser.add_argument("--signal-bandwidth-method", type=str, default="stage1_kfold_cv")
    parser.add_argument("--signal-bandwidth-grid", type=str, default=None)
    parser.add_argument("--variance-bandwidth", type=float, default=0.18)
    parser.add_argument("--variance-bandwidth-method", type=str, default="stage2_kfold_cv")
    parser.add_argument("--variance-bandwidth-grid", type=str, default=None)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--n-jobs", type=int, default=12)
    parser.add_argument("--run-root", type=Path, default=None)
    return parser.parse_args()


def prepare_run_root(run_root: Path | None, run_name: str) -> Path:
    base_root = run_root or Path(__file__).with_suffix("")
    resolved = base_root / run_name
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def load_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tasks: list[dict[str, object]] = []
    for row in rows:
        tasks.append(
            {
                "source_part": int(row["part"]),
                "bundle_block": str(row["bundle_block"]),
                "n_subject": int(row["n_subject"]),
                "coef_type": str(row["coef_type"]),
                "sigma2_function": str(row["sigma2_function"]),
                "rho_true": float(row["rho_true"]),
                "rep": int(row["rep"]),
                "seed": int(row["seed"]),
            }
        )
    return tasks


def write_run_config(run_root: Path, args: argparse.Namespace, tasks: list[dict[str, object]]) -> None:
    run_config = {
        "script": "src/experiments/paired_case2_altbase_repetition/paired_case2_altbase_manifest_run.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "manifest": str(args.manifest),
        "task_count": len(tasks),
        "parts": sorted({int(task["source_part"]) for task in tasks}),
        "bundle_blocks": sorted({str(task["bundle_block"]) for task in tasks}),
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "a_eval_mode": args.a_eval_mode,
        "a_eval_num_points": args.a_eval_num_points,
        "a_eval_grid": args.a_eval_grid,
        "a_interp": args.a_interp,
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
    }
    with output_paths(run_root)["config"].open("w", encoding="utf-8") as f:
        json.dump(to_json_safe(run_config), f, indent=2)


def build_shared_runner_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        n_subject_values=[],
        coef_types=[],
        n_rep=1,
        seed_base=0,
        R=args.R,
        S=args.S,
        p0=args.p0,
        a_eval_mode=args.a_eval_mode,
        a_eval_num_points=args.a_eval_num_points,
        a_eval_grid=args.a_eval_grid,
        a_interp=args.a_interp,
        beta=args.beta,
        sigma2=args.sigma2,
        sigma2_function=None,
        sigma2_functions=[],
        rho=0.0,
        rho_values=[],
        covariance_mode=args.covariance_mode,
        signal_bandwidth=args.signal_bandwidth,
        signal_bandwidth_method=args.signal_bandwidth_method,
        signal_bandwidth_grid=args.signal_bandwidth_grid,
        variance_bandwidth=args.variance_bandwidth,
        variance_bandwidth_method=args.variance_bandwidth_method,
        variance_bandwidth_grid=args.variance_bandwidth_grid,
        ridge=args.ridge,
        large_n_threshold=2000,
        prompt_accelerate_large_n=False,
        n_jobs=args.n_jobs,
        run_name=None,
        save_data=False,
        save_estimates=False,
        plot_functions=False,
        plot_a_indices="all",
        plot_max_a_panels=16,
    )


def main() -> None:
    args = parse_args()
    tasks = load_manifest(args.manifest)
    if not tasks:
        raise SystemExit("manifest contains no tasks")

    beta_true = parse_beta(args.beta, args.p0)
    if beta_true is None:
        raise ValueError("parsed beta must not be None for manifest runs.")
    signal_bandwidth_grid = parse_bandwidth_grid(args.signal_bandwidth_grid)
    variance_bandwidth_grid = parse_bandwidth_grid(args.variance_bandwidth_grid)
    run_root = prepare_run_root(args.run_root, args.run_name)
    write_run_config(run_root, args, tasks)
    initialize_raw_csv(run_root)

    runner_args = build_shared_runner_args(args)
    total_jobs = len(tasks)
    completed_jobs = 0
    global_start = time.perf_counter()
    records: list[PairedCase2AltbaseRecord] = []

    print(f"[run] total_jobs={total_jobs} n_jobs={args.n_jobs} run_dir={run_root}")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    submitted = [
        {
            "n_subject": int(task["n_subject"]),
            "coef_type": str(task["coef_type"]),
            "sigma2_function": str(task["sigma2_function"]),
            "rho_true": float(task["rho_true"]),
            "rep": int(task["rep"]),
            "seed": int(task["seed"]),
            "beta_true": beta_true,
            "args": runner_args,
            "signal_bandwidth_grid": signal_bandwidth_grid,
            "variance_bandwidth_grid": variance_bandwidth_grid,
            "output_root": run_root,
        }
        for task in tasks
    ]

    if args.n_jobs == 1:
        for task in submitted:
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
                n_rep=30,
            )
    else:
        max_workers = args.n_jobs if args.n_jobs > 0 else 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_one, **task) for task in submitted]
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
                    n_rep=30,
                )


if __name__ == "__main__":
    main()
