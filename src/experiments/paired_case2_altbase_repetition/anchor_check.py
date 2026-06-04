"""Local validation for n=5000 full vs anchor-grid paired Case 2 fits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.paired_case2_altbase_repetition import (
    PairedCase2AltbaseRecord,
    append_raw_record,
    initialize_raw_csv,
    print_summary,
    rewrite_summary_csv,
    summarize,
    write_progress_snapshot,
)
from src.experiments.paired_case2_altbase_repetition import run_one as run_case2_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-subject", type=int, default=5000)
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--S", type=int, default=27)
    parser.add_argument("--p0", type=int, default=4)
    parser.add_argument("--coef-type", type=str, default="base5")
    parser.add_argument("--rho", type=float, default=0.6)
    parser.add_argument("--sigma2", type=float, default=1.0)
    parser.add_argument("--sigma2-function", type=str, default="mixed")
    parser.add_argument("--beta", type=str, default="2.0,1.0,-1.0,0.5")
    parser.add_argument("--signal-bandwidth", type=float, default=0.18)
    parser.add_argument("--variance-bandwidth", type=float, default=0.18)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--full-seeds", type=int, nargs="+", default=[123, 124])
    parser.add_argument("--anchor-seed-start", type=int, default=123)
    parser.add_argument("--anchor-seed-end", type=int, default=127)
    parser.add_argument("--anchor-points", type=int, nargs="+", default=[250, 500, 1000])
    parser.add_argument("--full-workers", type=int, default=2)
    parser.add_argument("--anchor-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--plot-functions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plot-a-indices", type=str, default="all")
    parser.add_argument("--plot-max-a-panels", type=int, default=16)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional output directory name. Defaults to the script stem.",
    )
    return parser.parse_args()


def parse_beta(beta_arg: str, p0: int) -> tuple[float, ...]:
    parts = [part.strip() for part in beta_arg.split(",") if part.strip()]
    beta = tuple(float(part) for part in parts)
    if len(beta) != p0:
        raise ValueError("--beta must contain exactly p0 comma-separated values.")
    return beta


def output_root_from_args(args: argparse.Namespace) -> Path:
    if args.run_name:
        return Path(__file__).with_suffix("").parent / args.run_name
    return Path(__file__).with_suffix("")


def build_repetition_args(
    *,
    base_args: argparse.Namespace,
    seed_base: int,
    a_eval_mode: str,
    a_eval_num_points: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        n_subject_values=[base_args.n_subject],
        coef_types=[base_args.coef_type],
        n_rep=1,
        seed_base=seed_base,
        R=base_args.R,
        S=base_args.S,
        p0=base_args.p0,
        a_eval_mode=a_eval_mode,
        a_eval_num_points=a_eval_num_points,
        a_eval_grid="quantile",
        a_interp="linear",
        beta=base_args.beta,
        sigma2=base_args.sigma2,
        sigma2_function=base_args.sigma2_function,
        sigma2_functions=[base_args.sigma2_function],
        rho=base_args.rho,
        rho_values=[base_args.rho],
        covariance_mode="exchangeable_varying_sigma",
        signal_bandwidth=base_args.signal_bandwidth,
        signal_bandwidth_method="stage1_kfold_cv",
        signal_bandwidth_grid=None,
        variance_bandwidth=base_args.variance_bandwidth,
        variance_bandwidth_method="stage2_kfold_cv",
        variance_bandwidth_grid=None,
        ridge=base_args.ridge,
        large_n_threshold=2000,
        prompt_accelerate_large_n=False,
        n_jobs=1,
        run_name=None,
        save_data=False,
        save_estimates=False,
        plot_functions=base_args.plot_functions,
        plot_a_indices=base_args.plot_a_indices,
        plot_max_a_panels=base_args.plot_max_a_panels,
    )


def build_task(
    *,
    base_args: argparse.Namespace,
    beta_true: tuple[float, ...],
    seed: int,
    rep: int,
    a_eval_mode: str,
    a_eval_num_points: int,
    output_root: Path,
) -> dict[str, object]:
    repetition_args = build_repetition_args(
        base_args=base_args,
        seed_base=seed,
        a_eval_mode=a_eval_mode,
        a_eval_num_points=a_eval_num_points,
    )
    return {
        "n_subject": base_args.n_subject,
        "coef_type": base_args.coef_type,
        "sigma2_function": base_args.sigma2_function,
        "rho_true": base_args.rho,
        "rep": rep,
        "seed": seed,
        "beta_true": beta_true,
        "args": repetition_args,
        "signal_bandwidth_grid": None,
        "variance_bandwidth_grid": None,
        "output_root": output_root,
    }


def write_run_config(run_root: Path, args: argparse.Namespace) -> None:
    run_config = {
        "script": "src/experiments/paired_case2_altbase_repetition/test/test_anchor_eval_n5000_full_vs_anchor.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
        "n_subject": args.n_subject,
        "R": args.R,
        "S": args.S,
        "p0": args.p0,
        "coef_type": args.coef_type,
        "rho": args.rho,
        "sigma2": args.sigma2,
        "sigma2_function": args.sigma2_function,
        "beta": args.beta,
        "signal_bandwidth": args.signal_bandwidth,
        "variance_bandwidth": args.variance_bandwidth,
        "ridge": args.ridge,
        "plot_functions": args.plot_functions,
        "plot_a_indices": args.plot_a_indices,
        "plot_max_a_panels": args.plot_max_a_panels,
        "full_seeds": args.full_seeds,
        "anchor_seed_start": args.anchor_seed_start,
        "anchor_seed_end": args.anchor_seed_end,
        "anchor_points": args.anchor_points,
        "full_workers": args.full_workers,
        "anchor_workers": args.anchor_workers,
    }
    with (run_root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


def record_sort_key(record: PairedCase2AltbaseRecord) -> tuple[int, int, int]:
    mode_order = 0 if record.a_eval_mode == "full" else 1
    return (mode_order, record.a_eval_selected_points, record.seed)


def print_phase_summary(records: list[PairedCase2AltbaseRecord], phase_name: str) -> None:
    print(f"\n[{phase_name}] completed {len(records)} fits")
    for record in sorted(records, key=record_sort_key):
        status = "ok" if record.success == 1 else "fail"
        print(
            f"{record.a_eval_mode:11s} points={record.a_eval_selected_points:4d} "
            f"seed={record.seed} status={status:4s} elapsed={record.elapsed_seconds:.2f}s "
            f"miae_final={record.miae_final if record.miae_final is not None else 'NA'} "
            f"beta_rmse_final={record.beta_rmse_final if record.beta_rmse_final is not None else 'NA'}"
        )


def execute_phase(
    *,
    tasks: list[dict[str, object]],
    workers: int,
    phase_name: str,
    all_records: list[PairedCase2AltbaseRecord],
    run_root: Path,
    global_start: float,
    total_jobs: int,
) -> list[PairedCase2AltbaseRecord]:
    phase_records: list[PairedCase2AltbaseRecord] = []
    if workers <= 1:
        for task in tasks:
            record = run_case2_one(**task)
            all_records.append(record)
            phase_records.append(record)
            append_raw_record(run_root, record)
            rewrite_summary_csv(run_root, summarize(all_records))
            write_progress_snapshot(
                run_root,
                completed_jobs=len(all_records),
                total_jobs=total_jobs,
                records=all_records,
                global_start=global_start,
            )
    else:
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(run_case2_one, **task) for task in tasks]
                for future in as_completed(futures):
                    record = future.result()
                    all_records.append(record)
                    phase_records.append(record)
                    append_raw_record(run_root, record)
                    rewrite_summary_csv(run_root, summarize(all_records))
                    write_progress_snapshot(
                        run_root,
                        completed_jobs=len(all_records),
                        total_jobs=total_jobs,
                        records=all_records,
                        global_start=global_start,
                    )
        except PermissionError:
            print(f"[{phase_name}] process pool unavailable in this environment; falling back to serial execution")
            for task in tasks:
                record = run_case2_one(**task)
                all_records.append(record)
                phase_records.append(record)
                append_raw_record(run_root, record)
                rewrite_summary_csv(run_root, summarize(all_records))
                write_progress_snapshot(
                    run_root,
                    completed_jobs=len(all_records),
                    total_jobs=total_jobs,
                    records=all_records,
                    global_start=global_start,
                )
    print_phase_summary(phase_records, phase_name)
    return phase_records


def main() -> None:
    args = parse_args()
    beta_true = parse_beta(args.beta, args.p0)
    run_root = output_root_from_args(args)
    run_root.mkdir(parents=True, exist_ok=False)
    initialize_raw_csv(run_root)
    write_run_config(run_root, args)

    anchor_seeds = list(range(args.anchor_seed_start, args.anchor_seed_end + 1))
    if len(args.full_seeds) == 0:
        raise ValueError("--full-seeds must contain at least one seed.")
    if len(anchor_seeds) == 0:
        raise ValueError("anchor seed range must not be empty.")
    if any(points < 2 for points in args.anchor_points):
        raise ValueError("--anchor-points values must be at least 2.")

    full_tasks = [
        build_task(
            base_args=args,
            beta_true=beta_true,
            seed=seed,
            rep=index,
            a_eval_mode="full",
            a_eval_num_points=args.anchor_points[0],
            output_root=run_root,
        )
        for index, seed in enumerate(args.full_seeds)
    ]
    anchor_tasks = []
    for anchor_points in args.anchor_points:
        for seed in anchor_seeds:
            anchor_tasks.append(
                build_task(
                    base_args=args,
                    beta_true=beta_true,
                    seed=seed,
                    rep=seed - anchor_seeds[0],
                    a_eval_mode="anchor_grid",
                    a_eval_num_points=anchor_points,
                    output_root=run_root,
                )
            )

    total_jobs = len(full_tasks) + len(anchor_tasks)
    all_records: list[PairedCase2AltbaseRecord] = []
    global_start = time.perf_counter()

    print(
        f"[run] total_jobs={total_jobs} full_jobs={len(full_tasks)} anchor_jobs={len(anchor_tasks)} "
        f"full_workers={args.full_workers} anchor_workers={args.anchor_workers} run_dir={run_root}"
    )

    execute_phase(
        tasks=full_tasks,
        workers=args.full_workers,
        phase_name="full phase",
        all_records=all_records,
        run_root=run_root,
        global_start=global_start,
        total_jobs=total_jobs,
    )
    execute_phase(
        tasks=anchor_tasks,
        workers=args.anchor_workers,
        phase_name="anchor phase",
        all_records=all_records,
        run_root=run_root,
        global_start=global_start,
        total_jobs=total_jobs,
    )

    summary = summarize(all_records)
    rewrite_summary_csv(run_root, summary)
    print_summary(summary)

    with (run_root / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in sorted(all_records, key=record_sort_key)], f, indent=2)


if __name__ == "__main__":
    main()
