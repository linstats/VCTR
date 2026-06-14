"""Run the two-config local Case 2 Base 5 small-bandwidth pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.case2_3d_repetition import (
    Case23DRecord,
    append_raw_record,
    initialize_raw_csv,
    print_summary,
    rewrite_summary_csv,
    run_one as run_case2_one,
    summarize,
    write_progress_snapshot,
)


EXPERIMENT_PLAN = (
    {"n_subject": 2000, "signal_bandwidth": 0.14},
    {"n_subject": 5000, "signal_bandwidth": 0.08},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    """Build the fixed Case 2 arguments expected by repetition.run_one()."""

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
        save_estimates=True,
        plot_functions=True,
        plot_a_indices="all",
        plot_max_a_panels=16,
    )


def build_tasks(run_root: Path) -> list[dict[str, object]]:
    beta_true = (2.0, 1.0, -1.0, 0.5)
    tasks: list[dict[str, object]] = []
    for config in EXPERIMENT_PLAN:
        n_subject = int(config["n_subject"])
        signal_bandwidth = float(config["signal_bandwidth"])
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


def write_run_config(run_root: Path) -> None:
    config = {
        "script": "src/experiments/case2_3d_repetition/test/test_base5_small_h_pilot.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_root.name,
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
        "save_estimates": True,
        "plot_functions": True,
        "plot_a_indices": "all",
        "plot_max_a_panels": 16,
        "configurations": list(EXPERIMENT_PLAN),
    }
    with (run_root / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def print_record(record: Case23DRecord, completed: int, total: int) -> None:
    status = "done" if record.success else "fail"
    print(
        f"[{completed}/{total}] {status} n={record.n_subject} "
        f"h_A={record.best_signal_bandwidth if record.best_signal_bandwidth is not None else 'NA'} "
        f"h_sigma={record.best_variance_bandwidth if record.best_variance_bandwidth is not None else 'NA'} "
        f"elapsed={record.elapsed_seconds:.2f}s"
    )


def main() -> None:
    args = parse_args()
    run_root = output_root(args.run_name)
    run_root.mkdir(parents=True, exist_ok=False)
    initialize_raw_csv(run_root)
    write_run_config(run_root)

    tasks = build_tasks(run_root)
    records: list[Case23DRecord] = []
    global_start = time.perf_counter()
    total_jobs = len(tasks)

    print(f"[run] total_jobs={total_jobs} run_dir={run_root}")
    for completed, task in enumerate(tasks, start=1):
        record = run_case2_one(**task)
        records.append(record)
        append_raw_record(run_root, record)
        rewrite_summary_csv(run_root, summarize(records))
        write_progress_snapshot(
            run_root,
            completed_jobs=completed,
            total_jobs=total_jobs,
            records=records,
            global_start=global_start,
        )
        print_record(record, completed, total_jobs)

    summary = summarize(records)
    rewrite_summary_csv(run_root, summary)
    print_summary(summary)
    with (run_root / "records.json").open("w", encoding="utf-8") as file:
        json.dump([asdict(record) for record in records], file, indent=2)
    if any(record.success == 0 for record in records):
        raise RuntimeError(f"At least one pilot fit failed; inspect {run_root / 'records.json'}.")


if __name__ == "__main__":
    main()
