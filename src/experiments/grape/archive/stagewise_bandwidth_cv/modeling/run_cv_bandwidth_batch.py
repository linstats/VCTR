"""Run a balanced batch of GRAPE bandwidth-CV tasks."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GRAPE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = GRAPE_ROOT / "runs" / "cv_bandwidth"
DEFAULT_TASK_CSV = GRAPE_ROOT / "hpc" / "cv_bandwidth_tasks_v1.csv"
DEFAULT_BATCH_CSV = GRAPE_ROOT / "hpc" / "cv_bandwidth_batches_v1.csv"
CV_SCRIPT = GRAPE_ROOT / "modeling" / "cv_bandwidth.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-csv", type=Path, default=DEFAULT_BATCH_CSV)
    parser.add_argument("--batch-index", type=int, required=True)
    parser.add_argument("--task-csv", type=Path, default=DEFAULT_TASK_CSV)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--a-eval-mode", choices=["full", "anchor_grid"], default="anchor_grid")
    parser.add_argument("--a-eval-num-points", type=int, default=80)
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument("--fail-on-task-error", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (GRAPE_ROOT / path).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_batch(batch_csv: Path, batch_index: int) -> dict[str, Any]:
    for row in read_csv_rows(batch_csv):
        if int(row["batch_id"]) == int(batch_index):
            task_ids = [int(value) for value in row["task_ids"].split()]
            return {
                "batch_id": int(row["batch_id"]),
                "task_ids": task_ids,
                "n_tasks": int(row["n_tasks"]),
                "estimated_cost_sum": int(row["estimated_cost_sum"]),
            }
    raise ValueError(f"batch_id={batch_index} not found in {batch_csv}.")


def task_lookup(task_csv: Path) -> dict[int, dict[str, str]]:
    rows = read_csv_rows(task_csv)
    return {int(row["task_id"]): row for row in rows}


def rel_to_repo(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def run_one_task(
    *,
    task_id: int,
    task_csv: Path,
    run_root: Path,
    a_eval_mode: str,
    a_eval_num_points: int,
    ridge: float,
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    cmd = [
        sys.executable,
        str(CV_SCRIPT),
        "--task-csv",
        str(task_csv),
        "--task-index",
        str(task_id),
        "--run-root",
        str(run_root),
        "--a-eval-mode",
        a_eval_mode,
        "--a-eval-num-points",
        str(a_eval_num_points),
        "--ridge",
        str(ridge),
    ]
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - t0
    finished_at = datetime.now(timezone.utc)
    return {
        "task_id": int(task_id),
        "returncode": int(completed.returncode),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "subprocess_elapsed_seconds": elapsed,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def load_task_result(run_root: Path, run_name: str, task_id: int) -> dict[str, Any]:
    result_path = run_root / run_name / f"task_{task_id:04d}" / "result.json"
    if not result_path.exists():
        return {
            "task_id": int(task_id),
            "status": "missing_result_json",
            "elapsed_seconds": None,
            "result_json": result_path.as_posix(),
        }
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["result_json"] = result_path.as_posix()
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive.")

    batch_csv = resolve_path(args.batch_csv)
    task_csv = resolve_path(args.task_csv)
    run_root = resolve_path(args.run_root)
    batch = read_batch(batch_csv, args.batch_index)
    tasks = task_lookup(task_csv)
    task_rows = [tasks[task_id] for task_id in batch["task_ids"]]
    run_names = sorted({row["run_name"] for row in task_rows})
    if len(run_names) != 1:
        raise ValueError(f"Batch contains multiple run_name values: {run_names}")
    run_name = run_names[0]
    run_dir = run_root / run_name
    batch_dir = run_dir / f"batch_{int(args.batch_index):02d}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    batch_t0 = time.perf_counter()
    print(
        f"Starting batch_id={args.batch_index} n_tasks={len(batch['task_ids'])} "
        f"max_workers={args.max_workers} run_name={run_name}",
        flush=True,
    )

    subprocess_rows: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(batch["task_ids"]))) as executor:
        futures = {
            executor.submit(
                run_one_task,
                task_id=task_id,
                task_csv=task_csv,
                run_root=run_root,
                a_eval_mode=args.a_eval_mode,
                a_eval_num_points=int(args.a_eval_num_points),
                ridge=float(args.ridge),
            ): task_id
            for task_id in batch["task_ids"]
        }
        for future in as_completed(futures):
            task_id = futures[future]
            subprocess_row = future.result()
            subprocess_rows.append(subprocess_row)
            result = load_task_result(run_root, run_name, task_id)
            task_results.append(result)
            task_meta = tasks[task_id]
            print(
                "task_done "
                f"task_id={task_id} image_type={task_meta['image_type']} S={task_meta['S']} R={task_meta['R']} "
                f"status={result.get('status')} elapsed_seconds={result.get('elapsed_seconds')} "
                f"subprocess_elapsed_seconds={subprocess_row.get('subprocess_elapsed_seconds'):.3f} "
                f"returncode={subprocess_row.get('returncode')}",
                flush=True,
            )
            if args.fail_on_task_error and subprocess_row["returncode"] != 0:
                raise RuntimeError(f"task_id={task_id} failed with returncode={subprocess_row['returncode']}")

    finished_at = datetime.now(timezone.utc)
    batch_elapsed = time.perf_counter() - batch_t0
    task_results.sort(key=lambda row: int(row["task_id"]))
    subprocess_rows.sort(key=lambda row: int(row["task_id"]))
    n_success = sum(row.get("status") == "success" for row in task_results)
    n_failed = len(task_results) - n_success
    payload = {
        "batch_id": int(args.batch_index),
        "run_name": run_name,
        "batch_csv": rel_to_repo(batch_csv),
        "task_csv": rel_to_repo(task_csv),
        "task_ids": batch["task_ids"],
        "n_tasks": len(batch["task_ids"]),
        "estimated_cost_sum": batch["estimated_cost_sum"],
        "max_workers": int(args.max_workers),
        "a_eval_mode": args.a_eval_mode,
        "a_eval_num_points": int(args.a_eval_num_points),
        "ridge": float(args.ridge),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "batch_elapsed_seconds": batch_elapsed,
        "n_success": int(n_success),
        "n_failed": int(n_failed),
        "task_results": task_results,
        "subprocess_results": subprocess_rows,
    }
    write_json(batch_dir / "batch_result.json", payload)
    print(
        f"batch_done batch_id={args.batch_index} n_success={n_success} n_failed={n_failed} "
        f"batch_elapsed_seconds={batch_elapsed:.3f}",
        flush=True,
    )
    print(json.dumps({k: payload[k] for k in ("batch_id", "n_tasks", "n_success", "n_failed", "batch_elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
