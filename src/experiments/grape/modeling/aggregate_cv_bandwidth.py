"""Aggregate GRAPE bandwidth-CV task outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


GRAPE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = GRAPE_ROOT / "runs" / "cv_bandwidth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args()


def flatten_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def load_results(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(run_dir.glob("task_*/result.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["result_json"] = result_path.as_posix()
        rows.append({key: flatten_value(value) for key, value in result.items()})
    if not rows:
        raise FileNotFoundError(f"No task_*/result.json files found under {run_dir}.")
    return pd.DataFrame(rows)


def best_by_image(summary: pd.DataFrame) -> pd.DataFrame:
    success = summary[summary["status"] == "success"].copy()
    if success.empty:
        return pd.DataFrame(
            columns=[
                "image_type",
                "task_id",
                "S",
                "R",
                "best_signal_h",
                "best_variance_h",
                "signal_cv_score",
                "variance_cv_score",
            ]
        )
    success["signal_cv_score_numeric"] = pd.to_numeric(success["signal_cv_score"], errors="coerce")
    success["variance_cv_score_numeric"] = pd.to_numeric(success["variance_cv_score"], errors="coerce")
    success = success.sort_values(
        ["image_type", "signal_cv_score_numeric", "variance_cv_score_numeric", "R", "S"],
        kind="mergesort",
    )
    cols = [
        "image_type",
        "task_id",
        "S",
        "R",
        "best_signal_h",
        "best_variance_h",
        "signal_cv_score",
        "variance_cv_score",
        "elapsed_seconds",
        "feature_dir",
        "output_dir",
    ]
    return success.groupby("image_type", as_index=False).head(1)[cols]


def main() -> None:
    args = parse_args()
    run_dir = (args.run_root / args.run_name).resolve()
    summary = load_results(run_dir)
    if "task_id" in summary.columns:
        summary = summary.sort_values("task_id", kind="mergesort")

    failures = summary[summary["status"] != "success"].copy()
    best = best_by_image(summary)

    summary.to_csv(run_dir / "summary_all.csv", index=False)
    failures.to_csv(run_dir / "failures.csv", index=False)
    best.to_csv(run_dir / "summary_best_by_image.csv", index=False)

    print(
        json.dumps(
            {
                "run_dir": run_dir.as_posix(),
                "n_tasks": int(len(summary)),
                "n_success": int((summary["status"] == "success").sum()),
                "n_failures": int((summary["status"] != "success").sum()),
                "elapsed_seconds_sum": float(pd.to_numeric(summary.get("elapsed_seconds"), errors="coerce").sum()),
                "summary_all": (run_dir / "summary_all.csv").as_posix(),
                "summary_best_by_image": (run_dir / "summary_best_by_image.csv").as_posix(),
                "failures": (run_dir / "failures.csv").as_posix(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
