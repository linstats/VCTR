"""Merge flat part outputs for Case 1 base5/base6 all-sigma HPC runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).with_name("hpc_base56_allsigma"),
        help="Root directory that contains part1, part2, ..., part8 outputs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if merged row count does not match the sum of per-part total_jobs.",
    )
    return parser.parse_args()


def iter_part_roots(root: Path) -> list[Path]:
    part_roots: list[Path] = []
    for part_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        if (part_dir / "results" / "raw_results.csv").exists() and (part_dir / "run_config.json").exists():
            part_roots.append(part_dir)
    return part_roots


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_or_none(value: str | None) -> float | None:
    if value in ("", None):
        return None
    return float(value)


def summarize(rows: Iterable[dict[str, str]]) -> list[dict[str, float | int | str | None]]:
    grouped: dict[tuple[int, str, float, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                int(row["n_subject"]),
                str(row["coef_type"]),
                float(row["rho_true"]),
                str(row["sigma2_function"]),
                str(row["covariance_mode"]),
                str(row["signal_bandwidth_method"]),
            ),
            [],
        ).append(row)

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
        "rho_error",
        "best_signal_bandwidth",
        "best_variance_bandwidth",
        "elapsed_seconds",
    )

    summary_rows: list[dict[str, float | int | str | None]] = []
    for (
        n_subject,
        coef_type,
        rho_true,
        sigma2_function,
        covariance_mode,
        signal_bandwidth_method,
    ), vals in sorted(grouped.items()):
        row: dict[str, float | int | str | None] = {
            "n_subject": n_subject,
            "coef_type": coef_type,
            "rho_true": rho_true,
            "sigma2_function": sigma2_function,
            "covariance_mode": covariance_mode,
            "signal_bandwidth_method": signal_bandwidth_method,
            "n_rep": len(vals),
            "n_success": int(sum(int(v["success"]) for v in vals)),
            "n_fail": int(sum(1 - int(v["success"]) for v in vals)),
        }
        for field in metric_fields:
            arr = np.array(
                [numeric_or_none(v.get(field)) for v in vals if numeric_or_none(v.get(field)) is not None],
                dtype=float,
            )
            row[f"{field}_mean"] = float(np.mean(arr)) if arr.size else None
            row[f"{field}_std"] = float(np.std(arr, ddof=0)) if arr.size else None

        rho_errors = np.array(
            [numeric_or_none(v.get("rho_error")) for v in vals if numeric_or_none(v.get("rho_error")) is not None],
            dtype=float,
        )
        row["rho_mae"] = float(np.mean(np.abs(rho_errors))) if rho_errors.size else None
        row["rho_rmse"] = float(np.sqrt(np.mean(np.square(rho_errors)))) if rho_errors.size else None
        summary_rows.append(row)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_all_parts(
    root: Path,
    *,
    strict: bool = False,
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
    part_roots = iter_part_roots(root)
    if not part_roots:
        raise FileNotFoundError(f"No part outputs found under {root}")

    merged_rows: list[dict[str, str]] = []
    config_rows: list[dict[str, object]] = []
    merged_from_run_names: list[str] = []
    expected_total_records = 0
    part_record_counts: list[dict[str, int | str]] = []

    for part_root in part_roots:
        raw_path = part_root / "results" / "raw_results.csv"
        run_config_path = part_root / "run_config.json"
        rows = read_csv_rows(raw_path)
        merged_rows.extend(rows)

        with run_config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["_part_root"] = str(part_root)
        config_rows.append(cfg)
        merged_from_run_names.append(part_root.name)
        expected_total_records += int(cfg["total_jobs"])
        part_record_counts.append(
            {
                "part": part_root.name,
                "expected_records": int(cfg["total_jobs"]),
                "actual_records": len(rows),
                "missing_records": int(cfg["total_jobs"]) - len(rows),
            }
        )

    merged_rows.sort(
        key=lambda row: (
            row.get("sigma2_function", ""),
            int(row["seed"]),
            int(row["n_subject"]),
            row["coef_type"],
            float(row["rho_true"]),
            int(row["rep"]),
        )
    )

    if strict and len(merged_rows) != expected_total_records:
        raise ValueError(
            f"Merged row count {len(merged_rows)} does not match expected total_jobs {expected_total_records}."
        )

    config_rows_sorted = sorted(config_rows, key=lambda cfg: int(cfg["seed_base"]))
    reference = config_rows_sorted[0]
    aggregate_config = {
        "script": "src/experiments/paired_case1_altbase_repetition.py",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": root.name,
        "n_subject_values": reference["n_subject_values"],
        "coef_types": reference["coef_types"],
        "sigma2_functions": reference["sigma2_functions"],
        "rho_values": reference["rho_values"],
        "n_rep_per_part": [int(cfg["n_rep"]) for cfg in config_rows_sorted],
        "seed_bases_per_part": [int(cfg["seed_base"]) for cfg in config_rows_sorted],
        "n_rep_total": int(sum(int(cfg["n_rep"]) for cfg in config_rows_sorted)),
        "R": reference["R"],
        "S": reference["S"],
        "p0": reference["p0"],
        "beta": reference["beta"],
        "sigma2": reference["sigma2"],
        "covariance_mode": reference["covariance_mode"],
        "signal_bandwidth": reference["signal_bandwidth"],
        "signal_bandwidth_method": reference["signal_bandwidth_method"],
        "signal_bandwidth_grid": reference["signal_bandwidth_grid"],
        "variance_bandwidth": reference["variance_bandwidth"],
        "variance_bandwidth_method": reference["variance_bandwidth_method"],
        "variance_bandwidth_grid": reference["variance_bandwidth_grid"],
        "ridge": reference["ridge"],
        "n_jobs": reference["n_jobs"],
        "save_data": reference["save_data"],
        "save_estimates": reference["save_estimates"],
        "plot_functions": reference["plot_functions"],
        "plot_a_indices": reference["plot_a_indices"],
        "plot_max_a_panels": reference["plot_max_a_panels"],
        "created_at_first": min(str(cfg["created_at"]) for cfg in config_rows_sorted),
        "created_at_last": max(str(cfg["created_at"]) for cfg in config_rows_sorted),
        "merged_from_run_names": merged_from_run_names,
        "n_merged_parts": len(config_rows),
        "total_records_expected": expected_total_records,
        "total_records_merged": len(merged_rows),
        "total_records_missing": expected_total_records - len(merged_rows),
        "merge_note": "Merged from flat part1-part8 HPC runs. summary_results.csv recomputed from concatenated raw_results.csv.",
    }
    success_counts: dict[str, int] = {}
    for row in merged_rows:
        key = str(int(row["success"]))
        success_counts[key] = success_counts.get(key, 0) + 1
    merge_meta = {
        "n_records": len(merged_rows),
        "n_summary_rows": len(summarize(merged_rows)),
        "parts": merged_from_run_names,
        "expected_total_records": expected_total_records,
        "missing_total_records": expected_total_records - len(merged_rows),
        "part_record_counts": part_record_counts,
        "success_counts": dict(sorted(success_counts.items())),
    }
    return merged_rows, aggregate_config, merge_meta


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    merged_rows, aggregate_config, merge_meta = merge_all_parts(root, strict=args.strict)
    results_dir = root / "results"
    summary_rows = summarize(merged_rows)

    with (root / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate_config, f, indent=2)
    with (root / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(merge_meta, f, indent=2)

    if merged_rows:
        write_csv(results_dir / "raw_results.csv", merged_rows, merged_rows[0].keys())
    if summary_rows:
        write_csv(results_dir / "summary_results.csv", summary_rows, summary_rows[0].keys())

    print(f"Wrote aggregate run config to {root / 'run_config.json'}")
    print(f"Wrote merge metadata to {root / 'merge_meta.json'}")
    print(f"Wrote merged raw results to {results_dir / 'raw_results.csv'}")
    print(f"Wrote merged summary results to {results_dir / 'summary_results.csv'}")


if __name__ == "__main__":
    main()
