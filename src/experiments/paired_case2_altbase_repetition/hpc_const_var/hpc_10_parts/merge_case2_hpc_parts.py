"""Merge incomplete part1-8 snapshot, exact backfill runs, and full part9-10 runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import numpy as np


STANDARD_FIELDS = (
    "n_subject",
    "coef_type",
    "rep",
    "seed",
    "success",
    "error_message",
    "elapsed_seconds",
    "covariance_mode",
    "signal_bandwidth_input",
    "signal_bandwidth_method",
    "best_signal_bandwidth",
    "variance_bandwidth_input",
    "variance_bandwidth_method",
    "best_variance_bandwidth",
    "sigma2_true",
    "rho_true",
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
)


@dataclass(frozen=True, slots=True)
class TaskKey:
    n_subject: int
    coef_type: str
    rho_true: float
    rep: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-snapshot-root", type=Path, required=True)
    parser.add_argument("--backfill-root", type=Path, required=True)
    parser.add_argument("--part9-root", type=Path, required=True)
    parser.add_argument("--part10-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-part-success", type=int, default=96)
    parser.add_argument("--expected-total-success", type=int, default=960)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key_from_row(row: dict[str, str]) -> TaskKey:
    return TaskKey(
        n_subject=int(row["n_subject"]),
        coef_type=str(row["coef_type"]),
        rho_true=float(row["rho_true"]),
        rep=int(row["rep"]),
        seed=int(row["seed"]),
    )


def standardize_row(row: dict[str, str]) -> dict[str, str]:
    return {field: row.get(field, "") for field in STANDARD_FIELDS}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric_or_none(value: str):
    if value == "" or value is None:
        return None
    return float(value)


def summarize(rows: Iterable[dict[str, str]]) -> list[dict[str, float | int | str | None]]:
    grouped: dict[tuple[int, str, float, str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(
            (
                int(row["n_subject"]),
                str(row["coef_type"]),
                float(row["rho_true"]),
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
        "Sigma_fro_error",
        "best_signal_bandwidth",
        "best_variance_bandwidth",
        "elapsed_seconds",
    )
    summary_rows: list[dict[str, float | int | str | None]] = []
    for (n_subject, coef_type, rho_true, covariance_mode, signal_bandwidth_method), vals in sorted(grouped.items()):
        row: dict[str, float | int | str | None] = {
            "n_subject": n_subject,
            "coef_type": coef_type,
            "rho_true": rho_true,
            "covariance_mode": covariance_mode,
            "signal_bandwidth_method": signal_bandwidth_method,
            "n_rep": len(vals),
            "n_success": int(sum(int(v["success"]) for v in vals)),
            "n_fail": int(sum(1 - int(v["success"]) for v in vals)),
        }
        for field in metric_fields:
            arr = np.array([numeric_or_none(v[field]) for v in vals if numeric_or_none(v[field]) is not None], dtype=float)
            row[f"{field}_mean"] = float(np.mean(arr)) if arr.size else None
            row[f"{field}_std"] = float(np.std(arr, ddof=0)) if arr.size else None
        summary_rows.append(row)
    return summary_rows


def success_only(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if int(row["success"]) == 1]


def merge_sources(
    *,
    old_snapshot_root: Path,
    backfill_root: Path,
    part9_root: Path,
    part10_root: Path,
) -> tuple[list[dict[str, str]], dict]:
    merged_by_key: dict[TaskKey, dict[str, str]] = {}
    provenance_rows: list[dict[str, object]] = []
    duplicate_events: list[dict[str, object]] = []
    part_counts: dict[str, dict[str, int]] = {}

    def ingest(rows: Iterable[dict[str, str]], source_label: str, prefer_existing: bool) -> None:
        for raw_row in rows:
            std_row = standardize_row(raw_row)
            key = key_from_row(std_row)
            if key in merged_by_key:
                duplicate_events.append(
                    {
                        "source_label": source_label,
                        "existing_source_label": merged_by_key[key]["_source_label"],
                        "n_subject": key.n_subject,
                        "coef_type": key.coef_type,
                        "rho_true": key.rho_true,
                        "rep": key.rep,
                        "seed": key.seed,
                    }
                )
                if prefer_existing:
                    continue
            merged_by_key[key] = {**std_row, "_source_label": source_label}

    for part in range(1, 9):
        old_rows = success_only(read_rows(old_snapshot_root / f"part{part}" / "results" / "raw_results.csv"))
        backfill_rows = success_only(read_rows(backfill_root / f"part{part}" / "results" / "raw_results.csv"))
        ingest(old_rows, f"old_snapshot_part{part}", prefer_existing=True)
        ingest(backfill_rows, f"backfill_part{part}", prefer_existing=True)
        part_counts[f"part{part}"] = {
            "old_success": len(old_rows),
            "backfill_success": len(backfill_rows),
        }

    part9_rows = success_only(read_rows(part9_root / "results" / "raw_results.csv"))
    part10_rows = success_only(read_rows(part10_root / "results" / "raw_results.csv"))
    ingest(part9_rows, "full_part9", prefer_existing=False)
    ingest(part10_rows, "full_part10", prefer_existing=False)
    part_counts["part9"] = {"full_success": len(part9_rows)}
    part_counts["part10"] = {"full_success": len(part10_rows)}

    merged_rows: list[dict[str, str]] = []
    for key in sorted(merged_by_key, key=lambda item: (item.seed, item.n_subject, item.coef_type, item.rho_true, item.rep)):
        row = dict(merged_by_key[key])
        source_label = row.pop("_source_label")
        merged_rows.append(row)
        provenance_rows.append(
            {
                "source_label": source_label,
                "n_subject": key.n_subject,
                "coef_type": key.coef_type,
                "rho_true": key.rho_true,
                "rep": key.rep,
                "seed": key.seed,
            }
        )

    meta = {
        "part_counts": part_counts,
        "duplicate_events": duplicate_events,
        "provenance_count": len(provenance_rows),
    }
    return merged_rows, {"provenance_rows": provenance_rows, **meta}


def validate_merged(
    rows: list[dict[str, str]],
    expected_part_success: int,
    expected_total_success: int,
) -> dict[str, int]:
    counts: dict[str, int] = {f"part{i}": 0 for i in range(1, 11)}
    for row in rows:
        seed = int(row["seed"])
        # part1-10 used disjoint seed blocks of length 3:
        # part1=123-125, part2=126-128, ..., part10=150-152
        part = ((seed - 123) // 3) + 1
        counts[f"part{part}"] += 1
    for part in range(1, 11):
        label = f"part{part}"
        if counts[label] != expected_part_success:
            raise SystemExit(f"merged validation failed: {label} has {counts[label]} successes, expected {expected_part_success}")
    total = len(rows)
    if total != expected_total_success:
        raise SystemExit(f"merged validation failed: total {total} successes, expected {expected_total_success}")
    return counts


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=False)

    merged_rows, meta = merge_sources(
        old_snapshot_root=args.old_snapshot_root.resolve(),
        backfill_root=args.backfill_root.resolve(),
        part9_root=args.part9_root.resolve(),
        part10_root=args.part10_root.resolve(),
    )
    merged_counts = validate_merged(
        merged_rows,
        expected_part_success=args.expected_part_success,
        expected_total_success=args.expected_total_success,
    )
    summary_rows = summarize(merged_rows)

    write_csv(results_dir / "raw_results.csv", merged_rows, STANDARD_FIELDS)
    if summary_rows:
        write_csv(results_dir / "summary_results.csv", summary_rows, summary_rows[0].keys())
    write_csv(output_root / "provenance.csv", meta["provenance_rows"], ("source_label", "n_subject", "coef_type", "rho_true", "rep", "seed"))

    merge_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": "src/experiments/paired_case2_altbase_repetition/merge_case2_hpc_parts.py",
        "old_snapshot_root": str(args.old_snapshot_root.resolve()),
        "backfill_root": str(args.backfill_root.resolve()),
        "part9_root": str(args.part9_root.resolve()),
        "part10_root": str(args.part10_root.resolve()),
        "output_root": str(output_root),
        "expected_part_success": args.expected_part_success,
        "expected_total_success": args.expected_total_success,
        "merged_counts": merged_counts,
        "duplicate_events": meta["duplicate_events"],
        "part_counts": meta["part_counts"],
    }
    with (output_root / "merge_meta.json").open("w", encoding="utf-8") as f:
        json.dump(merge_meta, f, indent=2)

    print(f"Wrote merged raw results to {results_dir / 'raw_results.csv'}")
    print(f"Wrote merged summary results to {results_dir / 'summary_results.csv'}")
    print(f"Wrote provenance map to {output_root / 'provenance.csv'}")
    print(f"Wrote merge metadata to {output_root / 'merge_meta.json'}")


if __name__ == "__main__":
    main()
