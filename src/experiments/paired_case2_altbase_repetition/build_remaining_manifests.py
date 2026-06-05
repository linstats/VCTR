"""Build balanced 8-part manifests for the remaining Case 2 jobs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


N_SUBJECT_VALUES = (2000, 5000)
RHO_VALUES = (0.0, 0.3, 0.6, 0.9)
SEEDS = tuple(range(123, 153))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-parts", type=int, default=8)
    return parser.parse_args()


def build_bundles() -> dict[str, list[dict[str, object]]]:
    bundles: dict[str, list[dict[str, object]]] = {
        "a1a4_varying_var": [],
        "a5a6_allsigma": [],
    }
    for seed in SEEDS:
        rep = seed - SEEDS[0]
        for coef_type in ("base1", "base2", "base3", "base4"):
            for sigma2_function in ("sin", "sin2", "mixed"):
                bundles["a1a4_varying_var"].append(
                    {
                        "coef_type": coef_type,
                        "sigma2_function": sigma2_function,
                        "seed": seed,
                        "rep": rep,
                    }
                )
        for coef_type in ("base5", "base6"):
            for sigma2_function in ("constant", "sin", "sin2", "mixed"):
                bundles["a5a6_allsigma"].append(
                    {
                        "coef_type": coef_type,
                        "sigma2_function": sigma2_function,
                        "seed": seed,
                        "rep": rep,
                    }
                )
    return bundles


def distribute_round_robin(items: list[dict[str, object]], n_parts: int) -> list[list[dict[str, object]]]:
    parts = [[] for _ in range(n_parts)]
    for idx, item in enumerate(items):
        parts[idx % n_parts].append(item)
    return parts


def expand_bundle(part_id: int, block_name: str, bundle: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n_subject in N_SUBJECT_VALUES:
        for rho_true in RHO_VALUES:
            rows.append(
                {
                    "part": part_id,
                    "bundle_block": block_name,
                    "n_subject": n_subject,
                    "coef_type": str(bundle["coef_type"]),
                    "sigma2_function": str(bundle["sigma2_function"]),
                    "rho_true": rho_true,
                    "rep": int(bundle["rep"]),
                    "seed": int(bundle["seed"]),
                }
            )
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "part",
        "bundle_block",
        "n_subject",
        "coef_type",
        "sigma2_function",
        "rho_true",
        "rep",
        "seed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundles = build_bundles()
    a1a4_parts = distribute_round_robin(bundles["a1a4_varying_var"], args.n_parts)
    a5a6_parts = distribute_round_robin(bundles["a5a6_allsigma"], args.n_parts)

    summary: list[dict[str, object]] = []
    for idx in range(args.n_parts):
        part_id = idx + 1
        rows: list[dict[str, object]] = []
        for bundle in a1a4_parts[idx]:
            rows.extend(expand_bundle(part_id, "a1a4_varying_var", bundle))
        for bundle in a5a6_parts[idx]:
            rows.extend(expand_bundle(part_id, "a5a6_allsigma", bundle))
        rows.sort(key=lambda row: (int(row["seed"]), str(row["coef_type"]), str(row["sigma2_function"]), int(row["n_subject"]), float(row["rho_true"])))
        write_manifest(args.output_dir / f"part{part_id}.csv", rows)
        summary.append(
            {
                "part": part_id,
                "task_count": len(rows),
                "a1a4_bundle_count": len(a1a4_parts[idx]),
                "a5a6_bundle_count": len(a5a6_parts[idx]),
                "seed_min": min(int(row["seed"]) for row in rows),
                "seed_max": max(int(row["seed"]) for row in rows),
            }
        )

    with (args.output_dir / "manifest_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
