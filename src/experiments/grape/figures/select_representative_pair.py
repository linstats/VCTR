"""Rank representative GRAPE paired visits for descriptive image figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GRAPE_DIR = Path(__file__).resolve().parents[1]
PAIRED_CSV = GRAPE_DIR / "data" / "audit" / "processed_paired.csv"
TENSOR_ROOT = GRAPE_DIR / "data" / "tensors"
DEFAULT_OUTPUT = GRAPE_DIR / "runs" / "figures" / "paired_image_partitions_v1" / "candidate_pairs.csv"
FILTER_COLUMN = "include_primary_iop35"


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _median_absolute_deviation(values: pd.Series) -> float:
    median = float(values.median())
    mad = float((values - median).abs().median())
    return mad if mad > 0 else 1.0


def _available_pair_ids(tensor_root: Path) -> set[str]:
    pair_sets: list[set[str]] = []
    for image_type in ("cfp", "roi"):
        manifest_path = tensor_root / f"{image_type}_192_iop_le35" / "manifest.csv"
        manifest = pd.read_csv(manifest_path, dtype={"pair_id": str})
        pair_sets.append(set(manifest["pair_id"]))
    return set.intersection(*pair_sets)


def rank_candidates(paired_csv: Path, tensor_root: Path) -> pd.DataFrame:
    paired = pd.read_csv(paired_csv, dtype={"pair_id": str})
    paired = paired[_bool_series(paired[FILTER_COLUMN])].copy()
    paired = paired[paired["pair_id"].isin(_available_pair_ids(tensor_root))].copy()

    paired["mean_iop"] = paired[["iop_od", "iop_os"]].mean(axis=1)
    paired["iop_difference"] = (paired["iop_od"] - paired["iop_os"]).abs()

    age_median = float(paired["age_at_visit"].median())
    mean_iop_median = float(paired["mean_iop"].median())
    iop_difference_median = float(paired["iop_difference"].median())
    age_mad = _median_absolute_deviation(paired["age_at_visit"])
    mean_iop_mad = _median_absolute_deviation(paired["mean_iop"])
    iop_difference_mad = _median_absolute_deviation(paired["iop_difference"])

    paired["age_distance"] = (paired["age_at_visit"] - age_median).abs() / age_mad
    paired["mean_iop_distance"] = (paired["mean_iop"] - mean_iop_median).abs() / mean_iop_mad
    paired["paired_iop_distance"] = paired["iop_difference"] / iop_difference_mad
    paired["representative_score"] = (
        paired["age_distance"]
        + paired["mean_iop_distance"]
        + 0.25 * paired["paired_iop_distance"]
    )

    paired["age_median"] = age_median
    paired["mean_iop_median"] = mean_iop_median
    paired["iop_difference_median"] = iop_difference_median
    paired = paired.sort_values(
        ["representative_score", "subject_id", "interval_years"],
        kind="mergesort",
    ).reset_index(drop=True)
    paired.insert(0, "selection_rank", np.arange(1, len(paired) + 1, dtype=int))

    columns = [
        "selection_rank",
        "pair_id",
        "subject_id",
        "interval_years",
        "age_at_visit",
        "iop_od",
        "iop_os",
        "mean_iop",
        "iop_difference",
        "representative_score",
        "age_distance",
        "mean_iop_distance",
        "paired_iop_distance",
        "age_median",
        "mean_iop_median",
        "iop_difference_median",
        "cfp_path_od",
        "cfp_path_os",
        "roi_path_od",
        "roi_path_os",
    ]
    return paired[columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-csv", type=Path, default=PAIRED_CSV)
    parser.add_argument("--tensor-root", type=Path, default=TENSOR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top", type=int, default=20, help="Rows printed to stdout; the CSV retains all candidates.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranked = rank_candidates(args.paired_csv.resolve(), args.tensor_root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output, index=False)
    print(ranked.head(args.top).to_string(index=False))
    print(f"\nWrote {len(ranked)} ranked candidates to {output}")


if __name__ == "__main__":
    main()
