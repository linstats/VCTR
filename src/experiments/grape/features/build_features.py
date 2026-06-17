"""Build GRAPE Level-3 CP features from standardized image tensors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import CPProjectionConfig, PartitionSpec, blockwise_cp_project, partition_tensor_blocks


GRAPE_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
AUDIT_CSV = GRAPE_DATA_ROOT / "audit" / "processed_paired.csv"
TENSOR_ROOT = GRAPE_DATA_ROOT / "tensors"
FEATURE_ROOT = GRAPE_DATA_ROOT / "features"
FILTER_COLUMN = "include_primary_iop35"
DEFAULT_IMAGE_TYPES = ("cfp", "roi")


def _parse_s(value: str) -> tuple[int, ...]:
    try:
        parts = tuple(int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("S must have format like 3x3x1.") from exc
    if not parts or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("S must contain positive integers.")
    return parts


def _s_label(blocks: tuple[int, ...]) -> str:
    return "x".join(str(part) for part in blocks)


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _rel_to_grape_data(path: Path) -> str:
    try:
        return path.relative_to(GRAPE_DATA_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _zscore(values: np.ndarray, *, axis: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = values.mean(axis=axis, keepdims=True)
    sd = values.std(axis=axis, ddof=0, keepdims=True)
    sd_safe = sd.copy()
    sd_safe[sd_safe == 0] = 1.0
    return (values - mean) / sd_safe, np.squeeze(mean, axis=axis), np.squeeze(sd_safe, axis=axis)


def _load_audit_rows(audit_csv: Path) -> pd.DataFrame:
    paired = pd.read_csv(audit_csv)
    paired = paired[_bool_series(paired[FILTER_COLUMN])].copy()
    paired = paired.sort_values(["subject_id", "interval_years"]).reset_index(drop=True)
    paired.insert(0, "array_index", np.arange(len(paired), dtype=int))
    return paired


def _z_columns(paired: pd.DataFrame) -> list[str]:
    cols = [col for col in paired.columns if col.startswith("z_vf_") and col.endswith("_mean")]
    if not cols:
        raise ValueError("No z_vf_*_mean columns found in audit paired table.")
    return cols


def _build_model_arrays(paired: pd.DataFrame, z_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    y_raw = paired[["iop_od", "iop_os"]].to_numpy(dtype=np.float64)
    y_flat_std, y_mean, y_sd = _zscore(y_raw.reshape(-1), axis=0)
    y = y_flat_std.reshape(y_raw.shape)

    z_vf_raw = paired[z_cols].to_numpy(dtype=np.float64)
    z_vf, z_vf_mean, z_vf_sd = _zscore(z_vf_raw, axis=0)
    Z = np.column_stack([paired["is_female"].to_numpy(dtype=np.float64), z_vf])

    age = paired["age_at_visit"].to_numpy(dtype=np.float64)
    age_min = float(age.min())
    age_max = float(age.max())
    age_range = age_max - age_min
    if age_range == 0:
        raise ValueError("Cannot min-max normalize t because all ages are identical.")
    t = (age - age_min) / age_range

    transform_meta = {
        "y": {
            "source_columns": ["iop_od", "iop_os"],
            "transform": "zscore over flattened OD/OS responses",
            "mean": float(y_mean),
            "sd": float(y_sd),
        },
        "Z": {
            "columns": ["is_female", *z_cols],
            "is_female_transform": "none",
            "vf_transform": "columnwise zscore over paired rows",
            "vf_mean": z_vf_mean.tolist(),
            "vf_sd": z_vf_sd.tolist(),
        },
        "t": {
            "source_column": "age_at_visit",
            "transform": "min-max normalization over included paired rows",
            "age_min": age_min,
            "age_max": age_max,
        },
    }
    return y, Z, t, transform_meta


def _save_cp_components(output_dir: Path, result) -> None:
    arrays: dict[str, np.ndarray] = {}
    for block_idx, block_result in enumerate(result.blocks):
        prefix = f"block_{block_idx:03d}"
        arrays[f"{prefix}_lambdas"] = block_result.lambdas
        arrays[f"{prefix}_sample_scores"] = block_result.sample_scores
        for mode_idx, factor in enumerate(block_result.factors):
            arrays[f"{prefix}_factor_mode_{mode_idx}"] = factor
    np.savez_compressed(output_dir / "cp_components.npz", **arrays)


def _build_one(
    *,
    image_type: str,
    tensor_root: Path,
    feature_root: Path,
    paired: pd.DataFrame,
    y: np.ndarray,
    Z: np.ndarray,
    t: np.ndarray,
    transform_meta: dict,
    blocks_per_mode: tuple[int, ...],
    rank: int,
    max_iter: int,
    tol: float,
    random_state: int,
) -> dict[str, object]:
    tensor_dir = tensor_root / f"{image_type}_192_iop_le35"
    tensor_manifest_path = tensor_dir / "manifest.csv"
    tensor_path = tensor_dir / "X_image_uint8.npy"
    if not tensor_manifest_path.exists() or not tensor_path.exists():
        raise FileNotFoundError(f"Missing tensor inputs under {tensor_dir}.")

    tensor_manifest = pd.read_csv(tensor_manifest_path)
    if not paired["pair_id"].equals(tensor_manifest["pair_id"]):
        raise ValueError(f"Pair order mismatch between audit table and {tensor_manifest_path}.")

    X_image = np.load(tensor_path, mmap_mode="r")
    n_pair = len(paired)
    if X_image.shape[:2] != (n_pair, 2):
        raise ValueError(f"Unexpected tensor shape {X_image.shape}; expected first axes {(n_pair, 2)}.")

    X_flat = np.asarray(X_image).reshape(n_pair * 2, *X_image.shape[2:])
    partition_spec = PartitionSpec(blocks_per_mode=blocks_per_mode)
    blocks = partition_tensor_blocks(X_flat, partition_spec)

    cp_config = CPProjectionConfig(
        rank=rank,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        dtype="float64",
        standardize_sample_factors=True,
    )
    cp_result = blockwise_cp_project(blocks, cp_config)
    X_star = cp_result.X_star_flat.reshape(n_pair, 2, rank, len(blocks))

    feature_dir = feature_root / f"{image_type}_192_iop_le35" / f"S{_s_label(blocks_per_mode)}_R{rank}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    manifest = paired[
        [
            "array_index",
            "pair_id",
            "subject_id",
            "interval_years",
            "observation_id_od",
            "observation_id_os",
        ]
    ].copy()
    manifest["image_type"] = image_type
    manifest["tensor_manifest"] = _rel_to_grape_data(tensor_manifest_path.resolve())
    manifest.to_csv(feature_dir / "manifest.csv", index=False)

    np.save(feature_dir / "X_star.npy", X_star)
    np.save(feature_dir / "y.npy", y)
    np.save(feature_dir / "Z.npy", Z)
    np.save(feature_dir / "t.npy", t)
    _save_cp_components(feature_dir, cp_result)

    meta: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_type": image_type,
        "source_audit_csv": _rel_to_grape_data(AUDIT_CSV.resolve()),
        "source_tensor_dir": _rel_to_grape_data(tensor_dir.resolve()),
        "source_filter_column": FILTER_COLUMN,
        "source_filter_rule": "include rows where include_primary_iop35 is true",
        "output_dir": _rel_to_grape_data(feature_dir.resolve()),
        "n_pairs": int(n_pair),
        "eye_order": ["OD", "OS"],
        "blocks_per_mode": list(blocks_per_mode),
        "n_blocks": int(len(blocks)),
        "rank": int(rank),
        "X_star_shape": list(X_star.shape),
        "X_star_dtype": str(X_star.dtype),
        "y_shape": list(y.shape),
        "Z_shape": list(Z.shape),
        "t_shape": list(t.shape),
        "cp": {
            "method": "dependency-free CP-ALS",
            "max_iter": int(max_iter),
            "tol": float(tol),
            "random_state": int(random_state),
            "sample_factor_standardization": "divide each CP sample-score column by its population standard deviation",
            "block_iterations": [int(block.n_iter) for block in cp_result.blocks],
            "block_relative_changes": [float(block.relative_change) for block in cp_result.blocks],
        },
        "transforms": transform_meta,
        "notes": [
            "X_star is generated from tensors/X_image_uint8.npy after casting image values to float64 without dividing by 255.",
            "y, Z, and t are generated from audit/processed_paired.csv at the feature layer.",
        ],
    }
    (feature_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-types", nargs="+", choices=DEFAULT_IMAGE_TYPES, default=list(DEFAULT_IMAGE_TYPES))
    parser.add_argument("--S", type=_parse_s, default=_parse_s("3x3x1"))
    parser.add_argument("--R", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--random-state", type=int, default=20260617)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--tensor-root", type=Path, default=TENSOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.R <= 0:
        raise ValueError("R must be positive.")

    paired = _load_audit_rows(args.audit_csv.resolve())
    z_cols = _z_columns(paired)
    y, Z, t, transform_meta = _build_model_arrays(paired, z_cols)

    results = {}
    for image_type in args.image_types:
        results[image_type] = _build_one(
            image_type=image_type,
            tensor_root=args.tensor_root.resolve(),
            feature_root=args.feature_root.resolve(),
            paired=paired,
            y=y,
            Z=Z,
            t=t,
            transform_meta=transform_meta,
            blocks_per_mode=args.S,
            rank=args.R,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.random_state,
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
