"""Build a local grid of GRAPE Level-3 CP feature packages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.grape.features.build_features import (
    AUDIT_CSV,
    DEFAULT_IMAGE_TYPES,
    FEATURE_ROOT,
    TENSOR_ROOT,
    _build_model_arrays,
    _build_one,
    _load_audit_rows,
    _parse_s,
    _rel_to_grape_data,
    _s_label,
    _z_columns,
)


DEFAULT_S_GRID = ("2x2x1", "3x3x1", "4x4x1", "6x6x1", "8x8x1")
DEFAULT_R_GRID = (1, 2, 3, 4)
EXPECTED_FILES = (
    "manifest.csv",
    "X_star.npy",
    "y.npy",
    "Z.npy",
    "t.npy",
    "cp_components.npz",
    "meta.json",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-types", nargs="+", choices=DEFAULT_IMAGE_TYPES, default=list(DEFAULT_IMAGE_TYPES))
    parser.add_argument("--S-grid", nargs="+", type=_parse_s, default=[_parse_s(value) for value in DEFAULT_S_GRID])
    parser.add_argument("--R-grid", nargs="+", type=int, default=list(DEFAULT_R_GRID))
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--random-state", type=int, default=20260617)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV)
    parser.add_argument("--tensor-root", type=Path, default=TENSOR_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild feature packages even when all expected output files already exist.",
    )
    return parser.parse_args()


def _feature_dir(feature_root: Path, image_type: str, blocks_per_mode: tuple[int, ...], rank: int) -> Path:
    return feature_root / f"{image_type}_192_iop_le35" / f"S{_s_label(blocks_per_mode)}_R{rank}"


def _is_complete(feature_dir: Path) -> bool:
    return all((feature_dir / name).exists() for name in EXPECTED_FILES)


def _grid_size(image_types: Iterable[str], s_grid: Iterable[tuple[int, ...]], r_grid: Iterable[int]) -> int:
    return len(list(image_types)) * len(list(s_grid)) * len(list(r_grid))


def main() -> None:
    args = _parse_args()
    if any(rank <= 0 for rank in args.R_grid):
        raise ValueError("All R values must be positive.")

    paired = _load_audit_rows(args.audit_csv.resolve())
    z_cols = _z_columns(paired)
    y, Z, t, transform_meta = _build_model_arrays(paired, z_cols)

    feature_root = args.feature_root.resolve()
    tensor_root = args.tensor_root.resolve()
    total = _grid_size(args.image_types, args.S_grid, args.R_grid)
    records: list[dict[str, object]] = []
    started_at = datetime.now(timezone.utc)
    item_idx = 0

    for image_type in args.image_types:
        for blocks_per_mode in args.S_grid:
            for rank in args.R_grid:
                item_idx += 1
                output_dir = _feature_dir(feature_root, image_type, blocks_per_mode, rank)
                record: dict[str, object] = {
                    "image_type": image_type,
                    "S": _s_label(blocks_per_mode),
                    "R": int(rank),
                    "output_dir": _rel_to_grape_data(output_dir),
                }

                if not args.overwrite and _is_complete(output_dir):
                    record["status"] = "skipped_existing"
                    records.append(record)
                    print(f"[{item_idx}/{total}] skip {image_type} S{record['S']} R{rank}", flush=True)
                    continue

                print(f"[{item_idx}/{total}] build {image_type} S{record['S']} R{rank}", flush=True)
                t0 = time.perf_counter()
                try:
                    meta = _build_one(
                        image_type=image_type,
                        tensor_root=tensor_root,
                        feature_root=feature_root,
                        paired=paired,
                        y=y,
                        Z=Z,
                        t=t,
                        transform_meta=transform_meta,
                        blocks_per_mode=blocks_per_mode,
                        rank=rank,
                        max_iter=args.max_iter,
                        tol=args.tol,
                        random_state=args.random_state,
                    )
                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = repr(exc)
                    record["elapsed_seconds"] = time.perf_counter() - t0
                    records.append(record)
                    raise

                record["status"] = "built"
                record["elapsed_seconds"] = time.perf_counter() - t0
                record["X_star_shape"] = meta["X_star_shape"]
                record["n_blocks"] = meta["n_blocks"]
                record["max_block_iterations"] = max(meta["cp"]["block_iterations"])
                record["max_block_relative_change"] = max(meta["cp"]["block_relative_changes"])
                records.append(record)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at.isoformat(),
        "n_records": len(records),
        "n_built": sum(record["status"] == "built" for record in records),
        "n_skipped_existing": sum(record["status"] == "skipped_existing" for record in records),
        "image_types": list(args.image_types),
        "S_grid": [_s_label(value) for value in args.S_grid],
        "R_grid": [int(value) for value in args.R_grid],
        "max_iter": int(args.max_iter),
        "tol": float(args.tol),
        "random_state": int(args.random_state),
        "records": records,
    }
    feature_root.mkdir(parents=True, exist_ok=True)
    summary_path = feature_root / "build_feature_grid_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
