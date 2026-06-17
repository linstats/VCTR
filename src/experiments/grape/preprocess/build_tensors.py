"""Build GRAPE Level-2 image tensors from paired tables and raw images.

This script creates standardized CFP and ROI image tensors for the primary
paired-eye analysis. It does not partition images, compute CP features, or
write y/Z/t arrays.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


GRAPE_ROOT = Path(__file__).resolve().parents[1] / "data"
AUDIT_DIR = GRAPE_ROOT / "audit"
PAIRED_CSV = AUDIT_DIR / "processed_paired.csv"
TENSOR_ROOT = GRAPE_ROOT / "tensors"

FILTER_COLUMN = "include_primary_iop35"
IMAGE_SIZE = (192, 192)
EYE_ORDER = ("OD", "OS")
IMAGE_SPECS = {
    "cfp": {
        "output_dir": "cfp_192_iop_le35",
        "path_columns": ("cfp_path_od", "cfp_path_os"),
    },
    "roi": {
        "output_dir": "roi_192_iop_le35",
        "path_columns": ("roi_path_od", "roi_path_os"),
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-types",
        nargs="+",
        choices=sorted(IMAGE_SPECS),
        default=sorted(IMAGE_SPECS),
        help="Image inputs to build. Defaults to CFP and ROI.",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=PAIRED_CSV,
        help="Path to processed_paired.csv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=TENSOR_ROOT,
        help="Directory where tensor subdirectories are written.",
    )
    return parser.parse_args()


def _rel_to_grape_data(path: Path) -> str:
    return path.relative_to(GRAPE_ROOT).as_posix()


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _load_rgb(path: Path, *, flip_left_right: bool) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if flip_left_right:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        image = image.resize(IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def _build_manifest(paired: pd.DataFrame, image_type: str) -> pd.DataFrame:
    spec = IMAGE_SPECS[image_type]
    od_path_col, os_path_col = spec["path_columns"]
    manifest_cols = [
        "array_index",
        "pair_id",
        "subject_id",
        "interval_years",
        "visit_number_od",
        "visit_number_os",
        "observation_id_od",
        "observation_id_os",
        "include_primary_iop35",
        "pair_has_iop_gt35_visit",
        od_path_col,
        os_path_col,
    ]
    manifest = paired[manifest_cols].copy()
    manifest.insert(1, "image_type", image_type)
    manifest = manifest.rename(
        columns={
            od_path_col: "image_path_od",
            os_path_col: "image_path_os",
        }
    )
    return manifest


def _build_image_tensor(paired: pd.DataFrame, image_type: str) -> np.ndarray:
    spec = IMAGE_SPECS[image_type]
    od_path_col, os_path_col = spec["path_columns"]
    X = np.empty((len(paired), 2, IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)

    for row_idx, row in enumerate(paired.itertuples(index=False)):
        row_dict = row._asdict()
        od_path = GRAPE_ROOT / row_dict[od_path_col]
        os_path = GRAPE_ROOT / row_dict[os_path_col]
        X[row_idx, 0] = _load_rgb(od_path, flip_left_right=False)
        X[row_idx, 1] = _load_rgb(os_path, flip_left_right=True)

    return X


def _write_one_input(
    paired: pd.DataFrame,
    *,
    image_type: str,
    output_root: Path,
    source_paired_csv: Path,
) -> dict[str, object]:
    spec = IMAGE_SPECS[image_type]
    output_dir = output_root / spec["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = _build_manifest(paired, image_type)
    X = _build_image_tensor(paired, image_type)

    manifest.to_csv(output_dir / "manifest.csv", index=False)
    np.save(output_dir / "X_image_uint8.npy", X)

    meta: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_type": image_type,
        "source_paired_csv": _rel_to_grape_data(source_paired_csv.resolve()),
        "source_filter_column": FILTER_COLUMN,
        "source_filter_rule": "include rows where include_primary_iop35 is true",
        "output_dir": _rel_to_grape_data(output_dir.resolve()),
        "n_pairs": int(len(paired)),
        "eye_order": list(EYE_ORDER),
        "X_image_uint8_shape": list(X.shape),
        "X_image_uint8_dtype": str(X.dtype),
        "image_size": [IMAGE_SIZE[1], IMAGE_SIZE[0], 3],
        "resize": {
            "library": "PIL",
            "method": "Image.resize",
            "resample": "BILINEAR",
        },
        "os_horizontal_flip": True,
        "os_flip_before_resize": True,
        "image_values": {
            "dtype": "uint8",
            "range": [0, 255],
            "normalization": "none",
        },
        "notes": [
            "No cropping, masking, color enhancement, partitioning, or CP projection is applied.",
            "OD images are stored unchanged; OS images are horizontally flipped for orientation consistency.",
            "Response, vector covariates, and index variable remain in data/audit/processed_paired.csv and are not written at the tensor level.",
        ],
    }
    (output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return meta


def main() -> None:
    args = _parse_args()
    paired_csv = args.paired_csv.resolve()
    output_root = args.output_root.resolve()

    paired_all = pd.read_csv(paired_csv)
    paired = paired_all[_bool_series(paired_all[FILTER_COLUMN])].copy()
    paired = paired.sort_values(["subject_id", "interval_years"]).reset_index(drop=True)
    paired.insert(0, "array_index", np.arange(len(paired), dtype=int))

    results = {}
    for image_type in args.image_types:
        results[image_type] = _write_one_input(
            paired,
            image_type=image_type,
            output_root=output_root,
            source_paired_csv=paired_csv,
        )

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
