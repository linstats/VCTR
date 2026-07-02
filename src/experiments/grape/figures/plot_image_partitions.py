"""Create publication-ready paired-eye CFP/ROI tensor-partition figures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-grape")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
GRAPE_DIR = Path(__file__).resolve().parents[1]
PAIRED_CSV = GRAPE_DIR / "data" / "audit" / "processed_paired.csv"
TENSOR_ROOT = GRAPE_DIR / "data" / "tensors"
DEFAULT_CONFIG = GRAPE_DIR / "configs" / "figures" / "paired_image_partitions_v1.json"
DEFAULT_OUTPUT = GRAPE_DIR / "outputs" / "figures" / "paired_image_partitions_v1"
IMAGE_DIRS = {
    "cfp": "cfp_192_iop_le35",
    "roi": "roi_192_iop_le35",
}
EYE_INDEX = {"od": 0, "os": 1}


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("pair_id"):
        raise ValueError("Figure config must define pair_id.")
    for image_type in IMAGE_DIRS:
        partition = config.get("partitions", {}).get(image_type)
        if not isinstance(partition, list) or len(partition) != 3:
            raise ValueError(f"partitions.{image_type} must contain three block counts.")
        if any(int(value) <= 0 for value in partition):
            raise ValueError(f"partitions.{image_type} must contain positive integers.")
    return config


def _load_pair_images(tensor_root: Path, pair_id: str) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    images: dict[tuple[str, str], np.ndarray] = {}
    source_meta: dict[str, Any] = {}
    expected_array_index: int | None = None

    for image_type, directory_name in IMAGE_DIRS.items():
        tensor_dir = tensor_root / directory_name
        manifest_path = tensor_dir / "manifest.csv"
        tensor_path = tensor_dir / "X_image_uint8.npy"
        meta_path = tensor_dir / "meta.json"
        manifest = pd.read_csv(manifest_path, dtype={"pair_id": str})
        selected = manifest[manifest["pair_id"] == pair_id]
        if len(selected) != 1:
            raise ValueError(f"Expected one {image_type} manifest row for pair_id={pair_id!r}; found {len(selected)}.")
        row = selected.iloc[0]
        array_index = int(row["array_index"])
        if expected_array_index is None:
            expected_array_index = array_index
        elif array_index != expected_array_index:
            raise ValueError("CFP and ROI manifests disagree on array_index.")

        tensor = np.load(tensor_path, mmap_mode="r")
        pair_tensor = np.asarray(tensor[array_index])
        for eye, eye_index in EYE_INDEX.items():
            images[(image_type, eye)] = pair_tensor[eye_index]

        source_meta[image_type] = {
            "array_index": array_index,
            "manifest": _relative_to_project(manifest_path),
            "tensor": _relative_to_project(tensor_path),
            "image_path_od": str(row["image_path_od"]),
            "image_path_os": str(row["image_path_os"]),
            "tensor_meta": json.loads(meta_path.read_text(encoding="utf-8")),
        }

    return images, source_meta


def _partition_boundaries(image: np.ndarray, partition: list[int]) -> tuple[list[float], list[float]]:
    height, width, channels = image.shape
    row_blocks, column_blocks, channel_blocks = (int(value) for value in partition)
    for size, blocks, label in (
        (height, row_blocks, "height"),
        (width, column_blocks, "width"),
        (channels, channel_blocks, "channels"),
    ):
        if size % blocks != 0:
            raise ValueError(f"Image {label}={size} is not divisible by {blocks} blocks.")
    horizontal = [row * height / row_blocks - 0.5 for row in range(1, row_blocks)]
    vertical = [column * width / column_blocks - 0.5 for column in range(1, column_blocks)]
    return horizontal, vertical


def _draw_panel(
    ax: plt.Axes,
    image: np.ndarray,
    partition: list[int],
    style: dict[str, Any],
    *,
    title: str | None = None,
) -> None:
    ax.imshow(image)
    horizontal, vertical = _partition_boundaries(image, partition)
    line_kwargs = {
        "color": style.get("grid_color", "#FFFFFF"),
        "linewidth": float(style.get("grid_linewidth", 1.5)),
        "linestyle": tuple(style.get("grid_linestyle", [0, [7, 5]])),
        "solid_capstyle": "butt",
        "dash_capstyle": "butt",
    }
    for y_value in horizontal:
        ax.axhline(y_value, **line_kwargs)
    for x_value in vertical:
        ax.axvline(x_value, **line_kwargs)
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=4)


def _save_individual_panels(
    images: dict[tuple[str, str], np.ndarray],
    config: dict[str, Any],
    panel_dir: Path,
) -> list[str]:
    panel_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    style = config.get("style", {})
    dpi = int(style.get("panel_dpi", 600))
    for image_type in ("cfp", "roi"):
        for eye in ("od", "os"):
            fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
            _draw_panel(ax, images[(image_type, eye)], config["partitions"][image_type], style)
            output = panel_dir / f"{image_type}_{eye}_partition.png"
            fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            outputs.append(_relative_to_project(output))
    return outputs


def _save_composite(
    images: dict[tuple[str, str], np.ndarray],
    config: dict[str, Any],
    output_dir: Path,
) -> list[str]:
    style = config.get("style", {})
    dpi = int(style.get("figure_dpi", 600))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2), constrained_layout=True)
    layout = (
        ("cfp", "od", "CFP, OD"),
        ("roi", "od", "ROI, OD"),
        ("cfp", "os", "CFP, OS (horizontally flipped)"),
        ("roi", "os", "ROI, OS (horizontally flipped)"),
    )
    for ax, (image_type, eye, title) in zip(axes.flat, layout, strict=True):
        _draw_panel(ax, images[(image_type, eye)], config["partitions"][image_type], style, title=title)

    png_path = output_dir / "grape_paired_partitions.png"
    pdf_path = output_dir / "grape_paired_partitions.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return [_relative_to_project(png_path), _relative_to_project(pdf_path)]


def _selected_pair_record(paired_csv: Path, pair_id: str) -> dict[str, Any]:
    paired = pd.read_csv(paired_csv, dtype={"pair_id": str})
    selected = paired[paired["pair_id"] == pair_id]
    if len(selected) != 1:
        raise ValueError(f"Expected one audit row for pair_id={pair_id!r}; found {len(selected)}.")
    row = selected.iloc[0]
    return {
        "pair_id": pair_id,
        "subject_id": int(row["subject_id"]),
        "interval_years": float(row["interval_years"]),
        "age_at_visit": float(row["age_at_visit"]),
        "iop_od": float(row["iop_od"]),
        "iop_os": float(row["iop_os"]),
        "visit_number_od": int(row["visit_number_od"]),
        "visit_number_os": int(row["visit_number_os"]),
        "include_primary_iop35": _as_bool(row["include_primary_iop35"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--paired-csv", type=Path, default=PAIRED_CSV)
    parser.add_argument("--tensor-root", type=Path, default=TENSOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    images, source_meta = _load_pair_images(args.tensor_root.resolve(), str(config["pair_id"]))
    panel_outputs = _save_individual_panels(images, config, output_dir / "panels")
    composite_outputs = _save_composite(images, config, output_dir)
    selected_pair = _selected_pair_record(args.paired_csv.resolve(), str(config["pair_id"]))

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _relative_to_project(config_path),
        "selected_pair": selected_pair,
        "selection": config.get("selection", {}),
        "partitions": config["partitions"],
        "source": source_meta,
        "outputs": {
            "composite": composite_outputs,
            "panels": panel_outputs,
        },
        "notes": [
            "Images are read from the exact Level-2 uint8 tensors used by the paired-eye analysis.",
            "OS panels are already horizontally flipped in the Level-2 tensors.",
            "Partition boundaries are derived from the configured block counts and the tensor dimensions.",
            "Panel labels and manuscript subcaptions should be added in LaTeX rather than baked into individual panels.",
        ],
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
