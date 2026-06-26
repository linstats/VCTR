"""Export one PNG per A[r,s](t) curve from saved dataset/estimate artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import os

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Diagnostic fit directory that contains data/ and estimates/ subdirectories.",
    )
    parser.add_argument(
        "--output-dirname",
        type=str,
        default="plots_all_A",
        help="Subdirectory name under run_dir to store per-component A(t) plots.",
    )
    parser.add_argument(
        "--stem",
        type=str,
        default="A",
        help="Filename stem for exported plots.",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    cache_root = Path(tempfile.gettempdir()) / "matplotlib-codex"
    cache_root.mkdir(parents=True, exist_ok=True)
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg-cache")
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(cache_root / "mplconfig")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")


def find_single_npz(directory: Path) -> Path:
    files = sorted(directory.glob("*.npz"))
    if len(files) != 1:
        raise FileNotFoundError(f"Expected exactly one .npz under {directory}, found {len(files)}.")
    return files[0]


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    dataset_path = find_single_npz(run_dir / "data")
    estimate_path = find_single_npz(run_dir / "estimates")
    output_dir = run_dir / args.output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(dataset_path, allow_pickle=True)
    estimate = np.load(estimate_path, allow_pickle=True)

    t = np.asarray(dataset["t"], dtype=float).reshape(-1)
    order = np.argsort(t)
    t_sorted = t[order]
    a_true = np.asarray(dataset["A_true"], dtype=float)
    a_iid = np.asarray(estimate["A_hat_iid"], dtype=float)
    a_final = np.asarray(estimate["A_hat_final"], dtype=float)

    if a_final.ndim != 3:
        raise ValueError("A_hat_final must have shape (n_subject, R, S).")

    configure_matplotlib()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, n_row, n_col = a_final.shape
    saved = 0
    for row in range(n_row):
        for col in range(n_col):
            fig, ax = plt.subplots(figsize=(6.0, 3.8), constrained_layout=True)
            ax.plot(t_sorted, a_true[order, row, col], color="black", linewidth=1.8, label="true")
            ax.plot(t_sorted, a_iid[order, row, col], color="#6B7280", linewidth=1.2, linestyle="--", label="stage 1")
            ax.plot(t_sorted, a_final[order, row, col], color="#C2410C", linewidth=1.5, label="final")
            ax.set_title(f"A[{row},{col}](t)")
            ax.set_xlabel("t")
            ax.set_ylabel("value")
            ax.grid(alpha=0.25, linewidth=0.6)
            ax.legend(frameon=False)

            path = output_dir / f"{args.stem}_r{row:02d}_s{col:02d}.png"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved += 1

    print(f"Saved {saved} A(t) plots to {output_dir}")


if __name__ == "__main__":
    main()
