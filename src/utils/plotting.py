"""Plotting helpers for paired-eye VCTR experiment diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np


def parse_a_indices(spec: str | None, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Parse a component-index spec for ``A(t)`` plots.

    The accepted syntax is ``all`` or comma-separated zero-based ``r:s`` pairs,
    for example ``0:0,1:4``.
    """

    n_row, n_col = shape
    if spec is None or spec.strip().lower() == "all":
        return [(row, col) for row in range(n_row) for col in range(n_col)]

    indices: list[tuple[int, int]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.replace(";", ":").split(":")
        if len(parts) != 2:
            raise ValueError("--plot-a-indices must use zero-based r:s pairs, e.g. 0:0,1:4, or all.")
        row, col = (int(part) for part in parts)
        if not (0 <= row < n_row and 0 <= col < n_col):
            raise ValueError(f"A index ({row}, {col}) is outside shape ({n_row}, {n_col}).")
        indices.append((row, col))

    if not indices:
        raise ValueError("--plot-a-indices did not contain any valid indices.")
    return indices


def save_function_plots(
    *,
    output_dir: Path,
    stem: str,
    dataset,
    result,
    a_indices: Iterable[tuple[int, int]] | None = None,
    max_a_panels: int | None = 16,
    include_initial: bool = True,
    include_true: bool = True,
) -> list[Path]:
    """Save diagnostic plots for selected ``A_{rs}(t)`` curves and ``sigma^2(t)``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib_cache()
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    saved_paths: list[Path] = []
    t = np.asarray(dataset.t, dtype=float).reshape(-1)
    order = np.argsort(t)
    t_sorted = t[order]

    a_hat = np.asarray(result.A_hat, dtype=float)
    if a_hat.ndim < 3:
        raise ValueError("result.A_hat must have shape (n_subject, R, S).")
    a_shape = (a_hat.shape[-2], a_hat.shape[-1])
    resolved_indices = list(a_indices) if a_indices is not None else parse_a_indices("all", a_shape)
    if max_a_panels is not None:
        if max_a_panels < 0:
            raise ValueError("max_a_panels must be nonnegative when provided.")
        resolved_indices = resolved_indices[:max_a_panels]

    if resolved_indices:
        saved_paths.append(
            _save_a_plot(
                plt=plt,
                output_dir=output_dir,
                stem=stem,
                t_sorted=t_sorted,
                order=order,
                dataset=dataset,
                result=result,
                a_indices=resolved_indices,
                include_initial=include_initial,
                include_true=include_true,
            )
        )

    saved_paths.append(
        _save_sigma2_plot(
            plt=plt,
            output_dir=output_dir,
            stem=stem,
            t_sorted=t_sorted,
            order=order,
            dataset=dataset,
            result=result,
            include_true=include_true,
        )
    )
    plt.close("all")
    return saved_paths


def _save_a_plot(
    *,
    plt,
    output_dir: Path,
    stem: str,
    t_sorted: np.ndarray,
    order: np.ndarray,
    dataset,
    result,
    a_indices: list[tuple[int, int]],
    include_initial: bool,
    include_true: bool,
) -> Path:
    n_panel = len(a_indices)
    n_col = min(4, n_panel)
    n_row = int(np.ceil(n_panel / n_col))
    fig, axes = plt.subplots(
        n_row,
        n_col,
        figsize=(4.2 * n_col, 3.0 * n_row),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.80, wspace=0.16, hspace=0.42)
    axes_flat = axes.ravel()
    a_hat = np.asarray(result.A_hat, dtype=float)
    a_initial = None if result.initial.A_hat is None else np.asarray(result.initial.A_hat, dtype=float)
    a_true = None if dataset.A_true is None else np.asarray(dataset.A_true, dtype=float)

    for ax, (row, col) in zip(axes_flat, a_indices, strict=False):
        if include_true and a_true is not None:
            ax.plot(t_sorted, a_true[order, row, col], color="black", linewidth=1.8, label="true")
        if include_initial and a_initial is not None:
            ax.plot(t_sorted, a_initial[order, row, col], color="#6B7280", linewidth=1.2, linestyle="--", label="independence")
        ax.plot(t_sorted, a_hat[order, row, col], color="#C2410C", linewidth=1.5, label="paired")
        ax.set_title(rf"$a_{{{row + 1},{col + 1}}}(t)$")
        ax.set_xlabel("t")
        ax.set_ylabel("value")
        ax.grid(alpha=0.25, linewidth=0.6)

    for ax in axes_flat[n_panel:]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=len(handles), frameon=False)
    fig.suptitle("Estimated and true coefficient functions", y=0.99)
    path = output_dir / f"{stem}_A_functions.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_sigma2_plot(
    *,
    plt,
    output_dir: Path,
    stem: str,
    t_sorted: np.ndarray,
    order: np.ndarray,
    dataset,
    result,
    include_true: bool,
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)
    sigma2_hat = np.asarray(result.covariance.sigma2_hat_t, dtype=float).reshape(-1)
    if include_true:
        sigma2_true = _infer_sigma2_true(dataset, target_shape=sigma2_hat.shape)
        if sigma2_true is not None:
            ax.plot(t_sorted, sigma2_true[order], color="black", linewidth=1.8, label="true")
    ax.plot(t_sorted, sigma2_hat[order], color="#0369A1", linewidth=1.6, label="estimated")
    ax.set_title(r"$\sigma^2(t)$")
    ax.set_xlabel("t")
    ax.set_ylabel("variance")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    path = output_dir / f"{stem}_sigma2_function.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _infer_sigma2_true(dataset, target_shape: tuple[int, ...]) -> np.ndarray | None:
    meta = getattr(dataset, "meta", {}) or {}
    for key in ("sigma2_true_t", "sigma2_t", "sigma2_true"):
        if key in meta:
            value = np.asarray(meta[key], dtype=float)
            if value.ndim == 0:
                return np.full(target_shape, float(value), dtype=float)
            return value.reshape(target_shape)
    if "sigma2" in meta:
        return np.full(target_shape, float(meta["sigma2"]), dtype=float)

    sigma_true = getattr(dataset, "Sigma_true", None)
    if sigma_true is None:
        return None
    sigma_true = np.asarray(sigma_true, dtype=float)
    if sigma_true.shape == (2, 2):
        return np.full(target_shape, float(np.mean(np.diag(sigma_true))), dtype=float)
    if sigma_true.ndim == 3 and sigma_true.shape[1:] == (2, 2):
        return np.mean(np.diagonal(sigma_true, axis1=1, axis2=2), axis=1).reshape(target_shape)
    return None


def _configure_matplotlib_cache() -> None:
    cache_root = Path(tempfile.gettempdir()) / "matplotlib-codex"
    cache_root.mkdir(parents=True, exist_ok=True)
    if "XDG_CACHE_HOME" not in os.environ:
        os.environ["XDG_CACHE_HOME"] = str(cache_root / "xdg-cache")
        Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = str(cache_root / "mplconfig")
        Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
