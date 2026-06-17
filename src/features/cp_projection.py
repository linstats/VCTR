"""CP projection helpers for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class CPProjectionConfig:
    """Configuration for blockwise CP projection features."""

    rank: int
    max_iter: int = 50
    tol: float = 1e-5
    random_state: int = 0
    dtype: str = "float64"
    standardize_sample_factors: bool = True

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if self.tol < 0:
            raise ValueError("tol must be non-negative.")


@dataclass(slots=True)
class CPBlockResult:
    """CP decomposition result for one tensor block."""

    sample_scores: np.ndarray
    lambdas: np.ndarray
    factors: tuple[np.ndarray, ...]
    n_iter: int
    relative_change: float


@dataclass(slots=True)
class BlockwiseCPResult:
    """Blockwise CP projection output."""

    X_star_flat: np.ndarray
    blocks: tuple[CPBlockResult, ...]


def _mttkrp(X: np.ndarray, factors: list[np.ndarray], mode: int) -> np.ndarray:
    """Matricized tensor times Khatri-Rao product via optimized einsum."""

    ndim = X.ndim
    if ndim > 25:
        raise ValueError("CP helper supports tensors with at most 25 modes.")
    axis_labels = "abcdefghijklmnopqrstuvwxy"[:ndim]
    rank_label = "z"
    tensor_subscript = axis_labels
    factor_subscripts = [
        f"{axis_labels[axis]}{rank_label}" for axis in range(ndim) if axis != mode
    ]
    output_subscript = f"{axis_labels[mode]}{rank_label}"
    equation = f"{tensor_subscript},{','.join(factor_subscripts)}->{output_subscript}"
    operands = [X, *[factors[axis] for axis in range(ndim) if axis != mode]]
    return np.einsum(equation, *operands, optimize=True)


def _normalize_non_sample_modes(factors: list[np.ndarray]) -> None:
    """Keep scale in the sample mode while stabilizing spatial factors."""

    for mode in range(1, len(factors)):
        norms = np.linalg.norm(factors[mode], axis=0)
        norms[norms == 0] = 1.0
        factors[mode] = factors[mode] / norms
        factors[0] = factors[0] * norms


def _finalize_factors(factors: list[np.ndarray]) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    lambdas = np.ones(factors[0].shape[1], dtype=factors[0].dtype)
    normalized: list[np.ndarray] = []
    for factor in factors:
        norms = np.linalg.norm(factor, axis=0)
        norms[norms == 0] = 1.0
        lambdas *= norms
        normalized.append(factor / norms)
    return lambdas, tuple(normalized)


def cp_als(block: np.ndarray, config: CPProjectionConfig, *, seed: int) -> CPBlockResult:
    """Compute a small CP-ALS decomposition for one block.

    The implementation is intentionally dependency-free because the project
    does not currently include a Python tensor decomposition package.
    """

    X = np.asarray(block, dtype=np.dtype(config.dtype))
    if X.ndim < 2:
        raise ValueError("block must have at least a sample axis and one tensor axis.")
    if X.shape[0] <= 1:
        raise ValueError("block must contain at least two samples.")

    rng = np.random.default_rng(seed)
    factors = [rng.standard_normal((size, config.rank)).astype(X.dtype, copy=False) for size in X.shape]
    _normalize_non_sample_modes(factors)

    previous_sample = factors[0].copy()
    relative_change = np.inf
    n_iter = 0
    ridge = np.finfo(X.dtype).eps

    for iteration in range(1, config.max_iter + 1):
        for mode in range(X.ndim):
            gram = np.ones((config.rank, config.rank), dtype=X.dtype)
            for other_mode, factor in enumerate(factors):
                if other_mode != mode:
                    gram *= factor.T @ factor
            gram.flat[:: config.rank + 1] += ridge
            mttkrp = _mttkrp(X, factors, mode)
            factors[mode] = mttkrp @ np.linalg.pinv(gram)

        _normalize_non_sample_modes(factors)
        denom = np.linalg.norm(previous_sample)
        if denom == 0:
            denom = 1.0
        relative_change = float(np.linalg.norm(factors[0] - previous_sample) / denom)
        previous_sample = factors[0].copy()
        n_iter = iteration
        if relative_change <= config.tol:
            break

    lambdas, normalized_factors = _finalize_factors(factors)
    sample_scores = normalized_factors[0].copy()
    if config.standardize_sample_factors:
        sample_sd = sample_scores.std(axis=0, ddof=0)
        sample_sd[sample_sd == 0] = 1.0
        sample_scores = sample_scores / sample_sd

    return CPBlockResult(
        sample_scores=sample_scores,
        lambdas=lambdas,
        factors=normalized_factors,
        n_iter=n_iter,
        relative_change=relative_change,
    )


def blockwise_cp_project(blocks: Sequence[np.ndarray], config: CPProjectionConfig) -> BlockwiseCPResult:
    """Project partitioned tensor blocks into reduced CP sample scores.

    Returns ``X_star_flat`` with shape ``(n_samples, rank, n_blocks)``.
    """

    if not blocks:
        raise ValueError("blocks must not be empty.")
    n_samples = int(blocks[0].shape[0])
    if any(block.shape[0] != n_samples for block in blocks):
        raise ValueError("All blocks must have the same sample-axis length.")

    X_star = np.empty((n_samples, config.rank, len(blocks)), dtype=np.dtype(config.dtype))
    results: list[CPBlockResult] = []
    for block_idx, block in enumerate(blocks):
        result = cp_als(block, config, seed=config.random_state + block_idx)
        X_star[:, :, block_idx] = result.sample_scores
        results.append(result)

    return BlockwiseCPResult(X_star_flat=X_star, blocks=tuple(results))
