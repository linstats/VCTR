"""Penalty helpers for sparse VCTR estimators."""

from __future__ import annotations

import numpy as np


def penalty_derivative(value: np.ndarray | float, lam: float, penalty: str) -> np.ndarray:
    """Return the derivative magnitude of a penalty at ``|value|``."""

    penalty_key = penalty.strip().lower()
    abs_value = np.abs(np.asarray(value, dtype=float))

    if lam < 0:
        raise ValueError("lam must be nonnegative.")

    if penalty_key == "lasso":
        return np.full_like(abs_value, lam, dtype=float)

    if penalty_key == "scad":
        a = 3.7
        out = np.empty_like(abs_value, dtype=float)
        mask1 = abs_value <= lam
        mask2 = (abs_value > lam) & (abs_value <= a * lam)
        mask3 = abs_value > a * lam
        out[mask1] = lam
        out[mask2] = (a * lam - abs_value[mask2]) / (a - 1.0)
        out[mask3] = 0.0
        return out

    if penalty_key == "mcp":
        a = 3.0
        out = np.maximum(lam - abs_value / a, 0.0)
        return out

    raise ValueError(f"Unsupported penalty type: {penalty!r}")


def lqa_scalar_weight(
    value: np.ndarray | float,
    lam: float,
    penalty: str,
    *,
    eps: float = 1e-10,
) -> np.ndarray:
    """Return the local quadratic approximation weight ``p'(|x|)/|x|``."""

    abs_value = np.abs(np.asarray(value, dtype=float))
    deriv = penalty_derivative(abs_value, lam, penalty)
    return deriv / np.maximum(abs_value, eps)

