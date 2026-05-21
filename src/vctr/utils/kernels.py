"""Kernel helpers for VCTR estimators."""

from __future__ import annotations

import numpy as np


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Evaluate the Epanechnikov kernel at ``u``."""

    return np.maximum(0.75 * (1.0 - np.square(u)), 0.0)


def kernel_sqrt_weights(
    t: np.ndarray,
    t0: float,
    bandwidth: float,
    *,
    kernel: str = "epanechnikov",
) -> np.ndarray:
    """Return square-root kernel weights centered at ``t0``."""

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")
    if kernel != "epanechnikov":
        raise NotImplementedError("Only the Epanechnikov kernel is implemented.")

    scaled = (np.asarray(t, dtype=float).reshape(-1) - float(t0)) / bandwidth
    ker = epanechnikov_kernel(scaled)
    return np.sqrt(ker / bandwidth)
