"""Variance-function helpers for paired-eye DGPs."""

from __future__ import annotations

import numpy as np


SUPPORTED_SIGMA2_FUNCTIONS = ("constant", "sin", "sin2", "mixed")


def sigma2_curve(t: np.ndarray, *, base: float = 1.0, kind: str = "constant") -> np.ndarray:
    """Return a strictly positive variance curve evaluated at ``t``."""

    if base <= 0:
        raise ValueError("base must be positive.")

    t = np.asarray(t, dtype=float)
    if kind == "constant":
        curve = np.full(t.shape, float(base), dtype=float)
    elif kind == "sin":
        curve = float(base) * (1.0 + 0.3 * np.sin(2.0 * np.pi * t))
    elif kind == "sin2":
        curve = float(base) * (0.5 + 0.5 * np.square(np.sin(np.pi * t)))
    elif kind == "mixed":
        curve = float(base) * (
            1.0 + 0.25 * np.cos(2.0 * np.pi * t) + 0.1 * np.sin(4.0 * np.pi * t)
        )
    else:
        raise ValueError(
            "kind must be one of "
            f"{', '.join(repr(value) for value in SUPPORTED_SIGMA2_FUNCTIONS)}."
        )

    if np.any(~np.isfinite(curve)) or np.any(curve <= 0):
        raise ValueError("sigma2 curve must be finite and strictly positive.")
    return curve
