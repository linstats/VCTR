"""B-spline helpers for VCTR estimators."""

from __future__ import annotations

import numpy as np


def make_open_uniform_knots(
    order: int,
    n_knots: int,
    *,
    domain: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Construct a clamped knot vector matching the MATLAB setup.

    Parameters
    ----------
    order:
        B-spline order. ``order=4`` gives cubic splines.
    n_knots:
        Number of equally spaced knot locations including the endpoints.
    domain:
        Boundary interval for the spline basis.
    """

    if order < 1:
        raise ValueError("order must be at least 1.")
    if n_knots < 2:
        raise ValueError("n_knots must be at least 2.")

    start, end = map(float, domain)
    if end <= start:
        raise ValueError("domain must satisfy end > start.")

    interior = np.linspace(start, end, n_knots)
    left = np.full(order - 1, start, dtype=float)
    right = np.full(order - 1, end, dtype=float)
    return np.concatenate([left, interior, right])


def bspline_basis_matrix(order: int, knots: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate a B-spline basis design matrix.

    This mirrors the MATLAB helper ``bspline_basismatrix`` used in the
    original simulation code.
    """

    knots = np.asarray(knots, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float).reshape(-1)

    if order < 1:
        raise ValueError("order must be at least 1.")
    if knots.ndim != 1 or knots.size <= order:
        raise ValueError("knots must contain more than `order` entries.")

    n_basis = knots.size - order
    basis = np.zeros((x.size, n_basis), dtype=float)
    for j in range(n_basis):
        basis[:, j] = _bspline_basis(j, order, knots, x, n_basis)
    return basis


def _bspline_basis(
    index: int,
    order: int,
    knots: np.ndarray,
    x: np.ndarray,
    n_basis: int,
) -> np.ndarray:
    """Recursive Cox-de Boor basis evaluation."""

    if order == 1:
        left = knots[index]
        right = knots[index + 1]
        mask = (x >= left) & (x < right)
        if index == n_basis - 1:
            mask = mask | np.isclose(x, knots[-1])
        return mask.astype(float)

    left_denom = knots[index + order - 1] - knots[index]
    right_denom = knots[index + order] - knots[index + 1]

    left_term = np.zeros_like(x, dtype=float)
    right_term = np.zeros_like(x, dtype=float)

    if left_denom > 0:
        left_term = ((x - knots[index]) / left_denom) * _bspline_basis(
            index,
            order - 1,
            knots,
            x,
            n_basis,
        )
    if right_denom > 0:
        right_term = ((knots[index + order] - x) / right_denom) * _bspline_basis(
            index + 1,
            order - 1,
            knots,
            x,
            n_basis,
        )
    return left_term + right_term

