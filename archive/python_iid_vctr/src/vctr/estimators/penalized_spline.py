"""Penalized spline estimator for sparse VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseEstimator, EstimationResult, StructureResult
from .kruskal_init import KruskalRegInitializer
from ..utils.penalties import lqa_scalar_weight
from ..utils.splines import bspline_basis_matrix, make_open_uniform_knots


@dataclass(slots=True)
class PenalizedSplineVCREstimator(BaseEstimator):
    """Section 3 penalized spline estimator with structure identification.

    This first Python version follows the MATLAB Case III/IV logic at a
    practical level:

    - B-spline expansion for each reduced tensor feature.
    - Penalized beta update by local quadratic approximation.
    - Feature-wise penalized updates separating constant and varying parts.
    - Threshold-based classification into constant-zero, constant-nonzero,
      and varying coefficient groups.
    """

    order: int = 4
    n_knots: int = 6
    penalty: str = "SCAD"
    lambda_beta: float = 0.04
    lambda_const: float = 0.02
    lambda_vary: float = 0.07
    threshold: float = 1e-1
    ridge: float = 1e-4
    max_iter: int = 100
    tol: float = 1e-5
    init_rank: int = 2
    init_max_iter: int = 50
    init_replicates: int = 3
    init_seed: int = 0
    init_tol: float = 1e-5
    penalty_ridge: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit the penalized spline estimator and classify structure."""

        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        t = np.asarray(t, dtype=float).reshape(-1)

        n = t.shape[0]
        if X.shape[0] != n or Z.shape[0] != n or y.shape[0] != n:
            raise ValueError("X, Z, y, and t must have the same first dimension.")
        if X.ndim not in (2, 3):
            raise ValueError("X must have shape (n, p) or (n, R, S).")
        if Z.ndim != 2:
            raise ValueError("Z must have shape (n, p0).")
        if self.order < 1 or self.n_knots < 2:
            raise ValueError("Invalid spline configuration.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if self.tol <= 0:
            raise ValueError("tol must be positive.")

        x_shape = X.shape
        x_tensor, n_r, n_s = self._as_feature_tensor(X)
        n_features = n_r * n_s
        p0 = Z.shape[1]

        knots = make_open_uniform_knots(
            self.order,
            self.n_knots,
            domain=(float(np.min(t)), float(np.max(t))),
        )
        B = bspline_basis_matrix(self.order, knots, t)
        n_basis = B.shape[1]

        block_designs = self._build_block_designs(x_tensor, B)
        design_gamma = np.concatenate(block_designs, axis=1)
        gamma_old, beta_old = self._initialize_coefficients(
            x_shape,
            x_tensor,
            block_designs,
            design_gamma,
            Z,
            y,
            n_basis,
        )

        signal = self._signal_from_gamma(block_designs, gamma_old)

        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            beta_new = self._update_beta_given_gamma(
                Z=Z,
                y=y,
                beta_old=beta_old,
                gamma_old=gamma_old,
                block_designs=block_designs,
                n=n,
            )
            gamma_new = self._update_gamma_given_beta(
                Z=Z,
                y=y,
                beta_new=beta_new,
                gamma_old=gamma_old,
                block_designs=block_designs,
                n=n,
                n_basis=n_basis,
                n_r=n_r,
                n_s=n_s,
            )
            signal_current = self._signal_from_gamma(block_designs, gamma_new)

            dif = np.sqrt(np.mean((gamma_new - gamma_old) ** 2))
            gamma_old = gamma_new
            beta_old = beta_new
            signal = signal_current
            if dif <= self.tol:
                converged = True
                break

        gamma_hat = gamma_old
        beta_hat = beta_old
        A_hat = self._evaluate_A(B, gamma_hat)
        A_hat_flat = A_hat.reshape(A_hat.shape[0], -1)
        A_hat = self._restore_A_shape(A_hat_flat, x_shape)

        fitted_values = signal + Z @ beta_hat
        residuals = y - fitted_values
        structure = self._classify_structure(A_hat_flat, beta_hat, x_shape)

        return EstimationResult(
            A_hat=A_hat,
            beta_hat=beta_hat,
            fitted_values=fitted_values,
            residuals=residuals,
            structure=structure,
            meta={
                "n_samples": n,
                "x_shape": x_shape,
                "z_shape": Z.shape,
                "order": self.order,
                "n_knots": self.n_knots,
                "n_basis": n_basis,
                "penalty": self.penalty,
                "lambda_beta": self.lambda_beta,
                "lambda_const": self.lambda_const,
                "lambda_vary": self.lambda_vary,
                "threshold": self.threshold,
                "ridge": self.ridge,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "init_rank": self.init_rank,
                "init_max_iter": self.init_max_iter,
                "init_replicates": self.init_replicates,
                "init_seed": self.init_seed,
                "init_tol": self.init_tol,
                "penalty_ridge": self.penalty_ridge,
                "n_iter": n_iter,
                "converged": converged,
                "knots": knots,
                "basis_matrix": B,
                "gamma_hat": gamma_hat,
            },
        )

    def _initialize_coefficients(
        self,
        x_shape: tuple[int, ...],
        x_tensor: np.ndarray,
        block_designs: list[np.ndarray],
        design_gamma: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        n_basis: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Initialize coefficients by MATLAB-style Kruskal regression logic.

        The original MATLAB code uses ``kruskal_reg(..., r=2, 'normal')`` on
        the spline-expanded tensor design. Here we mimic that role by:

        1. fitting a full ridge coefficient tensor,
        2. projecting that tensor onto a rank-``init_rank`` CP model, and
        3. refitting ``beta`` conditional on the CP-approximated tensor.
        """

        full_design = np.concatenate([design_gamma, Z], axis=1)
        lhs = full_design.T @ full_design + self.ridge * np.eye(full_design.shape[1])
        rhs = full_design.T @ y
        coef = np.linalg.solve(lhs, rhs)

        n_r = x_tensor.shape[1]
        n_s = x_tensor.shape[2]
        gamma_full = coef[: n_basis * n_r * n_s].reshape(n_basis, n_r, n_s, order="F")
        beta_ridge = coef[n_basis * n_r * n_s :]

        initializer = KruskalRegInitializer(
            rank=self.init_rank,
            ridge=self.ridge,
            max_iter=self.init_max_iter,
            replicates=self.init_replicates,
            seed=self.init_seed,
            tol=self.init_tol,
        )
        init_result = initializer.initialize(
            block_designs=block_designs,
            Z=Z,
            y=y,
            gamma_full=gamma_full,
            beta_init=beta_ridge,
        )
        return init_result.gamma_hat, init_result.beta_hat

    def _build_block_designs(self, x_tensor: np.ndarray, basis: np.ndarray) -> list[np.ndarray]:
        """Return MATLAB-style block designs, one matrix per ``s`` slice."""

        n = x_tensor.shape[0]
        n_r = x_tensor.shape[1]
        block_designs: list[np.ndarray] = []
        for s_idx in range(x_tensor.shape[2]):
            parts = [basis * x_tensor[:, [r_idx], s_idx] for r_idx in range(n_r)]
            block_designs.append(np.concatenate(parts, axis=1).reshape(n, -1))
        return block_designs

    def _gamma_penalty_matrix_block(
        self,
        gamma_block: np.ndarray,
        n_basis: int,
        n_r: int,
    ) -> np.ndarray:
        """Construct MATLAB-style penalty matrices for one ``s`` block."""

        omega_const = np.zeros((n_basis * n_r, n_basis * n_r), dtype=float)
        omega_vary = np.zeros((n_basis * n_r, n_basis * n_r), dtype=float)
        ones = np.ones((n_basis, n_basis), dtype=float) / (n_basis * n_basis)
        center = np.eye(n_basis, dtype=float) - np.ones((n_basis, n_basis), dtype=float) / n_basis

        for r_idx in range(n_r):
            gamma_rs = gamma_block[:, r_idx]
            gamma_mean = float(np.mean(gamma_rs))
            gamma_dev = gamma_rs - gamma_mean

            weight_const = float(
                lqa_scalar_weight(abs(gamma_mean), self.lambda_const, self.penalty)
            )
            weight_vary = float(
                lqa_scalar_weight(np.linalg.norm(gamma_dev), self.lambda_vary, self.penalty)
            )
            block = slice(r_idx * n_basis, (r_idx + 1) * n_basis)
            omega_const[block, block] = weight_const * ones
            omega_vary[block, block] = weight_vary * center

        return omega_const + omega_vary

    def _update_beta_given_gamma(
        self,
        *,
        Z: np.ndarray,
        y: np.ndarray,
        beta_old: np.ndarray,
        gamma_old: np.ndarray,
        block_designs: list[np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Literal MATLAB-style beta update given the current gamma."""

        y_tilde = y - self._signal_from_gamma(block_designs, gamma_old)
        omega_beta = np.diag(
            lqa_scalar_weight(beta_old, self.lambda_beta, self.penalty)
        )
        left = Z.T @ Z + (n / 2.0) * omega_beta + self.penalty_ridge * np.eye(Z.shape[1])
        right = Z.T @ y_tilde
        return np.linalg.solve(left, right)

    def _update_gamma_given_beta(
        self,
        *,
        Z: np.ndarray,
        y: np.ndarray,
        beta_new: np.ndarray,
        gamma_old: np.ndarray,
        block_designs: list[np.ndarray],
        n: int,
        n_basis: int,
        n_r: int,
        n_s: int,
    ) -> np.ndarray:
        """Literal MATLAB-style blockwise gamma update given beta."""

        gamma_new = gamma_old.copy()
        z_beta = Z @ beta_new

        for s_idx in range(n_s):
            gamma_block_old = gamma_old[:, :, s_idx]
            block_signal_old = block_designs[s_idx] @ gamma_block_old.reshape(-1, order="F")
            y_tilde = (
                y
                - self._signal_from_gamma(block_designs, gamma_new)
                - z_beta
                + block_signal_old
            )

            omega_block = self._gamma_penalty_matrix_block(
                gamma_block_old,
                n_basis,
                n_r,
            )
            left = (
                block_designs[s_idx].T @ block_designs[s_idx]
                + (n / 2.0) * omega_block
                + self.penalty_ridge * np.eye(n_basis * n_r)
            )
            right = block_designs[s_idx].T @ y_tilde
            gamma_block_new = np.linalg.solve(left, right).reshape(
                n_basis,
                n_r,
                order="F",
            )
            gamma_new[:, :, s_idx] = gamma_block_new

        return gamma_new

    def _classify_structure(
        self,
        A_hat_flat: np.ndarray,
        beta_hat: np.ndarray,
        x_shape: tuple[int, ...],
    ) -> StructureResult:
        """Classify coefficient functions into varying / constant / zero."""

        mean_abs = np.abs(np.mean(A_hat_flat, axis=0))
        vary_strength = np.sqrt(np.mean((A_hat_flat - np.mean(A_hat_flat, axis=0)) ** 2, axis=0))

        varying_mask_flat = vary_strength >= self.threshold
        const_zero_mask_flat = (vary_strength < self.threshold) & (mean_abs < self.threshold)
        const_nonzero_mask_flat = (vary_strength < self.threshold) & (mean_abs >= self.threshold)
        beta_nonzero_mask = np.abs(beta_hat) > self.threshold

        target_shape = (x_shape[1],) if len(x_shape) == 2 else x_shape[1:]
        varying_mask = varying_mask_flat.reshape(target_shape)
        const_zero_mask = const_zero_mask_flat.reshape(target_shape)
        const_nonzero_mask = const_nonzero_mask_flat.reshape(target_shape)

        return StructureResult(
            varying_mask=varying_mask,
            const_nonzero_mask=const_nonzero_mask,
            const_zero_mask=const_zero_mask,
            beta_nonzero_mask=beta_nonzero_mask,
            meta={
                "threshold": self.threshold,
                "n_varying": int(np.sum(varying_mask_flat)),
                "n_const_nonzero": int(np.sum(const_nonzero_mask_flat)),
                "n_const_zero": int(np.sum(const_zero_mask_flat)),
                "n_beta_nonzero": int(np.sum(beta_nonzero_mask)),
                "mean_abs": mean_abs.reshape(target_shape),
                "vary_strength": vary_strength.reshape(target_shape),
            },
        )

    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        """Flatten reduced tensor features to shape ``(n, p)``."""

        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _restore_A_shape(self, A_hat_flat: np.ndarray, x_shape: tuple[int, ...]) -> np.ndarray:
        """Restore ``A_hat`` to match the reduced feature shape."""

        if len(x_shape) == 2:
            return A_hat_flat
        return A_hat_flat.reshape((A_hat_flat.shape[0],) + x_shape[1:])

    def _evaluate_A(self, basis: np.ndarray, gamma_hat: np.ndarray) -> np.ndarray:
        """Evaluate spline coefficients on the observed time grid."""

        n = basis.shape[0]
        n_r = gamma_hat.shape[1]
        n_s = gamma_hat.shape[2]
        A_hat = np.zeros((n, n_r, n_s), dtype=float)
        for s_idx in range(n_s):
            A_hat[:, :, s_idx] = basis @ gamma_hat[:, :, s_idx]
        return A_hat

    def _signal_from_gamma(
        self,
        block_designs: list[np.ndarray],
        gamma_hat: np.ndarray,
    ) -> np.ndarray:
        """Compute fitted tensor signal for a gamma tensor."""

        signal = np.zeros(block_designs[0].shape[0], dtype=float)
        for s_idx, design_block in enumerate(block_designs):
            signal += design_block @ gamma_hat[:, :, s_idx].reshape(-1, order="F")
        return signal

    def _as_feature_tensor(self, X: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Return a tensor view ``(n, R, S)`` for reduced features."""

        if X.ndim == 3:
            return X, X.shape[1], X.shape[2]
        return X[:, :, None], X.shape[1], 1
