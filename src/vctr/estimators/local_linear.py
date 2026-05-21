"""Local linear estimator for VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BaseEstimator, EstimationResult
from ..utils.kernels import kernel_sqrt_weights


@dataclass(slots=True)
class LocalLinearVCREstimator(BaseEstimator):
    """Section 2 local linear kernel smoothing estimator.

    This class is intentionally minimal at the current stage. It establishes
    the public API and stores configuration needed for the later numerical
    implementation.
    """

    bandwidth: float
    ridge: float = 1e-4
    kernel: str = "epanechnikov"
    eval_mode: str = "matlab_middle_random"
    eval_fraction: float = 0.2
    central_fraction: float = 0.7
    eval_seed: int | None = 0

    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        *,
        t_eval: np.ndarray | None = None,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit the local linear estimator.

        Parameters
        ----------
        X:
            Reduced tensor features with shape ``(n, R, S)`` or a flattened
            equivalent.
        Z:
            Scalar covariates with shape ``(n, p0)``.
        y:
            Response vector with shape ``(n,)`` or ``(n, 1)``.
        t:
            Index variable with shape ``(n,)``.
        t_eval:
            Optional evaluation points for ``A(t)``. Defaults to the observed
            index values ``t``.
        """

        X = np.asarray(X)
        Z = np.asarray(Z)
        y = np.asarray(y).reshape(-1)
        t = np.asarray(t).reshape(-1)

        n = t.shape[0]
        if X.shape[0] != n or Z.shape[0] != n or y.shape[0] != n:
            raise ValueError("X, Z, y, and t must have the same first dimension.")
        if X.ndim not in (2, 3):
            raise ValueError("X must have shape (n, p) or (n, R, S).")
        if Z.ndim != 2:
            raise ValueError("Z must have shape (n, p0).")
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.kernel != "epanechnikov":
            raise NotImplementedError("Only the Epanechnikov kernel is planned now.")

        eval_indices, t_eval_arr = self._resolve_eval_points(t, t_eval)
        x_shape = X.shape
        x_mat = self._flatten_X(X)
        n_features = x_mat.shape[1]
        p0 = Z.shape[1]

        A_hat_flat = np.zeros((t_eval_arr.shape[0], n_features), dtype=float)
        beta_local = np.zeros((t_eval_arr.shape[0], p0), dtype=float)

        for i, t0 in enumerate(t_eval_arr):
            ker = self._kernel_weights(t, t0)
            sst = (t - t0) / self.bandwidth

            y_star = y * ker
            z_star = Z * ker[:, None]
            x_mat_a = x_mat * ker[:, None]
            x_mat_b = (x_mat * sst[:, None]) * ker[:, None]

            xz_star = np.concatenate([x_mat_a, x_mat_b, z_star], axis=1)
            para_hat = self._solve_normal_equation(xz_star, y_star)

            A_hat_flat[i] = para_hat[:n_features]
            beta_local[i] = para_hat[2 * n_features :]

        X_eval = self._select_eval_rows(X, eval_indices)
        Z_eval = self._select_eval_rows(Z, eval_indices)
        y_eval = self._select_eval_rows(y, eval_indices)
        x_eval_mat = self._flatten_X(X_eval)

        signal_hat = np.sum(x_eval_mat * A_hat_flat, axis=1)
        beta_hat = self._estimate_beta(Z_eval, y_eval - signal_hat)
        fitted_values = signal_hat + Z_eval @ beta_hat
        residuals = y_eval - fitted_values
        A_hat = self._restore_A_shape(A_hat_flat, x_shape)

        return EstimationResult(
            A_hat=A_hat,
            beta_hat=beta_hat,
            fitted_values=fitted_values,
            residuals=residuals,
            meta={
                "n_samples": n,
                "x_shape": x_shape,
                "z_shape": Z.shape,
                "bandwidth": self.bandwidth,
                "ridge": self.ridge,
                "kernel": self.kernel,
                "eval_mode": self.eval_mode,
                "eval_fraction": self.eval_fraction,
                "central_fraction": self.central_fraction,
                "eval_seed": self.eval_seed,
                "n_eval": t_eval_arr.shape[0],
                "eval_indices": eval_indices,
                "t_eval": t_eval_arr,
                "beta_local": beta_local,
            },
        )

    def _kernel_weights(self, t: np.ndarray, t0: float) -> np.ndarray:
        """Return square-root kernel weights centered at ``t0``."""

        return kernel_sqrt_weights(t, t0, self.bandwidth, kernel=self.kernel)

    def _solve_normal_equation(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Solve the ridge-stabilized normal equation."""

        lhs = design.T @ design
        lhs = lhs + self.ridge * np.eye(lhs.shape[0])
        rhs = design.T @ target
        return np.linalg.solve(lhs, rhs)

    def _estimate_beta(self, Z: np.ndarray, y_resid: np.ndarray) -> np.ndarray:
        """Estimate the scalar coefficient vector after smoothing A(t)."""

        lhs = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        rhs = Z.T @ y_resid
        return np.linalg.solve(lhs, rhs)

    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        """Flatten reduced tensor features to shape ``(n, p)``."""

        X = np.asarray(X)
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _restore_A_shape(self, A_hat_flat: np.ndarray, x_shape: tuple[int, ...]) -> np.ndarray:
        """Restore ``A_hat`` to match the reduced feature shape."""

        if len(x_shape) == 2:
            return A_hat_flat
        return A_hat_flat.reshape((A_hat_flat.shape[0],) + x_shape[1:])

    def _resolve_eval_points(
        self,
        t: np.ndarray,
        t_eval: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return evaluation row indices and corresponding time points."""

        t = np.asarray(t).reshape(-1)
        n = t.shape[0]

        if t_eval is not None:
            t_eval_arr = np.asarray(t_eval).reshape(-1)
            if t_eval_arr.shape[0] != n:
                raise NotImplementedError(
                    "Custom t_eval is only supported when it matches observed t."
                )
            return np.arange(n), t_eval_arr

        if self.eval_mode == "all_points":
            return np.arange(n), t

        if self.eval_mode == "middle_subset":
            return self._middle_subset_indices(t)

        if self.eval_mode == "matlab_middle_random":
            return self._matlab_middle_random_indices(t)

        raise ValueError(
            "eval_mode must be 'matlab_middle_random', 'middle_subset', or 'all_points'."
        )

    def _middle_subset_indices(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic middle-subset evaluation design."""

        n = t.shape[0]
        if not 0 < self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be in (0, 1].")
        if not 0 < self.central_fraction <= 1:
            raise ValueError("central_fraction must be in (0, 1].")

        n_eval = max(1, int(round(self.eval_fraction * n)))
        start = int(np.floor((1.0 - self.central_fraction) * n / 2.0))
        end = n - start
        if end <= start:
            raise ValueError("central_fraction leaves no interior evaluation region.")

        middle_indices = np.arange(start, end)
        if n_eval >= middle_indices.shape[0]:
            eval_indices = middle_indices
        else:
            eval_indices = np.linspace(
                middle_indices[0],
                middle_indices[-1],
                num=n_eval,
                dtype=int,
            )
            eval_indices = np.unique(eval_indices)

        return eval_indices, t[eval_indices]

    def _matlab_middle_random_indices(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return MATLAB-style random middle-region evaluation indices.

        This mirrors the MATLAB pattern

        ``id_est = sort(n*0.15 + randperm(n*0.7, n_est))'``

        under the current defaults ``eval_fraction=0.2`` and
        ``central_fraction=0.7``.
        """

        n = t.shape[0]
        if not 0 < self.eval_fraction <= 1:
            raise ValueError("eval_fraction must be in (0, 1].")
        if not 0 < self.central_fraction <= 1:
            raise ValueError("central_fraction must be in (0, 1].")

        n_eval = max(1, int(self.eval_fraction * n))
        middle_size = max(1, int(self.central_fraction * n))
        start = int((1.0 - self.central_fraction) * n / 2.0)
        end = start + middle_size
        if end > n:
            end = n
            middle_size = end - start
        if middle_size <= 0:
            raise ValueError("central_fraction leaves no interior evaluation region.")

        rng = np.random.default_rng(self.eval_seed)
        if n_eval >= middle_size:
            eval_indices = np.arange(start, end)
        else:
            sampled = rng.choice(middle_size, size=n_eval, replace=False)
            eval_indices = np.sort(start + sampled)

        return eval_indices, t[eval_indices]

    def _select_eval_rows(self, array: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Select rows aligned with evaluation indices."""

        return np.asarray(array)[indices]
