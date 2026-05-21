"""Kernel-based refinement for sparse VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.dataset import SimulationDataset
from src.vctr.estimators.base import EstimationResult, StructureResult
from src.vctr.estimators.local_linear import LocalLinearVCREstimator


@dataclass(slots=True)
class KernelRefitter:
    """Refine a penalized structure estimate by local linear smoothing."""

    bandwidth: float
    ridge: float = 1e-4
    kernel: str = "epanechnikov"
    eval_mode: str = "matlab_middle_random"
    eval_fraction: float = 0.1
    central_fraction: float = 0.7
    eval_seed: int | None = 0

    def refit(
        self,
        dataset: SimulationDataset,
        result: EstimationResult,
        **kwargs: Any,
    ) -> EstimationResult:
        """Refit a sparse VCTR model using identified structure."""

        if result.structure is None or not isinstance(result.structure, StructureResult):
            raise ValueError("result.structure must be a StructureResult.")

        X = np.asarray(dataset.X, dtype=float)
        Z = np.asarray(dataset.Z, dtype=float)
        y = np.asarray(dataset.y, dtype=float).reshape(-1)
        t = np.asarray(dataset.t, dtype=float).reshape(-1)
        x_shape = X.shape

        x_mat = self._flatten_X(X)
        structure = result.structure

        varying_mask_flat = np.asarray(structure.varying_mask, dtype=bool).reshape(-1)
        const_nonzero_mask_flat = np.asarray(structure.const_nonzero_mask, dtype=bool).reshape(-1)
        beta_nonzero_mask = np.asarray(structure.beta_nonzero_mask, dtype=bool).reshape(-1)

        varying_ids = np.flatnonzero(varying_mask_flat)
        const_ids = np.flatnonzero(const_nonzero_mask_flat)
        beta_ids = np.flatnonzero(beta_nonzero_mask)

        X_vary = x_mat[:, varying_ids] if varying_ids.size else np.zeros((X.shape[0], 0))
        X_const = x_mat[:, const_ids] if const_ids.size else np.zeros((X.shape[0], 0))
        Z_nonzero = Z[:, beta_ids] if beta_ids.size else np.zeros((X.shape[0], 0))
        Z_refit = np.concatenate([X_const, Z_nonzero], axis=1)

        local = LocalLinearVCREstimator(
            bandwidth=self.bandwidth,
            ridge=self.ridge,
            kernel=self.kernel,
            eval_mode=self.eval_mode,
            eval_fraction=self.eval_fraction,
            central_fraction=self.central_fraction,
            eval_seed=self.eval_seed,
        )
        local_result = local.fit(X_vary, Z_refit, y, t)

        eval_indices = np.asarray(local_result.meta["eval_indices"])
        n_eval = eval_indices.shape[0]

        A_hat_full = np.zeros((n_eval, x_mat.shape[1]), dtype=float)
        if varying_ids.size:
            A_hat_full[:, varying_ids] = np.asarray(local_result.A_hat).reshape(n_eval, -1)

        linear_coef = np.asarray(local_result.beta_hat, dtype=float)
        const_coef = linear_coef[: const_ids.size] if const_ids.size else np.zeros(0)
        beta_nonzero_hat = linear_coef[const_ids.size :] if beta_ids.size else np.zeros(0)

        if const_ids.size:
            A_hat_full[:, const_ids] = np.repeat(const_coef.reshape(1, -1), n_eval, axis=0)

        beta_hat = np.zeros(Z.shape[1], dtype=float)
        if beta_ids.size:
            beta_hat[beta_ids] = beta_nonzero_hat

        X_eval = x_mat[eval_indices]
        Z_eval = Z[eval_indices]
        fitted_values = np.sum(X_eval * A_hat_full, axis=1) + Z_eval @ beta_hat
        residuals = y[eval_indices] - fitted_values

        A_hat = self._restore_A_shape(A_hat_full, x_shape)
        return EstimationResult(
            A_hat=A_hat,
            beta_hat=beta_hat,
            fitted_values=fitted_values,
            residuals=residuals,
            structure=structure,
            meta={
                "bandwidth": self.bandwidth,
                "ridge": self.ridge,
                "kernel": self.kernel,
                "eval_mode": self.eval_mode,
                "eval_fraction": self.eval_fraction,
                "central_fraction": self.central_fraction,
                "eval_seed": self.eval_seed,
                "eval_indices": eval_indices,
                "t_eval": local_result.meta["t_eval"],
                "varying_ids": varying_ids,
                "const_ids": const_ids,
                "beta_ids": beta_ids,
                "local_result": local_result,
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
