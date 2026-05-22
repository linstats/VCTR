"""Paired-eye VCTR model skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from src.data import PairedEyeDataset
from src.utils.kernels import kernel_sqrt_weights

from .base import BasePairedVCTRModel, CovarianceEstimate, InitialIidResult, PairedVCTRResult
from .covariance import estimate_exchangeable_covariance


@dataclass(slots=True)
class PairedEyeVCTRModel(BasePairedVCTRModel):
    """Skeleton estimator for the paired-eye VCTR target model.

    The intended model is

    ``y_{ij} = <X_{ij}, A(t_i)> + z_i^T beta + epsilon_{ij}``

    with within-subject dependence modeled through the covariance of
    ``(epsilon_{i1}, epsilon_{i2})^T``.
    """

    bandwidth: float | None = None
    bandwidth_method: str = "stage1_kfold_cv"
    bandwidth_grid: tuple[float, ...] | None = None
    bandwidth_cv_folds: int = 5
    bandwidth_cv_seed: int = 0
    ridge: float = 0.0
    spline_order: int = 4
    n_knots: int = 6
    penalty: str = "scad"

    DEFAULT_BANDWIDTH: ClassVar[float] = 0.13
    DEFAULT_BANDWIDTH_GRID: ClassVar[tuple[float, ...]] = (0.08, 0.10, 0.13, 0.16, 0.20)

    def initial_fit_iid(self, dataset: PairedEyeDataset) -> InitialIidResult:
        """Fit an initial iid working model on the flattened observation view."""

        selected_bandwidth, bandwidth_meta = self._resolve_bandwidth(dataset)
        result = self._fit_initial_iid_with_bandwidth(dataset, selected_bandwidth)
        result.meta.update(bandwidth_meta)
        return result

    def _fit_initial_iid_with_bandwidth(
        self,
        dataset: PairedEyeDataset,
        bandwidth: float,
    ) -> InitialIidResult:
        """Fit the stage-1 iid working model at a fixed bandwidth."""

        flat = dataset.to_iid_observations()
        x_mat = self._flatten_X(flat.X)
        n_subject = dataset.n_subject
        n_features = x_mat.shape[1]
        p0 = flat.Z.shape[1]
        x_subject_eye = dataset.X.reshape(n_subject, 2, n_features)

        A_hat_flat, beta_local = self._estimate_stage1_A(
            flat_Z=flat.Z,
            flat_X=x_mat,
            flat_y=flat.y,
            flat_t=flat.t,
            t_eval=dataset.t,
            p0=p0,
            bandwidth=bandwidth,
        )

        signal_hat = np.sum(x_subject_eye * A_hat_flat[:, None, :], axis=2)
        y_dagger = dataset.y - signal_hat
        beta_hat = self._solve_beta_ols(dataset.Z, y_dagger)
        fitted_values = signal_hat + dataset.Z @ beta_hat[:, None]
        residuals = dataset.y - fitted_values

        return InitialIidResult(
            A_hat=A_hat_flat.reshape((n_subject,) + dataset.X.shape[2:]),
            beta_hat=beta_hat,
            fitted_values=fitted_values.reshape(-1),
            residuals=residuals.reshape(-1),
            subject_ids=np.repeat(dataset.subject_ids, 2),
            eye_ids=np.tile(dataset.eye_ids, n_subject),
            meta={
                "bandwidth": bandwidth,
                "ridge": self.ridge,
                "t_eval": dataset.t.copy(),
                "beta_local": beta_local,
                "signal_hat": signal_hat,
                "y_dagger": y_dagger,
            },
        )

    def estimate_covariance(
        self,
        dataset: PairedEyeDataset,
        initial_result: InitialIidResult,
    ) -> CovarianceEstimate:
        """Estimate a subject-common covariance matrix from iid residuals."""

        _ = dataset
        return estimate_exchangeable_covariance(initial_result)

    def refit_with_covariance(
        self,
        dataset: PairedEyeDataset,
        covariance: CovarianceEstimate,
        initial_result: InitialIidResult | None = None,
    ) -> PairedVCTRResult:
        """Refit the paired-eye model using the estimated working covariance."""

        if initial_result is None:
            raise ValueError("initial_result is required for covariance-aware refitting.")

        n_subject = dataset.n_subject
        x_mat = dataset.X.reshape(n_subject, 2, -1)
        n_features = x_mat.shape[2]
        p0 = dataset.Z.shape[1]
        Sigma_inv = np.linalg.inv(covariance.Sigma_hat)
        selected_bandwidth = self._selected_bandwidth_from_initial(initial_result)

        A_hat_flat = np.zeros((n_subject, n_features), dtype=float)
        beta_local = np.zeros((n_subject, p0), dtype=float)

        for i, t0 in enumerate(dataset.t):
            lhs = np.zeros((p0 + 2 * n_features, p0 + 2 * n_features), dtype=float)
            rhs = np.zeros(p0 + 2 * n_features, dtype=float)
            for subj in range(n_subject):
                kh = self._kernel_scalar_weight(dataset.t[subj], t0, selected_bandwidth)
                if kh <= 0:
                    continue
                sst = (dataset.t[subj] - t0) / selected_bandwidth
                Vi = np.zeros((2, p0 + 2 * n_features), dtype=float)
                Vi[:, :p0] = dataset.Z[subj]
                Vi[:, p0 : p0 + n_features] = x_mat[subj]
                Vi[:, p0 + n_features :] = x_mat[subj] * sst
                Wi = kh * Sigma_inv
                yi = dataset.y[subj]
                lhs += Vi.T @ Wi @ Vi
                rhs += Vi.T @ Wi @ yi

            para_hat = np.linalg.solve(lhs + self.ridge * np.eye(lhs.shape[0]), rhs)
            beta_local[i] = para_hat[:p0]
            A_hat_flat[i] = para_hat[p0 : p0 + n_features]

        signal_hat = np.sum(x_mat * A_hat_flat[:, None, :], axis=2)
        y_star = dataset.y - signal_hat
        beta_hat = self._solve_beta_gls(dataset.Z, y_star, Sigma_inv)
        fitted_values = signal_hat + dataset.Z @ beta_hat[:, None]

        return PairedVCTRResult(
            initial=initial_result,
            covariance=covariance,
            A_hat=A_hat_flat.reshape((n_subject,) + dataset.X.shape[2:]),
            beta_hat=beta_hat,
            fitted_values=fitted_values,
            meta={
                "bandwidth": selected_bandwidth,
                "ridge": self.ridge,
                "Sigma_inv": Sigma_inv,
                "beta_local": beta_local,
                "signal_hat": signal_hat,
                "y_star": y_star,
                "bandwidth_selected": selected_bandwidth,
                "bandwidth_method": initial_result.meta.get("bandwidth_method"),
                "bandwidth_grid": initial_result.meta.get("bandwidth_grid"),
                "bandwidth_cv_scores": initial_result.meta.get("bandwidth_cv_scores"),
                "bandwidth_cv_metric": initial_result.meta.get("bandwidth_cv_metric"),
            },
        )

    def fit(self, dataset: PairedEyeDataset) -> PairedVCTRResult:
        """Run the default three-stage paired-eye workflow."""

        initial = self.initial_fit_iid(dataset)
        covariance = self.estimate_covariance(dataset, initial)
        return self.refit_with_covariance(dataset, covariance, initial)

    def fit_paired(self, dataset: PairedEyeDataset) -> PairedVCTRResult:
        """Backward-compatible alias for the default paired-eye workflow."""

        return self.fit(dataset)

    def _resolve_bandwidth(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        """Resolve the unique bandwidth used by all three stages."""

        if self.bandwidth is not None:
            selected = float(self.bandwidth)
            if selected <= 0:
                raise ValueError("bandwidth must be positive.")
            return selected, {
                "bandwidth_selected": selected,
                "bandwidth_method": "fixed",
                "bandwidth_grid": [selected],
                "bandwidth_cv_scores": [],
                "bandwidth_cv_metric": "subject_cv_mse",
            }

        if self.bandwidth_grid is None:
            selected = float(self.DEFAULT_BANDWIDTH)
            return selected, {
                "bandwidth_selected": selected,
                "bandwidth_method": "default_fixed",
                "bandwidth_grid": [selected],
                "bandwidth_cv_scores": [],
                "bandwidth_cv_metric": "subject_cv_mse",
            }

        if self.bandwidth_method == "stage1_kfold_cv":
            selected, cv_meta = self._select_bandwidth_stage1_kfold(dataset)
        elif self.bandwidth_method == "stage1_loo_cv":
            selected, cv_meta = self._select_bandwidth_stage1_loo(dataset)
        else:
            raise ValueError(
                "bandwidth_method must be 'stage1_kfold_cv' or 'stage1_loo_cv' when bandwidth is None."
            )
        return selected, cv_meta

    def _select_bandwidth_stage1_kfold(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        """Select bandwidth by stage-1 subject-level K-fold CV."""

        grid = self._resolved_bandwidth_grid()
        score_rows: list[dict[str, float]] = []
        for bandwidth in grid:
            try:
                cv_score = float(self._kfold_subject_cv_score(dataset, float(bandwidth)))
            except np.linalg.LinAlgError:
                cv_score = float("inf")
            score_rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": cv_score,
                }
            )
        best_row = min(score_rows, key=lambda row: row["cv_score"])
        if not np.isfinite(best_row["cv_score"]):
            raise np.linalg.LinAlgError("All candidate bandwidths failed during stage-1 K-fold CV.")
        return float(best_row["bandwidth"]), {
            "bandwidth_selected": float(best_row["bandwidth"]),
            "bandwidth_method": "stage1_kfold_cv",
            "bandwidth_grid": [float(value) for value in grid],
            "bandwidth_cv_scores": score_rows,
            "bandwidth_cv_metric": "kfold_subject_mse",
            "bandwidth_cv_folds": self._effective_bandwidth_cv_folds(dataset.n_subject),
            "bandwidth_cv_seed": self.bandwidth_cv_seed,
        }

    def _select_bandwidth_stage1_loo(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        """Select bandwidth by stage-1 leave-one-subject-out CV."""

        grid = self._resolved_bandwidth_grid()
        score_rows: list[dict[str, float]] = []
        for bandwidth in grid:
            try:
                cv_score = float(self._loo_subject_cv_score(dataset, float(bandwidth)))
            except np.linalg.LinAlgError:
                cv_score = float("inf")
            score_rows.append(
                {
                    "bandwidth": float(bandwidth),
                    "cv_score": cv_score,
                }
            )
        best_row = min(score_rows, key=lambda row: row["cv_score"])
        if not np.isfinite(best_row["cv_score"]):
            raise np.linalg.LinAlgError("All candidate bandwidths failed during stage-1 LOO CV.")
        return float(best_row["bandwidth"]), {
            "bandwidth_selected": float(best_row["bandwidth"]),
            "bandwidth_method": "stage1_loo_cv",
            "bandwidth_grid": [float(value) for value in grid],
            "bandwidth_cv_scores": score_rows,
            "bandwidth_cv_metric": "loo_subject_mse",
            "bandwidth_cv_folds": dataset.n_subject,
            "bandwidth_cv_seed": self.bandwidth_cv_seed,
        }

    def _resolved_bandwidth_grid(self) -> tuple[float, ...]:
        """Return the candidate bandwidth grid used in auto-selection."""

        grid = self.bandwidth_grid if self.bandwidth_grid is not None else self.DEFAULT_BANDWIDTH_GRID
        if len(grid) == 0:
            raise ValueError("bandwidth_grid must not be empty.")
        grid = tuple(float(value) for value in grid)
        if any(value <= 0 for value in grid):
            raise ValueError("All bandwidth_grid values must be positive.")
        return grid

    def _loo_subject_cv_score(self, dataset: PairedEyeDataset, bandwidth: float) -> float:
        """Return the stage-1 leave-one-subject-out CV score for one bandwidth."""

        fold_errors = [
            self._loo_subject_fold_mse(dataset, bandwidth, holdout_index)
            for holdout_index in range(dataset.n_subject)
        ]
        return float(np.mean(fold_errors))

    def _kfold_subject_cv_score(self, dataset: PairedEyeDataset, bandwidth: float) -> float:
        """Return the stage-1 subject-level K-fold CV score for one bandwidth."""

        fold_indices = self._subject_kfold_indices(dataset.n_subject)
        fold_errors = [self._subject_fold_mse(dataset, bandwidth, holdout_indices) for holdout_indices in fold_indices]
        return float(np.mean(fold_errors))

    def _subject_kfold_indices(self, n_subject: int) -> list[np.ndarray]:
        """Return subject-level validation folds."""

        n_folds = self._effective_bandwidth_cv_folds(n_subject)
        rng = np.random.default_rng(self.bandwidth_cv_seed)
        shuffled = rng.permutation(n_subject)
        return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled, n_folds) if len(fold) > 0]

    def _effective_bandwidth_cv_folds(self, n_subject: int) -> int:
        """Return the effective number of subject-level CV folds."""

        if n_subject < 2:
            raise ValueError("At least two subjects are required for bandwidth CV.")
        if self.bandwidth_cv_folds < 2:
            raise ValueError("bandwidth_cv_folds must be at least 2.")
        return min(self.bandwidth_cv_folds, n_subject)

    def _subject_fold_mse(
        self,
        dataset: PairedEyeDataset,
        bandwidth: float,
        holdout_indices: np.ndarray,
    ) -> float:
        """Return the stage-1 prediction MSE for one held-out subject fold."""

        holdout_indices = np.asarray(holdout_indices, dtype=int).reshape(-1)
        train_mask = np.ones(dataset.n_subject, dtype=bool)
        train_mask[holdout_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_dataset = self._subset_dataset(dataset, train_indices)
        holdout_dataset = self._subset_dataset(dataset, holdout_indices)

        train_result = self._fit_initial_iid_with_bandwidth(train_dataset, bandwidth)
        train_flat = train_dataset.to_iid_observations()
        A_holdout_flat, _ = self._estimate_stage1_A(
            flat_Z=train_flat.Z,
            flat_X=self._flatten_X(train_flat.X),
            flat_y=train_flat.y,
            flat_t=train_flat.t,
            t_eval=holdout_dataset.t,
            p0=train_dataset.Z.shape[1],
            bandwidth=bandwidth,
        )

        holdout_x = holdout_dataset.X.reshape(holdout_dataset.n_subject, 2, -1)
        holdout_signal = np.sum(holdout_x * A_holdout_flat[:, None, :], axis=2)
        holdout_pred = holdout_signal + holdout_dataset.Z @ train_result.beta_hat[:, None]
        return float(np.mean(np.square(holdout_dataset.y - holdout_pred)))

    def _loo_subject_fold_mse(
        self,
        dataset: PairedEyeDataset,
        bandwidth: float,
        holdout_index: int,
    ) -> float:
        """Return the stage-1 prediction MSE for one held-out subject."""

        return self._subject_fold_mse(
            dataset=dataset,
            bandwidth=bandwidth,
            holdout_indices=np.array([holdout_index], dtype=int),
        )

    def _subset_dataset(self, dataset: PairedEyeDataset, subject_indices: np.ndarray) -> PairedEyeDataset:
        """Return a subject-level subset of a paired dataset."""

        subject_indices = np.asarray(subject_indices, dtype=int).reshape(-1)
        A_true = None if dataset.A_true is None else dataset.A_true[subject_indices]
        beta_true = None if dataset.beta_true is None else np.asarray(dataset.beta_true, dtype=float).copy()
        Sigma_true = None if dataset.Sigma_true is None else np.asarray(dataset.Sigma_true, dtype=float).copy()
        return PairedEyeDataset(
            subject_ids=dataset.subject_ids[subject_indices],
            eye_ids=np.asarray(dataset.eye_ids).copy(),
            t=dataset.t[subject_indices],
            X=dataset.X[subject_indices],
            Z=dataset.Z[subject_indices],
            y=dataset.y[subject_indices],
            A_true=A_true,
            beta_true=beta_true,
            Sigma_true=Sigma_true,
            meta=dict(dataset.meta),
        )

    def _estimate_stage1_A(
        self,
        *,
        flat_Z: np.ndarray,
        flat_X: np.ndarray,
        flat_y: np.ndarray,
        flat_t: np.ndarray,
        t_eval: np.ndarray,
        p0: int,
        bandwidth: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate stage-1 local linear coefficients at arbitrary evaluation points."""

        n_eval = np.asarray(t_eval, dtype=float).shape[0]
        n_features = flat_X.shape[1]
        A_hat_flat = np.zeros((n_eval, n_features), dtype=float)
        beta_local = np.zeros((n_eval, p0), dtype=float)

        for i, t0 in enumerate(np.asarray(t_eval, dtype=float)):
            ker = self._kernel_sqrt_weights(flat_t, t0, bandwidth)
            sst = (flat_t - t0) / bandwidth
            design = np.concatenate([flat_Z, flat_X, flat_X * sst[:, None]], axis=1)
            design_star = design * ker[:, None]
            target_star = flat_y * ker
            para_hat = self._solve_normal_equation(design_star, target_star)
            beta_local[i] = para_hat[:p0]
            A_hat_flat[i] = para_hat[p0 : p0 + n_features]

        return A_hat_flat, beta_local

    def _selected_bandwidth_from_initial(self, initial_result: InitialIidResult) -> float:
        """Read the selected stage-1 bandwidth from the initial fit metadata."""

        selected = initial_result.meta.get("bandwidth_selected")
        if selected is None:
            selected = initial_result.meta.get("bandwidth")
        if selected is None:
            raise ValueError("initial_result.meta must contain the selected bandwidth.")
        return float(selected)

    def _kernel_sqrt_weights(self, t: np.ndarray, t0: float, bandwidth: float) -> np.ndarray:
        """Return square-root kernel weights at a given bandwidth."""

        return kernel_sqrt_weights(t, t0, bandwidth, kernel="epanechnikov")

    def _kernel_scalar_weight(self, ti: float, t0: float, bandwidth: float) -> float:
        """Return the scalar kernel weight ``K_h(t_i - t0)``."""

        return float(self._kernel_sqrt_weights(np.array([ti], dtype=float), t0, bandwidth)[0] ** 2)

    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        """Flatten reduced features to shape ``(n_obs, p)``."""

        X = np.asarray(X)
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _solve_normal_equation(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Solve the stage-1 normal equation.

        When ``ridge == 0``, this matches the unregularized paper formula.
        """

        lhs = design.T @ design + self.ridge * np.eye(design.shape[1])
        rhs = design.T @ target
        return np.linalg.solve(lhs, rhs)

    def _solve_beta_ols(self, Z: np.ndarray, y_resid: np.ndarray) -> np.ndarray:
        """Estimate the stage-1 global beta via duplicated-subject OLS.

        When ``ridge == 0``, this matches equation (13).
        """

        Z_rep = np.repeat(Z, 2, axis=0)
        y_rep = np.asarray(y_resid, dtype=float).reshape(-1)
        lhs = Z_rep.T @ Z_rep + self.ridge * np.eye(Z.shape[1])
        rhs = Z_rep.T @ y_rep
        return np.linalg.solve(lhs, rhs)

    def _solve_beta_gls(self, Z: np.ndarray, y_resid: np.ndarray, Sigma_inv: np.ndarray) -> np.ndarray:
        """Estimate the final beta via blockwise GLS.

        When ``ridge == 0``, this matches equation (15).
        """

        p0 = Z.shape[1]
        lhs = np.zeros((p0, p0), dtype=float)
        rhs = np.zeros(p0, dtype=float)
        for i in range(Z.shape[0]):
            Zi = np.repeat(Z[i][None, :], 2, axis=0)
            yi = y_resid[i]
            lhs += Zi.T @ Sigma_inv @ Zi
            rhs += Zi.T @ Sigma_inv @ yi
        lhs += self.ridge * np.eye(p0)
        return np.linalg.solve(lhs, rhs)
