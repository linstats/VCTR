"""Paired-eye VCTR estimator with configurable covariance modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from src.data import PairedEyeDataset
from src.utils.kernels import kernel_sqrt_weights

from .base import BasePairedVCTRModel, CovarianceEstimate, InitialIidResult, PairedVCTRResult
from .covariance import (
    estimate_exchangeable_covariance,
    estimate_exchangeable_varying_sigma_covariance,
    invert_blocks,
)


@dataclass(slots=True)
class PairedEyeVCTRModel(BasePairedVCTRModel):
    """Paired-eye VCTR estimator for reduced-feature paired data."""

    covariance_mode: str = "exchangeable_varying_sigma"
    signal_bandwidth: float | None = None
    signal_bandwidth_method: str = "stage1_kfold_cv"
    signal_bandwidth_grid: tuple[float, ...] | None = None
    signal_bandwidth_cv_folds: int = 5
    signal_bandwidth_cv_seed: int = 0
    variance_bandwidth: float | None = None
    variance_bandwidth_method: str = "stage2_kfold_cv"
    variance_bandwidth_grid: tuple[float, ...] | None = None
    variance_bandwidth_cv_folds: int = 5
    variance_bandwidth_cv_seed: int = 0
    ridge: float = 0.0
    spline_order: int = 4
    n_knots: int = 6
    penalty: str = "scad"

    DEFAULT_SIGNAL_BANDWIDTH: ClassVar[float] = 0.13
    DEFAULT_SIGNAL_BANDWIDTH_GRID: ClassVar[tuple[float, ...]] = (0.08, 0.10, 0.13, 0.16, 0.20)
    DEFAULT_VARIANCE_BANDWIDTH: ClassVar[float] = 0.13
    DEFAULT_VARIANCE_BANDWIDTH_GRID: ClassVar[tuple[float, ...]] = (0.08, 0.10, 0.13, 0.16, 0.20)

    def initial_fit_iid(self, dataset: PairedEyeDataset) -> InitialIidResult:
        """Fit the stage-1 iid working model on the flattened eye view."""

        selected_bandwidth, bandwidth_meta = self._resolve_signal_bandwidth(dataset)
        result = self._fit_initial_iid_with_bandwidth(dataset, selected_bandwidth)
        result.meta.update(bandwidth_meta)
        return result

    def _fit_initial_iid_with_bandwidth(
        self,
        dataset: PairedEyeDataset,
        bandwidth: float,
    ) -> InitialIidResult:
        """Fit the stage-1 iid working model at a fixed signal bandwidth."""

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
                "signal_bandwidth": bandwidth,
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
        """Estimate covariance blocks from stage-1 residuals."""

        self._validate_covariance_mode()
        if self.covariance_mode == "exchangeable_constant":
            covariance = estimate_exchangeable_covariance(initial_result)
            covariance.meta.update(self._constant_variance_meta())
            return covariance

        selected_bandwidth, bandwidth_meta = self._resolve_variance_bandwidth(dataset, initial_result)
        covariance = estimate_exchangeable_varying_sigma_covariance(
            initial_result=initial_result,
            t=dataset.t,
            bandwidth=selected_bandwidth,
        )
        covariance.meta.update(bandwidth_meta)
        return covariance

    def refit_with_covariance(
        self,
        dataset: PairedEyeDataset,
        covariance: CovarianceEstimate,
        initial_result: InitialIidResult | None = None,
    ) -> PairedVCTRResult:
        """Refit the paired-eye model using subject-specific covariance blocks."""

        if initial_result is None:
            raise ValueError("initial_result is required for covariance-aware refitting.")

        n_subject = dataset.n_subject
        x_mat = dataset.X.reshape(n_subject, 2, -1)
        n_features = x_mat.shape[2]
        p0 = dataset.Z.shape[1]
        Sigma_inv_blocks = invert_blocks(covariance.Sigma_hat_blocks)
        selected_bandwidth = self._selected_signal_bandwidth_from_initial(initial_result)

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
                Wi = kh * Sigma_inv_blocks[subj]
                yi = dataset.y[subj]
                lhs += Vi.T @ Wi @ Vi
                rhs += Vi.T @ Wi @ yi

            para_hat = np.linalg.solve(lhs + self.ridge * np.eye(lhs.shape[0]), rhs)
            beta_local[i] = para_hat[:p0]
            A_hat_flat[i] = para_hat[p0 : p0 + n_features]

        signal_hat = np.sum(x_mat * A_hat_flat[:, None, :], axis=2)
        y_star = dataset.y - signal_hat
        beta_hat = self._solve_beta_gls(dataset.Z, y_star, Sigma_inv_blocks)
        fitted_values = signal_hat + dataset.Z @ beta_hat[:, None]

        return PairedVCTRResult(
            initial=initial_result,
            covariance=covariance,
            A_hat=A_hat_flat.reshape((n_subject,) + dataset.X.shape[2:]),
            beta_hat=beta_hat,
            fitted_values=fitted_values,
            meta={
                "covariance_mode": covariance.covariance_mode,
                "signal_bandwidth": selected_bandwidth,
                "variance_bandwidth": covariance.meta.get("variance_bandwidth_selected"),
                "ridge": self.ridge,
                "Sigma_inv_blocks": Sigma_inv_blocks,
                "beta_local": beta_local,
                "signal_hat": signal_hat,
                "y_star": y_star,
                "signal_bandwidth_selected": selected_bandwidth,
                "signal_bandwidth_method": initial_result.meta.get("signal_bandwidth_method"),
                "signal_bandwidth_grid": initial_result.meta.get("signal_bandwidth_grid"),
                "signal_bandwidth_cv_scores": initial_result.meta.get("signal_bandwidth_cv_scores"),
                "signal_bandwidth_cv_metric": initial_result.meta.get("signal_bandwidth_cv_metric"),
                "variance_bandwidth_selected": covariance.meta.get("variance_bandwidth_selected"),
                "variance_bandwidth_method": covariance.meta.get("variance_bandwidth_method"),
                "variance_bandwidth_grid": covariance.meta.get("variance_bandwidth_grid"),
                "variance_bandwidth_cv_scores": covariance.meta.get("variance_bandwidth_cv_scores"),
                "variance_bandwidth_cv_metric": covariance.meta.get("variance_bandwidth_cv_metric"),
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

    def _resolve_signal_bandwidth(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        """Resolve the stage-1 and stage-3 signal bandwidth."""

        if self.signal_bandwidth is not None:
            selected = float(self.signal_bandwidth)
            if selected <= 0:
                raise ValueError("signal_bandwidth must be positive.")
            return selected, {
                "signal_bandwidth_selected": selected,
                "signal_bandwidth_method": "fixed",
                "signal_bandwidth_grid": [selected],
                "signal_bandwidth_cv_scores": [],
                "signal_bandwidth_cv_metric": "subject_cv_mse",
            }

        if self.signal_bandwidth_grid is None:
            selected = float(self.DEFAULT_SIGNAL_BANDWIDTH)
            return selected, {
                "signal_bandwidth_selected": selected,
                "signal_bandwidth_method": "default_fixed",
                "signal_bandwidth_grid": [selected],
                "signal_bandwidth_cv_scores": [],
                "signal_bandwidth_cv_metric": "subject_cv_mse",
            }

        if self.signal_bandwidth_method == "stage1_kfold_cv":
            selected, cv_meta = self._select_signal_bandwidth_stage1_kfold(dataset)
        elif self.signal_bandwidth_method == "stage1_loo_cv":
            selected, cv_meta = self._select_signal_bandwidth_stage1_loo(dataset)
        else:
            raise ValueError(
                "signal_bandwidth_method must be 'stage1_kfold_cv' or 'stage1_loo_cv' when signal_bandwidth is None."
            )
        return selected, cv_meta

    def _resolve_variance_bandwidth(
        self,
        dataset: PairedEyeDataset,
        initial_result: InitialIidResult,
    ) -> tuple[float, dict[str, Any]]:
        """Resolve the stage-2 variance bandwidth."""

        if self.variance_bandwidth is not None:
            selected = float(self.variance_bandwidth)
            if selected <= 0:
                raise ValueError("variance_bandwidth must be positive.")
            return selected, {
                "variance_bandwidth_selected": selected,
                "variance_bandwidth_method": "fixed",
                "variance_bandwidth_grid": [selected],
                "variance_bandwidth_cv_scores": [],
                "variance_bandwidth_cv_metric": "subject_cv_squared_residual_mse",
            }

        if self.variance_bandwidth_grid is None:
            selected = float(self.DEFAULT_VARIANCE_BANDWIDTH)
            return selected, {
                "variance_bandwidth_selected": selected,
                "variance_bandwidth_method": "default_fixed",
                "variance_bandwidth_grid": [selected],
                "variance_bandwidth_cv_scores": [],
                "variance_bandwidth_cv_metric": "subject_cv_squared_residual_mse",
            }

        if self.variance_bandwidth_method != "stage2_kfold_cv":
            raise ValueError(
                "variance_bandwidth_method must be 'stage2_kfold_cv' when variance_bandwidth is None."
            )
        return self._select_variance_bandwidth_stage2_kfold(dataset, initial_result)

    def _select_signal_bandwidth_stage1_kfold(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        grid = self._resolved_signal_bandwidth_grid()
        score_rows: list[dict[str, float]] = []
        for bandwidth in grid:
            try:
                cv_score = float(self._kfold_subject_cv_score(dataset, float(bandwidth)))
            except np.linalg.LinAlgError:
                cv_score = float("inf")
            score_rows.append({"bandwidth": float(bandwidth), "cv_score": cv_score})
        best_row = min(score_rows, key=lambda row: row["cv_score"])
        if not np.isfinite(best_row["cv_score"]):
            raise np.linalg.LinAlgError("All candidate signal bandwidths failed during stage-1 K-fold CV.")
        return float(best_row["bandwidth"]), {
            "signal_bandwidth_selected": float(best_row["bandwidth"]),
            "signal_bandwidth_method": "stage1_kfold_cv",
            "signal_bandwidth_grid": [float(value) for value in grid],
            "signal_bandwidth_cv_scores": score_rows,
            "signal_bandwidth_cv_metric": "kfold_subject_mse",
            "signal_bandwidth_cv_folds": self._effective_signal_bandwidth_cv_folds(dataset.n_subject),
            "signal_bandwidth_cv_seed": self.signal_bandwidth_cv_seed,
        }

    def _select_signal_bandwidth_stage1_loo(self, dataset: PairedEyeDataset) -> tuple[float, dict[str, Any]]:
        grid = self._resolved_signal_bandwidth_grid()
        score_rows: list[dict[str, float]] = []
        for bandwidth in grid:
            try:
                cv_score = float(self._loo_subject_cv_score(dataset, float(bandwidth)))
            except np.linalg.LinAlgError:
                cv_score = float("inf")
            score_rows.append({"bandwidth": float(bandwidth), "cv_score": cv_score})
        best_row = min(score_rows, key=lambda row: row["cv_score"])
        if not np.isfinite(best_row["cv_score"]):
            raise np.linalg.LinAlgError("All candidate signal bandwidths failed during stage-1 LOO CV.")
        return float(best_row["bandwidth"]), {
            "signal_bandwidth_selected": float(best_row["bandwidth"]),
            "signal_bandwidth_method": "stage1_loo_cv",
            "signal_bandwidth_grid": [float(value) for value in grid],
            "signal_bandwidth_cv_scores": score_rows,
            "signal_bandwidth_cv_metric": "loo_subject_mse",
            "signal_bandwidth_cv_folds": dataset.n_subject,
            "signal_bandwidth_cv_seed": self.signal_bandwidth_cv_seed,
        }

    def _select_variance_bandwidth_stage2_kfold(
        self,
        dataset: PairedEyeDataset,
        initial_result: InitialIidResult,
    ) -> tuple[float, dict[str, Any]]:
        grid = self._resolved_variance_bandwidth_grid()
        score_rows: list[dict[str, float]] = []
        residual_pairs = self._initial_residual_pairs(initial_result)
        for bandwidth in grid:
            try:
                cv_score = float(self._variance_kfold_subject_cv_score(dataset.t, residual_pairs, float(bandwidth)))
            except np.linalg.LinAlgError:
                cv_score = float("inf")
            score_rows.append({"bandwidth": float(bandwidth), "cv_score": cv_score})
        best_row = min(score_rows, key=lambda row: row["cv_score"])
        if not np.isfinite(best_row["cv_score"]):
            raise np.linalg.LinAlgError("All candidate variance bandwidths failed during stage-2 K-fold CV.")
        return float(best_row["bandwidth"]), {
            "variance_bandwidth_selected": float(best_row["bandwidth"]),
            "variance_bandwidth_method": "stage2_kfold_cv",
            "variance_bandwidth_grid": [float(value) for value in grid],
            "variance_bandwidth_cv_scores": score_rows,
            "variance_bandwidth_cv_metric": "kfold_subject_squared_residual_mse",
            "variance_bandwidth_cv_folds": self._effective_variance_bandwidth_cv_folds(dataset.n_subject),
            "variance_bandwidth_cv_seed": self.variance_bandwidth_cv_seed,
        }

    def _resolved_signal_bandwidth_grid(self) -> tuple[float, ...]:
        grid = self.signal_bandwidth_grid if self.signal_bandwidth_grid is not None else self.DEFAULT_SIGNAL_BANDWIDTH_GRID
        return self._validated_bandwidth_grid(grid, "signal_bandwidth_grid")

    def _resolved_variance_bandwidth_grid(self) -> tuple[float, ...]:
        grid = (
            self.variance_bandwidth_grid
            if self.variance_bandwidth_grid is not None
            else self.DEFAULT_VARIANCE_BANDWIDTH_GRID
        )
        return self._validated_bandwidth_grid(grid, "variance_bandwidth_grid")

    def _validated_bandwidth_grid(self, grid: tuple[float, ...], name: str) -> tuple[float, ...]:
        if len(grid) == 0:
            raise ValueError(f"{name} must not be empty.")
        resolved = tuple(float(value) for value in grid)
        if any(value <= 0 for value in resolved):
            raise ValueError(f"All {name} values must be positive.")
        return resolved

    def _loo_subject_cv_score(self, dataset: PairedEyeDataset, bandwidth: float) -> float:
        fold_errors = [
            self._loo_subject_fold_mse(dataset, bandwidth, holdout_index)
            for holdout_index in range(dataset.n_subject)
        ]
        return float(np.mean(fold_errors))

    def _kfold_subject_cv_score(self, dataset: PairedEyeDataset, bandwidth: float) -> float:
        fold_indices = self._subject_kfold_indices(dataset.n_subject, self.signal_bandwidth_cv_seed, self.signal_bandwidth_cv_folds)
        fold_errors = [self._subject_fold_mse(dataset, bandwidth, holdout_indices) for holdout_indices in fold_indices]
        return float(np.mean(fold_errors))

    def _variance_kfold_subject_cv_score(
        self,
        t: np.ndarray,
        residual_pairs: np.ndarray,
        bandwidth: float,
    ) -> float:
        fold_indices = self._subject_kfold_indices(
            len(t),
            self.variance_bandwidth_cv_seed,
            self.variance_bandwidth_cv_folds,
        )
        squared_pairs = np.square(residual_pairs)
        errors: list[float] = []
        for holdout_indices in fold_indices:
            train_mask = np.ones(len(t), dtype=bool)
            train_mask[holdout_indices] = False
            train_t = t[train_mask]
            train_sq = squared_pairs[train_mask]
            holdout_t = t[holdout_indices]
            holdout_sq = squared_pairs[holdout_indices]
            sigma_hat = self._smooth_variance_curve(train_t, train_sq, holdout_t, bandwidth)
            holdout_target = np.mean(holdout_sq, axis=1)
            errors.append(float(np.mean(np.square(holdout_target - sigma_hat))))
        return float(np.mean(errors))

    def _subject_kfold_indices(self, n_subject: int, seed: int, n_folds_param: int) -> list[np.ndarray]:
        n_folds = self._effective_cv_folds(n_subject, n_folds_param)
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(n_subject)
        return [np.asarray(fold, dtype=int) for fold in np.array_split(shuffled, n_folds) if len(fold) > 0]

    def _effective_cv_folds(self, n_subject: int, n_folds_param: int) -> int:
        if n_subject < 2:
            raise ValueError("At least two subjects are required for bandwidth CV.")
        if n_folds_param < 2:
            raise ValueError("bandwidth_cv_folds must be at least 2.")
        return min(n_folds_param, n_subject)

    def _effective_signal_bandwidth_cv_folds(self, n_subject: int) -> int:
        return self._effective_cv_folds(n_subject, self.signal_bandwidth_cv_folds)

    def _effective_variance_bandwidth_cv_folds(self, n_subject: int) -> int:
        return self._effective_cv_folds(n_subject, self.variance_bandwidth_cv_folds)

    def _subject_fold_mse(
        self,
        dataset: PairedEyeDataset,
        bandwidth: float,
        holdout_indices: np.ndarray,
    ) -> float:
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
        return self._subject_fold_mse(
            dataset=dataset,
            bandwidth=bandwidth,
            holdout_indices=np.array([holdout_index], dtype=int),
        )

    def _subset_dataset(self, dataset: PairedEyeDataset, subject_indices: np.ndarray) -> PairedEyeDataset:
        subject_indices = np.asarray(subject_indices, dtype=int).reshape(-1)
        A_true = None if dataset.A_true is None else dataset.A_true[subject_indices]
        beta_true = None if dataset.beta_true is None else np.asarray(dataset.beta_true, dtype=float).copy()
        if dataset.Sigma_true is None:
            Sigma_true = None
        else:
            Sigma_true_arr = np.asarray(dataset.Sigma_true, dtype=float)
            Sigma_true = Sigma_true_arr[subject_indices] if Sigma_true_arr.ndim == 3 else Sigma_true_arr.copy()
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

    def _selected_signal_bandwidth_from_initial(self, initial_result: InitialIidResult) -> float:
        selected = initial_result.meta.get("signal_bandwidth_selected")
        if selected is None:
            selected = initial_result.meta.get("signal_bandwidth")
        if selected is None:
            raise ValueError("initial_result.meta must contain the selected signal bandwidth.")
        return float(selected)

    def _initial_residual_pairs(self, initial_result: InitialIidResult) -> np.ndarray:
        if initial_result.residuals is None or initial_result.subject_ids is None or initial_result.eye_ids is None:
            raise ValueError("initial_result must contain residuals, subject_ids, and eye_ids.")
        from .covariance import regroup_residuals_by_subject

        return regroup_residuals_by_subject(
            initial_result.residuals,
            initial_result.subject_ids,
            initial_result.eye_ids,
        )

    def _smooth_variance_curve(
        self,
        train_t: np.ndarray,
        train_sq_pairs: np.ndarray,
        t_eval: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        train_t = np.asarray(train_t, dtype=float).reshape(-1)
        train_sq_pairs = np.asarray(train_sq_pairs, dtype=float)
        t_eval = np.asarray(t_eval, dtype=float).reshape(-1)
        sigma_hat = np.zeros(t_eval.shape[0], dtype=float)
        for idx, t0 in enumerate(t_eval):
            kernel = self._kernel_scalar_weights(train_t, t0, bandwidth)
            denom = 2.0 * np.sum(kernel)
            if denom <= 0:
                raise np.linalg.LinAlgError("Variance-kernel denominator is zero.")
            sigma_hat[idx] = max(float(np.sum(np.sum(train_sq_pairs, axis=1) * kernel) / denom), 1e-8)
        return sigma_hat

    def _constant_variance_meta(self) -> dict[str, Any]:
        return {
            "variance_bandwidth_selected": None,
            "variance_bandwidth_method": "not_used",
            "variance_bandwidth_grid": [],
            "variance_bandwidth_cv_scores": [],
            "variance_bandwidth_cv_metric": "not_used",
        }

    def _validate_covariance_mode(self) -> None:
        if self.covariance_mode not in {"exchangeable_constant", "exchangeable_varying_sigma"}:
            raise ValueError(
                "covariance_mode must be 'exchangeable_constant' or 'exchangeable_varying_sigma'."
            )

    def _kernel_sqrt_weights(self, t: np.ndarray, t0: float, bandwidth: float) -> np.ndarray:
        return kernel_sqrt_weights(t, t0, bandwidth, kernel="epanechnikov")

    def _kernel_scalar_weights(self, t: np.ndarray, t0: float, bandwidth: float) -> np.ndarray:
        return np.square(self._kernel_sqrt_weights(t, t0, bandwidth))

    def _kernel_scalar_weight(self, ti: float, t0: float, bandwidth: float) -> float:
        return float(self._kernel_scalar_weights(np.array([ti], dtype=float), t0, bandwidth)[0])

    def _flatten_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim == 2:
            return X
        return X.reshape(X.shape[0], -1)

    def _solve_normal_equation(self, design: np.ndarray, target: np.ndarray) -> np.ndarray:
        lhs = design.T @ design + self.ridge * np.eye(design.shape[1])
        rhs = design.T @ target
        return np.linalg.solve(lhs, rhs)

    def _solve_beta_ols(self, Z: np.ndarray, y_resid: np.ndarray) -> np.ndarray:
        Z_rep = np.repeat(Z, 2, axis=0)
        y_rep = np.asarray(y_resid, dtype=float).reshape(-1)
        lhs = Z_rep.T @ Z_rep + self.ridge * np.eye(Z.shape[1])
        rhs = Z_rep.T @ y_rep
        return np.linalg.solve(lhs, rhs)

    def _solve_beta_gls(self, Z: np.ndarray, y_resid: np.ndarray, Sigma_inv_blocks: np.ndarray) -> np.ndarray:
        p0 = Z.shape[1]
        lhs = np.zeros((p0, p0), dtype=float)
        rhs = np.zeros(p0, dtype=float)
        for i in range(Z.shape[0]):
            Zi = np.repeat(Z[i][None, :], 2, axis=0)
            yi = y_resid[i]
            lhs += Zi.T @ Sigma_inv_blocks[i] @ Zi
            rhs += Zi.T @ Sigma_inv_blocks[i] @ yi
        lhs += self.ridge * np.eye(p0)
        return np.linalg.solve(lhs, rhs)
