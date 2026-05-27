"""Covariance helpers for paired-eye VCTR workflows."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.utils.kernels import epanechnikov_kernel

from .base import CovarianceEstimate, InitialIidResult


def regroup_residuals_by_subject(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    eye_ids: np.ndarray,
) -> np.ndarray:
    """Regroup flat residuals into shape ``(n_subject, 2)``."""

    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    subject_ids = np.asarray(subject_ids).reshape(-1)
    eye_ids = np.asarray(eye_ids).reshape(-1)

    if not (residuals.shape == subject_ids.shape == eye_ids.shape):
        raise ValueError("residuals, subject_ids, and eye_ids must have the same shape.")

    eye_order = list(dict.fromkeys(eye_ids.tolist()))
    if len(eye_order) != 2:
        raise ValueError("Exactly two global eye labels are required.")

    per_subject: dict[object, dict[object, float]] = defaultdict(dict)
    for resid, subject_id, eye_id in zip(residuals, subject_ids, eye_ids, strict=True):
        subject_bucket = per_subject[subject_id]
        if eye_id in subject_bucket:
            raise ValueError(f"Duplicate eye label {eye_id!r} found for subject {subject_id!r}.")
        subject_bucket[eye_id] = float(resid)

    residual_pairs = np.empty((len(per_subject), 2), dtype=float)
    for row, (_, subject_bucket) in enumerate(per_subject.items()):
        if set(subject_bucket) != set(eye_order):
            raise ValueError("Each subject must have exactly two eyes.")
        residual_pairs[row, 0] = subject_bucket[eye_order[0]]
        residual_pairs[row, 1] = subject_bucket[eye_order[1]]
    return residual_pairs


def build_exchangeable_blocks(sigma2_hat_t: np.ndarray, rho_hat: float) -> np.ndarray:
    """Return subject-specific ``2 x 2`` exchangeable covariance blocks."""

    sigma2_hat_t = np.asarray(sigma2_hat_t, dtype=float).reshape(-1)
    base = np.array([[1.0, rho_hat], [rho_hat, 1.0]], dtype=float)
    return sigma2_hat_t[:, None, None] * base[None, :, :]


def invert_blocks(Sigma_hat_blocks: np.ndarray) -> np.ndarray:
    """Invert a stack of ``2 x 2`` covariance blocks."""

    Sigma_hat_blocks = np.asarray(Sigma_hat_blocks, dtype=float)
    if Sigma_hat_blocks.ndim != 3 or Sigma_hat_blocks.shape[1:] != (2, 2):
        raise ValueError("Sigma_hat_blocks must have shape (n_subject, 2, 2).")
    return np.linalg.inv(Sigma_hat_blocks)


def estimate_exchangeable_covariance(initial_result: InitialIidResult) -> CovarianceEstimate:
    """Estimate a shared exchangeable ``2 x 2`` covariance matrix from residuals."""

    if initial_result.residuals is None:
        raise ValueError("initial_result.residuals is required.")
    if initial_result.subject_ids is None:
        raise ValueError("initial_result.subject_ids is required.")
    if initial_result.eye_ids is None:
        raise ValueError("initial_result.eye_ids is required.")

    residual_pairs = regroup_residuals_by_subject(
        residuals=initial_result.residuals,
        subject_ids=initial_result.subject_ids,
        eye_ids=initial_result.eye_ids,
    )

    sigma2_hat = float(np.mean(np.square(residual_pairs)))
    rho_clipped = False
    if sigma2_hat <= 0:
        rho_hat = 0.0
    else:
        rho_hat = float(np.mean(residual_pairs[:, 0] * residual_pairs[:, 1]) / sigma2_hat)
    if abs(rho_hat) > 0.999:
        rho_hat = float(np.clip(rho_hat, -0.999, 0.999))
        rho_clipped = True

    sigma2_hat_t = np.full(residual_pairs.shape[0], sigma2_hat, dtype=float)
    Sigma_hat_blocks = build_exchangeable_blocks(sigma2_hat_t, rho_hat)
    Sigma_hat = Sigma_hat_blocks[0].copy()

    return CovarianceEstimate(
        covariance_mode="exchangeable_constant",
        rho_hat=rho_hat,
        sigma2_hat_t=sigma2_hat_t,
        Sigma_hat_blocks=Sigma_hat_blocks,
        Sigma_hat=Sigma_hat,
        sigma2_hat=sigma2_hat,
        residual_pairs=residual_pairs,
        meta={
            "method": "equations_16_17_constant",
            "rho_clipped": rho_clipped,
            "rho_clip_threshold": 0.999,
        },
    )


def estimate_exchangeable_varying_sigma_covariance(
    *,
    initial_result: InitialIidResult,
    t: np.ndarray,
    bandwidth: float,
) -> CovarianceEstimate:
    """Estimate ``sigma^2(t)`` and a shared ``rho`` from stage-1 residuals."""

    if initial_result.residuals is None:
        raise ValueError("initial_result.residuals is required.")
    if initial_result.subject_ids is None:
        raise ValueError("initial_result.subject_ids is required.")
    if initial_result.eye_ids is None:
        raise ValueError("initial_result.eye_ids is required.")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    t = np.asarray(t, dtype=float).reshape(-1)
    residual_pairs = regroup_residuals_by_subject(
        residuals=initial_result.residuals,
        subject_ids=initial_result.subject_ids,
        eye_ids=initial_result.eye_ids,
    )
    if residual_pairs.shape[0] != t.shape[0]:
        raise ValueError("t and regrouped residual pairs must have the same subject count.")

    squared_pairs = np.square(residual_pairs)
    sigma2_hat_t = np.zeros(t.shape[0], dtype=float)
    for idx, t0 in enumerate(t):
        scaled = (t - t0) / bandwidth
        kernel = epanechnikov_kernel(scaled) / bandwidth
        denom = 2.0 * np.sum(kernel)
        if denom <= 0:
            raise np.linalg.LinAlgError("Variance-kernel denominator is zero.")
        numer = float(np.sum(np.sum(squared_pairs, axis=1) * kernel))
        sigma2_hat_t[idx] = max(numer / denom, 1e-8)

    rho_clipped = False
    rho_denom = float(np.sum(sigma2_hat_t))
    if rho_denom <= 0:
        rho_hat = 0.0
    else:
        rho_hat = float(np.sum(residual_pairs[:, 0] * residual_pairs[:, 1]) / rho_denom)
    if abs(rho_hat) > 0.999:
        rho_hat = float(np.clip(rho_hat, -0.999, 0.999))
        rho_clipped = True

    Sigma_hat_blocks = build_exchangeable_blocks(sigma2_hat_t, rho_hat)
    sigma2_hat = float(np.mean(sigma2_hat_t))
    Sigma_hat = sigma2_hat * np.array([[1.0, rho_hat], [rho_hat, 1.0]], dtype=float)

    return CovarianceEstimate(
        covariance_mode="exchangeable_varying_sigma",
        rho_hat=rho_hat,
        sigma2_hat_t=sigma2_hat_t,
        Sigma_hat_blocks=Sigma_hat_blocks,
        Sigma_hat=Sigma_hat,
        sigma2_hat=sigma2_hat,
        residual_pairs=residual_pairs,
        meta={
            "method": "equations_16_17_varying_sigma",
            "variance_bandwidth": float(bandwidth),
            "rho_clipped": rho_clipped,
            "rho_clip_threshold": 0.999,
        },
    )
