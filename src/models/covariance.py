"""Covariance helpers for the paired-eye two-step workflow."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import CovarianceEstimate, InitialIidResult


def regroup_residuals_by_subject(
    residuals: np.ndarray,
    subject_ids: np.ndarray,
    eye_ids: np.ndarray,
) -> np.ndarray:
    """Regroup flat residuals into shape ``(n_subject, 2)``.

    The output eye ordering follows the first-appearance ordering in
    ``eye_ids`` and must be consistent across subjects.
    """

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
    for row, (subject_id, subject_bucket) in enumerate(per_subject.items()):
        if set(subject_bucket) != set(eye_order):
            raise ValueError(f"Subject {subject_id!r} does not have exactly two eyes.")
        residual_pairs[row, 0] = subject_bucket[eye_order[0]]
        residual_pairs[row, 1] = subject_bucket[eye_order[1]]
    return residual_pairs


def estimate_exchangeable_covariance(initial_result: InitialIidResult) -> CovarianceEstimate:
    """Estimate a shared exchangeable ``2 x 2`` covariance matrix from residuals.

    This follows equations (16) and (17) in Section 2.3 of the paired-eye VCTR
    manuscript: ``sigma2_hat`` is the average squared residual over the ``2n``
    eye-level observations, and ``rho_hat`` is the average within-subject
    residual product divided by ``sigma2_hat``.
    """

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
    Sigma_hat = sigma2_hat * np.array([[1.0, rho_hat], [rho_hat, 1.0]], dtype=float)

    return CovarianceEstimate(
        sigma2_hat=sigma2_hat,
        rho_hat=rho_hat,
        Sigma_hat=Sigma_hat,
        residual_pairs=residual_pairs,
        meta={
            "method": "equations_16_17",
            "rho_clipped": rho_clipped,
            "rho_clip_threshold": 0.999,
        },
    )
