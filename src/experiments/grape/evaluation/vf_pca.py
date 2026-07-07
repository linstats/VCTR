"""Fold-local PCA utilities for the GRAPE bilateral-mean VF covariates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def subject_equal_weights(groups: np.ndarray) -> np.ndarray:
    """Give every patient equal total weight across their retained visits."""

    groups = np.asarray(groups)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("groups must be a non-empty one-dimensional array.")
    _, inverse, counts = np.unique(groups.astype(str), return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse].astype(float)


@dataclass(frozen=True)
class FoldVFPCATransformer:
    """Training-fold VF standardization and PCA transformation."""

    mean_: np.ndarray
    scale_: np.ndarray
    components_: np.ndarray
    explained_variance_ratio_: np.ndarray
    singular_values_: np.ndarray
    n_training_rows_: int
    n_training_groups_: int

    @classmethod
    def fit(
        cls,
        vf: np.ndarray,
        groups: np.ndarray,
        *,
        n_components: int,
        weighting: str = "subject_equal",
    ) -> "FoldVFPCATransformer":
        vf = np.asarray(vf, dtype=float)
        groups = np.asarray(groups)
        if vf.ndim != 2 or vf.shape[0] == 0 or vf.shape[1] == 0:
            raise ValueError("vf must have shape (n_rows, n_vf) with both dimensions positive.")
        if groups.shape != (vf.shape[0],):
            raise ValueError("groups must have one entry per VF row.")
        if not np.isfinite(vf).all():
            raise ValueError("vf contains non-finite values.")
        max_components = min(vf.shape)
        if not 1 <= int(n_components) <= max_components:
            raise ValueError(f"n_components must be between 1 and {max_components}.")
        if weighting == "subject_equal":
            weights = subject_equal_weights(groups)
        elif weighting == "row_equal":
            weights = np.ones(vf.shape[0], dtype=float)
        else:
            raise ValueError("weighting must be 'subject_equal' or 'row_equal'.")

        weight_sum = float(weights.sum())
        mean = np.sum(vf * weights[:, None], axis=0) / weight_sum
        centered = vf - mean
        variance = np.sum(np.square(centered) * weights[:, None], axis=0) / weight_sum
        scale = np.sqrt(variance)
        if np.any(scale <= np.finfo(float).eps):
            bad = np.flatnonzero(scale <= np.finfo(float).eps).tolist()
            raise ValueError(f"VF columns have zero training-fold variance: {bad}")

        standardized = centered / scale
        weighted = standardized * np.sqrt(weights[:, None])
        _, singular_values, vt = np.linalg.svd(weighted, full_matrices=False)
        components = vt[: int(n_components)].copy()

        # Fix the otherwise arbitrary PC sign for stable fold and bootstrap output.
        for row in range(components.shape[0]):
            pivot = int(np.argmax(np.abs(components[row])))
            if components[row, pivot] < 0:
                components[row] *= -1.0

        eigenvalues = np.square(singular_values)
        total = float(eigenvalues.sum())
        explained = eigenvalues[: int(n_components)] / total if total > 0 else np.zeros(int(n_components))
        return cls(
            mean_=mean,
            scale_=scale,
            components_=components,
            explained_variance_ratio_=explained,
            singular_values_=singular_values[: int(n_components)],
            n_training_rows_=int(vf.shape[0]),
            n_training_groups_=int(np.unique(groups.astype(str)).size),
        )

    @property
    def n_components(self) -> int:
        return int(self.components_.shape[0])

    def transform(self, vf: np.ndarray) -> np.ndarray:
        return self.standardize(vf) @ self.components_.T

    def standardize(self, vf: np.ndarray) -> np.ndarray:
        """Apply training-fold VF centering and scaling without PCA projection."""

        vf = np.asarray(vf, dtype=float)
        if vf.ndim != 2 or vf.shape[1] != self.mean_.shape[0]:
            raise ValueError(f"vf must have shape (n_rows, {self.mean_.shape[0]}).")
        if not np.isfinite(vf).all():
            raise ValueError("vf contains non-finite values.")
        return (vf - self.mean_) / self.scale_


def split_sex_vf(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the current GRAPE Z matrix into sex and 59 bilateral-mean VF columns."""

    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[1] < 2:
        raise ValueError("Z must contain is_female followed by at least one VF column.")
    return Z[:, :1].copy(), Z[:, 1:].copy()


def compose_pca_covariates(*, sex: np.ndarray, scores: np.ndarray, include_sex: bool) -> np.ndarray:
    """Compose the fold-specific covariate matrix used by a PCA model."""

    sex = np.asarray(sex, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if sex.ndim != 2 or sex.shape[1] != 1:
        raise ValueError("sex must have shape (n_rows, 1).")
    if scores.ndim != 2 or scores.shape[0] != sex.shape[0]:
        raise ValueError("scores must have shape (n_rows, n_components).")
    return np.column_stack([sex, scores]) if include_sex else scores.copy()
