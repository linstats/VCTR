"""Base interfaces and result containers for VCTR estimators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class StructureResult:
    """Structure-identification output for sparse VCTR estimators.

    Attributes
    ----------
    varying_mask:
        Boolean mask marking entries classified as varying coefficients.
    const_nonzero_mask:
        Boolean mask marking entries classified as constant and nonzero.
    const_zero_mask:
        Boolean mask marking entries classified as constant zero.
    beta_nonzero_mask:
        Boolean mask marking scalar covariates classified as nonzero.
    meta:
        Free-form metadata such as thresholds and summary counts.
    """

    varying_mask: np.ndarray
    const_nonzero_mask: np.ndarray
    const_zero_mask: np.ndarray
    beta_nonzero_mask: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EstimationResult:
    """Container for estimator outputs.

    Attributes
    ----------
    A_hat:
        Estimated varying coefficient array evaluated at target time points.
    beta_hat:
        Estimated scalar covariate coefficients.
    fitted_values:
        In-sample fitted values when available.
    residuals:
        In-sample residuals when available.
    structure:
        Optional structure-identification result for sparse estimators.
    meta:
        Free-form metadata such as bandwidths, masks, and diagnostics.
    """

    A_hat: np.ndarray | None = None
    beta_hat: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    residuals: np.ndarray | None = None
    structure: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BaseEstimator(ABC):
    """Abstract interface shared by VCTR estimators."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        **kwargs: Any,
    ) -> EstimationResult:
        """Fit the estimator and return an ``EstimationResult``."""
