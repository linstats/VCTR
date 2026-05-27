"""Base interfaces for paired-eye VCTR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data import PairedEyeDataset


@dataclass(slots=True)
class InitialIidResult:
    """Stage-1 iid fit result on the flattened observation view."""

    A_hat: np.ndarray | None = None
    beta_hat: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    residuals: np.ndarray | None = None
    subject_ids: np.ndarray | None = None
    eye_ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CovarianceEstimate:
    """Estimated subject-level covariance blocks for paired-eye residuals."""

    covariance_mode: str
    rho_hat: float
    sigma2_hat_t: np.ndarray
    Sigma_hat_blocks: np.ndarray
    Sigma_hat: np.ndarray | None = None
    sigma2_hat: float | None = None
    residual_pairs: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairedVCTRResult:
    """Final paired-eye VCTR result after covariance-aware refitting."""

    initial: InitialIidResult
    covariance: CovarianceEstimate

    A_hat: np.ndarray | None = None
    beta_hat: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BasePairedVCTRModel(ABC):
    """Abstract interface for paired-eye VCTR estimators."""

    @abstractmethod
    def fit(self, dataset: PairedEyeDataset) -> PairedVCTRResult:
        """Fit a paired-eye VCTR model."""
