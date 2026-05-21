"""Base interfaces for paired-eye VCTR models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.data import PairedEyeDataset


@dataclass(slots=True)
class PairedVCTRResult:
    """Output container for paired-eye VCTR estimation."""

    A_hat: np.ndarray | None = None
    beta_hat: np.ndarray | None = None
    Sigma_hat: np.ndarray | None = None
    fitted_values: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class BasePairedVCTRModel(ABC):
    """Abstract interface for paired-eye VCTR estimators."""

    @abstractmethod
    def fit(self, dataset: PairedEyeDataset) -> PairedVCTRResult:
        """Fit a paired-eye VCTR model."""

