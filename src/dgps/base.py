"""Base interfaces for paired-eye VCTR data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data import PairedEyeDataset


class BasePairedDGP(ABC):
    """Abstract interface shared by paired-eye simulation DGPs."""

    @abstractmethod
    def sample(self, seed: int | None = None) -> PairedEyeDataset:
        """Generate one paired-eye simulated dataset."""
