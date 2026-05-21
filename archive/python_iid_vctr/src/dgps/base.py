"""Base interfaces for VCTR data-generating processes."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.data import SimulationDataset


class BaseDGP(ABC):
    """Abstract interface shared by all simulation DGPs."""

    @abstractmethod
    def sample(self, seed: int | None = None) -> SimulationDataset:
        """Generate one simulated dataset."""
