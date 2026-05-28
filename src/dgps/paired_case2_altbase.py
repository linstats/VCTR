"""Paired-eye Case 2 altbase DGP in reduced-feature space."""

from __future__ import annotations

from dataclasses import dataclass

from src.data import PairedEyeDataset

from .paired_case1_altbase import PairedCase1AltbaseDGP


@dataclass(slots=True)
class PairedCase2AltbaseDGP(PairedCase1AltbaseDGP):
    """Thin 3D-equivalent wrapper around the active altbase paired DGP.

    This class intentionally reuses the full data-generation mechanism from
    ``PairedCase1AltbaseDGP`` and only changes the default reduced-feature size
    and metadata labels so the experiment can be organized as a separate
    3D-style case.
    """

    R: int = 3
    S: int = 27

    def sample(self, seed: int | None = None) -> PairedEyeDataset:
        """Generate one paired-eye alternative-base Case 2 style dataset."""

        dataset = PairedCase1AltbaseDGP.sample(self, seed=seed)
        dataset.meta = {
            **dataset.meta,
            "dgp": "paired_case2_altbase",
            "case": "case2_altbase",
            "R": self.R,
            "S": self.S,
            "raw_equivalent_p1": 48,
            "raw_equivalent_p2": 48,
            "raw_equivalent_p3": 48,
            "raw_equivalent_p1_prime": 16,
            "raw_equivalent_p2_prime": 16,
            "raw_equivalent_p3_prime": 16,
        }
        return dataset
