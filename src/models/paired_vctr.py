"""Paired-eye VCTR model skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from src.data import PairedEyeDataset

from .base import BasePairedVCTRModel, PairedVCTRResult


@dataclass(slots=True)
class PairedEyeVCTRModel(BasePairedVCTRModel):
    """Skeleton estimator for the paired-eye VCTR target model.

    The intended model is

    ``y_{ij} = <X_{ij}, A(t_i)> + z_i^T beta + epsilon_{ij}``

    with within-subject dependence modeled through the covariance of
    ``(epsilon_{i1}, epsilon_{i2})^T``.
    """

    bandwidth: float
    spline_order: int = 4
    n_knots: int = 6
    penalty: str = "scad"

    def fit(self, dataset: PairedEyeDataset) -> PairedVCTRResult:
        raise NotImplementedError("Paired-eye VCTR estimation has not been implemented yet.")

