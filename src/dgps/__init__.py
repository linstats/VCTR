"""Paired-eye simulation DGPs."""

from .base import BasePairedDGP
from .archive.paired_case1 import PairedCase1DGP
from .archive.paired_case2 import PairedCase2DGP
from .paired_case1_altbase import PairedCase1AltbaseDGP
from .paired_case2_altbase import PairedCase2AltbaseDGP
from .variance_functions import SUPPORTED_SIGMA2_FUNCTIONS, sigma2_curve

__all__ = [
    "BasePairedDGP",
    "PairedCase1DGP",
    "PairedCase1AltbaseDGP",
    "PairedCase2DGP",
    "PairedCase2AltbaseDGP",
    "SUPPORTED_SIGMA2_FUNCTIONS",
    "sigma2_curve",
]
