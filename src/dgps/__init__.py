"""Paired-eye simulation DGPs."""

from .base import BasePairedDGP
from .paired_case1 import PairedCase1DGP
from .paired_case2 import PairedCase2DGP

__all__ = ["BasePairedDGP", "PairedCase1DGP", "PairedCase2DGP"]
