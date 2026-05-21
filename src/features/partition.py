"""Tensor partition helpers for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class PartitionSpec:
    """Describe the number of blocks along each tensor mode."""

    blocks_per_mode: tuple[int, ...]


def partition_tensor_blocks(X: np.ndarray, spec: PartitionSpec) -> Sequence[np.ndarray]:
    """Split a tensor batch into spatial blocks.

    This is a paired-eye oriented placeholder API. The actual blocking logic
    should be implemented once the paired simulation and real-data pipelines
    are fixed.
    """

    raise NotImplementedError("Tensor block partitioning has not been implemented yet.")

