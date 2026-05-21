"""CP projection helpers for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class CPProjectionConfig:
    """Configuration for blockwise CP projection features."""

    rank: int


def blockwise_cp_project(blocks: Sequence[np.ndarray], config: CPProjectionConfig) -> np.ndarray:
    """Project partitioned tensor blocks into a reduced paired-eye feature map."""

    raise NotImplementedError("Blockwise CP projection has not been implemented yet.")

