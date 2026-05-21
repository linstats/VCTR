"""Dataset containers for VCTR simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class SimulationDataset:
    """Container for one simulated dataset."""

    t: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    A_true: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
