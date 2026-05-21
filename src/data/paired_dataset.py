"""Dataset container for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PairedEyeDataset:
    """Container for paired-eye tensor regression data.

    The default shape conventions are:

    - ``subject_ids``: ``(n_subjects,)``
    - ``t``: ``(n_subjects,)``
    - ``Z``: ``(n_subjects, p0)``
    - ``X``: ``(n_subjects, n_eyes, *tensor_shape)``
    - ``y``: ``(n_subjects, n_eyes)``
    - ``eye_ids``: ``(n_eyes,)`` or ``(n_subjects, n_eyes)``
    """

    subject_ids: np.ndarray
    eye_ids: np.ndarray
    t: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    A_true: np.ndarray | None = None
    beta_true: np.ndarray | None = None
    Sigma_true: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

