"""Dataset containers for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FlattenedIIDView:
    """Observation-level view obtained from a paired-eye dataset."""

    X: np.ndarray
    y: np.ndarray
    t: np.ndarray
    Z: np.ndarray
    subject_ids: np.ndarray
    eye_ids: np.ndarray


@dataclass(slots=True)
class PairedEyeDataset:
    """Container for paired-eye tensor regression data.

    The default shape conventions are:

    - ``subject_ids``: ``(n_subjects,)``
    - ``eye_ids``: ``(2,)``
    - ``t``: ``(n_subjects,)``
    - ``Z``: ``(n_subjects, p0)``
    - ``X``: ``(n_subjects, 2, *tensor_shape)``
    - ``y``: ``(n_subjects, 2)``
    - ``Sigma_true``: optional ``(2, 2)`` matrix or ``(n_subjects, 2, 2)`` blocks
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

    def __post_init__(self) -> None:
        """Validate the paired-eye shape conventions."""

        self.subject_ids = np.asarray(self.subject_ids)
        self.eye_ids = np.asarray(self.eye_ids)
        self.t = np.asarray(self.t)
        self.X = np.asarray(self.X)
        self.Z = np.asarray(self.Z)
        self.y = np.asarray(self.y)

        n_subject = self.subject_ids.shape[0]
        if self.subject_ids.ndim != 1:
            raise ValueError("subject_ids must have shape (n_subject,).")
        if self.eye_ids.ndim != 1 or self.eye_ids.shape[0] != 2:
            raise ValueError("eye_ids must have shape (2,).")
        if self.t.shape != (n_subject,):
            raise ValueError("t must have shape (n_subject,).")
        if self.Z.ndim != 2 or self.Z.shape[0] != n_subject:
            raise ValueError("Z must have shape (n_subject, p0).")
        if self.X.ndim < 3 or self.X.shape[0] != n_subject or self.X.shape[1] != 2:
            raise ValueError("X must have shape (n_subject, 2, *tensor_shape).")
        if self.y.shape != (n_subject, 2):
            raise ValueError("y must have shape (n_subject, 2).")
        if self.Sigma_true is not None:
            self.Sigma_true = np.asarray(self.Sigma_true, dtype=float)
            if self.Sigma_true.shape != (2, 2) and self.Sigma_true.shape != (n_subject, 2, 2):
                raise ValueError("Sigma_true must have shape (2, 2) or (n_subject, 2, 2).")

    @property
    def n_subject(self) -> int:
        """Return the number of subjects."""

        return int(self.subject_ids.shape[0])

    def to_iid_observations(self) -> FlattenedIIDView:
        """Return an observation-level iid view without mutating the dataset."""

        n_subject = self.n_subject
        X_flat = self.X.reshape((n_subject * 2, *self.X.shape[2:]))
        y_flat = self.y.reshape(n_subject * 2)
        t_flat = np.repeat(self.t, 2)
        Z_flat = np.repeat(self.Z, 2, axis=0)
        subject_ids_flat = np.repeat(self.subject_ids, 2)
        eye_ids_flat = np.tile(self.eye_ids, n_subject)
        return FlattenedIIDView(
            X=X_flat,
            y=y_flat,
            t=t_flat,
            Z=Z_flat,
            subject_ids=subject_ids_flat,
            eye_ids=eye_ids_flat,
        )
