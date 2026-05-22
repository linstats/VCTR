"""Paired-eye Case 1 DGP in reduced-feature space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import PairedEyeDataset

from .base import BasePairedDGP


@dataclass(slots=True)
class PairedCase1DGP(BasePairedDGP):
    """Paired-eye analogue of the archived iid Case 1 reduced-feature DGP."""

    n_subject: int = 2000
    R: int = 10
    S: int = 16
    p0: int = 2
    coef_type: str = "quadratic"
    beta_true: tuple[float, ...] | None = None
    sigma2: float = 1.0
    rho: float = 0.3
    eye_ids: tuple[int, int] = (0, 1)

    def sample(self, seed: int | None = None) -> PairedEyeDataset:
        """Generate one paired-eye Case 1 dataset."""

        self._validate_parameters()
        rng = np.random.default_rng(seed)

        subject_ids = np.arange(self.n_subject)
        eye_ids = np.asarray(self.eye_ids)

        t = np.sort(rng.uniform(size=self.n_subject))
        Z = rng.normal(size=(self.n_subject, self.p0))
        X = rng.normal(size=(self.n_subject, 2, self.R, self.S))

        bR = np.sqrt(np.arange(1, self.R + 1) / self.R)
        bS = np.sqrt(np.arange(1, self.S + 1) / self.S)
        base = self._coefficient_base(t)
        A_true = base[:, None, None] * bR[None, :, None] * bS[None, None, :]

        beta_true = self._resolve_beta_true()
        signal = np.sum(X * A_true[:, None, :, :], axis=(2, 3)) + Z @ beta_true[:, None]

        Sigma_true = self.sigma2 * np.array(
            [[1.0, self.rho], [self.rho, 1.0]],
            dtype=float,
        )
        noise = rng.multivariate_normal(
            mean=np.zeros(2, dtype=float),
            cov=Sigma_true,
            size=self.n_subject,
        )
        y = signal + noise

        return PairedEyeDataset(
            subject_ids=subject_ids,
            eye_ids=eye_ids,
            t=t,
            X=X,
            Z=Z,
            y=y,
            A_true=A_true,
            beta_true=beta_true,
            Sigma_true=Sigma_true,
            meta={
                "dgp": "paired_case1",
                "case": "case1",
                "n_subject": self.n_subject,
                "R": self.R,
                "S": self.S,
                "p0": self.p0,
                "coef_type": self.coef_type,
                "beta_true": beta_true.tolist(),
                "sigma2": self.sigma2,
                "rho": self.rho,
            },
        )

    def _validate_parameters(self) -> None:
        """Validate Case 1 paired DGP parameters."""

        if self.n_subject <= 0:
            raise ValueError("n_subject must be positive.")
        if self.R <= 0 or self.S <= 0 or self.p0 <= 0:
            raise ValueError("R, S, and p0 must all be positive.")
        if self.sigma2 <= 0:
            raise ValueError("sigma2 must be positive.")
        if not (-1.0 < self.rho < 1.0):
            raise ValueError("rho must lie strictly between -1 and 1.")
        if len(self.eye_ids) != 2:
            raise ValueError("eye_ids must contain exactly two eye labels.")
        self._resolve_beta_true()
        self._coefficient_base(np.array([0.25], dtype=float))

    def _resolve_beta_true(self) -> np.ndarray:
        """Return the true beta vector for the paired Case 1 DGP.

        If no explicit vector is provided, default to the Case I paper setting
        where each of the ``p0`` scalar covariates has coefficient ``3.0``.
        """

        if self.beta_true is None:
            return np.full(self.p0, 3.0, dtype=float)

        beta_true = np.asarray(self.beta_true, dtype=float).reshape(-1)
        if beta_true.shape != (self.p0,):
            raise ValueError("beta_true must have shape (p0,).")
        return beta_true

    def _coefficient_base(self, t: np.ndarray) -> np.ndarray:
        """Return the scalar coefficient-function base used in Case 1."""

        if self.coef_type == "sqrt":
            return np.sqrt(t)
        if self.coef_type == "quadratic":
            return 4.0 * np.square(t - 0.5)
        if self.coef_type == "bump":
            return 1.75 * (
                np.exp(-np.square(3.0 * t - 1.0))
                + np.exp(-np.square(4.0 * t - 3.0))
                - 0.75
            )
        if self.coef_type == "sin":
            return np.sin(2.0 * np.pi * (t - 0.5))
        raise ValueError(
            "coef_type must be one of {'sqrt', 'quadratic', 'bump', 'sin'}."
        )
