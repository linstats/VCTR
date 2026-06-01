"""Paired-eye Case 1 altbase DGP in reduced-feature space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import PairedEyeDataset

from .base import BasePairedDGP
from .variance_functions import sigma2_curve


@dataclass(slots=True)
class PairedCase1AltbaseDGP(BasePairedDGP):
    """Paired-eye reduced-feature DGP with alternative Case 1 bases."""

    n_subject: int = 1000
    R: int = 4
    S: int = 25
    p0: int = 4
    coef_type: str = "base1"
    beta_true: tuple[float, ...] | None = None
    sigma2: float = 1.0
    sigma2_function: str = "constant"
    rho: float = 0.3
    eye_ids: tuple[int, int] = (0, 1)

    def sample(self, seed: int | None = None) -> PairedEyeDataset:
        """Generate one paired-eye alternative-base Case 1 dataset."""

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

        sigma2_true_t = sigma2_curve(t, base=self.sigma2, kind=self.sigma2_function)
        exchangeable_base = np.array([[1.0, self.rho], [self.rho, 1.0]], dtype=float)
        Sigma_true = sigma2_true_t[:, None, None] * exchangeable_base[None, :, :]
        noise = np.empty((self.n_subject, 2), dtype=float)
        for subject_idx in range(self.n_subject):
            noise[subject_idx] = rng.multivariate_normal(
                mean=np.zeros(2, dtype=float),
                cov=Sigma_true[subject_idx],
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
                "dgp": "paired_case1_altbase",
                "case": "case1_altbase",
                "n_subject": self.n_subject,
                "R": self.R,
                "S": self.S,
                "p0": self.p0,
                "coef_type": self.coef_type,
                "beta_true": beta_true.tolist(),
                "sigma2": self.sigma2,
                "sigma2_base": self.sigma2,
                "sigma2_function": self.sigma2_function,
                "sigma2_true_t": sigma2_true_t.tolist(),
                "rho": self.rho,
                "raw_equivalent_p1": 60,
                "raw_equivalent_p2": 60,
                "raw_equivalent_p1_prime": 12,
                "raw_equivalent_p2_prime": 12,
            },
        )

    def _validate_parameters(self) -> None:
        """Validate alternative-base Case 1 paired DGP parameters."""

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
        sigma2_curve(np.array([0.25], dtype=float), base=self.sigma2, kind=self.sigma2_function)

    def _resolve_beta_true(self) -> np.ndarray:
        """Return the true beta vector for the altbase paired Case 1 DGP."""

        if self.beta_true is None:
            return np.array([2.0, 1.0, -1.0, 0.5], dtype=float)

        beta_true = np.asarray(self.beta_true, dtype=float).reshape(-1)
        if beta_true.shape != (self.p0,):
            raise ValueError("beta_true must have shape (p0,).")
        return beta_true

    def _coefficient_base(self, t: np.ndarray) -> np.ndarray:
        """Return the scalar coefficient-function base used in altbase Case 1."""

        base1 = 5.0 * np.square(t - 0.2)
        base2 = np.exp(-np.square(3.0 * t - 1.0)) - 0.75
        base3 = np.sin(2.0 * np.pi * (t - 0.5))
        if self.coef_type == "base1":
            return base1
        if self.coef_type == "base2":
            return base2
        if self.coef_type == "base3":
            return base3
        if self.coef_type == "base4":
            return 0.45 * base1 + 0.35 * base2 + 0.20 * base3
        raise ValueError("coef_type must be one of {'base1', 'base2', 'base3', 'base4'}.")
