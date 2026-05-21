"""Case 2 experiment DGP for VCTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import SimulationDataset
from src.dgps.base import BaseDGP


@dataclass(slots=True)
class Case2BaselineDGP(BaseDGP):
    """Case 2 DGP aligned with the original reduced-feature simulation setup."""

    n: int = 2000
    R: int = 5
    S: int = 64
    p0: int = 2
    coef_type: str = "quadratic"
    noise_type: str = "gaussian"
    beta_value: float = 1.0
    noise_scale: float = 1.0

    def sample(self, seed: int | None = None) -> SimulationDataset:
        """Generate one Case 2 reduced-feature dataset."""

        rng = np.random.default_rng(seed)

        t = np.sort(rng.uniform(size=self.n))
        X = rng.normal(size=(self.n, self.R, self.S))
        Z = rng.normal(size=(self.n, self.p0))

        bR = np.sqrt(np.arange(1, self.R + 1) / self.R)
        bS = np.sqrt(np.arange(1, self.S + 1) / self.S)
        base = self._coefficient_base(t)
        A_true = base[:, None, None] * bR[None, :, None] * bS[None, None, :]

        beta_true = np.full(self.p0, self.beta_value, dtype=float)
        signal = np.sum(X * A_true, axis=(1, 2)) + Z @ beta_true
        noise = self._sample_noise(rng)
        y = signal + noise

        return SimulationDataset(
            t=t,
            X=X,
            Z=Z,
            y=y,
            A_true=A_true,
            beta_true=beta_true,
            meta={
                "dgp": "case2_baseline",
                "case": "case2",
                "n": self.n,
                "R": self.R,
                "S": self.S,
                "p0": self.p0,
                "coef_type": self.coef_type,
                "noise_type": self.noise_type,
                "beta_value": self.beta_value,
                "noise_scale": self.noise_scale,
            },
        )

    def _coefficient_base(self, t: np.ndarray) -> np.ndarray:
        """Return the scalar coefficient-function base used in Case 2."""

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

    def _sample_noise(self, rng: np.random.Generator) -> np.ndarray:
        """Sample the Case 2 noise term."""

        if self.noise_type == "gaussian":
            return rng.normal(scale=self.noise_scale, size=self.n)
        if self.noise_type == "heavy_tailed":
            return self.noise_scale * rng.standard_t(df=5, size=self.n)
        raise ValueError("noise_type must be one of {'gaussian', 'heavy_tailed'}.")
