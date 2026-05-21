"""Case 4 experiment DGP for sparse VCTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data import SimulationDataset
from src.dgps.base import BaseDGP


@dataclass(slots=True)
class Case4BaselineDGP(BaseDGP):
    """Case 4 DGP aligned with the MATLAB high-dimensional simulation."""

    n: int = 2000
    R: int = 20
    S: int = 64
    p0: int = 5
    sp: int = 10
    noise_scale: float = 1.0

    def sample(self, seed: int | None = None) -> SimulationDataset:
        """Generate one Case 4 reduced-feature dataset."""

        if self.sp > min(self.R, self.S):
            raise ValueError("sp must be no larger than min(R, S).")
        if self.p0 < 5:
            raise ValueError("Case 4 expects p0 >= 5.")

        rng = np.random.default_rng(seed)

        t = np.sort(rng.uniform(size=self.n))
        X = rng.normal(size=(self.n, self.R, self.S))
        Z = rng.normal(size=(self.n, self.p0))

        beta_true = np.zeros(self.p0, dtype=float)
        beta_true[:2] = 1.0

        A_true, varying_mask, const_nonzero_mask, const_zero_mask = self._build_coefficients(t)

        signal = np.sum(X * A_true, axis=(1, 2)) + Z @ beta_true
        y = signal + rng.normal(scale=self.noise_scale, size=self.n)

        return SimulationDataset(
            t=t,
            X=X,
            Z=Z,
            y=y,
            A_true=A_true,
            beta_true=beta_true,
            meta={
                "dgp": "case4_baseline",
                "case": "case4",
                "n": self.n,
                "R": self.R,
                "S": self.S,
                "p0": self.p0,
                "sp": self.sp,
                "noise_scale": self.noise_scale,
                "varying_mask_true": varying_mask,
                "const_nonzero_mask_true": const_nonzero_mask,
                "const_zero_mask_true": const_zero_mask,
                "beta_nonzero_mask_true": beta_true != 0.0,
            },
        )

    def _build_coefficients(
        self,
        t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct the true coefficient process and structure masks."""

        bR = np.sqrt(np.arange(1, self.R + 1) / self.R)
        bS = np.sqrt(np.arange(1, self.S + 1) / self.S)

        A_true = np.zeros((self.n, self.R, self.S), dtype=float)
        varying_mask = np.zeros((self.R, self.S), dtype=bool)
        const_nonzero_mask = np.zeros((self.R, self.S), dtype=bool)

        for r in range(self.sp):
            for s in range(self.sp):
                scale = bR[r] * bS[s]
                if s < r:
                    A_true[:, r, s] = np.sin(2.0 * np.pi * (t - 0.5)) * scale
                    varying_mask[r, s] = True
                else:
                    A_true[:, r, s] = scale
                    const_nonzero_mask[r, s] = True

        const_zero_mask = ~(varying_mask | const_nonzero_mask)
        return A_true, varying_mask, const_nonzero_mask, const_zero_mask

