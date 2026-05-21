"""Kruskal-style tensor-regression initialization for sparse VCTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class KruskalInitResult:
    """Initialization result returned by :class:`KruskalRegInitializer`."""

    gamma_hat: np.ndarray
    beta_hat: np.ndarray


@dataclass(slots=True)
class KruskalRegInitializer:
    """Approximate MATLAB ``kruskal_reg(..., r=2, 'normal')`` initialization.

    This module isolates the rank-r Kruskal tensor-regression style
    initialization used by the sparse VCTR estimator. It is not a full
    port of TensorReg, but it follows the same role in the MATLAB pipeline:

    1. start from a full coefficient tensor,
    2. alternate between beta updates and low-rank tensor approximation,
    3. return a rank-r tensor coefficient initializer and refit beta.
    """

    rank: int = 2
    ridge: float = 1e-4
    max_iter: int = 50
    replicates: int = 3
    seed: int = 0
    tol: float = 1e-5

    def initialize(
        self,
        block_designs: list[np.ndarray],
        Z: np.ndarray,
        y: np.ndarray,
        gamma_full: np.ndarray,
        beta_init: np.ndarray,
    ) -> KruskalInitResult:
        """Return Kruskal-style initial ``gamma`` and ``beta``."""

        gamma_old = np.asarray(gamma_full, dtype=float)
        beta_old = np.asarray(beta_init, dtype=float)

        for _ in range(max(1, self.max_iter)):
            signal_old = self.signal_from_gamma(block_designs, gamma_old)
            beta_new = self.refit_beta(
                block_designs=block_designs,
                gamma_hat=gamma_old,
                Z=Z,
                y=y,
                p0=beta_old.shape[0],
            )

            y_resid = y - Z @ beta_new
            full_design = np.concatenate(block_designs, axis=1)
            lhs = full_design.T @ full_design + self.ridge * np.eye(full_design.shape[1])
            rhs = full_design.T @ y_resid
            gamma_vec = np.linalg.solve(lhs, rhs)
            gamma_full_new = gamma_vec.reshape(gamma_old.shape, order="F")
            gamma_new = self.cp_rank_r_approximation(gamma_full_new)

            signal_new = self.signal_from_gamma(block_designs, gamma_new)
            diff = np.sqrt(np.mean((signal_new - signal_old) ** 2))
            gamma_old = gamma_new
            beta_old = beta_new
            if diff <= self.tol:
                break

        return KruskalInitResult(gamma_hat=gamma_old, beta_hat=beta_old)

    def refit_beta(
        self,
        *,
        block_designs: list[np.ndarray],
        gamma_hat: np.ndarray,
        Z: np.ndarray,
        y: np.ndarray,
        p0: int,
    ) -> np.ndarray:
        """Refit beta conditional on a tensor coefficient estimate."""

        signal = self.signal_from_gamma(block_designs, gamma_hat)
        lhs = Z.T @ Z + self.ridge * np.eye(p0)
        rhs = Z.T @ (y - signal)
        return np.linalg.solve(lhs, rhs)

    def signal_from_gamma(
        self,
        block_designs: list[np.ndarray],
        gamma_hat: np.ndarray,
    ) -> np.ndarray:
        """Compute the tensor signal implied by ``gamma_hat``."""

        signal = np.zeros(block_designs[0].shape[0], dtype=float)
        for s_idx, design_block in enumerate(block_designs):
            signal += design_block @ gamma_hat[:, :, s_idx].reshape(-1, order="F")
        return signal

    def cp_rank_r_approximation(self, tensor: np.ndarray) -> np.ndarray:
        """Return a rank-r CP approximation of ``tensor``."""

        if tensor.ndim <= 1 or self.rank <= 0:
            return np.asarray(tensor, dtype=float)

        best_fit = np.asarray(tensor, dtype=float)
        best_error = np.inf
        rng = np.random.default_rng(self.seed)

        for rep in range(max(1, self.replicates)):
            if rep == 0:
                factors = self._hosvd_init_factors(tensor)
            else:
                factors = [
                    rng.normal(scale=0.1, size=(dim, self.rank))
                    for dim in tensor.shape
                ]
            lambdas = np.ones(self.rank, dtype=float)

            prev_error = np.inf
            for _ in range(max(1, self.max_iter)):
                for mode in range(tensor.ndim):
                    unfolded = self._unfold_tensor(tensor, mode)
                    kr = self._khatri_rao_except(factors, mode)
                    gram = np.ones((self.rank, self.rank), dtype=float)
                    for other_mode, factor in enumerate(factors):
                        if other_mode == mode:
                            continue
                        gram *= factor.T @ factor
                    gram += self.ridge * np.eye(self.rank)
                    factors[mode] = (unfolded @ kr) @ np.linalg.inv(gram)

                factors, lambdas = self._normalize_cp_factors(factors)
                approx = self._reconstruct_cp_tensor(lambdas, factors)
                error = float(np.linalg.norm(tensor - approx))
                if abs(prev_error - error) <= self.tol * (prev_error + 1.0):
                    break
                prev_error = error

            approx = self._reconstruct_cp_tensor(lambdas, factors)
            error = float(np.linalg.norm(tensor - approx))
            if error < best_error:
                best_error = error
                best_fit = approx

        return best_fit

    def _unfold_tensor(self, tensor: np.ndarray, mode: int) -> np.ndarray:
        """Mode-``mode`` unfolding."""

        moved = np.moveaxis(tensor, mode, 0)
        return moved.reshape(moved.shape[0], -1)

    def _khatri_rao_except(
        self,
        factors: list[np.ndarray],
        skip_mode: int,
    ) -> np.ndarray:
        """Column-wise Kronecker product over all modes except ``skip_mode``."""

        matrices = [factors[idx] for idx in range(len(factors)) if idx != skip_mode]
        result = matrices[-1]
        for matrix in reversed(matrices[:-1]):
            result = self._khatri_rao_two(matrix, result)
        return result

    def _khatri_rao_two(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Column-wise Kronecker product of two factor matrices."""

        if left.shape[1] != right.shape[1]:
            raise ValueError("Khatri-Rao factors must have the same number of columns.")
        rank = left.shape[1]
        out = np.zeros((left.shape[0] * right.shape[0], rank), dtype=float)
        for col in range(rank):
            out[:, col] = np.kron(left[:, col], right[:, col])
        return out

    def _normalize_cp_factors(
        self,
        factors: list[np.ndarray],
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Normalize CP columns and collect weights."""

        rank = factors[0].shape[1]
        lambdas = np.ones(rank, dtype=float)
        normalized: list[np.ndarray] = []
        for factor in factors:
            factor_new = factor.copy()
            for col in range(rank):
                norm = np.linalg.norm(factor_new[:, col])
                if norm > 0:
                    factor_new[:, col] /= norm
                    lambdas[col] *= norm
            normalized.append(factor_new)
        return normalized, lambdas

    def _hosvd_init_factors(self, tensor: np.ndarray) -> list[np.ndarray]:
        """Initialize CP factors from leading singular vectors of unfoldings."""

        factors: list[np.ndarray] = []
        for mode in range(tensor.ndim):
            unfolded = self._unfold_tensor(tensor, mode)
            u, _, _ = np.linalg.svd(unfolded, full_matrices=False)
            rank = min(self.rank, u.shape[1])
            factor = np.zeros((u.shape[0], self.rank), dtype=float)
            factor[:, :rank] = u[:, :rank]
            factors.append(factor)
        return factors

    def _reconstruct_cp_tensor(
        self,
        lambdas: np.ndarray,
        factors: list[np.ndarray],
    ) -> np.ndarray:
        """Reconstruct a tensor from CP weights and factors."""

        shape = tuple(f.shape[0] for f in factors)
        out = np.zeros(shape, dtype=float)
        rank = lambdas.shape[0]
        for col in range(rank):
            component = lambdas[col]
            for factor in factors:
                component = np.multiply.outer(component, factor[:, col])
            out += component
        return out
