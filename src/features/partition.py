"""Tensor partition helpers for paired-eye VCTR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PartitionSpec:
    """Describe the number of blocks along each tensor mode."""

    blocks_per_mode: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.blocks_per_mode:
            raise ValueError("blocks_per_mode must not be empty.")
        if any(block <= 0 for block in self.blocks_per_mode):
            raise ValueError("All block counts must be positive.")


def partition_tensor_blocks(X: np.ndarray, spec: PartitionSpec) -> list[np.ndarray]:
    """Split a tensor batch into spatial blocks.

    Parameters
    ----------
    X:
        Tensor batch with shape ``(n_samples, d1, ..., dK)``. The first axis is
        the sample axis and is never partitioned.
    spec:
        Number of blocks for each non-sample tensor mode.

    Returns
    -------
    list[np.ndarray]
        Blocks ordered with the first tensor mode varying fastest. For images
        with ``blocks_per_mode=(3, 3, 1)``, this matches the inherited MATLAB
        ordering ``s = (s2 - 1) * S1 + s1``.
    """

    X = np.asarray(X)
    if X.ndim != len(spec.blocks_per_mode) + 1:
        raise ValueError(
            "X must have one sample axis plus one tensor axis per block count; "
            f"got X.ndim={X.ndim} and blocks_per_mode={spec.blocks_per_mode}."
        )

    tensor_shape = X.shape[1:]
    for mode_size, n_blocks in zip(tensor_shape, spec.blocks_per_mode, strict=True):
        if mode_size % n_blocks != 0:
            raise ValueError(
                f"Tensor mode size {mode_size} is not divisible by requested block count {n_blocks}."
            )

    block_sizes = tuple(mode_size // n_blocks for mode_size, n_blocks in zip(tensor_shape, spec.blocks_per_mode))
    blocks: list[np.ndarray] = []
    for block_indices in np.ndindex(*spec.blocks_per_mode[::-1]):
        # np.ndindex varies the last provided axis fastest. Reversing here and
        # then reversing back gives MATLAB-style order: mode 1 fastest.
        indices = block_indices[::-1]
        slices = [slice(None)]
        for block_idx, block_size in zip(indices, block_sizes, strict=True):
            start = block_idx * block_size
            stop = start + block_size
            slices.append(slice(start, stop))
        blocks.append(X[tuple(slices)])
    return blocks
