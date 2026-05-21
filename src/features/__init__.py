"""Feature extraction and projection modules for VCTR."""
"""Feature construction helpers for paired-eye VCTR."""

from .cp_projection import CPProjectionConfig, blockwise_cp_project
from .partition import PartitionSpec, partition_tensor_blocks

__all__ = [
    "CPProjectionConfig",
    "PartitionSpec",
    "blockwise_cp_project",
    "partition_tensor_blocks",
]
