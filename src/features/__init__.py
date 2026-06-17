"""Feature construction helpers for paired-eye VCTR."""

from .cp_projection import BlockwiseCPResult, CPBlockResult, CPProjectionConfig, blockwise_cp_project
from .partition import PartitionSpec, partition_tensor_blocks

__all__ = [
    "BlockwiseCPResult",
    "CPBlockResult",
    "CPProjectionConfig",
    "PartitionSpec",
    "blockwise_cp_project",
    "partition_tensor_blocks",
]
