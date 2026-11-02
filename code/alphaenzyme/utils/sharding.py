"""Contiguous rank assignment for offline evaluation datasets."""

from __future__ import annotations


def partition_indices(sample_count: int, gpu_id: int, world_size: int, rank: int | None = None) -> tuple[int, int, int]:
    sample_count = int(sample_count)
    gpu_id = int(gpu_id)
    world_size = int(world_size)
    if sample_count < 0:
        raise ValueError(f"sample_count must be nonnegative, got {sample_count}")
    if world_size < 1:
        raise ValueError(f"world_size must be positive, got {world_size}")

    resolved_rank = int(rank) if rank is not None else (gpu_id if world_size > 1 else 0)
    if resolved_rank < 0 or resolved_rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {resolved_rank}")

    per_rank = sample_count // world_size
    start = resolved_rank * per_rank
    end = start + per_rank if resolved_rank < world_size - 1 else sample_count
    return resolved_rank, start, end
