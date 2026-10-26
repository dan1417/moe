"""Distributed process group topology for Teutonic SN3 training."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch.distributed as dist

log = logging.getLogger(__name__)

GPUS_PER_NODE = 8

_PG: Optional["ProcessGroups"] = None


def _nccl_timeout() -> timedelta:
    """Match dist.init_process_group timeout; new_group defaults to 600s otherwise."""
    return timedelta(seconds=int(os.environ.get("NCCL_TIMEOUT", "1800")))


@dataclass
class ProcessGroups:
    """Central registry of distributed process groups."""

    ep_group: dist.ProcessGroup
    dp_group: dist.ProcessGroup
    ep_rank: int
    dp_rank: int
    ep_size: int
    dp_size: int
    local_rank: int
    global_rank: int
    world_size: int
    node_rank: int
    is_rank0: bool

    def __repr__(self) -> str:
        return (
            f"ProcessGroups(world_size={self.world_size}, node_rank={self.node_rank}, "
            f"ep_rank={self.ep_rank}/{self.ep_size}, dp_rank={self.dp_rank}/{self.dp_size})"
        )


def resolve_gpus_per_node(configured: int | None = None) -> int:
    """Resolve GPUs/node from env (torchrun) with optional config fallback."""
    for key in ("LOCAL_WORLD_SIZE", "GPUS_PER_NODE"):
        raw = os.environ.get(key)
        if raw is not None and raw.strip():
            value = int(raw)
            if value >= 1:
                return value
    if configured is not None and configured >= 1:
        return int(configured)
    if dist.is_initialized():
        return max(1, dist.get_world_size())
    return GPUS_PER_NODE


def init_process_groups(gpus_per_node: int | None = None) -> ProcessGroups:
    """Initialize EP/DP groups after ``dist.init_process_group``.

    Single-node: EP across all local GPUs, trivial DP size=1.
    Two-node: EP within each node, DP across matching local ranks.
    """
    global _PG

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before init_process_groups()")

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    gpus_per_node = resolve_gpus_per_node(gpus_per_node)
    local_rank = int(os.environ.get("LOCAL_RANK", global_rank % gpus_per_node))

    if world_size % gpus_per_node != 0:
        raise RuntimeError(
            f"world_size={world_size} is not divisible by gpus_per_node={gpus_per_node}"
        )

    num_nodes = world_size // gpus_per_node
    node_rank = global_rank // gpus_per_node
    ep_rank = local_rank
    dp_size = num_nodes
    dp_rank = node_rank

    timeout = _nccl_timeout()
    if num_nodes == 1:
        # Any single-node GPU count (1..N): FSDP across the full world.
        # ep_group is the FSDP process group (legacy name; not expert-parallel).
        # Must pass timeout: FA4 first-step JIT can exceed the 600s new_group default.
        ep_group = dist.new_group(list(range(world_size)), timeout=timeout)
        dp_groups = [dist.new_group([rank], timeout=timeout) for rank in range(world_size)]
        dp_group = dp_groups[global_rank]
    elif num_nodes == 2:
        ep_group_node0 = dist.new_group(list(range(gpus_per_node)), timeout=timeout)
        ep_group_node1 = dist.new_group(
            list(range(gpus_per_node, world_size)), timeout=timeout
        )
        ep_group = ep_group_node0 if node_rank == 0 else ep_group_node1
        dp_groups = [
            dist.new_group([idx, idx + gpus_per_node], timeout=timeout)
            for idx in range(gpus_per_node)
        ]
        dp_group = dp_groups[ep_rank]
    else:
        raise RuntimeError(
            f"Unsupported topology world_size={world_size} gpus_per_node={gpus_per_node} "
            f"(nodes={num_nodes}); expected 1 or 2 nodes"
        )

    pg = ProcessGroups(
        ep_group=ep_group,
        dp_group=dp_group,
        ep_rank=ep_rank,
        dp_rank=dp_rank,
        ep_size=gpus_per_node,
        dp_size=dp_size,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        node_rank=node_rank,
        is_rank0=global_rank == 0,
    )
    _PG = pg
    return pg


def get_ep_group() -> dist.ProcessGroup:
    if _PG is None:
        raise RuntimeError("Process groups not initialized")
    return _PG.ep_group


def get_dp_group() -> dist.ProcessGroup:
    if _PG is None:
        raise RuntimeError("Process groups not initialized")
    return _PG.dp_group


def get_process_groups() -> ProcessGroups:
    if _PG is None:
        raise RuntimeError("Process groups not initialized")
    return _PG


def log_topology(pg: ProcessGroups) -> None:
    """Print topology on rank 0."""
    if not pg.is_rank0:
        return
    log.info(
        "Topology: world_size=%d nodes=%d gpus_per_node=%d parallel=FSDP",
        pg.world_size,
        pg.dp_size,
        pg.ep_size,
    )
    log.info("  global_rank=%d local_rank=%d node_rank=%d", pg.global_rank, pg.local_rank, pg.node_rank)
    log.info("  ep_rank=%d/%d dp_rank=%d/%d", pg.ep_rank, pg.ep_size, pg.dp_rank, pg.dp_size)
