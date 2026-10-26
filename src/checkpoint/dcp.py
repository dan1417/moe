"""FSDP2 distributed training checkpoints without rank-0 consolidation."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)

from src.optim.optimizer import CombinedOptimizer
from src.dist.process_groups import ProcessGroups

FORMAT_VERSION = 3
COMMIT_MARKER = "COMMITTED"


def _state_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=True,
    )


def _rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])


def is_committed_checkpoint(path: str | Path) -> bool:
    return (Path(path) / COMMIT_MARKER).is_file()


def save_distributed_checkpoint(
    model: torch.nn.Module,
    optimizer: CombinedOptimizer,
    scheduler: Any,
    step: int,
    output_dir: Path,
    pg: ProcessGroups,
    meta: dict[str, Any],
    *,
    suffix: str = "",
    sparse_loco: Any | None = None,
) -> Path:
    """Save sharded model/optimizer state and atomically commit the directory."""
    final_dir = output_dir / f"step_{step:07d}{suffix}"
    tmp_dir = output_dir / f".step_{step:07d}{suffix}.tmp"
    if pg.is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True)
    dist.barrier(group=pg.ep_group)

    model_state, optim_state = get_state_dict(
        model,
        (optimizer.muon, optimizer.adamw),
        options=_state_options(),
    )
    dcp.save(
        {"model": model_state, "optimizer": optim_state},
        checkpoint_id=tmp_dir / "dcp",
        process_group=pg.ep_group,
    )
    torch.save(_rng_state(), tmp_dir / f"rng_rank_{pg.global_rank:05d}.pt")
    dist.barrier(group=pg.ep_group)

    if pg.is_rank0:
        torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
        if sparse_loco is not None:
            torch.save(sparse_loco.state_dict(), tmp_dir / "sparse_loco.pt")
        payload = {
            "step": step,
            "next_step": step + 1,
            **meta,
            "checkpoint_format_version": FORMAT_VERSION,
            "torch_version": torch.__version__,
            "world_size": pg.world_size,
            "gpus_per_node": pg.ep_size,
            "num_nodes": pg.dp_size,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (tmp_dir / "meta.json").write_text(json.dumps(payload, indent=2))
        (tmp_dir / COMMIT_MARKER).write_text("ok\n")
        if final_dir.exists():
            shutil.rmtree(final_dir)
        os.replace(tmp_dir, final_dir)
    dist.barrier(group=pg.ep_group)
    return final_dir


def load_distributed_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: CombinedOptimizer,
    scheduler: Any,
    pg: ProcessGroups,
    *,
    sparse_loco: Any | None = None,
) -> dict[str, Any]:
    """Restore one committed FSDP2 checkpoint on all ranks."""
    if not is_committed_checkpoint(checkpoint_dir):
        raise RuntimeError(f"Checkpoint is not committed: {checkpoint_dir}")
    meta = json.loads((checkpoint_dir / "meta.json").read_text())
    if int(meta.get("checkpoint_format_version", 0)) != FORMAT_VERSION:
        raise RuntimeError(
            f"Expected checkpoint format {FORMAT_VERSION}, "
            f"got {meta.get('checkpoint_format_version')}"
        )
    expected = (pg.world_size, pg.ep_size, pg.dp_size)
    saved = (
        int(meta.get("world_size", -1)),
        int(meta.get("gpus_per_node", -1)),
        int(meta.get("num_nodes", -1)),
    )
    if saved != expected:
        raise RuntimeError(f"Checkpoint topology mismatch: saved={saved}, runtime={expected}")

    model_state, optim_state = get_state_dict(
        model,
        (optimizer.muon, optimizer.adamw),
        options=_state_options(),
    )
    state: dict[str, Any] = {"model": model_state, "optimizer": optim_state}
    dcp.load(
        state,
        checkpoint_id=checkpoint_dir / "dcp",
        process_group=pg.ep_group,
    )
    incompatible = set_state_dict(
        model,
        (optimizer.muon, optimizer.adamw),
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=_state_options(),
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DCP state mismatch: "
            f"missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )
    scheduler.load_state_dict(
        torch.load(checkpoint_dir / "scheduler.pt", map_location="cpu", weights_only=True)
    )
    sparse_path = checkpoint_dir / "sparse_loco.pt"
    if sparse_loco is not None and sparse_path.exists():
        sparse_loco.load_state_dict(
            torch.load(sparse_path, map_location="cpu", weights_only=True)
        )
    rng = torch.load(
        checkpoint_dir / f"rng_rank_{pg.global_rank:05d}.pt",
        map_location="cpu",
        weights_only=False,
    )
    _restore_rng_state(rng)
    return meta
