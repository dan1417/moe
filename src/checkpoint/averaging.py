"""Pre-decay checkpoint averaging (model soup) for WSD stable phase."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.tensor import DTensor

from src.checkpoint.dcp import is_committed_checkpoint
from src.optim.scheduler import WSDScheduler

log = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)$")


def average_checkpoints(
    checkpoint_paths: list[str | Path],
    output_path: str | Path,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Linearly average legacy ``model.pt`` checkpoint state dicts on CPU."""
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must not be empty")
    averaged: dict[str, torch.Tensor] | None = None
    for path in checkpoint_paths:
        model_path = Path(path) / "model.pt"
        if not model_path.exists():
            model_path = Path(path)
        state = torch.load(model_path, map_location=device, weights_only=True)
        if averaged is None:
            averaged = {key: value.to(dtype=dtype).clone() for key, value in state.items()}
            continue
        if state.keys() != averaged.keys():
            missing = sorted(averaged.keys() - state.keys())
            unexpected = sorted(state.keys() - averaged.keys())
            raise RuntimeError(
                f"Checkpoint key mismatch while averaging: missing={missing[:10]} "
                f"unexpected={unexpected[:10]}"
            )
        for key, value in state.items():
            if value.shape != averaged[key].shape:
                raise RuntimeError(
                    f"Checkpoint shape mismatch for {key}: "
                    f"{tuple(value.shape)} != {tuple(averaged[key].shape)}"
                )
            averaged[key].add_(value.to(dtype=dtype))
    assert averaged is not None
    count = float(len(checkpoint_paths))
    for key in averaged:
        averaged[key].div_(count)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged, output)
    return averaged


def collect_stable_phase_checkpoints(
    output_dir: str | Path,
    scheduler: WSDScheduler,
    max_n: int = 5,
    min_step_gap: int = 50,
) -> list[str]:
    """Collect recent stable-phase checkpoint directories."""
    root = Path(output_dir)
    if not root.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.endswith("_outer"):
            continue
        match = CHECKPOINT_PATTERN.match(path.name)
        if not match:
            continue
        if not is_committed_checkpoint(path) and not (path / "model.pt").exists():
            continue
        step = int(match.group(1))
        phase = scheduler._phase_for_step(step)
        if phase != "stable":
            continue
        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("phase") not in (None, "stable"):
                continue
        candidates.append((step, path))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        return []
    selected: list[tuple[int, Path]] = [candidates[-1]]
    for step, path in reversed(candidates[:-1]):
        if len(selected) >= max_n:
            break
        if selected[-1][0] - step >= min_step_gap:
            selected.append((step, path))
    selected.reverse()
    return [str(path) for _, path in selected]


def _average_dcp_into_model(
    model: torch.nn.Module,
    paths: list[str],
    process_group: dist.ProcessGroup,
) -> None:
    """Average committed DCP model shards into the live FSDP2 parameters."""
    accum: dict[str, torch.Tensor] = {}
    options = StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True)
    for path in paths:
        model_state = get_model_state_dict(model, options=options)
        dcp.load(
            {"model": model_state},
            checkpoint_id=Path(path) / "dcp",
            process_group=process_group,
        )
        set_model_state_dict(model, model_state, options=options)
        with torch.no_grad():
            for name, param in model.named_parameters():
                local = param.to_local() if isinstance(param, DTensor) else param.data
                if name not in accum:
                    accum[name] = local.detach().float().clone()
                else:
                    accum[name].add_(local.detach().float())
    scale = 1.0 / float(len(paths))
    with torch.no_grad():
        for name, param in model.named_parameters():
            local = param.to_local() if isinstance(param, DTensor) else param.data
            local.copy_((accum[name] * scale).to(dtype=local.dtype))


def run_pre_decay_soup(
    model: torch.nn.Module,
    output_dir: str | Path,
    scheduler: WSDScheduler,
    rank: int,
    n_checkpoints: int = 5,
    min_step_gap: int = 50,
    process_group: dist.ProcessGroup | None = None,
) -> None:
    """Average stable checkpoints, load into model, and trigger decay."""
    paths = collect_stable_phase_checkpoints(
        output_dir=output_dir,
        scheduler=scheduler,
        max_n=n_checkpoints,
        min_step_gap=min_step_gap,
    )
    if not paths:
        if rank == 0:
            log.warning("No stable checkpoints found for pre-decay soup")
        scheduler.trigger_decay()
        return

    group = process_group if process_group is not None else dist.group.WORLD
    if all(is_committed_checkpoint(path) for path in paths):
        if rank == 0:
            log.info("Averaging %d stable DCP checkpoints for pre-decay soup", len(paths))
        _average_dcp_into_model(model, paths, group)
    else:
        soup_path = Path(output_dir) / "pre_decay_soup.pt"
        if rank == 0:
            log.info("Averaging %d legacy stable checkpoints for pre-decay soup", len(paths))
            averaged = average_checkpoints(paths, soup_path)
        else:
            averaged = {}
        dist.barrier(group=group)
        from src.model import load_state_dict_into_fsdp

        load_state_dict_into_fsdp(model, averaged)
    dist.barrier(group=group)
    scheduler.trigger_decay()
    if rank == 0:
        log.info("Loaded pre-decay soup and triggered decay phase")
