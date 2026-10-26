"""Export FSDP2-sharded training weights as a submission-shaped HF checkpoint."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import nn

from src.cpt.model import checkpoint_items

log = logging.getLogger(__name__)

CONTRACT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja.txt",
    "configuration_mimo_v2.py",
    "modeling_mimo_v2.py",
)
SHARD_BYTES = 5 * 1024**3


def _full(param: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return param.full_tensor() if isinstance(param, DTensor) else param.detach()


def expected_keys(king_dir: Path, n_layers: int | None) -> set[str]:
    """King tensor names, restricted to the first ``n_layers`` layers for sliced runs."""
    keys = set(json.loads((Path(king_dir) / "model.safetensors.index.json").read_text())["weight_map"])
    if n_layers is None:
        return keys
    kept = set()
    for key in keys:
        parts = key.split(".")
        if parts[:2] == ["model", "layers"] and int(parts[2]) >= n_layers:
            continue
        kept.add(key)
    return kept


@torch.no_grad()
def export_checkpoint(
    model: nn.Module, king_dir: Path, out_dir: Path, *, rank: int, n_layers: int | None = None
) -> None:
    """All ranks must call. Rank 0 writes bf16 safetensors + index + contract files."""
    out_dir = Path(out_dir)
    if rank == 0:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    files: list[tuple[str, list[str]]] = []

    def flush() -> None:
        nonlocal pending, pending_bytes
        if rank == 0 and pending:
            filename = f"model-{len(files) + 1:05d}.safetensors"
            save_file(pending, str(out_dir / filename), metadata={"format": "pt"})
            files.append((filename, list(pending)))
        pending, pending_bytes = {}, 0

    for name, param in model.named_parameters():
        full = _full(param)  # collective: every rank participates
        if rank == 0:
            full = full.to(torch.bfloat16).cpu()
            for key, tensor in checkpoint_items(name, full):
                pending[key] = tensor.clone()  # expert slices share storage; safetensors rejects that
                pending_bytes += tensor.numel() * 2
            if pending_bytes >= SHARD_BYTES:
                flush()
        del full
    flush()
    if rank == 0:
        total = len(files)
        weight_map, size = {}, 0
        for position, (filename, keys) in enumerate(files):
            final = f"model-{position + 1:05d}-of-{total:05d}.safetensors"
            (out_dir / filename).rename(out_dir / final)
            for key in keys:
                weight_map[key] = final
            size += (out_dir / final).stat().st_size
        index = {"metadata": {"total_size": size}, "weight_map": dict(sorted(weight_map.items()))}
        (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        for name in CONTRACT_FILES:
            shutil.copyfile(Path(king_dir) / name, out_dir / name)
        king_keys = expected_keys(king_dir, n_layers)
        if set(weight_map) != king_keys:
            raise RuntimeError(
                f"exported keys differ from king: missing={sorted(king_keys - set(weight_map))[:4]} "
                f"extra={sorted(set(weight_map) - king_keys)[:4]}"
            )
        log.info("exported %d tensors to %s (%.1f GB)", len(weight_map), out_dir, size / 1e9)
    if dist.is_initialized():
        dist.barrier()
