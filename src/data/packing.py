"""Document-boundary-aware sequence packing for variable-length attention."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from src.data.dataset import CollatedBatch

log = logging.getLogger(__name__)

IGNORE_INDEX = -100


def find_document_boundaries(token_ids: Tensor, eos_id: int) -> Tensor:
    """Return cumulative document lengths ``cu_seqlens`` for one sequence."""
    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must be 1D, got shape {tuple(token_ids.shape)}")
    seq_len = int(token_ids.numel())
    if seq_len == 0:
        return torch.zeros(1, dtype=torch.int32, device=token_ids.device)

    eos_positions = (token_ids == eos_id).nonzero(as_tuple=False).flatten()
    if eos_positions.numel() == 0:
        return torch.tensor([0, seq_len], dtype=torch.int32, device=token_ids.device)

    boundaries = [0]
    start = 0
    for pos_tensor in eos_positions:
        pos = int(pos_tensor.item()) + 1
        if pos > start:
            boundaries.append(pos)
            start = pos
    if boundaries[-1] != seq_len:
        boundaries.append(seq_len)
    return torch.tensor(boundaries, dtype=torch.int32, device=token_ids.device)


def _build_position_ids(cu_seqlens: Tensor) -> Tensor:
    position_chunks = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        length = int(end.item() - start.item())
        position_chunks.append(torch.arange(length, dtype=torch.long))
    if not position_chunks:
        return torch.zeros(0, dtype=torch.long)
    return torch.cat(position_chunks, dim=0)


def build_document_attention_mask(
    cu_seqlens: Tensor,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Build a per-row document + causal mask of shape ``(1, seq_len, seq_len)``."""
    mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device, dtype=dtype)
    cu = cu_seqlens.to(device=device)
    for start, end in zip(cu[:-1], cu[1:]):
        s, e = int(start.item()), int(end.item())
        doc_len = e - s
        if doc_len <= 0:
            continue
        causal = torch.triu(
            torch.ones(doc_len, doc_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        block = torch.zeros(doc_len, doc_len, device=device, dtype=dtype)
        block.masked_fill_(causal, torch.finfo(dtype).min)
        mask[s:e, s:e] = block
    return mask.unsqueeze(0)


def build_batch_document_attention_mask(
    input_ids: Tensor,
    eos_id: int,
    dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """Build ``(batch, 1, seq_len, seq_len)`` document-boundary masks for a batch."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    masks: list[Tensor] = []
    for row in range(batch_size):
        cu = find_document_boundaries(input_ids[row], eos_id)
        masks.append(build_document_attention_mask(cu, seq_len, device, dtype=dtype))
    return torch.cat(masks, dim=0).unsqueeze(1)


def pack_batch_with_masking(batch: CollatedBatch | dict[str, Any], eos_id: int) -> dict[str, Any]:
    """Transform a batch into a document-aware packed batch."""
    if isinstance(batch, CollatedBatch):
        input_ids = batch.input_ids
        labels = batch.labels
        datasets = batch.datasets
    else:
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids.clone())
        datasets = batch.get("datasets", [])

    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids shape (B, T), got {tuple(input_ids.shape)}")

    batch_size, seq_len = input_ids.shape
    packed_input: list[Tensor] = []
    packed_labels: list[Tensor] = []
    packed_positions: list[Tensor] = []
    cu_seqlens_list: list[Tensor] = []
    running_offset = 0
    max_seqlen = 0
    for index in range(batch_size):
        seq = input_ids[index]
        seq_labels = labels[index].clone()
        cu_local = find_document_boundaries(seq, eos_id)
        eos_mask = seq == eos_id
        seq_labels[eos_mask] = IGNORE_INDEX
        positions = _build_position_ids(cu_local.to(seq.device))

        packed_input.append(seq)
        packed_labels.append(seq_labels)
        packed_positions.append(positions.to(seq.device))
        if index == 0:
            cu_seqlens_list.append(cu_local)
        else:
            cu_seqlens_list.append(cu_local[1:] + running_offset)
        running_offset += seq_len
        max_seqlen = max(max_seqlen, int((cu_local[1:] - cu_local[:-1]).max().item()))

    input_flat = torch.cat(packed_input, dim=0)
    labels_flat = torch.cat(packed_labels, dim=0)
    position_flat = torch.cat(packed_positions, dim=0)
    cu_seqlens = torch.cat(cu_seqlens_list, dim=0).to(dtype=torch.int32)
    return {
        "input_ids": input_flat.reshape(batch_size, seq_len),
        "labels": labels_flat.reshape(batch_size, seq_len),
        "position_ids": position_flat.reshape(batch_size, seq_len),
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_per_row": cu_seqlens_list,
        "max_seqlen": max_seqlen,
        "datasets": datasets,
    }


def flash_attn_varlen_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    causal: bool = True,
    softmax_scale: float | None = None,
) -> Tensor:
    """Wrapper around ``flash_attn_varlen_func`` with a dense fallback."""
    try:
        from flash_attn import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=causal,
            softmax_scale=softmax_scale,
        )
    except ImportError:
        log.warning("flash_attn not installed; using standard varlen fallback")
        return _standard_varlen_attention_fallback(q, k, v, cu_seqlens, causal)


def _standard_varlen_attention_fallback(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,
    causal: bool,
) -> Tensor:
    """Block-diagonal causal attention fallback when flash-attn is unavailable."""
    total_tokens = q.shape[0]
    head_dim = q.shape[-1]
    n_heads = q.shape[1]
    output = torch.zeros_like(q)
    scale = head_dim**-0.5
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        s = int(start.item())
        e = int(end.item())
        q_doc = q[s:e]
        k_doc = k[s:e]
        v_doc = v[s:e]
        attn = torch.einsum("thd,shd->ths", q_doc, k_doc) * scale
        if causal:
            mask = torch.triu(torch.ones(e - s, e - s, device=q.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask.unsqueeze(0), float("-inf"))
        weights = torch.softmax(attn, dim=-1)
        output[s:e] = torch.einsum("ths,shd->thd", weights, v_doc)
    return output
