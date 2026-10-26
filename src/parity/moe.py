"""Grouped MiMo MoE matching the validator's ``grouped_mimo_moe`` numerics.

The validator runs each active expert's three projections as independent bf16 GEMMs
(``torch._foreach_mm``), scales the routed rows by fp32 router weights, accumulates
with ``index_add_`` into an fp32 buffer and casts back to bf16. This module performs
the same operations in the same order, but issues each projection as a single
``torch._grouped_mm`` over stacked expert weights (per-row results are identical to
per-expert GEMMs) and avoids host syncs so pipeline stages never stall the CPU.
"""

from __future__ import annotations

import types

import torch
from torch import nn

COMBINE_CHUNK_ROWS = 16384


def stack_expert_weights(module: nn.Module) -> None:
    """Re-home every expert projection as a view into stacked [E, out, in] tensors."""
    experts = module.experts
    stacked = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        first = getattr(experts[0], proj).weight
        buffer = torch.empty(
            (len(experts), *first.shape), dtype=first.dtype, device=first.device
        )
        for index, expert in enumerate(experts):
            buffer[index].copy_(getattr(expert, proj).weight)
            getattr(expert, proj).weight = nn.Parameter(buffer[index], requires_grad=False)
        stacked[proj] = buffer
    module._parity_stacked = stacked


def grouped_moe(
    module: nn.Module,
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    stacked = module._parity_stacked
    n_tokens = hidden_states.shape[0]
    top_k = topk_indices.shape[1]
    n_experts = stacked["gate_proj"].shape[0]

    flat_experts = topk_indices.reshape(-1)
    flat_tokens = torch.arange(n_tokens, device=hidden_states.device).repeat_interleave(top_k)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_weights = topk_weights.reshape(-1)[order]

    counts = torch.zeros(n_experts, dtype=torch.int64, device=hidden_states.device)
    counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts))
    offsets = torch.cumsum(counts, 0).to(torch.int32)

    routed_inputs = hidden_states[sorted_tokens]
    gate = torch._grouped_mm(routed_inputs, stacked["gate_proj"].transpose(1, 2), offs=offsets)
    up = torch._grouped_mm(routed_inputs, stacked["up_proj"].transpose(1, 2), offs=offsets)
    del routed_inputs
    activated = module.experts[0].act_fn(gate) * up
    del gate, up
    routed = torch._grouped_mm(activated, stacked["down_proj"].transpose(1, 2), offs=offsets)
    del activated

    # bf16 rows * fp32 router weights -> fp32, accumulated with index_add_ exactly as
    # the validator does; chunked only to bound the fp32 temporary.
    final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
    for start in range(0, routed.shape[0], COMBINE_CHUNK_ROWS):
        end = start + COMBINE_CHUNK_ROWS
        final_hidden_states.index_add_(
            0, sorted_tokens[start:end], routed[start:end] * sorted_weights[start:end].unsqueeze(-1)
        )
    return final_hidden_states.type(hidden_states.dtype)


def enable_grouped_moe(model: nn.Module) -> int:
    """Stack expert weights and route every MiMoV2MoE through :func:`grouped_moe`."""
    patched = 0
    for module in model.modules():
        if module.__class__.__name__ != "MiMoV2MoE":
            continue
        if not hasattr(module, "_parity_stacked"):
            stack_expert_weights(module)
        module.moe = types.MethodType(grouped_moe, module)
        patched += 1
    return patched
