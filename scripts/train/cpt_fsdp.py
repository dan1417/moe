#!/usr/bin/env python3
"""Continued pretraining of a Teutonic-II-110B-A7B (MiMo-V2) checkpoint with PyTorch FSDP — one node, N GPUs.

    python -m torch.distributed.run --nproc_per_node 8 scripts/train/cpt_fsdp.py --init /data/ckpt/start --pool /data/pools/text_cpt_v1 \
        --val /data/pools/text_val_v1 --out /data/ckpt/run1 --steps 800 --batch 2 --accum 16 --lr 1e-5

What it does
  * rank 0 loads the HF checkpoint on CPU (bf16 or fp32 master weights), the other ranks build the same module tree on the
    meta device; FSDP FULL_SHARD broadcasts rank 0's weights (`sync_module_states`), so host RAM ≈ one model copy.
  * the 256 routed experts of every MoE layer are re-homed as three stacked parameters [E, out, in] and run through
    `torch._grouped_mm` (same numerics as the validator's per-expert GEMMs, ~3 FSDP params per layer instead of 768).
  * routers (`mlp.gate.*`) are frozen by default (the lineage keeps them fixed); everything else trains with AdamW
    (8-bit states via bitsandbytes when available, else fp32), bf16 compute, per-decoder-layer activation checkpointing.
  * loss = mean token cross-entropy over fixed 2048-token sequences (= the validator's per-sequence mean), warmup + cosine.
  * every --eval-every steps: mean CE on the held-out pool (all ranks, all-reduced); every --save-every steps and at the end:
    a validator-ready bf16 safetensors checkpoint (per-expert tensor names restored, config/tokenizer/modeling files copied),
    written layer by layer through `summon_full_params` so rank 0 never holds more than one layer.
Data: a pool directory from scripts/data/build_diverse_pool.py (tokens.npy uint32 [N, 2048]).

LoRA mode (--lora-r > 0; the H100 80 GB route, where fp32 master weights of the full model do not fit):
  * the base model stays frozen in bf16 (27.5 GB/GPU sharded); rank-r adapters (fp32 master weights, their own FSDP units,
    gradients reduced in fp32) are attached to the attention projections, the shared expert, the layer-0 dense MLP and --
    as per-expert [E, r, in] / [E, out, r] adapters driven by the same grouped GEMMs -- the routed experts (--lora-targets).
  * B starts at zero, so step 0 is exactly the start model; at save time every adapter is folded into its base weight
    (W + alpha/r * B A), so the checkpoints are the same plain bf16 validator layout as in full-parameter mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

COMBINE_CHUNK_ROWS = 16384
INPLACE_COMBINE = False  # --inplace-combine
FUSE_GATE_UP = False  # --fuse-gate-up
FUSED_CE = False  # --fused-ce
COMBINE = "chunked"  # --combine: chunked (fp32 index_add per 16 384 rows, the validator's numerics) | bmm
EXPERT_RE = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(gate_proj|up_proj|down_proj)$")
# files of the init model that are not part of a checkpoint (identity markers of the ingested king, our own markers)
NOT_COPIED = ("model.safetensors.index.json", ".gitattributes", "README.md", "SAVED", "king_info.json", "INGESTED",
              "UNPERMUTED.json", "PERM.json", "STAGED", "manifest.json")


# ----------------------------------------------------------------------------- grouped experts
class GroupedExperts(nn.Module):
    """The routed experts of one MoE layer as three stacked parameters; forward = grouped GEMMs."""

    def __init__(self, experts: nn.ModuleList):
        super().__init__()
        self.act_fn = experts[0].act_fn
        self.n_experts = len(experts)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            first = getattr(experts[0], proj).weight
            if first.device.type == "meta":
                w = torch.empty((len(experts), *first.shape), dtype=first.dtype, device="meta")
            else:
                w = torch.empty((len(experts), *first.shape), dtype=first.dtype)
                for i, e in enumerate(experts):
                    w[i].copy_(getattr(e, proj).weight.detach())
            setattr(self, proj, nn.Parameter(w))
        self.lora_gate = self.lora_up = self.lora_down = None
        self.lora_scaling = 0.0

    def forward(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        n_tokens, top_k = hidden_states.shape[0], topk_indices.shape[1]
        flat_experts = topk_indices.reshape(-1)
        flat_tokens = torch.arange(n_tokens, device=hidden_states.device).repeat_interleave(top_k)
        order = torch.argsort(flat_experts, stable=True)
        sorted_experts, sorted_tokens = flat_experts[order], flat_tokens[order]
        sorted_weights = topk_weights.reshape(-1)[order]
        counts = torch.zeros(self.n_experts, dtype=torch.int64, device=hidden_states.device)
        counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts))
        offsets = torch.cumsum(counts, 0).to(torch.int32)
        routed_inputs = hidden_states[sorted_tokens]
        if FUSE_GATE_UP and self.lora_gate is None:
            # one grouped GEMM over [E, 2I, H] (N = 2I per expert): one pass over the routed inputs, and one input
            # gradient in backward instead of two plus an add over [N*k, H]
            inter = self.gate_proj.shape[1]
            gu = torch._grouped_mm(routed_inputs, torch.cat([self.gate_proj, self.up_proj], dim=1).transpose(1, 2), offs=offsets)
            gate, up = gu[:, :inter], gu[:, inter:]
        else:
            gate = torch._grouped_mm(routed_inputs, self.gate_proj.transpose(1, 2), offs=offsets)
            up = torch._grouped_mm(routed_inputs, self.up_proj.transpose(1, 2), offs=offsets)
        if self.lora_gate is not None:
            gate = gate + self.lora_scaling * self.lora_gate(routed_inputs, offsets)
            up = up + self.lora_scaling * self.lora_up(routed_inputs, offsets)
        activated = self.act_fn(gate) * up
        routed = torch._grouped_mm(activated, self.down_proj.transpose(1, 2), offs=offsets)
        if self.lora_down is not None:
            routed = routed + self.lora_scaling * self.lora_down(activated, offsets)
        if COMBINE in ("bmm", "bmm16"):
            # every token's top-k expert outputs back in token order (one gather), then one batched [1 x k] @ [k x H]
            # in fp32: no per-chunk slices (each chunk's slice-backward zero-filled and re-added the whole [N*k, H]
            # gradient), no out-of-place index_add copies, no atomics; ~20 % of a micro-step on the 8-layer profile
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=order.device)
            by_token = routed.index_select(0, inv).view(n_tokens, top_k, -1)
            if COMBINE == "bmm16":  # bf16 tensor-core matmul, fp32 accumulation; only the k routing weights are rounded
                return torch.bmm(topk_weights.view(n_tokens, 1, top_k).to(by_token.dtype), by_token).view(n_tokens, -1)
            out = torch.bmm(topk_weights.view(n_tokens, 1, top_k).float(), by_token.float()).view(n_tokens, -1)
            return out.type(hidden_states.dtype)
        out = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        for s in range(0, routed.shape[0], COMBINE_CHUNK_ROWS):
            e = s + COMBINE_CHUNK_ROWS
            if INPLACE_COMBINE:  # autograd-safe: `out` is a fresh non-leaf buffer, index_add_'s backward needs no inputs
                out.index_add_(0, sorted_tokens[s:e], routed[s:e] * sorted_weights[s:e].unsqueeze(-1))
            else:
                out = out.index_add(0, sorted_tokens[s:e], routed[s:e] * sorted_weights[s:e].unsqueeze(-1))
        return out.type(hidden_states.dtype)


def _moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    orig_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    flat = hidden_states.view(-1, hidden_states.shape[-1])
    routed = self.experts(flat, topk_indices, topk_weights)
    if self.shared_experts is not None:
        routed = routed + self.shared_experts(flat)
    return routed.view(*orig_shape)


def convert_experts(model: nn.Module) -> int:
    import types
    n = 0
    for module in model.modules():
        if module.__class__.__name__ == "MiMoV2MoE" and isinstance(module.experts, nn.ModuleList):
            grouped = GroupedExperts(module.experts)
            del module.experts
            module.experts = grouped
            module.forward = types.MethodType(_moe_forward, module)
            n += 1
    return n


# ----------------------------------------------------------------------------- LoRA
def _lora_init_(a: torch.Tensor, in_features: int) -> None:
    """kaiming_uniform(a=sqrt(5)) on the fan-in, as PEFT; B stays zero so the adapter starts as the identity."""
    if a.device.type != "meta":
        bound = 1.0 / math.sqrt(in_features)
        with torch.no_grad():
            a.uniform_(-bound, bound)


class LoRAAdapter(nn.Module):
    """Rank-r adapter of one nn.Linear: delta(x) = B(A(x)). fp32 master weights; wrapped as its own FSDP unit."""

    def __init__(self, in_features: int, out_features: int, r: int, device):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(r, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r, dtype=torch.float32, device=device))
        _lora_init_(self.lora_A, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype))


class GroupedLoRA(nn.Module):
    """Per-expert rank-r adapters of one stacked projection [E, out, in], applied with the same grouped GEMMs."""

    def __init__(self, n_experts: int, in_features: int, out_features: int, r: int, device):
        super().__init__()
        self.lora_A = nn.Parameter(torch.empty(n_experts, r, in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(n_experts, out_features, r, dtype=torch.float32, device=device))
        _lora_init_(self.lora_A, in_features)

    def forward(self, x_sorted: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        h = torch._grouped_mm(x_sorted, self.lora_A.transpose(1, 2).to(x_sorted.dtype), offs=offsets)
        return torch._grouped_mm(h, self.lora_B.transpose(1, 2).to(x_sorted.dtype), offs=offsets)


LORA_CLASSES = (LoRAAdapter, GroupedLoRA)
DENSE_MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate|up|down)_proj$")


def _linear_lora_forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight, self.bias) + self.lora_scaling * self.lora(x)


def add_lora(model: nn.Module, r: int, alpha: float, targets: set[str], device) -> tuple[int, int]:
    """Attach adapters in place (parameter names of the base stay unchanged; adapters live under <module>.lora*)."""
    import types
    scaling = alpha / r
    n_modules = n_params = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name != "lm_head":
            kind = ("attn" if ".self_attn." in name else "shared" if ".shared_experts." in name
                    else "dense" if DENSE_MLP_RE.match(name) else None)
            if kind not in targets:
                continue
            module.lora = LoRAAdapter(module.in_features, module.out_features, r, device)
            module.lora_scaling = scaling
            module.forward = types.MethodType(_linear_lora_forward, module)
            n_modules += 1
            n_params += r * (module.in_features + module.out_features)
        elif isinstance(module, GroupedExperts) and "experts" in targets:
            for proj in ("gate", "up", "down"):
                w = getattr(module, f"{proj}_proj")  # [E, out, in]
                setattr(module, f"lora_{proj}", GroupedLoRA(w.shape[0], w.shape[2], w.shape[1], r, device))
                n_params += w.shape[0] * r * (w.shape[1] + w.shape[2])
            module.lora_scaling = scaling
            n_modules += 3
    return n_modules, n_params


def fold_lora(name: str, weight: torch.Tensor, params: dict[str, torch.Tensor], scaling: float) -> torch.Tensor:
    """weight + scaling * B A if an adapter for this base weight is among `params` (full, unsharded), else weight."""
    if EXPERT_RE.match(name):
        base, proj = name.rsplit(".", 1)  # model.layers.N.mlp.experts, gate_proj
        a, b = params.get(f"{base}.lora_{proj[:-5]}.lora_A"), params.get(f"{base}.lora_{proj[:-5]}.lora_B")
        if a is None:
            return weight
        return (weight.float() + scaling * torch.bmm(b.float(), a.float())).to(weight.dtype)
    if name.endswith(".weight"):
        base = name[: -len(".weight")]
        a, b = params.get(f"{base}.lora.lora_A"), params.get(f"{base}.lora.lora_B")
        if a is not None:
            return (weight.float() + scaling * (b.float() @ a.float())).to(weight.dtype)
    return weight


# ----------------------------------------------------------------------------- model loading
def patch_masking_compat(model) -> list[str]:
    """The remote MiMo code passes cache_position= to transformers' mask builders; newer transformers (incl. the
    validator's fork) dropped that argument. Same adaptation the validator applies (engine.patch_mimo_masking_compat)."""
    import inspect
    import sys
    module = sys.modules[model.__class__.__module__]
    patched = []
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        fn = getattr(module, name, None)
        if fn is None or "cache_position" in inspect.signature(fn).parameters:
            continue

        def adapter(*args, _fn=fn, **kwargs):
            kwargs.pop("cache_position", None)
            return _fn(*args, **kwargs)

        setattr(module, name, adapter)
        patched.append(name)
    return patched


def build_model(init: Path, dtype: torch.dtype, rank: int, attn: str, lora_r: int = 0, lora_alpha: float = 32.0,
                lora_targets: set[str] | None = None, debug_layers: int = 0):
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = AutoConfig.from_pretrained(init, trust_remote_code=True)
    config.use_cache = False
    if debug_layers:  # truncated model for smoke tests (weights of the dropped layers are simply not loaded)
        config.num_hidden_layers = debug_layers
        config.hybrid_layer_pattern = list(config.hybrid_layer_pattern)[:debug_layers]
        config.moe_layer_freq = list(config.moe_layer_freq)[:debug_layers]
    cls = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], str(init))
    # the remote code implements sdpa / flash / flex paths itself (sdpa falls back to eager on sink-bias layers; flex takes the
    # sink bias as s_aux); only the HF capability flags are missing, and transformers checks them on the inner model class too
    import sys
    from transformers import PreTrainedModel
    for obj in vars(sys.modules[cls.__module__]).values():
        if isinstance(obj, type) and issubclass(obj, PreTrainedModel):
            obj._supports_sdpa = True
            obj._supports_flash_attn = True
            obj._supports_flex_attn = True
    config._attn_implementation = attn
    if rank == 0:
        model = cls.from_pretrained(init, config=config, torch_dtype=dtype, low_cpu_mem_usage=True)
    else:
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = cls(config)
        for p in model.parameters():  # parameters only: buffers (rotary inv_freq) stay fp32 exactly as on rank 0, so the
            p.data = p.data.to(dtype)  # coalesced state broadcast of every FSDP unit has the same layout on all ranks
    convert_experts(model)
    patch_masking_compat(model)
    import types
    model._cpt_orig_forward = model.forward
    model.forward = types.MethodType(_fused_causal_forward, model)  # ce_labels=None keeps the original behaviour
    if lora_r > 0:
        n_mod, n_par = add_lora(model, lora_r, lora_alpha, lora_targets or set(), "cpu" if rank == 0 else "meta")
        if rank == 0:
            print(f"LoRA r={lora_r} alpha={lora_alpha} on {sorted(lora_targets or [])}: {n_mod} modules, {n_par / 1e6:.1f}M params", flush=True)
    model.config.use_cache = False
    return model


# ----------------------------------------------------------------------------- data
# pool source labels: TEXT/CODE/MATH pools use the first three; MAIN pools label the five MAIN categories (before these were
# added every MAIN sequence fell into "text": one val split and no per-source loss weights for MAIN)
SOURCES = ("text", "math", "code", "code-reasoning", "math-reasoning", "pretrain-behaviors", "sft-reasoning", "txt360-v2")


class Pool:
    """One pool dir (tokens.npy [n, seq_len] + optional sources.npy), or several joined with commas: a long run then reads
    fresh data for more steps instead of repeating one pool (and never restarts from a bf16 checkpoint in between)."""

    def __init__(self, path: Path):
        self.parts = []
        for p in str(path).split(","):
            p = Path(p)
            tokens = np.load(p / "tokens.npy", mmap_mode="r")
            src = np.load(p / "sources.npy") if (p / "sources.npy").exists() else None
            sidx = np.asarray([SOURCES.index(x) if x in SOURCES else 0 for x in src]) if src is not None else np.zeros(len(tokens), dtype=np.int64)
            self.parts.append((tokens, sidx))
        self.seq_len = self.parts[0][0].shape[1]
        assert all(t.shape[1] == self.seq_len for t, _ in self.parts), "pools must share seq_len"
        self.starts = np.cumsum([0] + [len(t) for t, _ in self.parts])
        self.n = int(self.starts[-1])
        self.source_idx = np.concatenate([s for _, s in self.parts])

    def batch(self, idx: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = np.sort(idx)
        part = np.searchsorted(self.starts, idx, side="right") - 1
        rows = [np.asarray(self.parts[k][0][i - self.starts[k]], dtype=np.int64) for k, i in zip(part, idx)]
        ids = torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)
        return ids, torch.from_numpy(self.source_idx[idx]).to(device)


def _chunk_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), reduction="none")


DOCUMENT_EOS = 151645  # the validator's document boundary (teutonic.evaluation.masking.DOCUMENT_EOS_TOKEN_ID)


def document_layout(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validator document masking (teutonic.evaluator.document_masking.document_layout, policy
    document-masked-token-bootstrap-v1): EOS ends a fragment, positions restart at 0 after it, and a fragment's
    first token is never a target. Returns (position_ids [B, L], valid targets [B, L-1]). Given restarting
    position_ids, the MiMo forward's transformers mask builders add the same-fragment constraint to the causal and
    sliding-window masks (packed-sequence detection), so attention never crosses an EOS either."""
    starts = torch.ones_like(ids, dtype=torch.bool)
    starts[:, 1:] = ids[:, :-1] == DOCUMENT_EOS
    index = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    return index - torch.where(starts, index, 0).cummax(dim=1).values, ~starts[:, 1:]


def _fused_causal_forward(self, input_ids=None, position_ids=None, ce_labels=None, ce_valid=None, ce_chunk: int = 512, **kw):
    """CausalLM forward (installed by build_model). With ce_labels: the decoder, then lm_head + CE in position chunks,
    each checkpointed, returning the summed CE of the valid targets per sequence [B] (fp32). Only one chunk's logits
    are ever alive ([B, 512, V] instead of the full [B, L, V] bf16 logits, ~10 GB at batch 16, whose per-chunk slices
    also zero-filled the full logits gradient in backward). Runs inside the root FSDP unit's forward, so the head weight
    is gathered; the recompute reads it through the module (never a captured view). Without ce_labels: the original."""
    if ce_labels is None:
        return self._cpt_orig_forward(input_ids=input_ids, position_ids=position_ids, **kw)
    from torch.utils.checkpoint import checkpoint
    hidden = self.model(input_ids=input_ids, position_ids=position_ids, use_cache=False).last_hidden_state[:, :-1]
    head = self.lm_head

    def chunk_ce(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = head(h)  # bf16 logits, fp32 log-softmax: the same numerics as seq_ce's unfused path
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), reduction="none").view(y.shape)

    total = torch.zeros(hidden.shape[0], device=hidden.device, dtype=torch.float32)
    for s in range(0, hidden.shape[1], ce_chunk):
        e = min(s + ce_chunk, hidden.shape[1])
        part = (checkpoint(chunk_ce, hidden[:, s:e], ce_labels[:, s:e], use_reentrant=False) if torch.is_grad_enabled()
                else chunk_ce(hidden[:, s:e], ce_labels[:, s:e]))
        total = total + (part * ce_valid[:, s:e]).sum(1)
    return total


def seq_ce(model, ids: torch.Tensor, chunk: int = 512, doc_mask: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Summed cross-entropy and number of scored targets of every sequence, shapes [B] (logits chunked and up-cast to
    fp32); sum / count is the validator's per-sequence loss. With doc_mask (the live scoring policy), fragments are
    isolated at EOS and fragment starts are not targets; without it every one of the L-1 targets counts. With autograd
    on, each chunk's fp32 log-softmax is recomputed in backward instead of kept (B x 2047 x 152k fp32 = 1.25 GB per sequence)."""
    from torch.utils.checkpoint import checkpoint
    labels = ids[:, 1:]
    if FUSED_CE:
        positions, valid = document_layout(ids) if doc_mask else (None, torch.ones_like(labels, dtype=torch.bool))
        total = model(input_ids=ids, position_ids=positions, ce_labels=labels, ce_valid=valid, ce_chunk=chunk)
        return total, valid.sum(1).to(total.dtype)
    if doc_mask:
        positions, valid = document_layout(ids)
        logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
    else:
        valid = torch.ones_like(labels, dtype=torch.bool)
        logits = model(input_ids=ids, use_cache=False).logits
    n_pos = labels.shape[1]
    total = torch.zeros(ids.shape[0], device=ids.device)
    for s in range(0, n_pos, chunk):
        e = min(s + chunk, n_pos)
        if torch.is_grad_enabled():
            part = checkpoint(_chunk_ce, logits[:, s:e, :], labels[:, s:e], use_reentrant=False)
        else:
            part = _chunk_ce(logits[:, s:e, :], labels[:, s:e])
        total = total + (part.view(ids.shape[0], -1) * valid[:, s:e]).sum(1)
    return total, valid.sum(1).to(total.dtype)


# ----------------------------------------------------------------------------- saving
def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal safetensors writer (bf16, contiguous CPU tensors): header + raw bytes through plain file writes, which
    release the GIL -- a background writer thread does not slow the training loop's kernel dispatch."""
    header, offset = {}, 0
    for k, t in tensors.items():
        n = t.numel() * t.element_size()
        header[k] = {"dtype": "BF16", "shape": list(t.shape), "data_offsets": [offset, offset + n]}
        offset += n
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(len(blob).to_bytes(8, "little"))
        f.write(blob)
        for t in tensors.values():
            f.write(t.view(torch.int16).numpy().data)
    tmp.replace(path)


class AsyncSaver:
    """Checkpoint saves that cost the training loop only the gather (seconds), not the 220 GB of disk writes.

    Every FSDP unit is all-gathered as before, but only one rank (unit index mod world) copies it to host memory, as
    bf16 per-expert tensors; each rank then writes its own shard files from a background thread (8 writers in parallel,
    GIL released) and drops a .rank<r>.done marker. Rank 0's thread writes the index, the contract files and finally
    SAVED once every rank is done -- uploaders and scorers wait for SAVED. A new save first waits for the previous one."""

    def __init__(self, rank: int, world: int):
        self.rank, self.world, self.thread = rank, world, None
        # pinned host buffers, reused by every save (the previous save's writer has finished before they are refilled):
        # device->host copies run at full PCIe speed and asynchronously, overlapping the next unit's all-gather
        # (pageable copies, one unit at a time, made the B300 step-25 gather take 139 s)
        self.pinned: dict[str, torch.Tensor] = {}

    def _to_host(self, key: str, t: torch.Tensor) -> torch.Tensor:
        buf = self.pinned.get(key)
        if buf is None or buf.shape != t.shape:
            buf = self.pinned[key] = torch.empty(t.shape, dtype=torch.bfloat16, pin_memory=True)
        buf.copy_(t, non_blocking=True)  # stream-ordered: the source is freed only after the copy on this stream
        return buf

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def save(self, model, init: Path, out: Path, lora_scaling: float = 0.0, contract_dir: Path | None = None, sr: bool = False) -> None:
        import threading

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self.wait()
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        units = [m for m in model.modules() if isinstance(m, FSDP)]
        root = units[0]

        def clean(name: str) -> str:
            return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")

        prefix_of = {mod: clean(name) for name, mod in model.named_modules() if isinstance(mod, FSDP)}
        nested = [prefix_of[u] + "." for u in units if u is not root]
        mine: list[tuple[str, dict[str, torch.Tensor]]] = []  # (file name, tensors) this rank writes
        index = 0
        for unit in units:
            if isinstance(unit.module, LORA_CLASSES):
                continue  # gathered with its layer below
            writer = index % self.world
            name = f"model-{index + 1:05d}.safetensors"
            index += 1
            with FSDP.summon_full_params(unit, recurse=unit is not root, writeback=False, rank0_only=False, offload_to_cpu=False):
                if self.rank == writer:
                    pre = prefix_of[unit]
                    params = {clean(f"{pre}.{n}" if pre else n): p for n, p in unit.named_parameters(recurse=True)}
                    tensors: dict[str, torch.Tensor] = {}
                    for full, p in params.items():
                        if unit is root and any(full.startswith(n) for n in nested):
                            continue  # owned (and still sharded) by a nested unit
                        if ".lora" in full:
                            continue  # folded into its base weight
                        with torch.no_grad():
                            x = fold_lora(full, p, params, lora_scaling).detach()
                            if sr:
                                try:
                                    xb = sr_bf16(x)
                                except RuntimeError as e:  # e.g. OOM on a huge fused tensor: never lose the checkpoint over it
                                    print(f"stochastic rounding failed for {full} ({str(e)[:80]}); nearest instead", flush=True)
                                    xb = x.to(torch.bfloat16)
                            else:
                                xb = x.to(torch.bfloat16)
                            t = self._to_host(full, xb.contiguous())
                        m = EXPERT_RE.match(full)
                        for k, v in ([(f"{m.group(1)}.{e}.{m.group(2)}.weight", t[e]) for e in range(t.shape[0])] if m else [(full, t)]):
                            tensors[k] = v.contiguous()
                    if tensors:
                        mine.append((name, tensors))
                    # the bf16 casts and pinned copies must finish before summon_full_params frees the gathered
                    # parameters (FSDP's unshard stream could otherwise refill that memory under them); pinned copies
                    # make this ~0.2 s per layer
                    torch.cuda.current_stream().synchronize()
        torch.cuda.synchronize()  # every async device->host copy has landed in the pinned buffers
        torch.cuda.empty_cache()
        dist.barrier()
        gather_s = time.time() - t0
        cfg_model = model.module.config if hasattr(model, "module") else model.config
        cfg_overrides = {k: getattr(cfg_model, k) for k in ("num_hidden_layers", "hybrid_layer_pattern", "moe_layer_freq")}
        rank, world = self.rank, self.world

        def write() -> None:
            t1 = time.time()
            weight_map, total = {}, 0
            for name, tensors in mine:
                _write_safetensors(out / name, tensors)
                for k, v in tensors.items():
                    weight_map[k] = name
                    total += v.numel() * 2
            (out / f".rank{rank}.json").write_text(json.dumps({"weight_map": weight_map, "total": total}))
            mine.clear()
            if rank != 0:
                return
            while not all((out / f".rank{r}.json").exists() for r in range(world)):
                time.sleep(1)
            weight_map, total = {}, 0
            for r in range(world):
                part = json.loads((out / f".rank{r}.json").read_text())
                weight_map.update(part["weight_map"])
                total += part["total"]
                (out / f".rank{r}.json").unlink()
            (out / "model.safetensors.index.json").write_text(json.dumps(
                {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
            for f in init.iterdir():
                # never the init model's identity / bookkeeping files: king_info.json makes src/parity's loss cache file the
                # checkpoint under the KING's digest (the step-1000 MAIN screen of text_ap_10140501 read exactly 0 and three
                # samples' caches held our losses under M45's id, 2026-10-14)
                if f.is_file() and f.suffix != ".safetensors" and f.name not in NOT_COPIED:
                    shutil.copy2(f, out / f.name)
            cfg = json.loads((out / "config.json").read_text())
            cfg.update(cfg_overrides)  # identical unless --debug-layers
            cfg["use_cache"] = True
            cfg.pop("_attn_implementation", None)
            (out / "config.json").write_text(json.dumps(cfg, indent=2))
            if contract_dir is not None and cfg_overrides == {k: json.loads((contract_dir / "config.json").read_text()).get(k) for k in cfg_overrides}:
                # submission-ready: the validator byte-locks these files to genesis (a re-serialized config.json fails)
                for name in ("config.json", "configuration_mimo_v2.py", "modeling_mimo_v2.py", "tokenizer.json",
                             "tokenizer_config.json", "chat_template.jinja.txt"):
                    if (contract_dir / name).exists():
                        shutil.copy2(contract_dir / name, out / name)
            (out / "SAVED").write_text(json.dumps({"total_size": total, "files": len(set(weight_map.values())),
                                                   "gather_s": round(gather_s, 1), "write_s": round(time.time() - t1, 1)}))
            print(f"saved {out} ({total / 1e9:.1f} GB, {len(set(weight_map.values()))} shards; training paused "
                  f"{gather_s:.0f}s for the gather, files written in the background in {time.time() - t1:.0f}s)", flush=True)

        self.thread = threading.Thread(target=write, name="ckpt-writer", daemon=False)
        self.thread.start()


def save_checkpoint(model, init: Path, out: Path, rank: int, shard_bytes: int = 5 * 2**30, lora_scaling: float = 0.0) -> None:
    """Validator-ready bf16 safetensors (HF names, per-expert tensors), gathered one FSDP unit at a time; LoRA adapters
    (nested FSDP units) are gathered together with their layer and folded into the base weights."""
    from safetensors.torch import save_file
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    units = [m for m in model.modules() if isinstance(m, FSDP)]
    root = units[0]
    shard, shard_size, shard_idx, weight_map, total = {}, 0, 0, {}, 0
    files: list[str] = []

    def flush():
        nonlocal shard, shard_size, shard_idx
        if not shard:
            return
        name = f"model-{shard_idx + 1:05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        files.append(name)
        shard, shard_size, shard_idx = {}, 0, shard_idx + 1

    def emit(name: str, t: torch.Tensor):
        nonlocal shard_size, total
        m = EXPERT_RE.match(name)
        t = t.detach().to(torch.bfloat16).contiguous().cpu()  # one device->host copy per tensor (folded on the GPU)
        items = [(f"{m.group(1)}.{e}.{m.group(2)}.weight", t[e]) for e in range(t.shape[0])] if m else [(name, t)]
        for k, v in items:
            v = v.contiguous()
            shard[k] = v
            shard_size += v.numel() * 2
            total += v.numel() * 2
            if shard_size >= shard_bytes:
                flush()

    def clean(name: str) -> str:
        return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")

    prefix_of = {mod: clean(name) for name, mod in model.named_modules() if isinstance(mod, FSDP)}
    nested = [prefix_of[u] + "." for u in units if u is not root]
    for unit in units:
        if isinstance(unit.module, LORA_CLASSES):
            continue  # gathered with its layer below
        with FSDP.summon_full_params(unit, recurse=unit is not root, writeback=False, rank0_only=True, offload_to_cpu=False):
            if rank == 0:
                pre = prefix_of[unit]
                params = {clean(f"{pre}.{n}" if pre else n): p for n, p in unit.named_parameters(recurse=True)}
                for full, p in params.items():
                    if unit is root and any(full.startswith(n) for n in nested):
                        continue  # owned (and still sharded) by a nested unit
                    if ".lora" in full:
                        continue  # folded into its base weight
                    with torch.no_grad():
                        emit(full, fold_lora(full, p, params, lora_scaling))
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        flush()
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
        for f in init.iterdir():
            if f.is_file() and f.suffix != ".safetensors" and f.name not in NOT_COPIED:
                shutil.copy2(f, out / f.name)
        cfg = json.loads((out / "config.json").read_text())
        for k in ("num_hidden_layers", "hybrid_layer_pattern", "moe_layer_freq"):  # identical unless --debug-layers
            cfg[k] = getattr(model.module.config if hasattr(model, "module") else model.config, k)
        cfg["use_cache"] = True
        cfg.pop("_attn_implementation", None)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"saved {out} ({total / 1e9:.1f} GB, {len(files)} shards)", flush=True)
    dist.barrier()


# ----------------------------------------------------------------------------- optimizer-state offload
class StateOffload:
    """Optimizer state tensors live in pinned host memory between steps and on the GPU only around opt.step().
    Full-parameter B200 runs: frees the 8-bit Adam states (~26 GiB/GPU) during forward/backward, so a bigger micro-batch
    fits under the allocator ceiling; costs one D2H + one H2D copy of the states per optimizer step."""

    def __init__(self, opt: torch.optim.Optimizer, min_numel: int = 1 << 16):
        self.opt, self.min_numel, self.host = opt, min_numel, {}

    def _big(self, v) -> bool:
        return torch.is_tensor(v) and v.numel() >= self.min_numel  # small tensors (qmaps, step counters) stay put

    def to_device(self, device) -> None:
        for st in self.opt.state.values():
            for k, v in st.items():
                if self._big(v) and not v.is_cuda:
                    st[k] = v.to(device, non_blocking=True)

    def to_host(self) -> None:
        for p, st in self.opt.state.items():
            for k, v in st.items():
                if self._big(v) and v.is_cuda:
                    buf = self.host.get((id(p), k))
                    if buf is None:
                        buf = self.host[(id(p), k)] = torch.empty(v.shape, dtype=v.dtype, pin_memory=True)
                    buf.copy_(v, non_blocking=True)
                    st[k] = buf
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- weight EMA
def sr_bf16(x: torch.Tensor) -> torch.Tensor:
    """Stochastic rounding fp32 -> bf16: unbiased (E[sr(x)] = x), so updates smaller than half a bf16 ulp survive in
    expectation instead of rounding back to the start; values already on the bf16 grid (frozen / untouched weights) stay
    exact (their low 16 bits are 0, the added noise never carries). Sign-magnitude floats: adding to the raw bits rounds
    the magnitude up with probability (low bits / 2^16), for either sign."""
    if x.dtype == torch.bfloat16:
        return x
    xi = x.float().contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, xi.shape, dtype=torch.int32, device=x.device)
    return ((xi + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class DeviceEMA:
    """HostEMA's interface with the fp32 EMA kept next to the shards in HBM (--ema-on-gpu): no per-step PCIe round trip
    of every trainable shard (2 x ~46 GB per rank per step, ~1.7 s), at +4 bytes per trainable parameter per GPU.
    `host` is the list of EMA tensors (here on the GPU), so state saves and resumes treat both classes alike."""

    def __init__(self, params, decay: float):
        self.params = [p for p in params if p.numel() > 0]
        self.decay = decay
        self.host = [p.detach().float().clone() for p in self.params]

    @torch.no_grad()
    def update(self) -> None:
        torch._foreach_lerp_(self.host, [p.detach().float() for p in self.params], 1.0 - self.decay)

    @torch.no_grad()
    def swap(self) -> None:
        for p, h in zip(self.params, self.host):
            raw = p.detach().float().clone()
            p.copy_(h)
            h.copy_(raw)


class HostEMA:
    """fp32 exponential moving average of the trainable parameters (this rank's FSDP shards) in pinned host memory.

    update(): ema = d·ema + (1−d)·param, tensor by tensor on the GPU (one H2D + one D2H per tensor; the GPU only ever
    holds one tensor's copy). swap(): exchanges parameters and EMA in place and is its own inverse, so
    swap(); evaluate()/save_checkpoint(); swap() runs on the EMA weights and training continues on the raw ones.
    Why: at lr 3e-6 the raw weights jitter ±0.0003 on the screen from step to step while the EMA (0.99/step) extracted
    the drift (+0.00066 vs start at step 200 vs +0.00031 raw; UPDATE.md 2026-09-25)."""

    def __init__(self, params, decay: float):
        self.params = [p for p in params if p.numel() > 0]
        self.decay = decay
        self.host = [torch.empty(p.shape, dtype=torch.float32, pin_memory=True) for p in self.params]
        with torch.no_grad():
            for p, h in zip(self.params, self.host):
                h.copy_(p.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def update(self) -> None:
        for p, h in zip(self.params, self.host):
            e = h.to(p.device, non_blocking=True)
            e.lerp_(p.detach().float(), 1.0 - self.decay)
            h.copy_(e, non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def swap(self) -> None:
        for p, h in zip(self.params, self.host):
            e = h.to(p.device, non_blocking=True)
            raw = p.detach().float().clone()
            p.copy_(e)
            h.copy_(raw, non_blocking=True)
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- pause (lets the parity scorer borrow the GPUs)
def pause_if_requested(out: Path, rank: int, device) -> None:
    """If <out>/PAUSE exists after a step: free the activation cache, write <out>/PAUSED and wait until PAUSE is removed.
    All ranks follow rank 0's view of the file (broadcast), so no rank can run ahead into the next collective."""
    flag = torch.zeros(1, device=device)
    if rank == 0 and (out / "PAUSE").exists():
        flag.fill_(1)
    dist.broadcast(flag, 0)
    if not flag.item():
        return
    torch.cuda.empty_cache()
    if rank == 0:
        (out / "PAUSED").touch()
        print(f"paused ({time.strftime('%H:%M:%S')}); GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB allocated, "
              f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved", flush=True)
    while True:
        time.sleep(5)
        if rank == 0:
            flag.fill_(1 if (out / "PAUSE").exists() else 0)
        dist.broadcast(flag, 0)
        if not flag.item():
            break
    if rank == 0:
        (out / "PAUSED").unlink(missing_ok=True)
        print(f"resumed ({time.strftime('%H:%M:%S')})", flush=True)


# ----------------------------------------------------------------------------- live control (scripts/train/autopilot.py)
class Control:
    """<out>/CONTROL.json, written by the autopilot and read by rank 0 after every step (parsed only when its mtime
    changes), broadcast to every rank so all of them apply the same decision at the same step. Fields (all optional):
      id        int    a new id applies the one-shot actions below once (the persistent fields apply on every change)
      lr_scale  float  multiplies the scheduled LR of every param group (persistent; default 1)
      anneal    int    K > 0: from the next step, cosine the LR from its current value down to min-lr-frac x lr x lr_scale
                       over K steps, then hold; 0 cancels it (persistent)
      save      bool   one-shot: write a checkpoint step-N after this step
      state     bool   one-shot: write the exact training state after this step
      stop      bool   one-shot: write a checkpoint and the exact state after this step, then exit cleanly
    Every applied change is acknowledged in <out>/CONTROL.ack.json (id, step, effective LR of the first group)."""

    FIELDS = ("id", "lr_scale", "anneal", "save", "state", "stop")

    def __init__(self, out: Path, rank: int, device):
        self.path, self.ack, self.rank, self.device = out / "CONTROL.json", out / "CONTROL.ack.json", rank, device
        self.mtime, self.id = None, None
        self.lr_scale, self.anneal_k, self.anneal_from_step, self.anneal_from_lr = 1.0, 0, None, None

    def poll(self) -> dict | None:
        """Changed values (all ranks), or None when nothing changed."""
        vec = torch.full((len(self.FIELDS) + 1,), float("nan"), device=self.device)
        if self.rank == 0 and self.path.exists():
            try:
                mtime = self.path.stat().st_mtime
                if mtime != self.mtime:
                    raw = json.loads(self.path.read_text())
                    self.mtime = mtime
                    vec[0] = 1.0
                    for i, k in enumerate(self.FIELDS):
                        if k in raw and raw[k] is not None:
                            vec[i + 1] = float(raw[k])
            except (OSError, ValueError) as exc:  # half-written file: read it again after the next step
                print(f"CONTROL.json unreadable ({exc}); retrying next step", flush=True)
        dist.broadcast(vec, 0)
        if not vec[0].item() == 1.0:
            return None
        got = {k: float(vec[i + 1]) for i, k in enumerate(self.FIELDS) if not math.isnan(float(vec[i + 1]))}
        new_id = got.get("id")
        one_shot = new_id is not None and new_id != self.id
        self.id = new_id if new_id is not None else self.id
        out = {"lr_scale": got.get("lr_scale"), "anneal": got.get("anneal")}
        for k in ("save", "state", "stop"):
            out[k] = bool(got.get(k)) and one_shot
        return out

    def write_ack(self, step: int, lr: float, note: str) -> None:
        if self.rank == 0:
            self.ack.write_text(json.dumps({"id": self.id, "step": step, "lr": lr, "lr_scale": self.lr_scale,
                                            "anneal": self.anneal_k, "note": note, "t": time.time()}))


# ----------------------------------------------------------------------------- exact training state (resume without a reset)
def _balanced_parts(tensors: list, k: int) -> list[list[int]]:
    """Indices of `tensors` split into k groups of similar byte size (greedy, largest first)."""
    order = sorted(range(len(tensors)), key=lambda i: -tensors[i].numel() * tensors[i].element_size())
    parts, sizes = [[] for _ in range(k)], [0] * k
    for i in order:
        j = sizes.index(min(sizes))
        parts[j].append(i)
        sizes[j] += tensors[i].numel() * tensors[i].element_size()
    return [sorted(p) for p in parts if p]


def _parallel(fn, items, threads: int = 8) -> list:
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(threads) as ex:
        return list(ex.map(fn, items))


class AsyncStateSaver:
    """Exact state of this rank -- fp32 master shards of the trainable parameters, the optimizer state (8-bit Adam
    moments included), the EMA and the loop's bookkeeping -- so a resume continues the run as if it never stopped (no
    fresh Adam moments, no re-warmup, no bf16 rounding of sub-ulp progress).

    The training loop only pays for a snapshot (GPU masters -> pinned host buffers, host EMA / moments -> host buffers,
    all reused between saves; seconds); background threads then write ~115 GB per rank as part files (parallel
    torch.save without CRC32) while training goes on. The synchronous version cost 222 s per save on the full model
    (11 % of the run at one save per 20 steps). Ranks coordinate through marker files, never collectives (a collective
    from a writer thread could interleave with the training loop's): rank 0's writer waits for every rank's marker, then
    writes STATE.json, renames <state>.tmp into place and drops the older states. No STATE.json = incomplete."""

    def __init__(self, rank: int, world: int, threads: int = 8):
        self.rank, self.world, self.threads = rank, world, threads
        self.thread, self.pinned, self.host_ema, self.host_opt = None, [], [], {}
        try:  # the CRC32 of torch's zip writer halves single-stream throughput; nothing here reads it back
            from torch.utils.serialization import config as ser_config
            ser_config.save.compute_crc32 = False
        except ImportError:
            pass

    def wait(self) -> None:
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def _snap_opt(self, obj, path=()):
        """Copy of an optimizer state (nested dicts / lists of tensors) into reusable host buffers."""
        if torch.is_tensor(obj):
            buf = self.host_opt.get(path)
            if buf is None or buf.shape != obj.shape or buf.dtype != obj.dtype:
                buf = self.host_opt[path] = torch.empty(obj.shape, dtype=obj.dtype, device="cpu")
            buf.copy_(obj)
            return buf
        if isinstance(obj, dict):
            return {k: self._snap_opt(v, path + (k,)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._snap_opt(v, path + (i,)) for i, v in enumerate(obj))
        return obj

    def save(self, state_dir: Path, step: int, params: list, opt, ema, extra: dict, keep: int = 2) -> float:
        """Snapshot now, write in the background; returns the seconds the training loop was held."""
        self.wait()  # the previous state is complete before its buffers are refilled
        t0 = time.time()
        rank, world, threads = self.rank, self.world, self.threads
        tmp = state_dir.with_name(state_dir.name + ".tmp")
        if rank == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        dist.barrier()
        if len(self.pinned) != len(params):
            self.pinned = [torch.empty(p.shape, dtype=p.dtype, pin_memory=True) for p in params]
        for buf, p in zip(self.pinned, params):
            buf.copy_(p.detach(), non_blocking=True)
        if ema is not None:
            if len(self.host_ema) != len(ema.host):
                self.host_ema = [torch.empty(h.shape, dtype=h.dtype, device="cpu") for h in ema.host]  # also for DeviceEMA
            _parallel(lambda i: self.host_ema[i].copy_(ema.host[i]), range(len(ema.host)), threads)
        sd = opt.state_dict()
        state = {k: self._snap_opt(v, ("s", k)) for k, v in sd["state"].items()}
        rest = {k: v for k, v in sd.items() if k != "state"}
        torch.cuda.synchronize()  # every pinned device->host copy has landed
        held = time.time() - t0
        host, host_ema = self.pinned, (self.host_ema if ema is not None else None)
        meta_shapes = [tuple(p.shape) for p in params]

        def write() -> None:
            t1 = time.time()
            try:
                jobs = [("params", i, idx, host) for i, idx in enumerate(_balanced_parts(host, threads))]
                if host_ema is not None:
                    jobs += [("ema", i, idx, host_ema) for i, idx in enumerate(_balanced_parts(host_ema, threads))]
                keys = sorted(state)
                opt_parts = [keys[i::threads] for i in range(threads) if keys[i::threads]]

                def write_job(job):
                    kind, i, idx, src = job
                    torch.save({"idx": idx, "tensors": [src[k] for k in idx]}, tmp / f"rank{rank}.{kind}{i}.pt")

                _parallel(write_job, jobs, threads)
                _parallel(lambda i: torch.save({k: state[k] for k in opt_parts[i]}, tmp / f"rank{rank}.opt{i}.pt"),
                          range(len(opt_parts)), threads)
                meta = {"step": step, "world": world, "shapes": meta_shapes, "extra": extra, "opt_rest": rest,
                        "parts": {"params": sum(1 for j in jobs if j[0] == "params"),
                                  "ema": sum(1 for j in jobs if j[0] == "ema"), "opt": len(opt_parts)}}
                torch.save(meta, tmp / f"rank{rank}.meta.pt")
                (tmp / f".rank{rank}.state_done").touch()
                if rank != 0:
                    return
                t_wait = time.time()
                while not all((tmp / f".rank{r}.state_done").exists() for r in range(world)):
                    if time.time() - t_wait > 7200:
                        print(f"state-{step}: other ranks never finished writing; {tmp} left incomplete", flush=True)
                        return
                    time.sleep(2)
                for r in range(world):
                    (tmp / f".rank{r}.state_done").unlink(missing_ok=True)
                (tmp / "STATE.json").write_text(json.dumps({"step": step, "world": world, "t": time.time(), "extra": extra}))
                shutil.rmtree(state_dir, ignore_errors=True)
                tmp.rename(state_dir)
                olds = sorted((p for p in state_dir.parent.glob("state-*") if p.is_dir() and (p / "STATE.json").exists()),
                              key=lambda p: int(p.name.split("-")[1]))
                for p in olds[:-keep]:
                    shutil.rmtree(p, ignore_errors=True)
                print(f"state-{step} written in the background in {time.time() - t1:.0f}s (training held {held:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001 - a lost state is not worth the run: the newest complete one stays
                print(f"state-{step} write FAILED on rank {rank}: {exc}", flush=True)

        import threading
        self.thread = threading.Thread(target=write, name="state-writer", daemon=False)
        self.thread.start()
        return held


def load_state(state_dir: Path, rank: int, world: int, params: list, opt, offload, ema, device) -> tuple[int, dict]:
    meta_json = json.loads((state_dir / "STATE.json").read_text())
    if meta_json["world"] != world:
        raise SystemExit(f"state {state_dir} was written by {meta_json['world']} ranks, this run has {world}")
    meta = torch.load(state_dir / f"rank{rank}.meta.pt", map_location="cpu", weights_only=False)
    shapes = [tuple(p.shape) for p in params]
    if [tuple(s) for s in meta["shapes"]] != shapes:
        raise SystemExit(f"state {state_dir}: trainable parameter shards differ (another --freeze / model / world size?)")
    load = lambda name: torch.load(state_dir / name, map_location="cpu", weights_only=False)  # noqa: E731
    parts = meta["parts"]
    with torch.no_grad():
        for part in _parallel(load, [f"rank{rank}.params{i}.pt" for i in range(parts["params"])]):
            for k, t in zip(part["idx"], part["tensors"]):
                params[k].data.copy_(t.to(device, non_blocking=True))
        if ema is not None and parts["ema"]:
            for part in _parallel(load, [f"rank{rank}.ema{i}.pt" for i in range(parts["ema"])]):
                for k, t in zip(part["idx"], part["tensors"]):
                    ema.host[k].copy_(t)
    state = {}
    for part in _parallel(load, [f"rank{rank}.opt{i}.pt" for i in range(parts["opt"])]):
        state.update(part)
    opt.load_state_dict({**meta["opt_rest"], "state": state})
    if offload is None:  # bitsandbytes keeps the loaded moments on the CPU; the plain path expects them on the GPU
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v) and not v.is_cuda and v.numel() >= (1 << 16):
                    st[k] = v.to(device)
    torch.cuda.synchronize()
    return int(meta["step"]), meta.get("extra") or {}


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, type=Path, help="start checkpoint dir (HF layout, bf16 safetensors)")
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=2, help="sequences per GPU per micro-step")
    ap.add_argument("--accum", type=int, default=16, help="micro-steps per optimizer step")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--decay-start", type=int, default=None,
                    help="warmup-stable-decay: hold --lr from the end of warmup to this step, then cosine down to "
                         "--min-lr-frac x --lr at --steps (default: cosine over the whole run after warmup)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="master weight dtype (fp32 recommended)")
    ap.add_argument("--optim", default="adamw8bit", choices=["adamw8bit", "adamw"])
    ap.add_argument("--offload-opt", type=int, default=0, help="1: optimizer states in pinned host memory between steps (frees ~26 GiB/GPU in full-parameter mode)")
    ap.add_argument("--attn", default="eager", help="eager (safe) | sdpa (9 full-attention layers only; the 36 sliding-window layers with sink bias use eager anyway)")
    ap.add_argument("--freeze", default=r"\.mlp\.gate\.", help="regex of parameter names to freeze (routers by default)")
    ap.add_argument("--train-only", default=None, help="regex: train ONLY matching parameters (smoke tests)")
    ap.add_argument("--lora-r", type=int, default=0, help="> 0: freeze the bf16 base and train rank-r adapters (H100 route)")
    ap.add_argument("--lora-alpha", type=float, default=32.0, help="adapter scale = alpha / r")
    ap.add_argument("--lora-targets", default="attn,shared,dense,experts", help="comma list of attn|shared|dense|experts")
    ap.add_argument("--debug-layers", type=int, default=0, help="smoke tests: keep only the first N decoder layers")
    ap.add_argument("--profile", type=int, default=0, help="profile this many micro-steps of step 1 (rank 0 prints the kernel table)")
    ap.add_argument("--limit-all-gathers", type=int, default=1, help="0 lets FSDP issue all-gathers ahead of compute (more overlap, +1 layer of GPU memory)")
    ap.add_argument("--lr-dense-frac", type=float, default=1.0, help="lr multiplier for the non-expert adapters (attention, shared expert, dense MLP)")
    ap.add_argument("--contract-dir", type=Path, default=None,
                    help="copy the six byte-locked contract files from this dir (e.g. the king) into every checkpoint")
    ap.add_argument("--val-raw", type=int, default=1, help="with --ema: also validate the raw weights (a second val pass); 0 = EMA only")
    ap.add_argument("--ema", type=float, default=0.0, help="> 0: keep an fp32 EMA of the trainable weights (decay per step, host memory); "
                                                             "val is reported for raw and EMA, checkpoints are the EMA weights")
    ap.add_argument("--loss-weights", default=None, help="per-source loss weights, e.g. text=1,math=2,code=2 (pool needs sources.npy)")
    ap.add_argument("--doc-mask", type=int, default=1, help="1: validator document masking (>= 14a9bb2): positions restart and "
                    "attention stops at EOS, fragment starts are not targets, losses per scored token; 0: plain causal 2048-token rows")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--val-n", type=int, default=512)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--start-val", type=float, default=None, help="skip the step-0 val pass and use this as the baseline (probes from an "
                    "already-measured start: saves ~5 min of idle GPUs per launch)")
    ap.add_argument("--save-sr", type=int, default=0, help="bf16 rounding of saved checkpoints: 0 = nearest (default), 1 = "
                    "stochastic (unbiased: sub-ulp updates survive in expectation), 2 = both (nearest as step-N, stochastic as sr-step-N)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-seqs", type=int, default=0,
                    help="continue a run's data order: drop the first N sequences of the --seed order (resume = steps done x seqs/step)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--state-every", type=int, default=0, help="write the exact training state (fp32 masters, Adam moments, EMA) "
                    "every N steps to <out>/state-N (the newest --state-keep are kept); 0 = only on a CONTROL stop/state")
    ap.add_argument("--state-keep", type=int, default=2)
    ap.add_argument("--resume-state", type=Path, default=None, help="continue a run exactly from <out>/state-N (same --out, "
                    "--steps, global batch, --freeze and world size; the loop continues at step N+1 with the same data order)")
    ap.add_argument("--no-ckpt-last", type=int, default=0, help="keep the activations of the top N decoder layers (no "
                    "recompute in backward; uses HBM headroom)")
    ap.add_argument("--ema-on-gpu", type=int, default=0, help="1: keep the fp32 EMA in HBM instead of pinned host memory")
    ap.add_argument("--fuse-gate-up", type=int, default=0, help="1: one grouped GEMM over the concatenated gate/up weights")
    ap.add_argument("--fused-ce", type=int, default=0, help="1: chunked lm_head + CE inside the forward (no full logits)")
    ap.add_argument("--inplace-combine", type=int, default=0, help="1: accumulate the routed experts' outputs in place")
    ap.add_argument("--combine", default="chunked", choices=["chunked", "bmm", "bmm16"], help="routed-expert "
                    "combine: chunked fp32 index_add (validator numerics), one gather + batched fp32 matmul (same fp32 sum), "
                    "or bmm16 (bf16 tensor cores, fp32 accumulation, routing weights rounded to bf16)")
    ap.add_argument("--log-micro-gnorm", type=int, default=1, help="1: log |g| of the first micro-step next to the full "
                    "step's (gradient signal-to-noise; ~10 ms per step)")
    ap.add_argument("--spike-factor", type=float, default=5.0, help="skip the optimizer step when |g| exceeds this multiple "
                    "of the median of the last 20 accepted steps (after warmup); non-finite gradients are always skipped. 0 = off")
    args = ap.parse_args()
    global INPLACE_COMBINE, COMBINE, FUSE_GATE_UP, FUSED_CE
    INPLACE_COMBINE, COMBINE = bool(args.inplace_combine), args.combine
    FUSE_GATE_UP, FUSED_CE = bool(args.fuse_gate_up), bool(args.fused_ce)

    import datetime
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))  # saves / pauses can exceed the 10-min default
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    lora = args.lora_r > 0
    lora_targets = {t for t in args.lora_targets.split(",") if t}
    if lora and dtype != torch.bfloat16:
        if rank == 0:
            print("LoRA: base weights are frozen, using bf16 for them (--dtype fp32 ignored)", flush=True)
        dtype = torch.bfloat16
    t0 = time.time()

    model = build_model(args.init, dtype, rank, args.attn, args.lora_r, args.lora_alpha, lora_targets, args.debug_layers)
    freeze = re.compile(args.freeze) if args.freeze else None
    only = re.compile(args.train_only) if args.train_only else None
    n_train = n_all = 0
    for name, p in model.named_parameters():
        train = (only.search(name) is not None) if only else True
        if lora:
            train = train and ".lora" in name
        if freeze and freeze.search(name):
            train = False
        p.requires_grad_(train)
        n_all += p.numel()
        n_train += p.numel() if train else 0
    if rank == 0:
        print(f"model built in {time.time() - t0:.0f}s: {n_all / 1e9:.1f}B params, {n_train / 1e9:.2f}B trainable, master {args.dtype}, attn {args.attn}", flush=True)

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import CustomPolicy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (CheckpointImpl, apply_activation_checkpointing,
                                                                             checkpoint_wrapper)
    import functools
    layer_cls = type(model.model.layers[0])
    layer_index = {id(layer): i for i, layer in enumerate(model.model.layers)}  # FSDP keeps these module objects
    mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)  # buffers (rotary inv_freq) stay fp32
    mp_lora = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    def wrap_policy(m: nn.Module):
        if isinstance(m, LORA_CLASSES):
            return {"mixed_precision": mp_lora}  # own unit: fp32 master + fp32 gradient reduction, bf16 compute
        return isinstance(m, layer_cls)

    model = FSDP(model, auto_wrap_policy=CustomPolicy(wrap_policy), sharding_strategy=ShardingStrategy.FULL_SHARD,
                 mixed_precision=mp, device_id=device, sync_module_states=True, use_orig_params=True,
                 limit_all_gathers=bool(args.limit_all_gathers), forward_prefetch=True,
                 param_init_fn=(lambda m: m.to_empty(device=device, recurse=False)) if rank != 0 else None)
    # selective activation checkpointing: the top --no-ckpt-last decoder layers keep their activations (no recompute in
    # backward) when HBM allows; every other layer recomputes its forward
    first_kept = len(layer_index) - args.no_ckpt_last
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=functools.partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                                   check_fn=lambda m: isinstance(m, layer_cls) and layer_index.get(id(m), 0) < first_kept)
    model.eval()  # no dropout anywhere; MiMo-V2 noaux_tc routing refuses train mode — gradients flow regardless
    if rank == 0:
        print(f"FSDP ready in {time.time() - t0:.0f}s; GPU mem {torch.cuda.memory_allocated() / 2**30:.1f} GB", flush=True)

    expert_params = [p for n, p in model.named_parameters() if p.requires_grad and ".mlp.experts." in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and ".mlp.experts." not in n]
    groups = [{"params": expert_params, "scale": 1.0, "lr": args.lr}, {"params": other_params, "scale": args.lr_dense_frac, "lr": args.lr * args.lr_dense_frac}]
    groups = [g for g in groups if g["params"]]
    if rank == 0:
        print(f"param groups: experts {world * sum(p.numel() for p in expert_params) / 1e6:.0f}M at lr x1, "
              f"other {world * sum(p.numel() for p in other_params) / 1e6:.0f}M at lr x{args.lr_dense_frac}", flush=True)
    if args.optim == "adamw8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f"bitsandbytes unavailable ({exc}); using torch AdamW (fp32 states)", flush=True)
            opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd, foreach=True)
    else:
        opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd, foreach=True)
    offload = StateOffload(opt) if args.offload_opt else None
    saver = AsyncSaver(rank, world)
    ema = None
    if args.ema > 0:
        t_e = time.time()
        ema = (DeviceEMA if args.ema_on_gpu else HostEMA)([p for p in model.parameters() if p.requires_grad], args.ema)
        if rank == 0:
            print(f"EMA {args.ema} of {world * sum(h.numel() for h in ema.host) / 1e9:.1f}B params in "
                  f"{'GPU memory' if args.ema_on_gpu else 'pinned host memory'} "
                  f"({time.time() - t_e:.0f}s); val reports EMA (raw in brackets), checkpoints are EMA weights", flush=True)
    if offload and rank == 0:
        print("optimizer states offloaded to pinned host memory between steps", flush=True)
    loss_w = torch.ones(len(SOURCES), device=device)
    if args.loss_weights:
        for kv in args.loss_weights.split(","):
            k, v = kv.split("=")
            loss_w[SOURCES.index(k.strip())] = float(v)
        if rank == 0:
            print(f"loss weights {dict(zip(SOURCES, loss_w.tolist()))}", flush=True)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        start = args.warmup if args.decay_start is None else max(args.warmup, args.decay_start)
        if step < start:
            return args.lr
        prog = (step - start) / max(1, args.steps - start)
        return args.lr * (args.min_lr_frac + (1 - args.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))

    pool = Pool(args.pool)
    val = Pool(args.val) if args.val else None
    rng = np.random.default_rng(args.seed)
    per_step = args.batch * args.accum * world
    need = args.steps * per_step
    order = np.concatenate([rng.permutation(pool.n) for _ in range(math.ceil((need + args.skip_seqs) / pool.n))])
    order = order[args.skip_seqs:args.skip_seqs + need]
    tokens_per_step = per_step * pool.seq_len
    if rank == 0:
        print(f"data: {pool.n} seqs x {pool.seq_len}; {per_step} seqs/step ({tokens_per_step / 1e6:.2f}M tokens); {args.steps} steps = "
              f"{need / pool.n:.2f} epochs; val {val.n if val else 0} seqs", flush=True)

    def evaluate() -> tuple[float, dict[str, float]]:
        """Held-out CE per scored token and the per-source means (the pool mirrors the competition mix, so the mean is
        the mix; token-weighted like the validator's verdict)."""
        if val is None:
            return float("nan"), {}
        n = min(args.val_n, val.n)
        mine = np.arange(rank, n, world)
        tot = torch.zeros(len(SOURCES), 2, device=device)
        with torch.no_grad():
            for s in range(0, len(mine), args.batch):
                ids, src = val.batch(mine[s:s + args.batch], device)
                ce, count = seq_ce(model, ids, doc_mask=bool(args.doc_mask))
                tot[:, 0].index_add_(0, src, ce.float())
                tot[:, 1].index_add_(0, src, count.float())
        dist.all_reduce(tot)
        per = {k: float(tot[i, 0] / tot[i, 1]) for i, k in enumerate(SOURCES) if tot[i, 1] > 0}
        return float(tot[:, 0].sum() / tot[:, 1].sum()), per

    log = (args.out / "train_log.jsonl") if rank == 0 else None
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    control = Control(args.out, rank, device)
    start_step, resumed = 0, {}
    if args.resume_state is not None:
        t_r = time.time()
        start_step, resumed = load_state(args.resume_state, rank, world, trainable, opt, offload, ema, device)
        control.lr_scale = float(resumed.get("lr_scale", 1.0))
        control.anneal_k = int(resumed.get("anneal_k", 0))
        control.anneal_from_step, control.anneal_from_lr = resumed.get("anneal_from_step"), resumed.get("anneal_from_lr")
        control.id = resumed.get("control_id")
        control.mtime = None  # re-read CONTROL.json once: the persistent fields written while the run was down apply now
        if rank == 0:
            print(f"resumed exactly from {args.resume_state} (step {start_step}, lr_scale {control.lr_scale}, "
                  f"anneal {control.anneal_k}) in {time.time() - t_r:.0f}s", flush=True)

    def eff_lr(step0: int) -> float:
        """LR of the first group at 0-based step `step0`: the schedule x CONTROL lr_scale, under a CONTROL anneal."""
        lr = lr_at(step0) * control.lr_scale
        if control.anneal_k > 0 and control.anneal_from_step is not None:
            floor = args.lr * args.min_lr_frac * control.lr_scale
            prog = min(1.0, (step0 - control.anneal_from_step) / control.anneal_k)
            lr = min(lr, floor + (control.anneal_from_lr - floor) * 0.5 * (1 + math.cos(math.pi * prog)))
        return lr

    if resumed.get("v0") is not None:
        v0, per0 = float(resumed["v0"]), dict(resumed.get("per0") or {})
    else:
        v0, per0 = (args.start_val, {}) if args.start_val is not None else evaluate()
    fmt_per = lambda per: " ".join(f"{k} {v:.5f}" for k, v in per.items())  # noqa: E731
    if rank == 0 and not resumed:
        print(f"step 0: val {v0:.5f} [{fmt_per(per0)}] ({time.time() - t0:.0f}s)", flush=True)
        log.open("a").write(json.dumps({"step": 0, "val": v0, "val_split": per0, "t": time.time()}) + "\n")
    t_train = time.time()
    run = torch.zeros(1, device=device)
    gnorm_hist: list[float] = list(resumed.get("gnorm_hist") or [])
    skipped, bad_in_row = int(resumed.get("skipped", 0)), 0

    def state_extra() -> dict:
        return {"v0": v0, "per0": per0, "lr_scale": control.lr_scale, "anneal_k": control.anneal_k,
                "anneal_from_step": control.anneal_from_step, "anneal_from_lr": control.anneal_from_lr,
                "control_id": control.id, "gnorm_hist": gnorm_hist[-20:], "skipped": skipped}

    state_saver = AsyncStateSaver(rank, world)

    def write_state(step: int, wait: bool = False) -> None:
        held = state_saver.save(args.out / f"state-{step}", step, trainable, opt, ema, state_extra(), keep=args.state_keep)
        if rank == 0:
            print(f"state-{step} snapshot in {held:.0f}s, files written in the background", flush=True)
        if wait:
            state_saver.wait()
            dist.barrier()

    def apply_anneal(k: int, step_done: int) -> None:
        """Anneal over k steps starting after step `step_done` (0 cancels)."""
        control.anneal_k = k
        if k:
            control.anneal_from_step, control.anneal_from_lr = step_done, lr_at(step_done) * control.lr_scale
        else:
            control.anneal_from_step = control.anneal_from_lr = None

    pre = control.poll()  # persistent fields written before the launch (e.g. the LR scale of a resume) apply from the first step
    if pre is not None:
        if pre["lr_scale"] is not None:
            control.lr_scale = pre["lr_scale"]
        if pre["anneal"] is not None and int(pre["anneal"]) != control.anneal_k:
            apply_anneal(int(pre["anneal"]), start_step)
        control.write_ack(start_step, eff_lr(start_step), "stopped before the first step" if pre["stop"] else "at launch")
        if pre["stop"]:  # asked to stop while loading: nothing new to save (the start / resumed state is unchanged)
            if rank == 0:
                print("stopped by CONTROL before the first step", flush=True)
            dist.barrier()
            dist.destroy_process_group()
            return
    for step in range(start_step + 1, args.steps + 1):
        g_micro = None
        t_step = time.time()  # after the previous step's val/save/pause, so step_s is pure training time
        lr_now = eff_lr(step - 1)
        for g in opt.param_groups:
            g["lr"] = lr_now * g["scale"]
        prof = None
        if args.profile and step == 1:
            from torch.profiler import ProfilerActivity, profile
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=bool(os.environ.get("PROFILE_SHAPES")))
        for a in range(args.accum):
            if prof is not None and a == 1:
                torch.cuda.synchronize(); prof.__enter__(); t_prof = time.time()
            base = ((step - 1) * args.accum + a) * args.batch * world + rank * args.batch
            ids, src = pool.batch(order[base: base + args.batch], device)
            ce, count = seq_ce(model, ids, doc_mask=bool(args.doc_mask))
            w = loss_w[src] * count  # token-weighted, like the validator's verdict
            loss = (ce * loss_w[src]).sum() / w.sum() / args.accum
            loss.backward()
            if a == 0 and args.accum > 1 and args.log_micro_gnorm:
                # gradient signal vs noise (McCandlish et al. 2018): after the first micro-step .grad holds g_b / accum, the
                # gradient of b = batch x world sequences; with the full step's g_B (B = accum x b) over many steps,
                # |G|^2 = (K |g_B|^2 - |g_b|^2) / (K - 1) and tr(Sigma)/B = (|g_b|^2 - |g_B|^2) / (K - 1), K = accum
                sq = torch.zeros(1, device=device, dtype=torch.float64)
                for p_ in trainable:
                    if p_.grad is not None and p_.grad.numel():
                        # a reduction kernel, no full-size temporary (the run sits at 226 of 262 GB per GPU)
                        sq += torch.linalg.vector_norm(p_.grad.detach(), 2, dtype=torch.float32).double() ** 2
                dist.all_reduce(sq)
                g_micro = math.sqrt(float(sq)) * args.accum
            run += (ce.detach().sum() / count.sum() / args.accum)  # logged loss: CE per scored token, unweighted by source
            if prof is not None and a == args.profile:
                torch.cuda.synchronize(); prof.__exit__(None, None, None)
                if rank == 0:
                    print(f"profiled micro-steps 1..{args.profile}: {(time.time() - t_prof) / args.profile:.2f}s per micro-step", flush=True)
                    print(prof.key_averages(group_by_input_shape=bool(os.environ.get("PROFILE_SHAPES"))).table(
                        sort_by=os.environ.get("PROFILE_SORT", "self_cuda_time_total"), row_limit=45), flush=True)
                prof = None
        gnorm = model.clip_grad_norm_(args.clip) if args.clip > 0 else torch.tensor(0.0)
        dist.all_reduce(run)
        mean = float(run) / world
        run.zero_()
        g_now = float(gnorm)  # the total norm is all-reduced by FSDP: every rank takes the same decision below
        # step guard: a non-finite gradient would turn every weight into NaN (and every later checkpoint with it); a spike
        # far above the recent |g| is skipped as well. Three non-finite steps in a row: exit (the autopilot resumes the
        # newest exact state at a lower LR) instead of training on garbage.
        median = sorted(gnorm_hist)[len(gnorm_hist) // 2] if len(gnorm_hist) >= 10 else None
        spike = (args.spike_factor > 0 and median is not None and step > args.warmup and g_now > args.spike_factor * median)
        bad = not (math.isfinite(g_now) and math.isfinite(mean))
        if bad or spike:
            opt.zero_grad(set_to_none=True)
            skipped += 1
            bad_in_row = bad_in_row + 1 if bad else 0
            if rank == 0:
                why = "non-finite gradient/loss" if bad else f"|g| {g_now:.3f} > {args.spike_factor} x median {median:.3f}"
                print(f"step {step}: SKIPPED optimizer step ({why}); {skipped} skipped so far", flush=True)
            if bad_in_row >= 3:
                if rank == 0:
                    print(f"step {step}: 3 non-finite steps in a row; exiting without saving (resume from the newest state)", flush=True)
                saver.wait()
                state_saver.wait()  # a state in flight was snapshot before the bad steps: let it complete
                dist.barrier()
                dist.destroy_process_group()
                raise SystemExit(5)
        else:
            bad_in_row = 0
            gnorm_hist = (gnorm_hist + [g_now])[-20:]
            if offload:
                offload.to_device(device)
            opt.step()
            if offload:
                offload.to_host()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                t_e = time.time()
                ema.update()
                if rank == 0 and step == start_step + 1:
                    print(f"EMA update {time.time() - t_e:.1f}s per step", flush=True)
        # every step goes to train_log.jsonl (with wall time and this step's own throughput, for the live dashboard);
        # the console line stays every --log-every steps (and the first step, which chain_runs.py parses)
        now = time.time()
        el = now - t_train
        done = step - start_step
        tps = done * tokens_per_step / el  # cumulative over this process, includes val/save/pause time
        tps_step = tokens_per_step / (now - t_step)
        eta = (args.steps - step) * el / done
        if rank == 0:
            if step % args.log_every == 0 or step == 1 or step == start_step + 1:
                print(f"step {step}/{args.steps}: train {mean:.5f} |g| {g_now:.3f} lr {lr_now:.2e} "
                      f"{tps / 1e3:.1f}k tok/s eta {eta / 3600:.2f}h mem {torch.cuda.max_memory_allocated() / 2**30:.1f}/"
                      f"{torch.cuda.max_memory_reserved() / 2**30:.1f} GB", flush=True)
            log.open("a").write(json.dumps({"step": step, "train": mean, "gnorm": g_now, "lr": lr_now, "lr_scale": control.lr_scale,
                                            "skipped": bool(bad or spike), "tok_s": tps, "tok_s_step": tps_step,
                                            **({"g_micro": g_micro} if g_micro is not None else {}),
                                            "step_s": now - t_step, "t": now}) + "\n")
        # live control: applied between steps, identically on every rank
        ctl = control.poll()
        want_save = want_state = want_stop = False
        if ctl is not None:
            notes = []
            if ctl["lr_scale"] is not None and ctl["lr_scale"] != control.lr_scale:
                notes.append(f"lr_scale {control.lr_scale:g} -> {ctl['lr_scale']:g}")
                control.lr_scale = ctl["lr_scale"]
            if ctl["anneal"] is not None and int(ctl["anneal"]) != control.anneal_k:
                apply_anneal(int(ctl["anneal"]), step)
                notes.append(f"anneal over {control.anneal_k} steps from step {step}" if control.anneal_k else "anneal cancelled")
            want_save, want_state, want_stop = ctl["save"], ctl["state"], ctl["stop"]
            notes += [k for k, v in (("save", want_save), ("state", want_state), ("stop", want_stop)) if v]
            control.write_ack(step, eff_lr(step), "; ".join(notes) or "no change")
            if rank == 0 and notes:
                print(f"step {step}: CONTROL {'; '.join(notes)}", flush=True)
        do_eval = step % args.eval_every == 0 or step == args.steps
        do_save = step % args.save_every == 0 or step == args.steps or want_save or want_stop
        v_raw = per_raw = None
        if ema is not None and (do_eval or do_save):
            if do_eval and args.val_raw:
                v_raw, per_raw = evaluate()  # raw weights (optional: costs a second val pass)
            ema.swap()  # one swap pair: val and save both run on the EMA weights
        if do_eval:
            v, per = evaluate()
            extra, raw = "", {}
            if v_raw is not None:
                extra, raw = f" [raw {v_raw:.5f}, delta {v_raw - v0:+.5f}]", {"val_raw": v_raw, "val_split_raw": per_raw}
            if rank == 0:
                deltas = " ".join(f"{k} {per[k] - per0[k]:+.5f}" for k in per if k in per0)
                print(f"step {step}: val {v:.5f} (start {v0:.5f}, delta {v - v0:+.5f}) [{deltas}]{extra}", flush=True)
                log.open("a").write(json.dumps({"step": step, "val": v, "val_split": per, **raw, "lr": lr_now,
                                                "lr_scale": control.lr_scale, "t": time.time()}) + "\n")
        if do_save:
            ck = args.out / (f"step-{step}" if step != args.steps else "final")
            ls = args.lora_alpha / args.lora_r if lora else 0.0
            saver.save(model, args.init, ck, lora_scaling=ls, contract_dir=args.contract_dir, sr=args.save_sr == 1)
            if args.save_sr == 2:  # A/B of the rounding: the same EMA weights, stochastically rounded (waits for the first writer)
                try:
                    saver.save(model, args.init, ck.parent / f"sr-{ck.name}", lora_scaling=ls, contract_dir=args.contract_dir, sr=True)
                except Exception as e:  # the A/B twin is optional: training goes on without it
                    if rank == 0:
                        print(f"sr twin save failed: {str(e)[:200]}", flush=True)
        if ema is not None and (do_eval or do_save):
            ema.swap()
        # exact state on the raw (training) weights, after the EMA swap-back; never at the last step (the run is over)
        if step < args.steps and (want_state or want_stop or (args.state_every and step % args.state_every == 0)):
            write_state(step, wait=want_stop)  # a stop exits only once its state is complete
        if want_stop:
            if rank == 0:
                print(f"step {step}: stopped by CONTROL (checkpoint step-{step} + state-{step})", flush=True)
            break
        pause_if_requested(args.out, rank, device)
    saver.wait()  # the last checkpoint's files are written before the process group goes away
    state_saver.wait()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
