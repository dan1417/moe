"""Attention backends for parity scoring.

``eager``  -- the locked MiMo ``eager_attention_forward`` (validator-supported).
``fa4sim`` -- a FlashAttention-4-faithful reference for GPUs FA4 does not support
              (e.g. sm_120). Production evaluators run ``flash_attention_4``; FA4
              keeps QK^T scores in fp32, applies the learnable sink in the softmax
              denominator, feeds bf16 probabilities to the PV GEMM with fp32
              accumulation and normalizes in fp32. Eager instead rounds the scores to
              bf16 before the softmax, which perturbs per-token losses noticeably.

Sliding-window layers (window 128) are computed blockwise: each 128-query block
only sees its own and the previous key block, so scores cover 256 keys instead of
the full 2048 -- the same dot products and sums, 8x less work.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FA4SIM = "fa4sim"
# Upper bound on one fp32 score tensor; full-attention batches are chunked below it.
SCORE_CHUNK_BYTES = 512 * 1024**2
_mask_cache: dict[tuple, torch.Tensor] = {}


def _causal_blocked(seq: int, window: int | None, device: torch.device) -> torch.Tensor:
    """True where attention is NOT allowed (transformers' causal/sliding semantics)."""
    key = ("dense", seq, window, device)
    mask = _mask_cache.get(key)
    if mask is None:
        positions = torch.arange(seq, device=device)
        allowed = positions[None, :] <= positions[:, None]
        if window is not None:
            # sliding_window_overlay: kv_idx > q_idx - window (128 keys incl. self), the
            # same span FA4 gets from window_size=(window - 1, window - 1) plus causal.
            allowed &= positions[None, :] > positions[:, None] - window
        mask = _mask_cache[key] = ~allowed
    return mask


def _band_blocked(n_blocks: int, block: int, groups: int, device: torch.device) -> torch.Tensor:
    """[n_blocks, groups*block, 2*block] mask for the blockwise sliding window."""
    key = ("band", n_blocks, block, groups, device)
    mask = _mask_cache.get(key)
    if mask is None:
        w = torch.arange(block, device=device)[:, None]  # query offset in its block
        j = torch.arange(2 * block, device=device)[None, :]  # key offset in the window
        allowed = (j > w) & (j <= w + block)  # q - k in [0, block): the 128-key band
        allowed = allowed[None].repeat(n_blocks, 1, 1)
        allowed[0, :, :block] = False  # block 0's previous-block keys are padding
        mask = ~allowed.repeat(1, groups, 1)
        _mask_cache[key] = mask
    return mask


def _softmax_pv(scores, sink, value_windows, out_dtype):
    """FA-style: fp32 probs vs (max incl. sink), bf16 P @ V with fp32 accum, fp32 norm."""
    row_max = scores.amax(dim=-1, keepdim=True)
    if sink is not None:
        row_max = torch.maximum(row_max, sink)
    probs = scores.sub_(row_max).exp_()
    denom = probs.sum(dim=-1, keepdim=True)
    if sink is not None:
        denom = denom + torch.exp(sink - row_max)
    context = torch.bmm(
        probs.to(value_windows.dtype).flatten(0, -3), value_windows.flatten(0, -3), out_dtype=torch.float32
    ).view(*probs.shape[:-1], value_windows.shape[-1])
    return context.div_(denom).to(out_dtype)


def _sliding(query, key, value, scaling, window, s_aux):
    bsz, n_heads, seq, head_dim = query.shape
    n_kv, v_dim = key.shape[1], value.shape[-1]
    groups = n_heads // n_kv
    n_blocks = seq // window
    # queries: [B, KV, NB, G*W, D]
    q = query.reshape(bsz, n_kv, groups, n_blocks, window, head_dim).permute(0, 1, 3, 2, 4, 5)
    q = q.reshape(bsz * n_kv * n_blocks, groups * window, head_dim)
    # keys/values of blocks (n-1, n): [B*KV*NB, D, 2W] and [B*KV*NB, 2W, Dv]
    k = F.pad(key, (0, 0, window, 0)).unfold(2, 2 * window, window)
    k = k.reshape(bsz * n_kv * n_blocks, head_dim, 2 * window)
    v = F.pad(value, (0, 0, window, 0)).unfold(2, 2 * window, window).transpose(-1, -2)
    v = v.reshape(bsz * n_kv * n_blocks, 2 * window, v_dim)
    scores = torch.bmm(q, k, out_dtype=torch.float32).mul_(scaling)
    scores = scores.view(bsz, n_kv, n_blocks, groups * window, 2 * window)
    scores.masked_fill_(_band_blocked(n_blocks, window, groups, query.device), float("-inf"))
    sink = None
    if s_aux is not None:
        sink = s_aux.float().view(1, n_kv, 1, groups, 1, 1).expand(1, n_kv, 1, groups, window, 1)
        sink = sink.reshape(1, n_kv, 1, groups * window, 1)
    v = v.view(bsz, n_kv, n_blocks, 2 * window, v_dim)
    context = _softmax_pv(scores, sink, v, query.dtype)  # [B, KV, NB, G*W, Dv]
    context = context.view(bsz, n_kv, n_blocks, groups, window, v_dim).permute(0, 2, 4, 1, 3, 5)
    return context.reshape(bsz, seq, n_heads, v_dim)


def _dense(query, key, value, scaling, window, s_aux):
    bsz, n_heads, seq, head_dim = query.shape
    n_kv, v_dim = key.shape[1], value.shape[-1]
    groups = n_heads // n_kv
    # one GEMM per (batch, kv head): its `groups` query heads stacked along rows
    q = query.reshape(bsz * n_kv, groups * seq, head_dim)
    k = key.reshape(bsz * n_kv, seq, head_dim).transpose(1, 2)
    v = value.reshape(bsz * n_kv, seq, v_dim)
    blocked = _causal_blocked(seq, window, query.device)
    sink_all = None if s_aux is None else s_aux.float().view(n_kv, groups, 1, 1)
    out = torch.empty(bsz * n_kv, groups * seq, v_dim, dtype=query.dtype, device=query.device)
    step = max(1, SCORE_CHUNK_BYTES // (groups * seq * seq * 4))
    for start in range(0, bsz * n_kv, step):
        end = min(start + step, bsz * n_kv)
        count = end - start
        scores = torch.bmm(q[start:end], k[start:end], out_dtype=torch.float32).mul_(scaling)
        scores.view(count, groups, seq, seq).masked_fill_(blocked, float("-inf"))
        sink = None
        if sink_all is not None:
            rows = torch.arange(start, end, device=query.device) % n_kv
            sink = sink_all[rows].expand(count, groups, seq, 1).reshape(count, groups * seq, 1)
        out[start:end] = _softmax_pv(scores, sink, v[start:end], query.dtype)
    out = out.view(bsz, n_kv, groups, seq, v_dim).permute(0, 3, 1, 2, 4)
    return out.reshape(bsz, seq, n_heads, v_dim)


def fa4sim_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
):
    if attention_mask is not None and attention_mask.dim() == 2 and not bool(attention_mask.all()):
        raise ValueError("fa4sim supports unpadded evaluation sequences only")
    scaling = query.shape[-1] ** -0.5 if scaling is None else scaling
    seq = query.shape[2]
    if sliding_window is not None and seq % sliding_window == 0 and seq > sliding_window:
        return _sliding(query, key, value, scaling, sliding_window, s_aux), None
    return _dense(query, key, value, scaling, sliding_window, s_aux), None


def fa4sim_reference(query, key, value, scaling, sliding_window=None, s_aux=None):
    """Straightforward per-head-group version; the definition fa4sim must match."""
    bsz, n_heads, seq, head_dim = query.shape
    n_kv, v_dim = key.shape[1], value.shape[-1]
    groups = n_heads // n_kv
    blocked = _causal_blocked(seq, sliding_window, query.device)
    out = torch.empty(bsz, seq, n_heads, v_dim, dtype=query.dtype, device=query.device)
    for b in range(bsz):
        for kv in range(n_kv):
            heads = slice(kv * groups, (kv + 1) * groups)
            scores = torch.bmm(
                query[b, heads], key[b, kv].transpose(0, 1).expand(groups, head_dim, seq), out_dtype=torch.float32
            ).mul_(scaling)
            scores.masked_fill_(blocked, float("-inf"))
            sink = None if s_aux is None else s_aux[heads].float().view(groups, 1, 1)
            context = _softmax_pv(scores, sink, value[b, kv].expand(groups, seq, v_dim), query.dtype)
            out[b, :, heads, :] = context.transpose(0, 1)
    return out


def register_fa4sim() -> None:
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask

    AttentionInterface.register(FA4SIM, fa4sim_attention_forward)
    ALL_MASK_ATTENTION_FUNCTIONS.register(FA4SIM, flash_attention_mask)
