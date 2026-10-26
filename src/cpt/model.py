"""MiMo-V2 model for full-parameter CPT under FSDP2.

The locked remote code (config.json / modeling_mimo_v2.py) is used as-is except for
three runtime patches that never touch the exported files:
  * each MiMoV2MoE's 256 ``experts`` become one :class:`GroupedExperts` module whose
    stacked weights run through ``torch._grouped_mm`` (bit-identical per expert to
    the per-expert Linear layers, including autograd) and whose combine matches the
    validator (fp32 router weights, fp32 ``index_add_`` accumulation);
  * the noaux_tc gate, which refuses ``training=True``, runs its inference routing;
  * decoder layers use non-reentrant activation checkpointing.
Checkpoint key names are mapped back to ``...mlp.experts.{e}.{proj}.weight`` on load
and export, so exported checkpoints are indistinguishable from the king's layout.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import sys
import types
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

log = logging.getLogger(__name__)

GROUPED_PARAM = re.compile(r"^(model\.layers\.\d+\.mlp)\.grouped\.(gate_proj|up_proj|down_proj)$")
EXPERT_KEY = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


_CONFIG_SKIP = {"architectures", "auto_map", "transformers_version", "torch_dtype", "dtype", "quantization_config",
                "vision_config", "audio_config", "processor_config", "_name_or_path"}


def load_config(king_dir: Path, *, n_layers: int | None = None, code_dir: Path | None = None):
    """King config; ``n_layers`` keeps only the first layers (smoke tests only).

    ``code_dir``: build with that checkpoint's (validator-matched) MiMo-V2 code and take every language-model
    field from ``king_dir/config.json`` -- e.g. to run a larger MiMo-V2 teacher (MiMo-V2.5-Base) whose own
    remote code also carries vision/audio towers."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(code_dir or king_dir), trust_remote_code=True)
    if code_dir is not None:
        source = json.loads((Path(king_dir) / "config.json").read_text())
        rope = dict(getattr(config, "rope_parameters", None) or {})
        for key, value in source.items():
            # rope_scaling aliases transformers-5 rope_parameters: assigning it would drop rope_theta and
            # partial_rotary_factor (-> full-width RoPE, a scrambled model). Rebuilt explicitly below.
            if key not in _CONFIG_SKIP and key not in ("rope_scaling", "rope_parameters") and hasattr(config, key):
                setattr(config, key, value)
        rope["rope_theta"] = source.get("rope_theta", rope.get("rope_theta"))
        rope["partial_rotary_factor"] = source.get("partial_rotary_factor", rope.get("partial_rotary_factor"))
        config.rope_parameters = rope
        if not source.get("n_shared_experts"):
            config.n_shared_experts = None
    config.use_cache = False
    if n_layers is not None and n_layers < config.num_hidden_layers:
        config.num_hidden_layers = n_layers
        config.hybrid_layer_pattern = list(config.hybrid_layer_pattern[:n_layers])
        config.moe_layer_freq = list(config.moe_layer_freq[:n_layers])
        if getattr(config, "layer_types", None):
            config.layer_types = list(config.layer_types[:n_layers])
    return config


class GroupedExperts(nn.Module):
    """Routed experts as stacked [E, out, in] weights (FSDP2 shards the expert dim)."""

    def __init__(self, n_experts: int, hidden: int, inter: int, act_fn, *, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.gate_proj = nn.Parameter(torch.empty(n_experts, inter, hidden, **factory))
        self.up_proj = nn.Parameter(torch.empty(n_experts, inter, hidden, **factory))
        self.down_proj = nn.Parameter(torch.empty(n_experts, hidden, inter, **factory))
        self.act_fn = act_fn

    def forward(self, routed_inputs: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        gate = torch._grouped_mm(routed_inputs, self.gate_proj.transpose(1, 2), offs=offsets)
        up = torch._grouped_mm(routed_inputs, self.up_proj.transpose(1, 2), offs=offsets)
        return torch._grouped_mm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2), offs=offsets)


def _moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    flat = hidden_states.reshape(-1, shape[-1])
    n_tokens, top_k = topk_indices.shape
    n_experts = self.grouped.gate_proj.shape[0]
    flat_experts = topk_indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_tokens = torch.arange(n_tokens, device=flat.device).repeat_interleave(top_k)[order]
    sorted_weights = topk_weights.reshape(-1)[order]
    counts = torch.zeros(n_experts, dtype=torch.int64, device=flat.device)
    counts.scatter_add_(0, flat_experts, torch.ones_like(flat_experts))
    offsets = torch.cumsum(counts, 0).to(torch.int32)
    routed = self.grouped(flat[sorted_tokens], offsets)
    combined = torch.zeros(flat.shape, dtype=sorted_weights.dtype, device=flat.device)
    combined = combined.index_add(0, sorted_tokens, routed * sorted_weights.unsqueeze(-1))
    out = combined.to(flat.dtype)
    if self.shared_experts is not None:
        out = out + self.shared_experts(flat)
    return out.view(shape)


def _gate_training_forward(self, hidden_states: torch.Tensor):
    # MiMoV2MoEGate raises in training mode; routing itself is mode-independent.
    was_training = self.training
    self.training = False
    try:
        return type(self).forward(self, hidden_states)
    finally:
        self.training = was_training


def _checkpointed_forward(self, *args, **kwargs):
    return checkpoint(self._cpt_inner_forward, *args, use_reentrant=False, **kwargs)


CE_CHUNK = 512  # positions per lm_head chunk: [B, 512, 152576] fp32 logits instead of [B, 2047, 152576]


def _loss_forward(self, input_ids: torch.Tensor, kd=None):
    """Per-token next-token CE [B, T-1] in fp32 (the eval objective) for the whole causal LM.

    ``kd=(idx [B,T-1,k], logp [B,T-1,k], weight [B])`` -> (ce, ce + weight * soft-CE vs the teacher's top-k);
    with ``lm_head._cpt_topk = k`` set -> (top-k log-probs, top-k ids) of this model (teacher extraction)."""
    hidden = self.model(input_ids=input_ids, use_cache=False).last_hidden_state
    return self.lm_head(hidden[:, :-1], input_ids[:, 1:], kd)


def _head_loss_forward(self, hidden: torch.Tensor, labels: torch.Tensor, kd=None):
    """lm_head + CE in checkpointed position chunks: only one chunk's logits are ever alive.

    Runs inside the lm_head FSDP unit, so the gathered weight is used in the forward and in the
    backward recompute exactly like the decoder layers' checkpointed forwards. Per-token CE equals
    ``cross_entropy(lm_head(hidden).float())`` (bf16 logits, fp32 log-softmax)."""
    topk = int(getattr(self, "_cpt_topk", 0) or 0)
    slm = float(getattr(self, "_cpt_slm", 0.0) or 0.0)  # fraction of teacher-scored tokens kept (selective LM)

    def soft_ce(logp: torch.Tensor, idx: torch.Tensor, target_logp: torch.Tensor) -> torch.Tensor:
        """Cross-entropy against a top-k target WITH a tail bucket: the stored log-probs are exact (unnormalized over
        the top-k), and the remaining mass 1 - sum(top-k) is matched as one bucket. Renormalizing the top-k instead
        tells the model the tail has zero mass -> it squashes the tail and pays on the ~5% of true tokens outside it."""
        p_target = target_logp.float().exp()
        mass_target = (1.0 - p_target.sum(-1)).clamp_min(1e-6)
        picked = logp.gather(-1, idx)
        mass_model = (1.0 - picked.exp().sum(-1)).clamp_min(1e-6)
        return -(p_target * picked).sum(-1) - mass_target * torch.log(mass_model)

    def head_ce(h: torch.Tensor, y: torch.Tensor, *kd_chunk):
        logp = torch.log_softmax(torch.nn.functional.linear(h, self.weight).float(), dim=-1)
        if topk:
            return torch.topk(logp, topk, dim=-1)
        ce = -logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        if not kd_chunk:
            return ce
        if len(kd_chunk) == 3 and slm:  # selective LM (Rho-1): CE on the tokens with the largest excess loss
            kd_idx, kd_logp, kd_w = kd_chunk
            hit = kd_idx == y.unsqueeze(-1)
            ref = torch.where(hit.any(-1), -(kd_logp.float() * hit).sum(-1), torch.full_like(ce, 1e4))  # teacher CE of y
            excess = (ce.detach() - ref).masked_fill(kd_w.unsqueeze(-1) <= 0, float("-inf"))
            rows = kd_w > 0
            keep = torch.zeros_like(ce)
            if rows.any():
                cut = torch.quantile(excess[rows].float().flatten(), 1.0 - slm)
                keep = (excess >= cut).float()
            selected = keep * ce / slm  # E[weight] = 1 on teacher-scored rows
            return ce, torch.where(kd_w.unsqueeze(-1) > 0, selected, ce)
        if len(kd_chunk) == 3:  # CE + w * soft-CE(teacher)
            kd_idx, kd_logp, kd_w = kd_chunk
            soft = soft_ce(logp, kd_idx, kd_logp)
            return ce, ce + kd_w.unsqueeze(-1) * soft
        # ensemble distillation: soft-CE(p_mix) = (1-mix)*soft-CE(frozen student) + mix*soft-CE(teacher), whose optimum is
        # the probability mixture itself; KD rows: alpha*CE + beta*soft-CE(p_mix), other rows: CE
        t_idx, t_logp, s_idx, s_logp, row = kd_chunk
        alpha, beta, mix = getattr(self, "_cpt_kd", (0.5, 1.0, 0.5))
        soft_t = soft_ce(logp, t_idx, t_logp)
        soft_s = soft_ce(logp, s_idx, s_logp)
        kd_loss = alpha * ce + beta * ((1.0 - mix) * soft_s + mix * soft_t)
        row = row.unsqueeze(-1)
        return ce, (1.0 - row) * ce + row * kd_loss, soft_t, soft_s

    outs = []
    for start in range(0, hidden.shape[1], CE_CHUNK):
        part = slice(start, start + CE_CHUNK)
        args = [hidden[:, part], labels[:, part]]
        if kd is not None and not topk:
            args += [t[:, part] if t.dim() > 1 else t for t in kd]
        outs.append(checkpoint(head_ce, *args, use_reentrant=False) if torch.is_grad_enabled() else head_ce(*args))
    if isinstance(outs[0], tuple):
        return tuple(torch.cat(parts, dim=1) for parts in zip(*outs))
    return torch.cat(outs, dim=1)


def install_loss_head(model: nn.Module) -> None:
    """``model(input_ids)`` returns per-token CE [B, T-1]; call before ``fully_shard`` or after, either works."""
    model.forward = types.MethodType(_loss_forward, model)
    model.lm_head.forward = types.MethodType(_head_loss_forward, model.lm_head)


def patch_masking_compat(model: nn.Module) -> None:
    """Drop ``cache_position`` for mask builders that no longer take it (transformers 5.x).

    Same adaptation the validator applies (``engine.patch_mimo_masking_compat``).
    """
    module = sys.modules[type(model).__module__]
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(module, name)
        if getattr(function, "_cpt_compat", False) or "cache_position" in inspect.signature(function).parameters:
            continue

        def adapter(*args, _function=function, **kwargs):
            kwargs.pop("cache_position", None)
            return _function(*args, **kwargs)

        adapter._cpt_compat = True
        setattr(module, name, adapter)


def build_model(config, *, activation_checkpointing: bool = True) -> nn.Module:
    """Meta-device fp32 skeleton with grouped experts and training patches."""
    from transformers import AutoModelForCausalLM

    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    patch_masking_compat(model)
    model = model.to(torch.float32)
    for module in model.modules():
        name = type(module).__name__
        if name == "MiMoV2MoE":
            first = module.experts[0]
            module.grouped = GroupedExperts(
                len(module.experts),
                first.gate_proj.weight.shape[1],
                first.gate_proj.weight.shape[0],
                first.act_fn,
                device="meta",
                dtype=torch.float32,
            )
            del module.experts
            module.forward = types.MethodType(_moe_forward, module)
        elif name == "MiMoV2MoEGate":
            module.forward = types.MethodType(_gate_training_forward, module)
    if activation_checkpointing:
        for layer in model.model.layers:
            layer._cpt_inner_forward = layer.forward
            layer.forward = types.MethodType(_checkpointed_forward, layer)
    install_loss_head(model)
    model.config.use_cache = False
    return model


def set_attention(model: nn.Module, implementation: str) -> None:
    """Select the attention backend after construction (as the validator does)."""
    if implementation == "flash_attention_4":
        model.__class__._supports_flash_attn = True
    model.config._attn_implementation = implementation


def freeze(model: nn.Module, patterns: list[str]) -> tuple[int, int]:
    """Freeze parameters whose name matches any regex; return (trainable, frozen) counts."""
    compiled = [re.compile(p) for p in patterns]
    trainable = frozen = 0
    for name, param in model.named_parameters():
        if any(p.search(name) for p in compiled):
            param.requires_grad_(False)
            frozen += param.numel()
        else:
            trainable += param.numel()
    return trainable, frozen


def init_rotary_buffers(model: nn.Module, device: torch.device) -> None:
    """Recompute non-persistent RoPE buffers (``to_empty`` leaves them garbage)."""
    for module in model.modules():
        if type(module).__name__ == "MiMoV2RotaryEmbedding":
            inv_freq, scaling = module.rope_init_fn(module.config, device)
            module._buffers["inv_freq"] = inv_freq.to(device=device, dtype=torch.float32)
            module.original_inv_freq = module._buffers["inv_freq"]
            module.attention_scaling = scaling


def _local_rows(param) -> tuple[torch.Tensor, int]:
    """Local shard of a (possibly FSDP2 DTensor) parameter and its dim-0 offset."""
    from torch.distributed.tensor import DTensor

    if not isinstance(param, DTensor):
        return param.data, 0
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, param.placements)
    return param.to_local(), int(offset[0]) if offset else 0


QKV_KEY = re.compile(r"^model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight$")


def deinterleave_qkv(weight: torch.Tensor, config, layer: int, ckpt_tp: int) -> torch.Tensor:
    """Xiaomi's MiMo-V2 checkpoints store fused qkv_proj rows as ``ckpt_tp`` (= num_key_value_heads) tensor-parallel
    chunks [q_i | k_i | v_i]; the HF modeling code expects [all q | all k | all v] (SGLang de-interleaves on load).
    Our own checkpoints were trained with the HF layout and must not be passed through this."""
    swa = config.hybrid_layer_pattern[layer] == 1
    heads = config.swa_num_attention_heads if swa else config.num_attention_heads
    kv_heads = config.swa_num_key_value_heads if swa else config.num_key_value_heads
    head_dim = (getattr(config, "swa_head_dim", None) or config.head_dim) if swa else config.head_dim
    v_dim = (getattr(config, "swa_v_head_dim", None) or config.v_head_dim) if swa else config.v_head_dim
    q, k = heads // ckpt_tp * head_dim, max(1, kv_heads // ckpt_tp) * head_dim
    shards = weight.chunk(ckpt_tp, dim=0)
    return torch.cat([s[:q] for s in shards] + [s[q : q + k] for s in shards] + [s[q + k :] for s in shards], dim=0)


@torch.no_grad()
def load_checkpoint_shards(model: nn.Module, checkpoint_dir: Path, *, qkv_ckpt_tp: int | None = None) -> int:
    """Copy each rank's rows of every parameter straight from safetensors (no full copy).

    ``qkv_ckpt_tp``: de-interleave fused qkv_proj rows of an external MiMo-V2 checkpoint (see ``deinterleave_qkv``)."""
    from safetensors import safe_open

    weight_map = json.loads((checkpoint_dir / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def slice_of(key: str):
        filename = weight_map[key]
        if filename not in handles:
            handles[filename] = safe_open(str(checkpoint_dir / filename), framework="pt", device="cpu")
        return handles[filename].get_slice(key)

    def full(key: str) -> torch.Tensor:
        """Whole tensor; FP8 weights with 128x128 block scales (``<key>_scale_inv``) are dequantized to fp32."""
        tensor = slice_of(key)[:]
        scale_key = f"{key}_scale_inv"
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and scale_key in weight_map:
            scale = slice_of(scale_key)[:].to(torch.float32)
            rows_b, cols_b = -(-tensor.shape[0] // scale.shape[0]), -(-tensor.shape[1] // scale.shape[1])
            scale = scale.repeat_interleave(rows_b, 0)[: tensor.shape[0]].repeat_interleave(cols_b, 1)[:, : tensor.shape[1]]
            tensor = tensor.to(torch.float32) * scale
        return tensor

    quantized = any(k.endswith("_scale_inv") for k in weight_map)
    loaded = 0
    for name, param in model.named_parameters():
        local, offset = _local_rows(param)
        rows = local.shape[0] if local.dim() else 1
        grouped = GROUPED_PARAM.match(name)
        if grouped:
            prefix, proj = grouped.groups()
            for row in range(rows):
                key = f"{prefix}.experts.{offset + row}.{proj}.weight"
                local[row].copy_(full(key) if quantized else slice_of(key)[:])
                loaded += 1
        else:
            qkv = QKV_KEY.match(name) if qkv_ckpt_tp else None
            if qkv:
                data = deinterleave_qkv(full(name), model.config, int(qkv.group(1)), qkv_ckpt_tp)[offset : offset + rows]
            elif quantized:
                data = full(name)[offset : offset + rows] if local.dim() else full(name)
            else:
                data = slice_of(name)[offset : offset + rows] if local.dim() else slice_of(name)[:]
            if tuple(data.shape) != tuple(local.shape):
                raise RuntimeError(f"{name}: checkpoint rows {tuple(data.shape)} != local shard {tuple(local.shape)}")
            local.copy_(data)
            loaded += 1
    return loaded


def checkpoint_items(name: str, full: torch.Tensor):
    """Yield (checkpoint key, tensor) pairs for one full parameter."""
    grouped = GROUPED_PARAM.match(name)
    if grouped:
        prefix, proj = grouped.groups()
        for expert in range(full.shape[0]):
            yield f"{prefix}.experts.{expert}.{proj}.weight", full[expert]
    else:
        yield name, full
