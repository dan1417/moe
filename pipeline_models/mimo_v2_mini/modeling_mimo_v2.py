"""Depth-shrunk full-width MiMo-V2 causal LM for local pipeline testing.

Uses Teutonic-II 110B tensor shapes with fewer layers. Not byte-identical to
genesis remote code and not for Teutonic mining submission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from .configuration_mimo_v2 import MiMoV2Config
except ImportError:  # direct import when building from scripts/
    from configuration_mimo_v2 import MiMoV2Config

logger = logging.get_logger(__name__)


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


def _build_rope_cache(seq_len: int, head_dim: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class MiMoV2MoEGate(nn.Module):
    """DeepSeek-style grouped top-k gate (noaux_tc, sigmoid scores)."""

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.scoring_func = config.scoring_func
        self.topk_method = config.topk_method
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        nn.init.normal_(self.weight, std=config.initializer_range)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        logits = F.linear(flat.float(), self.weight.float(), None)
        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
        elif self.scoring_func == "softmax":
            scores = torch.softmax(logits, dim=-1)
        else:
            raise ValueError(f"Unsupported scoring_func={self.scoring_func!r}")

        if self.topk_method != "noaux_tc":
            raise NotImplementedError(f"Unsupported topk_method={self.topk_method!r}")

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        experts_per_group = self.n_routed_experts // self.n_group
        grouped = scores_for_choice.view(-1, self.n_group, experts_per_group)
        group_scores = grouped.topk(2, dim=-1).values.sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, experts_per_group)
            .reshape(-1, self.n_routed_experts)
        )
        masked = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(masked, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx.view(bsz, seqlen, self.top_k), topk_weight.view(bsz, seqlen, self.top_k)


class MiMoV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act = nn.SiLU() if hidden_act == "silu" else nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class MiMoV2MoE(nn.Module):
    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__()
        self.config = config
        self.gate = MiMoV2MoEGate(config)
        self.experts = nn.ModuleList(
            [MiMoV2MLP(config.hidden_size, config.moe_intermediate_size, config.hidden_act) for _ in range(config.n_routed_experts)]
        )
        if config.n_shared_experts > 0:
            self.shared_experts = MiMoV2MLP(
                config.hidden_size,
                config.moe_intermediate_size * config.n_shared_experts,
                config.hidden_act,
            )
        else:
            self.shared_experts = None

    def forward(self, hidden_states: Tensor) -> Tensor:
        bsz, seqlen, hidden = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden)
        topk_idx, topk_weight = self.gate(hidden_states)
        idx = topk_idx.reshape(-1, self.config.num_experts_per_tok)
        weight = topk_weight.reshape(-1, self.config.num_experts_per_tok).to(dtype=flat.dtype)
        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            mask = (idx == expert_id).any(dim=-1)
            if not bool(mask.any()):
                continue
            token_idx = mask.nonzero(as_tuple=False).squeeze(-1)
            expert_input = flat.index_select(0, token_idx)
            expert_out = expert(expert_input)
            for k in range(self.config.num_experts_per_tok):
                slot = (idx[token_idx, k] == expert_id).to(dtype=flat.dtype).unsqueeze(-1)
                out.index_add_(0, token_idx, expert_out * weight[token_idx, k].unsqueeze(-1) * slot)
        if self.shared_experts is not None:
            out = out + self.shared_experts(flat)
        return out.view(bsz, seqlen, hidden)


class MiMoV2Attention(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_swa = config.layer_is_swa(layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.layer_num_key_value_heads(layer_idx)
        self.head_dim = config.head_dim
        self.v_head_dim = config.v_head_dim
        self.q_size = self.num_heads * self.head_dim
        self.k_size = self.num_kv_heads * self.head_dim
        self.v_size = self.num_kv_heads * self.v_head_dim
        self.o_size = self.num_heads * self.v_head_dim
        self.qkv_proj = nn.Linear(config.hidden_size, self.q_size + self.k_size + self.v_size, bias=False)
        self.o_proj = nn.Linear(self.o_size, config.hidden_size, bias=False)
        self.attention_sink_bias = (
            nn.Parameter(torch.zeros(self.num_heads)) if config.layer_wants_sink_bias(layer_idx) else None
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        bsz, seqlen, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = torch.split(qkv, [self.q_size, self.k_size, self.v_size], dim=-1)
        q = q.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_kv_heads, self.v_head_dim).transpose(1, 2)

        if position_ids is None:
            position_ids = torch.arange(seqlen, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = _build_rope_cache(seqlen, self.head_dim, hidden_states.device, hidden_states.dtype)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        if self.is_swa and self.config.sliding_window > 0:
            window = int(self.config.sliding_window)
            idx = torch.arange(seqlen, device=hidden_states.device)
            dist = idx.unsqueeze(0) - idx.unsqueeze(1)
            swa_mask = dist < 0
            swa_mask = swa_mask | (dist >= window)
            attn_scores = attn_scores.masked_fill(swa_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        if self.attention_sink_bias is not None:
            sink = self.attention_sink_bias.view(1, -1, 1, 1).to(attn_scores.dtype)
            attn_scores[..., :1] = attn_scores[..., :1] + sink
        attn_probs = torch.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).reshape(bsz, seqlen, self.o_size)
        return self.o_proj(attn_out)


class MiMoV2DecoderLayer(nn.Module):
    def __init__(self, config: MiMoV2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MiMoV2Attention(config, layer_idx)
        self.input_layernorm = nn.Parameter(torch.ones(config.hidden_size))
        self.post_attention_layernorm = nn.Parameter(torch.ones(config.hidden_size))
        if config.layer_is_moe(layer_idx):
            self.mlp = MiMoV2MoE(config)
        else:
            self.mlp = MiMoV2MLP(config.hidden_size, config.intermediate_size, config.hidden_act)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.input_layernorm, self.self_attn.config.rms_norm_eps)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = _rms_norm(hidden_states, self.post_attention_layernorm, self.self_attn.config.rms_norm_eps)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiMoV2Model(PreTrainedModel):
    config_class = MiMoV2Config

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiMoV2DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.Parameter(torch.ones(config.hidden_size))
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del use_cache, kwargs
        hidden_states = self.embed_tokens(input_ids)
        if attention_mask is not None and attention_mask.ndim == 2:
            # (bsz, seq) -> additive causal mask
            bsz, seqlen = input_ids.shape
            causal = torch.triu(
                torch.ones(seqlen, seqlen, device=input_ids.device, dtype=torch.bool),
                diagonal=1,
            )
            pad = attention_mask[:, None, None, :] == 0
            attn_mask = torch.zeros(bsz, 1, seqlen, seqlen, device=input_ids.device, dtype=hidden_states.dtype)
            attn_mask = attn_mask.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn_mask = attn_mask.masked_fill(pad, float("-inf"))
            attention_mask = attn_mask
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        return _rms_norm(hidden_states, self.norm, self.config.rms_norm_eps)


class MiMoV2MTPHead(nn.Module):
    """Dummy MTP block so export/resume can exercise mtp stripping."""

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.proj(hidden_states)


class MiMoV2ForCausalLM(PreTrainedModel):
    config_class = MiMoV2Config
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MiMoV2Config) -> None:
        super().__init__(config)
        self.model = MiMoV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.mtp_num_hidden_layers > 0:
            self.mtp = nn.ModuleList(
                [MiMoV2MTPHead(config) for _ in range(config.mtp_num_hidden_layers)]
            )
        else:
            self.mtp = None
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del output_router_logits, kwargs
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)


@dataclass
class MiMoV2FlashTopkRouter:
    """Type alias placeholder for router_utils discovery tests."""

    pass
