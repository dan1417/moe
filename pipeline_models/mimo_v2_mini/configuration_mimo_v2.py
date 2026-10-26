"""MiMo-V2 configuration for pipeline-test depth-shrunk models.

Full-width Teutonic-II / genesis tensor shapes with depth cut to 4 layers
mirroring genesis layers 0, 1, 43, 44 (not a width-shrunk toy).
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

# Genesis roles remapped onto mini indices 0..3:
#   L0 ← genesis 0  (dense + global attn)
#   L1 ← genesis 1  (MoE + SWA)
#   L2 ← genesis 43 (MoE + SWA)
#   L3 ← genesis 44 (MoE + global attn)
_DEFAULT_NUM_HIDDEN_LAYERS = 4
_DEFAULT_MOE_LAYER_FREQ = [0, 1, 1, 1]
_DEFAULT_HYBRID_LAYER_PATTERN = [0, 1, 1, 0]


class MiMoV2Config(PretrainedConfig):
    """Depth-shrunk MiMo-V2 layout matching Teutonic genesis tensor naming."""

    model_type = "mimo_v2"

    def __init__(
        self,
        vocab_size: int = 152576,
        hidden_size: int = 3072,
        intermediate_size: int = 16384,
        moe_intermediate_size: int = 1024,
        num_hidden_layers: int = _DEFAULT_NUM_HIDDEN_LAYERS,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 4,
        num_key_value_heads_swa: int = 8,
        head_dim: int = 192,
        v_head_dim: int = 128,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 1,
        topk_group: int = 1,
        moe_layer_freq: list[int] | None = None,
        hybrid_layer_pattern: list[int] | None = None,
        attention_projection_layout: str = "fused_qkv",
        sliding_window: int = 128,
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-5,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        use_cache: bool = False,
        tie_word_embeddings: bool = False,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.0,
        add_swa_attention_sink_bias: bool = True,
        add_full_attention_sink_bias: bool = False,
        output_router_logits: bool = False,
        mtp_num_hidden_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_heads_swa = num_key_value_heads_swa
        self.head_dim = head_dim
        self.v_head_dim = v_head_dim
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_layer_freq = moe_layer_freq if moe_layer_freq is not None else self._default_moe_layer_freq()
        self.hybrid_layer_pattern = (
            hybrid_layer_pattern if hybrid_layer_pattern is not None else self._default_hybrid_pattern()
        )
        self.attention_projection_layout = attention_projection_layout
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.output_router_logits = output_router_logits
        self.mtp_num_hidden_layers = mtp_num_hidden_layers

    def _default_moe_layer_freq(self) -> list[int]:
        """Depth-shrink defaults: dense L0, MoE elsewhere (genesis 0/1/43/44 roles)."""
        if self.num_hidden_layers == _DEFAULT_NUM_HIDDEN_LAYERS:
            return list(_DEFAULT_MOE_LAYER_FREQ)
        if self.num_hidden_layers == 3:
            # Genesis roles 0, 1, 44
            return [0, 1, 1]
        return [0] + [1] * (self.num_hidden_layers - 1)

    def _default_hybrid_pattern(self) -> list[int]:
        """Depth-shrink defaults mirroring genesis hybrid roles for 3/4 layers."""
        if self.num_hidden_layers == _DEFAULT_NUM_HIDDEN_LAYERS:
            return list(_DEFAULT_HYBRID_LAYER_PATTERN)
        if self.num_hidden_layers == 3:
            # Genesis roles 0, 1, 44 → GA, SWA, GA
            return [0, 1, 0]
        pattern = [0]
        for _ in range((self.num_hidden_layers - 1) // 6):
            pattern.extend([1, 1, 1, 1, 1, 0])
        while len(pattern) < self.num_hidden_layers:
            pattern.append(1)
        return pattern[: self.num_hidden_layers]

    def layer_is_moe(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.moe_layer_freq):
            return False
        return bool(self.moe_layer_freq[layer_idx])

    def layer_is_swa(self, layer_idx: int) -> bool:
        if layer_idx >= len(self.hybrid_layer_pattern):
            return False
        return int(self.hybrid_layer_pattern[layer_idx]) == 1

    def layer_num_key_value_heads(self, layer_idx: int) -> int:
        return self.num_key_value_heads_swa if self.layer_is_swa(layer_idx) else self.num_key_value_heads

    def layer_wants_sink_bias(self, layer_idx: int) -> bool:
        if self.layer_is_swa(layer_idx):
            return bool(self.add_swa_attention_sink_bias)
        return bool(self.add_full_attention_sink_bias)
