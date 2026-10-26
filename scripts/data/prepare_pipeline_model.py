#!/usr/bin/env python3
"""Build a depth-shrunk full-width MiMo-V2 checkpoint for local pipeline tests.

Keeps Teutonic-II 110B tensor shapes (hidden, experts, heads, lm_head) and cuts
depth to 3/4/6/8 layers (default 4 ≈ genesis layers 0, 1, 43, 44).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
MINI_SRC = ROOT / "pipeline_models" / "mimo_v2_mini"

log = logging.getLogger(__name__)


def _import_mini_classes():
    if str(MINI_SRC) not in sys.path:
        sys.path.insert(0, str(MINI_SRC))
    from configuration_mimo_v2 import MiMoV2Config  # noqa: PLC0415
    from modeling_mimo_v2 import MiMoV2ForCausalLM  # noqa: PLC0415

    return MiMoV2Config, MiMoV2ForCausalLM


def _write_tokenizer(dest: Path, vocab_size: int) -> None:
    """Emit a minimal fixed vocab tokenizer (no BPE training).

    Synthetic smoke data only needs valid token ids in ``[0, vocab_size)``.
    Training a 152k BPE from a toy corpus is unnecessary and impractical.
    """
    special = {
        "<|endoftext|>": 0,
        "<|unk|>": 1,
        "<|pad|>": 2,
    }
    vocab = dict(special)
    # Reserve remaining ids as opaque tokens so vocab_size matches the model.
    for i in range(len(special), vocab_size):
        vocab[f"<tok_{i}>"] = i

    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<|unk|>"))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        model_max_length=2048,
    )
    fast.save_pretrained(dest)


def _config_dict(config) -> dict:
    payload = config.to_dict()
    payload["architectures"] = ["MiMoV2ForCausalLM"]
    payload["auto_map"] = {
        "AutoConfig": "configuration_mimo_v2.MiMoV2Config",
        "AutoModelForCausalLM": "modeling_mimo_v2.MiMoV2ForCausalLM",
    }
    payload["torch_dtype"] = "bfloat16"
    return payload


def _estimate_activated_params(config) -> int:
    """Rough activated-param count (embeds + all attn + dense FFN + top-k experts)."""
    h = config.hidden_size
    vocab = config.vocab_size
    e = config.moe_intermediate_size
    i = config.intermediate_size
    heads = config.num_attention_heads
    qd, vd = config.head_dim, config.v_head_dim
    top_k = config.num_experts_per_tok
    n_shared = config.n_shared_experts

    def attn(kv: int) -> int:
        return h * (heads * qd + kv * qd + kv * vd) + (heads * vd) * h

    def expert() -> int:
        return 2 * h * e + e * h

    def dense_ffn() -> int:
        return 2 * h * i + i * h

    total = vocab * h  # embed
    if not config.tie_word_embeddings:
        total += vocab * h  # lm_head
    for layer_idx in range(config.num_hidden_layers):
        kv = config.layer_num_key_value_heads(layer_idx)
        total += attn(kv)
        if config.layer_is_moe(layer_idx):
            total += (top_k + n_shared) * expert()
            total += h * config.n_routed_experts  # router always used
        else:
            total += dense_ffn()
    return total


# Depths used for 1×B300 layer-speed ladders. 3/4 mirror genesis roles;
# 6/8 use the config's generic dense-L0 + MoE-rest / hybrid tiling.
ALLOWED_NUM_LAYERS = (3, 4, 6, 8)


def _genesis_layer_roles(num_layers: int) -> list[int] | None:
    if num_layers == 6:
        return [0, 1, 23, 24, 43, 44]
    if num_layers == 8:
        return [0, 1, 15, 16, 31, 32, 47, 48]
    if num_layers == 4:
        return [0, 1, 43, 44]
    if num_layers == 3:
        return [0, 1, 44]
    return None


def build_pipeline_model(
    output_dir: Path,
    *,
    seed: int = 42,
    vocab_size: int = 152576,
    num_layers: int = 4,
) -> dict:
    """Materialize a HuggingFace-style depth-shrunk full-width MiMo-V2 king directory."""
    if num_layers not in ALLOWED_NUM_LAYERS:
        raise ValueError(f"num_layers must be one of {ALLOWED_NUM_LAYERS}, got {num_layers}")

    MiMoV2Config, MiMoV2ForCausalLM = _import_mini_classes()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ("configuration_mimo_v2.py", "modeling_mimo_v2.py"):
        shutil.copy2(MINI_SRC / name, output_dir / name)

    torch.manual_seed(seed)
    config = MiMoV2Config(vocab_size=vocab_size, num_hidden_layers=num_layers)
    model = MiMoV2ForCausalLM(config)
    model.to(dtype=torch.bfloat16)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    activated = _estimate_activated_params(config)
    log.info(
        "Depth-shrink MiMo-V2 params: total=%d (~%.2fB) trainable=%d activated≈%d (~%.2fB) layers=%d",
        total,
        total / 1e9,
        trainable,
        activated,
        activated / 1e9,
        config.num_hidden_layers,
    )

    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    save_file(state, output_dir / "model.safetensors")

    (output_dir / "config.json").write_text(json.dumps(_config_dict(config), indent=2) + "\n")
    _write_tokenizer(output_dir, vocab_size)

    digest = hashlib.sha256((output_dir / "model.safetensors").read_bytes()).hexdigest()
    roles = _genesis_layer_roles(num_layers)
    king_info = {
        "digest": digest,
        "source": "pipeline_mini_depth_shrink",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "param_count": total,
        "activated_param_estimate": activated,
        "vocab_size": vocab_size,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "n_routed_experts": config.n_routed_experts,
        "n_shared_experts": config.n_shared_experts,
        "num_experts_per_tok": config.num_experts_per_tok,
        "moe_layer_freq": list(config.moe_layer_freq),
        "hybrid_layer_pattern": list(config.hybrid_layer_pattern),
        "genesis_layer_roles": roles,
        "note": (
            "Full-width Teutonic-II slice; simplified local modeling (not byte-identical genesis)."
            if roles is not None
            else "Full-width depth ladder (generic MoE/hybrid tiling; not genesis role remap)."
        ),
    }
    (output_dir / "king_info.json").write_text(json.dumps(king_info, indent=2) + "\n")

    # Quick forward sanity check (short seq on CPU).
    with torch.no_grad():
        ids = torch.randint(0, vocab_size, (1, 8))
        out = model(input_ids=ids, labels=ids)
        assert out.loss is not None and torch.isfinite(out.loss)

    return {
        "output_dir": str(output_dir),
        "param_count": total,
        "activated_param_estimate": activated,
        "digest": digest,
        "moe_layers": sum(1 for i in range(config.num_hidden_layers) if config.layer_is_moe(i)),
        "swa_layers": sum(1 for i in range(config.num_hidden_layers) if config.layer_is_swa(i)),
        "num_hidden_layers": config.num_hidden_layers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build depth-shrunk full-width MiMo-V2 mini model (Teutonic-II shapes)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "pipeline_model",
        help="Output directory (default: ./pipeline_model)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=152576)
    parser.add_argument(
        "--num-layers",
        type=int,
        choices=ALLOWED_NUM_LAYERS,
        default=4,
        help=(
            "Depth: 4≈genesis[0,1,43,44] (~8.6B), 3≈genesis[0,1,44] (~6.1B), "
            "6/8=speed-ladder depths with generic MoE/hybrid tiling"
        ),
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    summary = build_pipeline_model(
        args.output,
        seed=args.seed,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
    )
    log.info(
        "Pipeline model ready at %s (%d params ≈%.2fB, %d layers, %d MoE, %d SWA, digest=%s)",
        summary["output_dir"],
        summary["param_count"],
        summary["param_count"] / 1e9,
        summary["num_hidden_layers"],
        summary["moe_layers"],
        summary["swa_layers"],
        summary["digest"][:16],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
