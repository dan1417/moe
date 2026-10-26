"""Model loading, FSDP wrapping, and training helpers for MiMo-V2."""

from __future__ import annotations

import functools
import logging
import os
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    FSDPModule,
    MixedPrecisionPolicy,
    fully_shard,
)
from transformers import AutoModelForCausalLM

from src.moe.utils import freeze_expert_load_biases
from src.dist.process_groups import ProcessGroups

log = logging.getLogger(__name__)

ROUTER_KEYWORDS = ("router", "gate")
NORM_KEYWORDS = ("norm", "ln")
EMBED_KEYWORDS = ("embed",)
# Skip numerically sensitive / non-GEMM modules for TE low-precision Linear swap.
LP_SKIP_KEYWORDS = ("router", "gate", "embed", "lm_head", "norm", "layernorm")
# Hopper+ can run delayed FP8; MXFP8/NVFP4 need datacenter Blackwell (sm_100 / sm_103).
MIN_DELAYED_FP8_MAJOR = 9
BLACKWELL_DC_MAJOR = 10  # B100/B200=10.0, B300=10.3 — not consumer sm_120
_MASKING_COMPAT_PATCHED = False
_FA4_SM120_SINK_PATCHED = False
_FA4_ASYMMETRIC_V_PATCHED = False
_GATE_NOAUX_PATCHED = False
_MIMO_MOE_CLASS: type[nn.Module] | None = None
_MIMO_GATE_CLASS: type[nn.Module] | None = None


@dataclass(frozen=True)
class LowPrecisionState:
    """Resolved low-precision training mode after hardware/TE gates."""

    mode: str  # none | mxfp8 | nvfp4
    recipe: Any | None = None
    replaced_linears: int = 0
    strict: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode != "none" and self.recipe is not None


def _patch_transformers_masking_compat() -> None:
    """Strip cache_position for MiMo-V2 modeling vs transformers 5.x API drift."""
    global _MASKING_COMPAT_PATCHED
    if _MASKING_COMPAT_PATCHED:
        return

    import functools
    import sys

    import transformers.masking_utils as masking_utils

    def _wrap(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs.pop("cache_position", None)
            return fn(*args, **kwargs)

        wrapper._mimo_mask_compat = True  # type: ignore[attr-defined]
        return wrapper

    originals: dict[str, Any] = {}
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        original = getattr(masking_utils, name)
        if getattr(original, "_mimo_mask_compat", False):
            originals[name] = original
            continue
        patched = _wrap(original)
        originals[name] = original
        setattr(masking_utils, name, patched)

    # Remote-code modules may have bound the old functions before this patch.
    for module in list(sys.modules.values()):
        if module is None:
            continue
        mod_name = getattr(module, "__name__", "")
        if not mod_name.endswith("modeling_mimo_v2"):
            continue
        for name, original in originals.items():
            if getattr(module, name, None) is original:
                setattr(module, name, getattr(masking_utils, name))

    _MASKING_COMPAT_PATCHED = True
    if _is_rank0():
        log.debug("Patched transformers masking_utils for MiMo-V2 cache_position compat")


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError):
        pass
    return True


def _flash_attn_4_available() -> bool:
    try:
        from transformers.utils import is_flash_attn_4_available

        return bool(is_flash_attn_4_available())
    except Exception:
        return False


def _cuda_arch_major() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        return int(torch.cuda.get_device_capability(0)[0])
    except Exception:
        return None


def _needs_fa4_sm120_sink_workaround() -> bool:
    """SM120 FA4 kernels assert learnable_sink is None; SM90/SM100 support sinks natively."""
    major = _cuda_arch_major()
    return major is not None and major >= 12


def _apply_learnable_sink_lse_correction(
    out: torch.Tensor,
    lse: torch.Tensor,
    learnable_sink: torch.Tensor,
) -> torch.Tensor:
    """Rescale FA output so it matches softmax([scores, sink]) without a V for the sink.

    ``out_corrected = out * sigmoid(lse - sink)`` where ``lse = logsumexp(scores)``.
    """
    sink = learnable_sink.to(dtype=torch.float32)
    lse_f = lse.to(dtype=torch.float32)
    if out.ndim == 4 and lse_f.ndim == 3:
        # dense: out (B,S,H,D), lse (B,H,S)
        scale = torch.sigmoid(lse_f - sink.view(1, -1, 1)).transpose(1, 2).unsqueeze(-1)
    elif out.ndim == 3 and lse_f.ndim == 2:
        # varlen: out (T,H,D), lse (H,T)
        scale = torch.sigmoid(lse_f - sink.view(-1, 1)).transpose(0, 1).unsqueeze(-1)
    else:
        raise ValueError(f"Unexpected FA shapes out={tuple(out.shape)} lse={tuple(lse.shape)}")
    return out * scale.to(dtype=out.dtype)


def _patch_fa4_sm120_learnable_sink() -> None:
    """Make FA4 ``learnable_sink`` work on RTX PRO 6000 (sm_120) via LSE correction.

    Upstream CuTe SM120 forward rejects sinks. We strip ``learnable_sink``, run FA4 with
    ``return_lse=True``, then rescale — numerically matches eager sink math in bf16.
    """
    global _FA4_SM120_SINK_PATCHED
    if _FA4_SM120_SINK_PATCHED or not _needs_fa4_sm120_sink_workaround():
        return

    import functools

    import flash_attn.cute as fa4_cute
    import flash_attn.cute.interface as fa4_iface

    def _wrap(fn: Any) -> Any:
        if getattr(fn, "_teutonic_sm120_sink", False):
            return fn

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            sink = kwargs.get("learnable_sink", None)
            if sink is None:
                return fn(*args, **kwargs)
            want_lse = bool(kwargs.get("return_lse", False))
            kwargs = dict(kwargs)
            kwargs["learnable_sink"] = None
            kwargs["return_lse"] = True
            out, lse = fn(*args, **kwargs)
            out = _apply_learnable_sink_lse_correction(out, lse, sink)
            if want_lse:
                sink_f = sink.to(dtype=torch.float32)
                if lse.ndim == 3:
                    lse = torch.logaddexp(lse.to(dtype=torch.float32), sink_f.view(1, -1, 1))
                else:
                    lse = torch.logaddexp(lse.to(dtype=torch.float32), sink_f.view(-1, 1))
                return out, lse.to(dtype=out.dtype)
            return out

        wrapper._teutonic_sm120_sink = True  # type: ignore[attr-defined]
        return wrapper

    for mod in (fa4_iface, fa4_cute):
        mod.flash_attn_func = _wrap(mod.flash_attn_func)
        mod.flash_attn_varlen_func = _wrap(mod.flash_attn_varlen_func)

    # Force transformers to re-import FA4 callables after our wrap.
    try:
        import transformers.modeling_flash_attention_utils as fa_utils

        fa_utils._loaded_implementation = None
        for attr in (
            "_flash_fn",
            "_flash_varlen_fn",
            "_flash_with_kvcache_fn",
            "_pad_fn",
            "_unpad_fn",
            "_process_flash_kwargs_fn",
        ):
            if hasattr(fa_utils, attr):
                setattr(fa_utils, attr, None)
    except Exception as exc:
        if _is_rank0():
            log.warning("Could not reset transformers FA cache after SM120 sink patch: %s", exc)

    _FA4_SM120_SINK_PATCHED = True
    if _is_rank0():
        log.info("Patched FA4 SM120 learnable_sink via LSE correction")


def _patch_fa4_asymmetric_value_heads() -> None:
    """Stop transformers from padding V up to QK head_dim under FA4.

    Transformers' FA integration pads ``v`` from MiMo's 128 → 192 so FA2 sees a
    shared head dim. FA4 supports asymmetric QK/V, and the padded square-192
    backward exceeds SM120 shared memory (115712 > 101376). Keep native 192/128.
    """
    global _FA4_ASYMMETRIC_V_PATCHED
    if _FA4_ASYMMETRIC_V_PATCHED:
        return

    try:
        import transformers.integrations.flash_attention as fa_int
    except Exception as exc:  # noqa: BLE001
        if _is_rank0():
            log.warning("Could not patch FA asymmetric V heads: %s", exc)
        return

    original = fa_int.flash_attention_forward
    if getattr(original, "_teutonic_fa4_asym_v", False):
        _FA4_ASYMMETRIC_V_PATCHED = True
        return

    def flash_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        is_causal: bool | None = None,
        s_aux: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        impl = getattr(getattr(module, "config", None), "_attn_implementation", "")
        if impl != "flash_attention_4":
            return original(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
                is_causal=is_causal,
                s_aux=s_aux,
                **kwargs,
            )

        from transformers.integrations.flash_attention import (
            _use_top_left_mask,
            get_target_dtype,
        )
        from transformers.modeling_flash_attention_utils import _flash_attention_forward

        if kwargs.get("output_attentions", False):
            log.warning(
                "Flash Attention does not support output_attentions=True; continuing without weights"
            )

        seq_len = query.shape[2]
        if any(dim == 0 for dim in query.shape):
            raise ValueError("FlashAttention does not support inputs with a zero dimension")

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # Intentionally do NOT pad value to query head_dim — FA4 handles asymmetric V.
        target_dtype = get_target_dtype(query, module)
        is_causal_flag = is_causal if is_causal is not None else module.is_causal
        attn_output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=is_causal_flag,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=_use_top_left_mask,
            target_dtype=target_dtype,
            attn_implementation=impl,
            layer_idx=getattr(module, "layer_idx", None),
            s_aux=(s_aux.to(query.dtype) if s_aux is not None else None),
            **kwargs,
        )
        return attn_output, None

    flash_attention_forward._teutonic_fa4_asym_v = True  # type: ignore[attr-defined]
    fa_int.flash_attention_forward = flash_attention_forward

    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        ALL_ATTENTION_FUNCTIONS.register("flash_attention_4", flash_attention_forward)
    except Exception as exc:  # noqa: BLE001
        if _is_rank0():
            log.warning("Could not refresh ALL_ATTENTION_FUNCTIONS FA4 entry: %s", exc)

    _FA4_ASYMMETRIC_V_PATCHED = True
    if _is_rank0():
        log.info("Patched FA4 to keep asymmetric V head dim (no pad 128→192)")


def _resolve_mimo_moe_classes(model: nn.Module) -> tuple[type[nn.Module] | None, type[nn.Module] | None]:
    """Discover MiMo MoE/gate classes from loaded remote-code modules."""
    global _MIMO_MOE_CLASS, _MIMO_GATE_CLASS
    if _MIMO_MOE_CLASS is not None and _MIMO_GATE_CLASS is not None:
        return _MIMO_MOE_CLASS, _MIMO_GATE_CLASS
    moe_cls: type[nn.Module] | None = None
    gate_cls: type[nn.Module] | None = None
    for module in model.modules():
        cls = type(module)
        name = cls.__name__
        if name == "MiMoV2MoE":
            moe_cls = cls
        elif name == "MiMoV2MoEGate":
            gate_cls = cls
    _MIMO_MOE_CLASS = moe_cls
    _MIMO_GATE_CLASS = gate_cls
    return moe_cls, gate_cls


def _patch_mimo_gate_noaux_training(model: nn.Module) -> None:
    """Allow noaux_tc gate routing during training without editing contract code."""
    global _GATE_NOAUX_PATCHED
    _, gate_cls = _resolve_mimo_moe_classes(model)
    if gate_cls is None:
        return

    patched_count = 0
    for module in model.modules():
        if type(module) is not gate_cls:
            continue
        if getattr(module, "topk_method", None) != "noaux_tc":
            continue
        if getattr(module.forward, "_teutonic_noaux_train", False):
            patched_count += 1
            continue

        original_forward = module.forward

        @functools.wraps(original_forward)
        def training_forward(self: nn.Module, hidden_states: torch.Tensor, _orig=original_forward) -> Any:
            was_training = self.training
            try:
                self.training = False
                return _orig(hidden_states)
            finally:
                self.training = was_training

        training_forward._teutonic_noaux_train = True  # type: ignore[attr-defined]
        module.forward = types.MethodType(training_forward, module)
        patched_count += 1

    if patched_count > 0 and _is_rank0():
        log.info("Patched %d MiMoV2MoEGate modules for noaux_tc training", patched_count)
    _GATE_NOAUX_PATCHED = True


# TE DelayedScaling historically wanted leading dims % 8; MXFP8 block scaling
# requires % 32 (see transformer_engine quantizer assert MXFP8_BLOCK_SIZE).
TE_FP8_TOKEN_ALIGN = 8
TE_MXFP8_TOKEN_ALIGN = 32


def _pad_tokens_for_fp8(x: torch.Tensor, *, multiple: int = TE_MXFP8_TOKEN_ALIGN) -> tuple[torch.Tensor, int]:
    """Pad leading token dim to a TE FP8/MXFP8-safe multiple; return (padded, original_n).

    MoE experts often receive irregular token counts (e.g. 7). MXFP8 block scaling
    needs ``flat_first_dim % 32 == 0`` and last dim % 32 == 0.
    """
    if x.ndim < 2:
        raise ValueError(f"Expected rank>=2 token tensor, got shape={tuple(x.shape)}")
    n = int(x.shape[0])
    if n == 0 or n % multiple == 0:
        return x, n
    pad = multiple - (n % multiple)
    pad_shape = (pad, *x.shape[1:])
    padded = torch.cat([x, x.new_zeros(pad_shape)], dim=0)
    return padded, n


def _patch_mimo_moe_fsdp_expert_alignment(model: nn.Module) -> None:
    """Force every rank to enter every expert under per-expert FSDP2.

    Native MiMo MoE skips experts with no local tokens. With ``fully_shard`` on
    each expert that creates uneven all-gather counts across ranks and deadlocks.
    Empty-token forwards keep FSDP collectives and the autograd graph aligned.

    Also pads per-expert token batches to a multiple of 8 so TE MXFP8/NVFP4
    Linear kernels accept irregular MoE dispatch sizes.
    """
    moe_cls, _ = _resolve_mimo_moe_classes(model)
    if moe_cls is None:
        return

    patched_count = 0
    for module in model.modules():
        if type(module) is not moe_cls:
            continue
        if getattr(module.forward, "_teutonic_fsdp_expert_align", False):
            patched_count += 1
            continue

        def aligned_forward(self: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            bsz, seqlen, hidden = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden)
            topk_idx, topk_weight = self.gate(hidden_states)
            idx = topk_idx.reshape(-1, self.config.num_experts_per_tok)
            weight = topk_weight.reshape(-1, self.config.num_experts_per_tok).to(dtype=flat.dtype)
            out = torch.zeros_like(flat)
            for expert_id, expert in enumerate(self.experts):
                mask = (idx == expert_id).any(dim=-1)
                token_idx = mask.nonzero(as_tuple=False).squeeze(-1)
                if token_idx.numel() == 0:
                    # Enter the FSDP unit and stay on the graph for backward collectives.
                    out = out + expert(flat[:0]).sum() * 0
                    continue
                expert_input = flat.index_select(0, token_idx)
                expert_input, n_tok = _pad_tokens_for_fp8(expert_input)
                expert_out = expert(expert_input)[:n_tok]
                for k in range(self.config.num_experts_per_tok):
                    slot = (idx[token_idx, k] == expert_id).to(dtype=flat.dtype).unsqueeze(-1)
                    out.index_add_(
                        0,
                        token_idx,
                        expert_out * weight[token_idx, k].unsqueeze(-1) * slot,
                    )
            if self.shared_experts is not None:
                shared_in, n_flat = _pad_tokens_for_fp8(flat)
                out = out + self.shared_experts(shared_in)[:n_flat]
            return out.view(bsz, seqlen, hidden)

        aligned_forward._teutonic_fsdp_expert_align = True  # type: ignore[attr-defined]
        module.forward = types.MethodType(aligned_forward, module)
        patched_count += 1

    if patched_count > 0 and _is_rank0():
        log.info(
            "Patched %d MiMoV2MoE modules for FSDP2 per-expert collective alignment + FP8 token pad",
            patched_count,
        )


def partition_experts_for_ep(model: nn.Module, pg: ProcessGroups) -> list[nn.Module]:
    """DISABLED — custom EP expert partition is unsafe for this MiMo MoE."""
    raise RuntimeError(
        "partition_experts_for_ep is disabled. Keep moe.dispatch_overlap=false "
        "and use native MiMo MoE + FSDP."
    )


def patch_mimo_moe_ep_forward(model: nn.Module, dispatcher: Any) -> None:
    """DISABLED — custom EP forward patch is unsafe for this MiMo MoE."""
    raise RuntimeError(
        "patch_mimo_moe_ep_forward is disabled. Keep moe.dispatch_overlap=false "
        "and use native MiMo MoE + FSDP."
    )


def _patch_mimo_moe_ep_forward(model: nn.Module, dispatcher: Any) -> None:
    """DISABLED stub retained only for import compatibility."""
    raise RuntimeError(
        "_patch_mimo_moe_ep_forward is disabled. Keep moe.dispatch_overlap=false."
    )

def _resolve_attn_implementation(preferred: str | None = None) -> str | None:
    """Pick attention backend for MiMo-V2.

    Prefer FA4 when installed (correct sinks + asymmetric QK/V). FA2 is wrong for
    this model (CE ~8). Default without FA4 is eager.
    """
    import os

    env = os.environ.get("TEUTONIC_ATTN", "").strip().lower()
    # Explicit preferred/env wins; otherwise auto-pick FA4 if present.
    if preferred is not None:
        choice = preferred.strip().lower()
    elif env:
        choice = env
    elif _flash_attn_4_available():
        choice = "flash_attention_4"
    else:
        choice = "eager"

    if choice in {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "flash_attention_4"}:
        if choice == "flash_attention_4" and not _flash_attn_4_available():
            if _is_rank0():
                log.warning("TEUTONIC_ATTN=flash_attention_4 but FA4 not installed; falling back to eager")
            return "eager"
        if choice == "flash_attention_2" and _is_rank0():
            log.warning(
                "flash_attention_2 ignores MiMo attention sinks (expect CE~8); "
                "use flash_attention_4 or eager"
            )
        return choice
    if choice in {"auto", "none", "default", ""}:
        return "flash_attention_4" if _flash_attn_4_available() else "eager"
    if _is_rank0():
        log.warning("Unknown TEUTONIC_ATTN=%r; using eager", choice)
    return "eager"


def _ensure_fa4_cute_disk_cache() -> None:
    """Enable FA4 CuTe-DSL disk cache and lengthen cross-process lock waits.

    Upstream defaults ``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=0``, so every
    rank JITs from scratch. Under 8-GPU contention that is pathologically slow.
    Cache lock waits also default to 15s, which is too short while another rank
    is exporting a freshly compiled kernel.
    """
    os.environ.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    cache_dir = os.environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")
    if not cache_dir:
        cache_dir = str(Path.cwd() / ".cache" / "fa4_cute")
        os.environ["FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"] = cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        from flash_attn.cute import cache_utils as fa_cache

        # Must set before get_jit_cache() materializes caches.
        fa_cache.CUTE_DSL_CACHE_ENABLED = True
        fa_cache.CUTE_DSL_CACHE_DIR = cache_dir
        fa_cache.JITPersistentCache.LOCK_TIMEOUT_SECONDS = int(
            os.environ.get("FLASH_ATTENTION_CUTE_DSL_LOCK_TIMEOUT", "3600")
        )
    except Exception as exc:  # noqa: BLE001
        if _is_rank0():
            log.warning("Could not configure FA4 CuTe disk cache: %s", exc)


def prewarm_fa4_mimo_kernels(
    seq_len: int = 256,
    *,
    device: torch.device | None = None,
) -> None:
    """Compile MiMo-V2 FA4 kernels once on a single GPU into the CuTe disk cache.

    Covers full (GQA 64/4) and SWA (GQA 64/8, window=128) with and without LSE
    (SM120 sink workaround forces ``return_lse=True`` on SWA).
    """
    _ensure_fa4_cute_disk_cache()
    if not torch.cuda.is_available():
        return
    from flash_attn.cute.interface import flash_attn_func

    if device is None:
        device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    dqk, dv = 192, 128
    jobs = (
        dict(name="full_gqa16", nq=64, nkv=4, window=None, return_lse=False),
        dict(name="swa_gqa8_lse", nq=64, nkv=8, window=(128, 0), return_lse=True),
        dict(name="swa_gqa8", nq=64, nkv=8, window=(128, 0), return_lse=False),
    )
    for job in jobs:
        nq, nkv = int(job["nq"]), int(job["nkv"])
        q = torch.randn(1, seq_len, nq, dqk, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, seq_len, nkv, dqk, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, seq_len, nkv, dv, device=device, dtype=torch.bfloat16, requires_grad=True)
        kwargs: dict[str, Any] = {"causal": True, "return_lse": bool(job["return_lse"])}
        if job["window"] is not None:
            kwargs["window_size"] = job["window"]
        t0 = time.perf_counter()
        out = flash_attn_func(q, k, v, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        out.float().sum().backward()
        torch.cuda.synchronize(device)
        log.info("FA4 prewarm %s ok in %.1fs", job["name"], time.perf_counter() - t0)
        del q, k, v, out
        torch.cuda.empty_cache()


def _enable_flash_attention_on_model(model: nn.Module, implementation: str = "flash_attention_4") -> None:
    """Force a flash attn backend on MiMo-V2 after load.

    Transformers rejects flash for remote-code models that omit
    ``_supports_flash_attn``, but MiMo-V2's attention already dispatches
    through ``ALL_ATTENTION_FUNCTIONS`` when ``_attn_implementation`` is set.
    Official MiMo-V2-Flash only lists FA4 as compatible.
    """
    if implementation == "flash_attention_4":
        _ensure_fa4_cute_disk_cache()
        _patch_fa4_sm120_learnable_sink()
        _patch_fa4_asymmetric_value_heads()
    model.__class__._supports_flash_attn = True
    model.__class__._compatible_flash_implementations = ["flash_attention_4"]
    # Prefer the internal setter so we do not re-trigger recursive validation.
    if hasattr(model.config, "_attn_implementation_internal"):
        model.config._attn_implementation_internal = implementation
    else:
        model.config._attn_implementation = implementation
    if _is_rank0():
        log.info("Enabled %s on %s", implementation, model.__class__.__name__)


def _apply_attn_implementation(model: nn.Module, resolved_attn: str | None) -> None:
    if resolved_attn in {"flash_attention_2", "flash_attention_3", "flash_attention_4"}:
        _enable_flash_attention_on_model(model, resolved_attn)
    elif resolved_attn is not None:
        if hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = resolved_attn
        else:
            model.config._attn_implementation = resolved_attn
        if _is_rank0():
            log.info("Using attention backend %s on %s", resolved_attn, model.__class__.__name__)
    # Prefer fused SDPA kernels when available (does not use flash_attn package).
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def load_challenger_model(
    king_dir: str | Path,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> nn.Module:
    """Load a MiMo-V2 model from a HuggingFace-style checkpoint directory."""
    king_path = Path(king_dir)
    if not (king_path / "config.json").exists():
        raise FileNotFoundError(
            f"config.json not found in {king_path}. Expected a HuggingFace model directory."
        )
    if not (king_path / "king_info.json").exists() and _is_rank0():
        log.debug("king_info.json not found in %s (ok for merged or checkpoint dirs)", king_path)
    _patch_transformers_masking_compat()
    resolved_attn = _resolve_attn_implementation(attn_implementation)
    # Do not pass attn_implementation into from_pretrained: transformers rejects
    # flash for remote-code MiMo until _supports_flash_attn is set (post-load).
    model = AutoModelForCausalLM.from_pretrained(
        king_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    _apply_attn_implementation(model, resolved_attn)
    _patch_mimo_gate_noaux_training(model)
    _patch_mimo_moe_fsdp_expert_alignment(model)
    freeze_expert_load_biases(model)
    model.config.output_router_logits = True
    model.config.use_cache = False
    return model


def load_challenger_model_distributed(
    king_dir: str | Path,
    pg: ProcessGroups,
    torch_dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str | None = None,
) -> nn.Module:
    """Load a checkpoint for FSDP wrap.

    Every rank loads from disk. A prior meta-device + ``sync_module_states`` shortcut
    produced random logits (CE ≈ ln(vocab) ≈ 11) on this MiMo stack, so we keep the
    correct all-rank load. OS page cache usually makes ranks 1..N much faster after
    rank0's first read.
    """
    del pg  # API kept for call sites; all ranks must materialize real weights.
    return load_challenger_model(
        king_dir,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )


def get_parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Return parameter groups for logging/reference (WD=0 on specials + routers)."""
    from src.optim.optimizer import _is_expert_load_bias_name
    from src.moe.router import is_router_param_name

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    router_params: list[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _is_expert_load_bias_name(name):
            continue
        lower = name.lower()
        if is_router_param_name(name):
            router_params.append(param)
        elif param.ndim <= 1 or any(keyword in lower for keyword in NORM_KEYWORDS + EMBED_KEYWORDS + ("bias", "lm_head")):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": router_params, "weight_decay": 0.0, "lr_scale": 0.1},
    ]


def _count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return total, trainable


def print_model_summary(model: nn.Module, low_precision: str | LowPrecisionState = "none") -> None:
    """Print a concise model summary on rank 0."""
    if not _is_rank0():
        return
    total, trainable = _count_parameters(model)
    config = getattr(model, "config", None)
    n_layers = getattr(config, "num_hidden_layers", "unknown")
    n_experts = getattr(config, "n_routed_experts", getattr(config, "num_experts", "unknown"))
    hidden = getattr(config, "hidden_size", "unknown")
    top_k = getattr(config, "num_experts_per_tok", getattr(config, "moe_top_k", "unknown"))
    lp_mode = low_precision.mode if isinstance(low_precision, LowPrecisionState) else str(low_precision)
    log.info(
        "Model summary: total_params=%.2fB trainable=%.2fB layers=%s experts=%s top_k=%s hidden=%s",
        total / 1e9,
        trainable / 1e9,
        n_layers,
        n_experts,
        top_k,
        hidden,
    )
    log.info("Attention pattern: hybrid SWA/full (MiMo-V2); low_precision=%s", lp_mode)
    if torch.cuda.is_available():
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        log.info("Per-GPU memory: %.1f GB total (per-node ~%.1f GB on 8 GPUs)", total_mem, total_mem * 8)


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    """Return the inner module when activation checkpointing has wrapped a layer."""
    current = module
    for _ in range(8):
        if not type(current).__name__.startswith("Checkpoint"):
            return current
        inner = getattr(current, "_checkpoint_wrapped_module", None)
        if inner is None:
            inner = getattr(current, "module", None)
        if inner is None:
            return current
        current = inner
    return current


def _find_decoder_layer_class(model: nn.Module) -> type[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers") and len(model.model.layers) > 0:
        return type(_unwrap_checkpoint_module(model.model.layers[0]))
    raise RuntimeError("Could not detect decoder layer class from model.model.layers[0]")


def _get_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def resolve_low_precision_mode(cfg: dict[str, Any]) -> str:
    """Resolve training.low_precision (preferred) or legacy training.fp8_training."""
    training_cfg = cfg.get("training", {})
    raw = training_cfg.get("low_precision", None)
    if raw is not None and str(raw).strip() != "":
        mode = str(raw).lower().strip()
    elif bool(training_cfg.get("fp8_training", False)):
        mode = "mxfp8"
    else:
        mode = "none"
    aliases = {"fp8": "mxfp8", "fp4": "nvfp4", "none": "none", "bf16": "none", "off": "none"}
    mode = aliases.get(mode, mode)
    if mode not in {"none", "mxfp8", "nvfp4"}:
        raise ValueError(
            f"Unsupported training.low_precision={raw!r}; expected none|mxfp8|nvfp4 "
            "(aliases: fp8→mxfp8, fp4→nvfp4)"
        )
    return mode


def _capability_supports_low_precision(mode: str, capability: tuple[int, int]) -> bool:
    """Gate TE recipes to verified GPU families.

    MXFP8 / NVFP4 training: datacenter Blackwell sm_100 / sm_103 (B100/B200/B300).
    Consumer/workstation sm_120 is explicitly unsupported for these recipes.
    """
    major, _minor = capability
    if mode in {"mxfp8", "nvfp4"}:
        return major == BLACKWELL_DC_MAJOR
    return False


def _layer_index_from_module_name(name: str) -> int | None:
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def _should_enable_lp_layer(
    name: str,
    module: nn.Module,
    *,
    n_layers: int | None,
    high_precision_tail: int,
    mode: str,
) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    lower = name.lower()
    if any(keyword in lower for keyword in LP_SKIP_KEYWORDS):
        return False
    # TE Linear requires dims divisible by 16.
    if module.in_features % 16 != 0 or module.out_features % 16 != 0:
        return False
    # NVFP4 paper / TE docs: keep the last few layers in higher precision (BF16 here).
    if mode == "nvfp4" and n_layers is not None and high_precision_tail > 0:
        layer_idx = _layer_index_from_module_name(name)
        if layer_idx is not None and layer_idx >= n_layers - high_precision_tail:
            return False
    return True


def _build_te_recipe(mode: str, capability: tuple[int, int]) -> Any:
    from transformer_engine.common import recipe as te_recipe

    del capability  # reserved for future sm_120 workarounds
    if mode == "mxfp8":
        return te_recipe.MXFP8BlockScaling()
    if mode == "nvfp4":
        # Default NVFP4BlockScaling (RHT + stochastic rounding) is the verified
        # B100/B300 training recipe in Transformer Engine docs.
        return te_recipe.NVFP4BlockScaling()
    raise ValueError(f"No TE recipe for mode={mode}")


def _te_autocast(recipe_obj: Any):
    """Return TE autocast context (prefer te.autocast, fallback fp8_autocast)."""
    import transformer_engine.pytorch as te

    if hasattr(te, "autocast"):
        return te.autocast(enabled=True, recipe=recipe_obj)
    return te.fp8_autocast(enabled=True, fp8_recipe=recipe_obj)


def enable_low_precision_training(
    model: nn.Module,
    pg: ProcessGroups,
    cfg: dict[str, Any],
) -> LowPrecisionState:
    """Swap eligible Linears to TE and return the active MXFP8/NVFP4 recipe.

    Verified path: Transformer Engine ``te.Linear`` + ``MXFP8BlockScaling`` /
    ``NVFP4BlockScaling`` on datacenter Blackwell (8×B300 = sm_103). Falls back
    to BF16 when TE/hardware/config is unavailable.
    """
    del pg
    mode = resolve_low_precision_mode(cfg)
    strict = bool(cfg.get("training", {}).get("low_precision_strict", True))

    def unavailable(reason: str, exc: Exception | None = None) -> LowPrecisionState:
        message = f"low_precision={mode} unavailable: {reason}"
        if strict:
            raise RuntimeError(message) from exc
        if _is_rank0():
            log.warning("%s; falling back to BF16", message)
        return LowPrecisionState(mode="none", strict=strict)

    if mode == "none":
        if _is_rank0():
            log.info("Low-precision training disabled (BF16)")
        return LowPrecisionState(mode="none", strict=strict)

    if not torch.cuda.is_available():
        return unavailable("CUDA is unavailable")

    capability = torch.cuda.get_device_capability()
    if not _capability_supports_low_precision(mode, capability):
        return unavailable(
            f"CUDA capability {capability} is unsupported; datacenter Blackwell sm_10x is required"
        )

    try:
        import transformer_engine.pytorch as te
        if not hasattr(te, "Linear"):
            raise RuntimeError("transformer_engine.pytorch.Linear is missing")
    except (ImportError, RuntimeError) as exc:
        return unavailable("Transformer Engine PyTorch extension is unavailable", exc)

    try:
        recipe_obj = _build_te_recipe(mode, capability)
    except Exception as exc:  # noqa: BLE001 — recipe missing in older TE
        return unavailable(f"Transformer Engine recipe construction failed: {exc}", exc)

    training_cfg = cfg.get("training", {})
    high_precision_tail = int(training_cfg.get("nvfp4_high_precision_layers", 4))
    n_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)

    replaced = 0
    for name, module in list(model.named_modules()):
        if not _should_enable_lp_layer(
            name,
            module,
            n_layers=n_layers if isinstance(n_layers, int) else None,
            high_precision_tail=high_precision_tail,
            mode=mode,
        ):
            continue
        parent, leaf = _get_parent_module(model, name)
        bias = module.bias is not None
        te_linear = te.Linear(
            module.in_features,
            module.out_features,
            bias=bias,
            params_dtype=module.weight.dtype,
            device=module.weight.device,
        )
        te_linear.weight.data.copy_(module.weight.data)
        if bias and module.bias is not None and te_linear.bias is not None:
            te_linear.bias.data.copy_(module.bias.data)
        setattr(parent, leaf, te_linear)
        replaced += 1

    if replaced == 0:
        return unavailable("no eligible Linear layers were found")

    if _is_rank0():
        tail_note = (
            f", nvfp4 BF16 tail layers={high_precision_tail}" if mode == "nvfp4" else ""
        )
        log.info(
            "Enabled %s on %d linear layers (capability=%s%s)",
            mode.upper(),
            replaced,
            capability,
            tail_note,
        )
    return LowPrecisionState(
        mode=mode,
        recipe=recipe_obj,
        replaced_linears=replaced,
        strict=strict,
    )


def enable_fp8_training(model: nn.Module, pg: ProcessGroups, cfg: dict[str, Any]) -> bool:
    """Backward-compatible wrapper; prefer ``enable_low_precision_training``."""
    return enable_low_precision_training(model, pg, cfg).enabled


def _unwrap_model(model: nn.Module) -> nn.Module:
    # FSDP2 mutates modules in place instead of adding an FSDP1 wrapper.
    return model


def load_state_dict_strict_except_mtp(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load weights while allowing only intentionally stripped MTP tensors."""
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in incompatible.missing_keys if ".mtp." not in key and not key.startswith("mtp.")]
    unexpected = [
        key for key in incompatible.unexpected_keys if ".mtp." not in key and not key.startswith("mtp.")
    ]
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is incompatible: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )


def load_state_dict_into_fsdp(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Load a rank-0 full state dict into an FSDP2 model (legacy/eval compatibility)."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=True,
        broadcast_from_rank0=True,
    )
    incompatible = set_model_state_dict(model, state_dict, options=options)
    missing = [key for key in incompatible.missing_keys if "mtp" not in key.lower()]
    unexpected = [key for key in incompatible.unexpected_keys if "mtp" not in key.lower()]
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch: missing={missing[:10]} unexpected={unexpected[:10]}"
        )


def apply_fsdp(
    model: nn.Module,
    pg: ProcessGroups,
    cfg: dict[str, Any],
    ignored_modules: list[nn.Module] | None = None,
) -> nn.Module:
    """Apply composable FSDP2 bottom-up across decoder layers and model root."""
    parallel_cfg = cfg.get("parallelism", {})
    decoder_layer_cls = _find_decoder_layer_class(model)
    mixed_precision = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    )
    offload_policy = (
        CPUOffloadPolicy(pin_memory=True)
        if bool(parallel_cfg.get("fsdp_cpu_offload", False))
        else None
    )
    mesh = DeviceMesh.from_group(
        pg.ep_group,
        device_type="cuda",
        mesh_dim_names=("fsdp",),
    )
    ignored_params: set[nn.Parameter] = set()
    if ignored_modules:
        for module in ignored_modules:
            ignored_params.update(module.parameters())

    shard_kwargs: dict[str, Any] = {
        "mesh": mesh,
        "mp_policy": mixed_precision,
    }
    if offload_policy is not None:
        shard_kwargs["offload_policy"] = offload_policy
    if ignored_params:
        shard_kwargs["ignored_params"] = ignored_params

    decoder_layers = [
        module for module in model.modules() if isinstance(module, decoder_layer_cls)
    ]
    # Keep the full replica on CPU until FSDP2 places local shards on CUDA.
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Bottom-up MoE wrapping: shard each expert alone so forward/backward only
    # all-gathers one ~9M-param expert (~18MB BF16) at a time instead of the
    # entire 256-expert block (~5GB). Critical for 24GB GPUs and 110B.
    expert_units = 0
    for layer in decoder_layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        experts = getattr(mlp, "experts", None)
        if isinstance(experts, nn.ModuleList):
            for expert in experts:
                fully_shard(expert, reshard_after_forward=True, **shard_kwargs)
                expert_units += 1
        shared = getattr(mlp, "shared_experts", None)
        if isinstance(shared, nn.Module):
            fully_shard(shared, reshard_after_forward=True, **shard_kwargs)
            expert_units += 1

    for layer in decoder_layers:
        fully_shard(layer, reshard_after_forward=True, **shard_kwargs)
    # Root (embed/lm_head) must reshard after forward on 24GB GPUs; keeping it
    # gathered costs ~vocab*hidden*2 BF16 on every rank.
    fully_shard(
        model,
        reshard_after_forward=bool(
            parallel_cfg.get("fsdp_root_reshard_after_forward", True)
        ),
        **shard_kwargs,
    )
    if pg.is_rank0:
        ignore_note = f", ignored_modules={len(ignored_modules)}" if ignored_modules else ""
        log.info(
            "Applied FSDP2 FULL_SHARD to %d expert units + %d %s blocks + root%s%s",
            expert_units,
            len(decoder_layers),
            decoder_layer_cls.__name__,
            ignore_note,
            ", cpu_offload=params" if offload_policy is not None else "",
        )
    return model


def _is_mlp_or_moe_child_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ("mlp", "moe", "feed_forward", "feedforward", "ffn"))


def _apply_selective_activation_checkpointing(model: nn.Module) -> int:
    """Checkpoint MLP/MoE blocks only; keep attention activations (FlashAttn).

    Verified memory↔speed trade used by selective-AC recipes: avoid recomputing
    attention while still cutting activation memory on the expert FFNs.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    inner = _unwrap_model(model)
    if not hasattr(inner, "model") or not hasattr(inner.model, "layers"):
        raise RuntimeError("Selective activation checkpointing needs model.model.layers")

    wrapped = 0
    for layer in inner.model.layers:
        for name, child in list(layer.named_children()):
            if not _is_mlp_or_moe_child_name(name):
                continue
            if type(child).__name__.startswith("Checkpoint"):
                continue
            setattr(layer, name, checkpoint_wrapper(child))
            wrapped += 1
    return wrapped


def enable_gradient_checkpointing(
    model: nn.Module,
    decoder_layer_cls: type[nn.Module],
    mode: str = "full",
) -> None:
    """Enable activation checkpointing on decoder layers when configured.

    Modes:
      - ``full``: checkpoint each decoder layer (HF GC or FSDP wrapper)
      - ``selective``: checkpoint MLP/MoE only (skip attention recompute)
    """
    mode = str(mode).lower().strip()
    if mode not in {"full", "selective"}:
        raise ValueError(f"activation_checkpointing_mode must be full|selective, got {mode!r}")

    if mode == "selective":
        wrapped = _apply_selective_activation_checkpointing(model)
        if _is_rank0():
            log.info("Enabled selective activation checkpointing on %d MLP/MoE blocks", wrapped)
        return

    supports_hf_gc = getattr(model, "supports_gradient_checkpointing", False) or getattr(
        type(model), "_supports_gradient_checkpointing", False
    )
    if supports_hf_gc and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if _is_rank0():
            log.info("Enabled HuggingFace gradient checkpointing on %s", type(model).__name__)
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda submodule: isinstance(submodule, decoder_layer_cls),
    )
    if _is_rank0():
        log.info(
            "Enabled full FSDP activation checkpointing on %s layers",
            decoder_layer_cls.__name__,
        )


@contextmanager
def low_precision_context(state: LowPrecisionState | bool) -> Iterator[None]:
    """TE autocast for MXFP8/NVFP4; no-op when disabled.

    Backward: a bare ``True`` enables MXFP8 (legacy ``fp8_context(True)`` callers).
    """
    if isinstance(state, bool):
        if not state:
            with nullcontext():
                yield
            return
        try:
            from transformer_engine.common import recipe as te_recipe

            state = LowPrecisionState(mode="mxfp8", recipe=te_recipe.MXFP8BlockScaling())
        except Exception:  # noqa: BLE001
            if _is_rank0():
                log.warning("transformer_engine recipe unavailable; falling back to BF16")
            with nullcontext():
                yield
            return

    if not state.enabled or state.recipe is None:
        with nullcontext():
            yield
        return

    try:
        with _te_autocast(state.recipe):
            yield
    except (ImportError, RuntimeError) as exc:
        if state.strict:
            raise RuntimeError("Transformer Engine autocast failed in strict mode") from exc
        if _is_rank0():
            log.warning("Transformer Engine autocast failed; falling back to BF16")
        with nullcontext():
            yield


@contextmanager
def fp8_context(enabled: bool | LowPrecisionState) -> Iterator[None]:
    """Backward-compatible alias for ``low_precision_context``."""
    with low_precision_context(enabled):
        yield


def compile_decoder_layers(model: nn.Module, pg: ProcessGroups, cfg: dict[str, Any]) -> None:
    """Compile decoder attention and MoE submodules for throughput."""
    del pg
    compile_cfg = cfg.get("compilation", {})
    if not bool(compile_cfg.get("enabled", False)):
        return
    if not hasattr(torch, "compile"):
        if _is_rank0():
            log.warning("torch.compile unavailable in this PyTorch build")
        return

    torch._dynamo.config.cache_size_limit = int(compile_cfg.get("cache_size_limit", 64))
    attention_mode = str(compile_cfg.get("attention_mode", "max-autotune"))
    moe_mode = str(compile_cfg.get("decoder_mode", "reduce-overhead"))
    inner = _unwrap_model(model)
    if not hasattr(inner, "model") or not hasattr(inner.model, "layers"):
        if _is_rank0():
            log.warning("Could not find decoder layers to compile")
        return

    compiled = 0
    for layer in inner.model.layers:
        for name, submodule in layer.named_children():
            lower = name.lower()
            if "attn" in lower or "attention" in lower:
                setattr(layer, name, torch.compile(submodule, mode=attention_mode))
                compiled += 1
            elif "mlp" in lower or "moe" in lower:
                setattr(layer, name, torch.compile(submodule, mode=moe_mode))
                compiled += 1
    if _is_rank0():
        log.info("torch.compile applied to %d decoder submodules", compiled)


def maybe_compile_optimizer(optimizer: Any, cfg: dict[str, Any]) -> None:
    """Optionally compile Muon and AdamW optimizer steps."""
    compile_cfg = cfg.get("compilation", {})
    if not bool(compile_cfg.get("compile_optimizer", False)):
        return
    if not hasattr(torch, "compile"):
        return
    optimizer.muon.step = torch.compile(optimizer.muon.step, fullgraph=False)
    optimizer.adamw.step = torch.compile(optimizer.adamw.step, fullgraph=False)
    if _is_rank0():
        log.info("Compiled optimizer step functions")
