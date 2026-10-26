"""QK normalization check and optional injection for MiMo-V2."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

QK_CONFIG_KEYS = ("qk_norm", "qk_layernorm", "use_qk_norm")


def _config_has_qk_norm(config: Any) -> bool | None:
    for key in QK_CONFIG_KEYS:
        if hasattr(config, key):
            return bool(getattr(config, key))
    if isinstance(config, dict):
        for key in QK_CONFIG_KEYS:
            if key in config:
                return bool(config[key])
    return None


def _is_attention_module(module: nn.Module, name: str) -> bool:
    class_name = type(module).__name__.lower()
    lower = name.lower()
    return "attention" in class_name or "attn" in lower


class _QKNormHookState:
    def __init__(self) -> None:
        self.handles: list[torch.utils.hooks.RemovableHandle] = []


def check_and_inject_qk_norm(model: nn.Module, cfg: dict[str, Any]) -> bool:
    """Inject QK norm only when explicitly enabled and absent from the king config."""
    qk_cfg = cfg.get("qk_norm", {})
    inject_requested = bool(qk_cfg.get("enabled", False))
    config = getattr(model, "config", None)
    existing = _config_has_qk_norm(config) if config is not None else None

    if existing is True:
        log.info("Model already has QK norm enabled in config")
        return True
    if not inject_requested:
        if existing is False:
            log.info("QK norm disabled in king config; leaving architecture unchanged")
        return False
    raise RuntimeError(
        "qk_norm.enabled is unsupported: the current forward pre-hook changes the "
        "MiMo attention call signature. Keep qk_norm.enabled=false."
    )

    # Retained for reference; unreachable until a model-native implementation is validated.
    state = _QKNormHookState()
    eps = float(qk_cfg.get("eps", 1e-6))
    for name, module in model.named_modules():
        if not _is_attention_module(module, name):
            continue
        if not hasattr(module, "q_proj") or not hasattr(module, "k_proj"):
            continue

        def make_hook(attn_module: nn.Module):
            def pre_hook(_module: nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
                if not inputs:
                    return inputs
                hidden_states = inputs[0]
                q = attn_module.q_proj(hidden_states)
                k = attn_module.k_proj(hidden_states)
                q = _rms_norm(q, eps=eps)
                k = _rms_norm(k, eps=eps)
                return (hidden_states, q, k, *inputs[1:])

            return pre_hook

        handle = module.register_forward_pre_hook(make_hook(module))
        state.handles.append(handle)

    model._qk_norm_hook_state = state  # type: ignore[attr-defined]
    log.warning("Injected forward-time QK RMSNorm hooks; verify this matches eval expectations")
    return True


def _rms_norm(tensor: torch.Tensor, eps: float) -> torch.Tensor:
    variance = tensor.pow(2).mean(dim=-1, keepdim=True)
    return tensor * torch.rsqrt(variance + eps)
