"""Shared, conservative router identification helpers."""

from __future__ import annotations

import torch.nn as nn

ROUTER_CLASS_NAMES = frozenset(
    {
        "MiMoV2MoEGate",
        "MiMoV2FlashTopkRouter",
        "MoEGate",
        "TopKRouter",
        "Router",
    }
)

_ROUTER_PARAM_NAMES = frozenset(
    {
        "e_score_correction_bias",
        "expert_bias",
        "router_bias",
    }
)


def is_router_module(module: nn.Module, qualified_name: str = "") -> bool:
    """Return whether *module* is a routing module, never an expert gate projection."""
    if type(module).__name__ in ROUTER_CLASS_NAMES:
        return True
    segments = tuple(part.lower() for part in qualified_name.split(".") if part)
    if any(part in {"gate_proj", "gate_up_proj"} for part in segments):
        return False
    return bool(segments and segments[-1] in {"gate", "router"})


def is_router_param_name(name: str) -> bool:
    """Classify router parameters using path segments instead of substrings."""
    segments = tuple(part.lower() for part in name.split(".") if part)
    if not segments:
        return False
    if any(part in {"gate_proj", "gate_up_proj"} for part in segments):
        return False
    if segments[-1] in _ROUTER_PARAM_NAMES:
        return True
    return any(part in {"gate", "router"} for part in segments[:-1])
