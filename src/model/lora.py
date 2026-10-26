"""Minimal LoRA adapters for memory-safe smoke / adapter training.

Full 110B continued pre-training is not viable on 8× RTX PRO 6000 (96 GB).
This module freezes the base weights and trains low-rank adapters on attention
projections. Routers can stay trainable so MoE z/aux losses still update bias.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.optim.optimizer import _is_expert_load_bias_name
from src.moe.router import is_router_param_name

log = logging.getLogger(__name__)

DEFAULT_TARGET_MODULES = ("qkv_proj", "q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):
    """Wrap an existing ``nn.Linear`` with a frozen base and trainable A/B."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        in_features = int(base.in_features)
        out_features = int(base.out_features)
        dtype = base.weight.dtype
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=dtype, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora_x = self.dropout(x)
        lora_out = F.linear(F.linear(lora_x, self.lora_A), self.lora_B)
        return result + lora_out * self.scaling


def _is_target_linear(child_name: str, targets: Sequence[str]) -> bool:
    return child_name in targets


def apply_lora(model: nn.Module, cfg: dict[str, Any]) -> int:
    """Inject LoRA adapters and freeze the base model.

    Returns the number of Linear modules wrapped.
    """
    lora_cfg = cfg.get("lora") or {}
    if not bool(lora_cfg.get("enabled", False)):
        return 0

    rank = int(lora_cfg.get("rank", 8))
    alpha = float(lora_cfg.get("alpha", 16))
    dropout = float(lora_cfg.get("dropout", 0.0))
    targets = tuple(lora_cfg.get("target_modules") or DEFAULT_TARGET_MODULES)
    freeze_base = bool(lora_cfg.get("freeze_base", True))
    train_routers = bool(lora_cfg.get("train_routers", True))
    if rank < 1:
        raise ValueError(f"lora.rank must be >= 1, got {rank}")

    wrapped = 0
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and _is_target_linear(name, targets):
                setattr(parent, name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                wrapped += 1

    if wrapped == 0:
        raise RuntimeError(
            "lora.enabled=true but no target Linear modules were wrapped. "
            f"targets={targets}"
        )

    if freeze_base:
        for name, param in model.named_parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            lower = name.lower()
            if "lora_a" in lower or "lora_b" in lower:
                param.requires_grad_(True)
            elif (
                train_routers
                and is_router_param_name(name)
                and not _is_expert_load_bias_name(name)
            ):
                param.requires_grad_(True)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(
        "LoRA enabled: wrapped=%d rank=%d alpha=%s trainable=%d frozen=%d train_routers=%s",
        wrapped,
        rank,
        alpha,
        trainable,
        frozen,
        train_routers,
    )
    if trainable <= 0:
        raise RuntimeError("LoRA freeze left zero trainable parameters")
    return wrapped


def lora_enabled(cfg: dict[str, Any]) -> bool:
    return bool((cfg.get("lora") or {}).get("enabled", False))


def should_save_full_checkpoint(cfg: dict[str, Any]) -> bool:
    """Full 110B FSDP gathers are not part of LoRA smoke."""
    lora_cfg = cfg.get("lora") or {}
    if not bool(lora_cfg.get("enabled", False)):
        return True
    return bool(lora_cfg.get("save_full_checkpoint", False))
