"""SparseLoCo inter-node gradient synchronization for multi-node training."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

QUANT_LEVELS = 4  # 2-bit => 4 quantization levels


@dataclass
class SparseLoCoStats:
    """Metrics from the most recent outer sync step."""

    compression_ratio: float = 1.0
    sync_time_ms: float = 0.0
    pseudo_grad_norm: float = 0.0
    enabled: bool = False

    def __repr__(self) -> str:
        return (
            f"SparseLoCoStats(enabled={self.enabled}, compression={self.compression_ratio:.4f}, "
            f"sync_ms={self.sync_time_ms:.2f}, pseudo_grad_norm={self.pseudo_grad_norm:.4f})"
        )


class SparseLoCo:
    """SparseLoCo outer optimizer for inter-node DP synchronization."""

    def __init__(
        self,
        model: nn.Module,
        dp_group: dist.ProcessGroup,
        inner_steps: int = 200,
        sparsity_k: float = 0.01,
        quant_bits: int = 2,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.dp_group = dp_group
        self.inner_steps = inner_steps
        self.sparsity_k = sparsity_k
        self.quant_bits = quant_bits
        self.outer_lr = outer_lr
        self.outer_momentum = outer_momentum
        self.enabled = enabled and dist.is_initialized() and dist.get_world_size(dp_group) > 1
        self._theta_init: dict[int, Tensor] = {}
        self._error_feedback: dict[int, Tensor] = {}
        self._velocity: dict[int, Tensor] = {}
        self.last_stats = SparseLoCoStats(enabled=self.enabled)

    def __repr__(self) -> str:
        return (
            f"SparseLoCo(enabled={self.enabled}, inner_steps={self.inner_steps}, "
            f"sparsity_k={self.sparsity_k}, quant_bits={self.quant_bits})"
        )

    def should_sync(self, step: int) -> bool:
        return self.enabled and step > 0 and step % self.inner_steps == 0

    def snapshot_params(self) -> None:
        """Save theta_init snapshot for the next inner loop."""
        if not self.enabled:
            return
        self._theta_init.clear()
        for param in self._trainable_params():
            self._theta_init[id(param)] = param.detach().float().clone()

    @torch.no_grad()
    def outer_step(self, step: int) -> SparseLoCoStats:
        """Run SparseLoCo outer sync across the DP group."""
        del step
        if not self.enabled:
            self.last_stats = SparseLoCoStats(enabled=False)
            return self.last_stats

        start = time.monotonic()
        total_numel = 0
        kept_numel = 0
        pseudo_norm_sq = 0.0

        for param in self._trainable_params():
            key = id(param)
            if key not in self._theta_init:
                self._theta_init[key] = param.detach().float().clone()

            pseudo_grad = self._theta_init[key] - param.detach().float()
            if key in self._error_feedback:
                pseudo_grad = pseudo_grad + self._error_feedback[key]

            sparse_grad, mask = self._top_k_sparsify(pseudo_grad, self.sparsity_k)
            self._error_feedback[key] = pseudo_grad * (1.0 - mask)

            quantized, scale, offset = self._quantize(sparse_grad, self.quant_bits)
            dist.all_reduce(quantized, op=dist.ReduceOp.SUM, group=self.dp_group)
            quantized /= dist.get_world_size(self.dp_group)
            synced_grad = self._dequantize(quantized, scale, offset) * mask

            velocity = self._velocity.get(key)
            if velocity is None:
                velocity = torch.zeros_like(synced_grad)
            nesterov_grad = self.outer_momentum * velocity + synced_grad
            param.data.add_(nesterov_grad.to(dtype=param.dtype), alpha=-self.outer_lr)
            self._velocity[key] = self.outer_momentum * velocity + synced_grad

            numel = pseudo_grad.numel()
            total_numel += numel
            kept_numel += int(mask.sum().item())
            pseudo_norm_sq += float(pseudo_grad.pow(2).sum().item())

        if self.dp_group is not None and dist.get_world_size(self.dp_group) > 1:
            norm_tensor = torch.tensor([pseudo_norm_sq], device=self._device(), dtype=torch.float64)
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=self.dp_group)
            pseudo_norm_sq = float(norm_tensor.item())

        sync_time_ms = (time.monotonic() - start) * 1000.0
        compression_ratio = kept_numel / max(total_numel, 1)
        stats = SparseLoCoStats(
            compression_ratio=compression_ratio,
            sync_time_ms=sync_time_ms,
            pseudo_grad_norm=pseudo_norm_sq**0.5,
            enabled=True,
        )
        self.last_stats = stats
        self.snapshot_params()

        if self._is_rank0():
            log.info(
                "SparseLoCo outer sync: compression_ratio=%.4f sync_time_ms=%.2f pseudo_grad_norm=%.4f",
                stats.compression_ratio,
                stats.sync_time_ms,
                stats.pseudo_grad_norm,
            )
        return stats

    def _trainable_params(self) -> list[nn.Parameter]:
        return [param for param in self.model.parameters() if param.requires_grad]

    def _device(self) -> torch.device:
        for param in self.model.parameters():
            return param.device
        return torch.device("cpu")

    def _is_rank0(self) -> bool:
        return dist.get_rank() == 0 if dist.is_initialized() else True

    def _top_k_sparsify(self, tensor: Tensor, keep_fraction: float) -> tuple[Tensor, Tensor]:
        flat = tensor.reshape(-1)
        numel = flat.numel()
        if numel == 0:
            return tensor, torch.zeros_like(tensor)
        keep = max(1, int(numel * keep_fraction))
        threshold = torch.topk(flat.abs(), keep, largest=True).values.min()
        mask = (flat.abs() >= threshold).reshape(tensor.shape).to(tensor.dtype)
        return tensor * mask, mask

    def _quantize(self, tensor: Tensor, quant_bits: int) -> tuple[Tensor, float, float]:
        if quant_bits != 2:
            return tensor, 1.0, 0.0
        active = tensor[tensor != 0]
        if active.numel() == 0:
            return tensor, 1.0, 0.0
        min_t = active.min().detach().clone()
        max_t = active.max().detach().clone()
        if self.enabled and dist.is_initialized() and dist.get_world_size(self.dp_group) > 1:
            dist.all_reduce(min_t, op=dist.ReduceOp.MIN, group=self.dp_group)
            dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=self.dp_group)
        min_val = float(min_t.item())
        max_val = float(max_t.item())
        if min_val == max_val:
            return torch.zeros_like(tensor), 1.0, min_val
        scale = (max_val - min_val) / (QUANT_LEVELS - 1)
        offset = min_val
        quantized = torch.round((tensor - offset) / scale).clamp(0, QUANT_LEVELS - 1)
        return quantized, scale, offset

    def _dequantize(self, quantized: Tensor, scale: float, offset: float) -> Tensor:
        return quantized * scale + offset

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "inner_steps": self.inner_steps,
            "sparsity_k": self.sparsity_k,
            "quant_bits": self.quant_bits,
            "outer_lr": self.outer_lr,
            "outer_momentum": self.outer_momentum,
            "velocity": {str(key): value.cpu() for key, value in self._velocity.items()},
            "error_feedback": {str(key): value.cpu() for key, value in self._error_feedback.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.enabled = bool(state.get("enabled", self.enabled))
        self.inner_steps = int(state.get("inner_steps", self.inner_steps))
        self.sparsity_k = float(state.get("sparsity_k", self.sparsity_k))
        self.quant_bits = int(state.get("quant_bits", self.quant_bits))
        self.outer_lr = float(state.get("outer_lr", self.outer_lr))
        self.outer_momentum = float(state.get("outer_momentum", self.outer_momentum))
        self._velocity = {
            int(key): value.to(self._device())
            for key, value in state.get("velocity", {}).items()
        }
        self._error_feedback = {
            int(key): value.to(self._device())
            for key, value in state.get("error_feedback", {}).items()
        }
