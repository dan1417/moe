"""Learning-rate schedulers for Teutonic SN3 continued pretraining."""

from __future__ import annotations

import logging
import math
from typing import Any, Literal

import torch
from torch.optim import Optimizer

log = logging.getLogger(__name__)

PhaseName = Literal["warmup", "stable", "decay"]


class WSDScheduler:
    """Warmup → Stable → Decay (cosine) scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_max: float,
        lr_min: float,
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._step_count = last_epoch + 1
        self._decay_triggered = False
        self._extra_stable_steps = 0

    def __repr__(self) -> str:
        return (
            f"WSDScheduler(step={self._step_count}, phase={self.phase!r}, "
            f"lr={self.get_lr(self._step_count):.3e})"
        )

    @property
    def phase(self) -> PhaseName:
        return self._phase_for_step(self._step_count)

    @property
    def step_count(self) -> int:
        return self._step_count

    def total_steps(self) -> int:
        return self.warmup_steps + self.stable_steps + self._extra_stable_steps + self.decay_steps

    def _phase_for_step(self, step: int) -> PhaseName:
        if step < self.warmup_steps:
            return "warmup"
        stable_end = self.warmup_steps + self.stable_steps + self._extra_stable_steps
        if step < stable_end:
            return "stable"
        return "decay"

    def get_lr(self, step: int) -> float:
        """Pure LR lookup with no optimizer side effects."""
        if step < 0:
            return 0.0
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.lr_max
            progress = step / self.warmup_steps
            return self.lr_max * progress

        stable_end = self.warmup_steps + self.stable_steps + self._extra_stable_steps
        if step < stable_end:
            return self.lr_max

        decay_start = stable_end
        if self.decay_steps == 0:
            return self.lr_min
        decay_step = min(step - decay_start, self.decay_steps)
        progress = decay_step / self.decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def _set_lr(self, lr: float) -> float:
        for group in self.optimizer.param_groups:
            ratio = group.get("lr_ratio", 1.0)
            group["lr"] = lr * ratio
        return lr

    def step(self) -> float:
        """Advance one step and update optimizer param groups."""
        lr = self.get_lr(self._step_count)
        self._set_lr(lr)
        self._step_count += 1
        return lr

    def trigger_decay(self) -> None:
        """Enter the decay phase immediately."""
        stable_end = self.warmup_steps + self.stable_steps + self._extra_stable_steps
        if self._step_count < stable_end:
            skipped = stable_end - self._step_count
            self._extra_stable_steps = max(0, self._extra_stable_steps - skipped)
            self._step_count = stable_end
        self._decay_triggered = True

    def extend_stable(self, n: int) -> None:
        """Add more stable-phase steps (useful when reusing checkpoints)."""
        self._extra_stable_steps += max(0, n)

    def state_dict(self) -> dict[str, Any]:
        return {
            "warmup_steps": self.warmup_steps,
            "stable_steps": self.stable_steps,
            "decay_steps": self.decay_steps,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "step_count": self._step_count,
            "decay_triggered": self._decay_triggered,
            "extra_stable_steps": self._extra_stable_steps,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.warmup_steps = int(state["warmup_steps"])
        self.stable_steps = int(state["stable_steps"])
        self.decay_steps = int(state["decay_steps"])
        self.lr_max = float(state["lr_max"])
        self.lr_min = float(state["lr_min"])
        self._step_count = int(state["step_count"])
        self._decay_triggered = bool(state.get("decay_triggered", False))
        self._extra_stable_steps = int(state.get("extra_stable_steps", 0))
        self._set_lr(self.get_lr(self._step_count))


class CosineWarmupScheduler:
    """Standard cosine schedule with linear warmup."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        lr_max: float,
        lr_min: float,
        last_epoch: int = -1,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self._step_count = last_epoch + 1

    def __repr__(self) -> str:
        return (
            f"CosineWarmupScheduler(step={self._step_count}/{self.total_steps}, "
            f"lr={self.get_lr(self._step_count):.3e})"
        )

    def get_lr(self, step: int) -> float:
        if step < 0:
            return 0.0
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.lr_max
            return self.lr_max * (step / self.warmup_steps)
        if self.total_steps <= self.warmup_steps:
            return self.lr_min
        progress = min(1.0, (step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def step(self) -> float:
        lr = self.get_lr(self._step_count)
        for group in self.optimizer.param_groups:
            ratio = group.get("lr_ratio", 1.0)
            group["lr"] = lr * ratio
        self._step_count += 1
        return lr

    def extend_total_steps(self, n: int) -> None:
        self.total_steps += max(0, n)

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.total_steps = int(state["total_steps"])
        self.warmup_steps = int(state["warmup_steps"])
        self.lr_max = float(state["lr_max"])
        self.lr_min = float(state["lr_min"])
        self._step_count = int(state["step_count"])
        lr = self.get_lr(self._step_count)
        for group in self.optimizer.param_groups:
            ratio = group.get("lr_ratio", 1.0)
            group["lr"] = lr * ratio


def build_scheduler(optimizer: Optimizer, cfg: dict[str, Any]) -> WSDScheduler | CosineWarmupScheduler:
    """Build the configured LR scheduler."""
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "wsd")
    lr_max = float(scheduler_cfg.get("lr_max", 1.5e-5))
    lr_min = float(scheduler_cfg.get("lr_min", 1.5e-6))
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 200))
    stable_steps = int(scheduler_cfg.get("stable_steps", 4000))
    decay_steps = int(scheduler_cfg.get("decay_steps", 800))

    if scheduler_type == "cosine":
        total_steps = int(scheduler_cfg.get("total_steps", warmup_steps + stable_steps + decay_steps))
        scheduler = CosineWarmupScheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr_max=lr_max,
            lr_min=lr_min,
        )
    else:
        scheduler = WSDScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            lr_max=lr_max,
            lr_min=lr_min,
        )
    log.info("Built %s scheduler", scheduler)
    return scheduler
