"""Hybrid Muon + AdamW optimizer for Teutonic SN3 continued pretraining.

FSDP2 / 110B memory contract for Muon:
  - Persistent ``momentum_buffer`` is ALWAYS a local shard
    (``numel == param.to_local().numel()`` under DTensor, or ``param.numel()``
    under legacy flat shards). Never store a full ``(M, N)`` FP32 replica.
  - Newton–Schulz needs one full matrix; gather it transiently in BF16, apply the
    local rows, then free it before the next parameter.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import Shard
from torch.optim import Optimizer

from src.dist.process_groups import ProcessGroups
from src.moe.router import is_router_param_name

log = logging.getLogger(__name__)

NS_COEFF_A = 3.4445
NS_COEFF_B = -4.7750
NS_COEFF_C = 2.0315

SPECIAL_KEYWORDS = ("embed", "norm", "bias", "lm_head")
EXPERT_BIAS_PARAM_NAMES = frozenset(
    {
        "e_score_correction_bias",
        "expert_bias",
        "router_bias",
    }
)
MAX_ROUTER_PARAMS = 500_000_000


def _is_expert_load_bias_name(name: str) -> bool:
    """True for aux-loss-free load-balancing bias parameters."""
    leaf = name.rsplit(".", 1)[-1].lower()
    return leaf in EXPERT_BIAS_PARAM_NAMES


def _is_no_decay_param(name: str, param: Tensor) -> bool:
    """Norms, biases, embeddings, and LM head should not receive AdamW WD."""
    if param.ndim <= 1:
        return True
    lower = name.lower()
    return any(keyword in lower for keyword in SPECIAL_KEYWORDS)


class NewtonSchulzOrthogonalizer:
    """Five-step Newton-Schulz orthogonalization."""

    def __init__(self, ns_steps: int = 5) -> None:
        self.ns_steps = ns_steps

    def __repr__(self) -> str:
        return f"NewtonSchulzOrthogonalizer(ns_steps={self.ns_steps})"

    def orthogonalize(self, matrix: Tensor) -> Tensor:
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape {tuple(matrix.shape)}")
        transposed = False
        x = matrix.bfloat16()
        # Transpose so rows >= cols (matches reference: KellerJordan/Muon).
        if x.size(0) > x.size(1):
            x = x.T
            transposed = True
        # Normalize so spectral norm is at most 1.
        x = x / (x.norm() + 1e-7)
        for _ in range(self.ns_steps):
            a_mat = x @ x.T
            b_mat = NS_COEFF_B * a_mat + NS_COEFF_C * a_mat @ a_mat
            x = NS_COEFF_A * x + b_mat @ x
        if transposed:
            x = x.T
        return x.to(dtype=matrix.dtype)


def _muon_matrix_shape(param: Tensor, state: dict[str, Any]) -> tuple[int, ...]:
    """Return the logical 2D shape recorded at optimizer construction."""
    shape = state.get("matrix_shape")
    if shape is not None:
        return tuple(shape)
    if param.ndim == 2:
        return tuple(param.shape)
    raise RuntimeError(
        "DistributedMuon missing matrix_shape for flattened parameter; "
        f"param shape={tuple(param.shape)}"
    )


def _reshape_muon_grad(grad: Tensor, matrix_shape: tuple[int, ...]) -> Tensor:
    """View a possibly flattened gradient as the original 2D matrix."""
    if grad.ndim == 2 and tuple(grad.shape) == matrix_shape:
        return grad
    if grad.ndim == 1 and grad.numel() == math.prod(matrix_shape):
        return grad.reshape(matrix_shape)
    raise RuntimeError(
        "DistributedMuon cannot reshape gradient to matrix form: "
        f"grad={tuple(grad.shape)} matrix_shape={matrix_shape}"
    )


def _fsdp_flat_shard_meta(numel: int, world_size: int) -> tuple[int, int]:
    """Return ``(shard_size, padded_numel)`` using ceil sharding."""
    shard_size = (numel + world_size - 1) // world_size
    return shard_size, shard_size * world_size


def _all_gather_flat_shards(
    local: Tensor,
    group: dist.ProcessGroup,
    expected_numel: int,
) -> tuple[Tensor, list[int]]:
    """All-gather variable-size local flat shards into one contiguous tensor."""
    world_size = dist.get_world_size(group)
    local_flat = local.reshape(-1).contiguous()
    size_t = torch.tensor([local_flat.numel()], device=local.device, dtype=torch.long)
    size_list = [torch.zeros_like(size_t) for _ in range(world_size)]
    dist.all_gather(size_list, size_t, group=group)
    shard_sizes = [int(x.item()) for x in size_list]
    if sum(shard_sizes) != expected_numel:
        raise RuntimeError(
            "DistributedMuon shard sizes do not reconstruct matrix numel: "
            f"shard_sizes={shard_sizes} sum={sum(shard_sizes)} expected={expected_numel}"
        )
    if max(shard_sizes) == expected_numel and sum(1 for s in shard_sizes if s > 0) == 1:
        owner = shard_sizes.index(expected_numel)
        full = torch.zeros(expected_numel, device=local.device, dtype=local.dtype)
        if dist.get_rank(group) == owner and local_flat.numel() > 0:
            full.copy_(local_flat)
        dist.broadcast(full, src=owner, group=group)
        return full, shard_sizes

    max_size = max(shard_sizes)
    padded = torch.zeros(max_size, device=local.device, dtype=local.dtype)
    if local_flat.numel() > 0:
        padded[: local_flat.numel()] = local_flat
    gather_list = [torch.empty_like(padded) for _ in range(world_size)]
    dist.all_gather(gather_list, padded, group=group)
    full = torch.cat([gather_list[i][: shard_sizes[i]] for i in range(world_size)], dim=0)
    return full, shard_sizes


def _gather_matrix_grad(
    param: Tensor,
    group: dist.ProcessGroup | None,
    matrix_shape: tuple[int, ...],
) -> tuple[Tensor | None, list[int] | None]:
    """Reconstruct a full 2D gradient from local shards (legacy / unit-test path)."""
    grad = param.grad
    numel = math.prod(matrix_shape)
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        if grad is None:
            return None, None
        return _reshape_muon_grad(grad.detach().float(), matrix_shape), None

    local = (
        grad.detach().float().reshape(-1)
        if grad is not None and param.numel() > 0
        else torch.zeros(0, device=param.device, dtype=torch.float32)
    )
    full_flat, shard_sizes = _all_gather_flat_shards(local, group, numel)
    if float(full_flat.abs().sum().item()) == 0.0:
        return None, shard_sizes
    return full_flat.reshape(matrix_shape), shard_sizes


def _extract_local_flat_shard(
    full_flat: Tensor,
    rank: int,
    shard_sizes: list[int],
) -> Tensor:
    """Slice this rank's contiguous flat shard out of a gathered tensor."""
    local_size = shard_sizes[rank]
    if local_size == 0:
        return full_flat.new_empty(0)
    offset = sum(shard_sizes[:rank])
    return full_flat[offset : offset + local_size]


def _apply_muon_update(
    param: Tensor,
    full_update: Tensor,
    lr: float,
    weight_decay: float,
    group: dist.ProcessGroup | None,
    matrix_shape: tuple[int, ...] | None = None,
    shard_sizes: list[int] | None = None,
) -> None:
    """Apply Muon update (+ optional WD) to the locally owned shard."""
    if group is None or not dist.is_initialized() or dist.get_world_size(group) == 1:
        if weight_decay != 0:
            param.data.mul_(1.0 - lr * weight_decay)
        update = full_update.reshape(param.shape).to(param.dtype)
        param.data.add_(update, alpha=-lr)
        return

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if shard_sizes is not None:
        local_size = shard_sizes[rank]
        if local_size == 0:
            return
        offset = sum(shard_sizes[:rank])
        local_update = full_update.reshape(-1)[offset : offset + local_size]
        if weight_decay != 0:
            param.data.mul_(1.0 - lr * weight_decay)
        param.data.add_(local_update.to(param.dtype), alpha=-lr)
        return

    numel = full_update.numel()
    shard_size, _ = _fsdp_flat_shard_meta(numel, world_size)

    if (
        matrix_shape is not None
        and full_update.ndim == 2
        and param.ndim == 1
        and param.numel() == shard_size
        and numel == math.prod(matrix_shape)
    ):
        start = rank * shard_size
        end = min(start + shard_size, numel)
        local_update = full_update.reshape(-1)[start:end]
        if weight_decay != 0:
            param.data.mul_(1.0 - lr * weight_decay)
        param.data.add_(local_update.to(param.dtype), alpha=-lr)
        return

    if full_update.ndim == 2 and param.ndim == 1 and param.numel() == numel:
        full_update = full_update.reshape(-1)

    if param.shape != full_update.shape:
        if (
            full_update.ndim == 2
            and full_update.shape[0] % world_size == 0
            and full_update.shape[1] == param.shape[1]
            and full_update.shape[0] // world_size == param.shape[0]
        ):
            local_update = full_update.chunk(world_size, dim=0)[rank]
            if weight_decay != 0:
                param.data.mul_(1.0 - lr * weight_decay)
            param.data.add_(local_update.to(param.dtype), alpha=-lr)
            return
        raise RuntimeError(
            "DistributedMuon cannot apply update: "
            f"param={tuple(param.shape)} update={tuple(full_update.shape)} world={world_size}"
        )

    shard_size, _ = _fsdp_flat_shard_meta(param.numel(), world_size)
    start = rank * shard_size
    end = min(start + shard_size, param.numel())
    if end <= start:
        return
    param_flat = param.data.view(-1)
    upd_flat = full_update.reshape(-1).to(dtype=param.dtype)
    if weight_decay != 0:
        param_flat[start:end].mul_(1.0 - lr * weight_decay)
    param_flat[start:end].add_(upd_flat[start:end], alpha=-lr)


def _scale_muon_update(update: Tensor, rms_scale: float | None) -> Tensor:
    m_dim, n_dim = update.shape
    if rms_scale is not None:
        update.mul_(rms_scale * math.sqrt(max(m_dim, n_dim)))
    else:
        update.mul_(max(1.0, m_dim / n_dim) ** 0.5)
    return update


def _ensure_local_momentum(
    state: dict[str, Any],
    *,
    local_numel: int,
    device: torch.device,
    ref: Tensor | None = None,
) -> Tensor:
    """Allocate or reshape persistent momentum to the local shard size only."""
    buf = state.get("momentum_buffer")
    if isinstance(buf, DTensor):
        # Upgrade older DTensor momentum to a plain local shard tensor.
        buf = buf.to_local().detach().float().reshape(-1).clone()
        state["momentum_buffer"] = buf
    if not isinstance(buf, Tensor) or buf.numel() != local_numel:
        if ref is not None and isinstance(ref, DTensor):
            buf = torch.zeros_like(ref.to_local(), dtype=torch.float32).reshape(-1)
        else:
            buf = torch.zeros(local_numel, device=device, dtype=torch.float32)
        state["momentum_buffer"] = buf
    return buf.view(-1)


class DistributedMuon(Optimizer):
    """Distributed Muon with sharded FP32 momentum and transient NS gathers.

    Uses Moonlight RMS-matched update scaling by default:
        update *= rms_scale * sqrt(max(fan_in, fan_out))
    Set ``rms_scale=None`` to fall back to Keller's original scaling.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        ep_group: dist.ProcessGroup | None = None,
        rms_scale: float | None = 0.2,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
        self.ep_group = ep_group
        self.rms_scale = rms_scale
        self.orthogonalizer = NewtonSchulzOrthogonalizer(ns_steps=ns_steps)
        for group in self.param_groups:
            for param in group["params"]:
                # FSDP2 DTensor params keep logical ndim==2 even when locally flat.
                if param.ndim != 2:
                    raise ValueError(
                        "DistributedMuon params must be logical 2D matrices; "
                        f"got shape={tuple(param.shape)}"
                    )
                self.state[param]["matrix_shape"] = tuple(param.shape)
                # Momentum allocated lazily on first step (local shard only).

    def __repr__(self) -> str:
        return f"DistributedMuon(lr={self.defaults['lr']}, momentum={self.defaults['momentum']})"

    def _step_dtensor_param(
        self,
        param: DTensor,
        state: dict[str, Any],
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
    ) -> None:
        grad = param.grad
        if grad is None:
            return
        if not isinstance(grad, DTensor):
            raise RuntimeError(
                f"FSDP2 Muon expected a DTensor gradient, got {type(grad).__name__}"
            )
        if len(param.placements) != 1 or not isinstance(param.placements[0], Shard):
            raise RuntimeError(
                "FSDP2 Muon requires a one-dimensional Shard placement; "
                f"got placements={param.placements}"
            )
        if param.placements[0].dim != 0:
            raise RuntimeError(
                f"FSDP2 Muon requires row-wise Shard(0); got {param.placements}"
            )

        local_param = param.to_local()
        local_grad = grad.to_local().detach().float().reshape(-1)
        local_numel = local_param.numel()
        buf = _ensure_local_momentum(
            state,
            local_numel=local_numel,
            device=local_param.device,
            ref=param,
        )
        # Momentum stays local: buf <- m*buf + g
        buf.mul_(momentum).add_(local_grad)
        if nesterov:
            local_source = local_grad.add(buf, alpha=momentum)
        else:
            local_source = buf

        # Build a sharded BF16 DTensor view of the local source, gather once for NS.
        source_dt = DTensor.from_local(
            local_source.to(torch.bfloat16).reshape_as(local_param),
            device_mesh=param.device_mesh,
            placements=param.placements,
            shape=param.shape,
            stride=param.stride(),
        )
        full_source = source_dt.full_tensor()
        update = self.orthogonalizer.orthogonalize(full_source)
        _scale_muon_update(update, self.rms_scale)

        local_shape, offset = compute_local_shape_and_global_offset(
            param.shape,
            param.device_mesh,
            param.placements,
        )
        local_rows = int(local_shape[0])
        if local_rows > 0:
            row_start = int(offset[0])
            local_update = update[row_start : row_start + local_rows]
            if tuple(local_update.shape) != tuple(local_param.shape):
                raise RuntimeError(
                    "FSDP2 Muon local update shape mismatch: "
                    f"param={tuple(local_param.shape)} update={tuple(local_update.shape)}"
                )
            if weight_decay != 0:
                local_param.mul_(1.0 - lr * weight_decay)
            local_param.add_(local_update.to(dtype=local_param.dtype), alpha=-lr)

        # Free transient full-matrix workspace before the next parameter.
        del local_grad, local_source, source_dt, full_source, update

    def _step_local_param(
        self,
        param: Tensor,
        state: dict[str, Any],
        matrix_shape: tuple[int, ...],
        *,
        lr: float,
        momentum: float,
        nesterov: bool,
        weight_decay: float,
    ) -> None:
        if param.grad is None and (self.ep_group is None or param.numel() > 0):
            # Empty local shard ranks still participate in collectives below.
            if self.ep_group is None or not dist.is_initialized() or dist.get_world_size(self.ep_group) == 1:
                return

        local_numel = param.numel()
        buf = _ensure_local_momentum(
            state,
            local_numel=local_numel,
            device=param.device,
        )

        if self.ep_group is None or not dist.is_initialized() or dist.get_world_size(self.ep_group) == 1:
            if param.grad is None:
                return
            full_grad = _reshape_muon_grad(param.grad.detach().float(), matrix_shape)
            # Single-process: local shard IS the full matrix.
            if buf.numel() != full_grad.numel():
                buf = full_grad.new_zeros(full_grad.numel())
                state["momentum_buffer"] = buf
            buf_view = buf.view_as(full_grad)
            buf_view.mul_(momentum).add_(full_grad)
            source = full_grad.add(buf_view, alpha=momentum) if nesterov else buf_view
            update = self.orthogonalizer.orthogonalize(source.bfloat16())
            _scale_muon_update(update, self.rms_scale)
            _apply_muon_update(param, update, lr, weight_decay, None, matrix_shape=matrix_shape)
            del full_grad, source, update
            return

        # Distributed legacy/flat path: gather grad, update full momentum transiently,
        # persist only this rank's flat shard.
        full_grad, shard_sizes = _gather_matrix_grad(param, self.ep_group, matrix_shape)
        if full_grad is None or shard_sizes is None:
            return
        rank = dist.get_rank(self.ep_group)
        # Reconstruct full momentum via all-gather of local shards only.
        full_buf_flat, _ = _all_gather_flat_shards(buf, self.ep_group, full_grad.numel())
        full_buf = full_buf_flat.reshape(matrix_shape)
        full_buf.mul_(momentum).add_(full_grad)
        source = full_grad.add(full_buf, alpha=momentum) if nesterov else full_buf
        update = self.orthogonalizer.orthogonalize(source.bfloat16())
        _scale_muon_update(update, self.rms_scale)
        # Persist only the local momentum shard.
        state["momentum_buffer"] = _extract_local_flat_shard(
            full_buf.reshape(-1), rank, shard_sizes
        ).clone()
        _apply_muon_update(
            param,
            update,
            lr,
            weight_decay,
            self.ep_group,
            matrix_shape=matrix_shape,
            shard_sizes=shard_sizes,
        )
        del full_grad, full_buf, full_buf_flat, source, update

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                state = self.state[param]
                matrix_shape = _muon_matrix_shape(param, state)
                if isinstance(param, DTensor):
                    self._step_dtensor_param(
                        param,
                        state,
                        lr=lr,
                        momentum=momentum,
                        nesterov=nesterov,
                        weight_decay=weight_decay,
                    )
                else:
                    self._step_local_param(
                        param,
                        state,
                        matrix_shape,
                        lr=lr,
                        momentum=momentum,
                        nesterov=nesterov,
                        weight_decay=weight_decay,
                    )
        return loss


class BF16AdamW(torch.optim.AdamW):
    """AdamW that stores optimizer states in BF16 when enabled."""

    def __init__(self, *args: Any, bf16_states: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bf16_states = bf16_states

    def __repr__(self) -> str:
        return f"BF16AdamW(lr={self.defaults['lr']}, bf16_states={self.bf16_states})"

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = super().step(closure)
        if self.bf16_states:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state.get(param)
                    if not state:
                        continue
                    if "exp_avg" in state:
                        state["exp_avg"] = state["exp_avg"].to(torch.bfloat16)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"] = state["exp_avg_sq"].to(torch.bfloat16)
        return loss


class CombinedOptimizer:
    """Unified interface over Muon and AdamW optimizers.

    Exposes *all* param groups (Muon + AdamW) so the LR scheduler can
    reach every group.  Each group carries an ``"lr_ratio"`` field that
    records its initial LR relative to the scheduler's ``lr_max``.  The
    scheduler sets a single "base" LR value; each group's actual LR is
    ``base_lr * lr_ratio``, preserving the intended ratio throughout
    warmup → stable → decay.
    """

    def __init__(
        self,
        muon: DistributedMuon,
        adamw: BF16AdamW,
        scheduler_lr_max: float,
    ) -> None:
        self.muon = muon
        self.adamw = adamw
        for group in self.muon.param_groups + self.adamw.param_groups:
            if scheduler_lr_max > 0:
                group["lr_ratio"] = group["lr"] / scheduler_lr_max
            else:
                group["lr_ratio"] = 1.0

    def __repr__(self) -> str:
        return f"CombinedOptimizer(muon={self.muon}, adamw={self.adamw})"

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.muon.param_groups + self.adamw.param_groups

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])


def _classify_param(name: str, param: Tensor) -> str:
    """Classify a parameter into an optimizer bucket.

    Returns one of: ``expert_bias`` (skip), ``router``, ``muon``, ``no_decay``.
    Under the Muon hybrid, all non-router AdamW params are no-decay (norms,
    biases, embeddings, lm_head, and any residual 1D tensors).
    """
    if _is_expert_load_bias_name(name):
        return "expert_bias"
    if is_router_param_name(name):
        return "router"
    if _is_no_decay_param(name, param):
        return "no_decay"
    if param.ndim == 2:
        return "muon"
    return "no_decay"


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any], pg: ProcessGroups) -> CombinedOptimizer:
    """Build the hybrid Muon + AdamW optimizer."""
    opt_cfg = cfg.get("optimizer", {})
    muon_params: list[Tensor] = []
    adamw_no_decay_params: list[Tensor] = []
    router_params: list[Tensor] = []
    counts = {"muon": 0, "adamw_no_decay": 0, "router": 0, "expert_bias_skipped": 0}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            if _is_expert_load_bias_name(name):
                counts["expert_bias_skipped"] += param.numel()
            continue
        category = _classify_param(name, param)
        if category == "expert_bias":
            param.requires_grad_(False)
            counts["expert_bias_skipped"] += param.numel()
            continue
        if category == "muon":
            muon_params.append(param)
            counts["muon"] += param.numel()
        elif category == "router":
            router_params.append(param)
            counts["router"] += param.numel()
        else:
            adamw_no_decay_params.append(param)
            counts["adamw_no_decay"] += param.numel()

    if counts["router"] >= MAX_ROUTER_PARAMS:
        raise RuntimeError(
            "Router parameter classification is implausibly large: "
            f"{counts['router']:,} parameters (limit {MAX_ROUTER_PARAMS:,}). "
            "Expert gate projections may have been misclassified."
        )
    if not muon_params and not adamw_no_decay_params and not router_params:
        raise RuntimeError("No trainable parameters found for the optimizer")
    if not muon_params:
        raise RuntimeError(
            "Muon parameter list is empty. Full-model training expects 2D matrices; "
            "LoRA adapters are 2D and should be classified as Muon."
        )
    if not adamw_no_decay_params and not router_params:
        raise RuntimeError(
            "AdamW parameter groups are empty. For LoRA smoke keep lora.train_routers=true "
            "so router params remain on AdamW."
        )
    invalid_muon = [
        tuple(param.shape) for param in muon_params if getattr(param, "ndim", 0) != 2
    ]
    if invalid_muon:
        raise RuntimeError(f"Muon only supports 2D parameters; invalid shapes: {invalid_muon[:5]}")

    rms_scale_raw = opt_cfg.get("muon_rms_scale", 0.2)
    rms_scale: float | None = float(rms_scale_raw) if rms_scale_raw is not None else None
    muon = DistributedMuon(
        muon_params,
        lr=float(opt_cfg.get("muon_lr", 0.005)),
        momentum=float(opt_cfg.get("muon_momentum", 0.95)),
        nesterov=bool(opt_cfg.get("muon_nesterov", True)),
        ns_steps=int(opt_cfg.get("muon_ns_steps", 5)),
        weight_decay=float(opt_cfg.get("muon_weight_decay", 0.0)),
        ep_group=pg.ep_group,
        rms_scale=rms_scale,
    )
    adamw_lr = float(opt_cfg.get("adamw_lr", 1.5e-5))
    router_lr_scale = float(opt_cfg.get("router_lr_scale", 0.1))
    _ = float(opt_cfg.get("adamw_weight_decay", 0.1))
    adamw_groups: list[dict[str, Any]] = []
    if adamw_no_decay_params:
        adamw_groups.append({"params": adamw_no_decay_params, "lr": adamw_lr, "weight_decay": 0.0})
    if router_params:
        adamw_groups.append(
            {
                "params": router_params,
                "lr": adamw_lr * router_lr_scale,
                "weight_decay": 0.0,
            }
        )
    betas = opt_cfg.get("adamw_betas", [0.9, 0.95])
    adamw = BF16AdamW(
        adamw_groups,
        betas=(float(betas[0]), float(betas[1])),
        eps=float(opt_cfg.get("adamw_eps", 1e-8)),
        bf16_states=bool(opt_cfg.get("adamw_bf16_states", True)),
    )
    scheduler_cfg = cfg.get("scheduler", {})
    scheduler_lr_max = float(scheduler_cfg.get("lr_max", adamw_lr))
    if pg.is_rank0:
        log.info(
            "Optimizer groups: muon=%d adamw_no_decay=%d router=%d expert_bias_skipped=%d",
            counts["muon"],
            counts["adamw_no_decay"],
            counts["router"],
            counts["expert_bias_skipped"],
        )
    return CombinedOptimizer(muon=muon, adamw=adamw, scheduler_lr_max=scheduler_lr_max)
