"""MoE router monitoring, stability losses, and gradient utilities.

For MiMo-V2 / DeepSeek-style ``noaux_tc`` gates this module implements:
  - ST-MoE router z-loss on *live* fp32 logits (gradients reach the gate)
  - DeepSeek-V3 complementary sequence-wise aux loss (eqs. 17–20)
  - Aux-loss-free expert-bias updates with cross-rank load sync
  - Freezing of ``e_score_correction_bias`` so only ±γ updates touch it
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from src.dist.process_groups import ProcessGroups
from src.moe.router import is_router_module, is_router_param_name

log = logging.getLogger(__name__)

DEAD_EXPERT_THRESHOLD = 0.001
BIAS_CLAMP = 10.0


@dataclass
class RouterLayerStats:
    """Routing statistics for one MoE layer."""

    layer_index: int
    expert_load: Tensor
    load_balance_loss: float
    max_expert_load: float
    dead_expert_count: int

    def __repr__(self) -> str:
        return (
            f"RouterLayerStats(layer={self.layer_index}, max_load={self.max_expert_load:.4f}, "
            f"dead={self.dead_expert_count})"
        )


@dataclass
class _HookRecord:
    layer_index: int
    # Live logits for z-loss / soft routing probs (must retain grad for training).
    router_logits: Tensor | None = None
    # Detached copies for metrics / bias load accounting.
    routing_weights: Tensor | None = None
    expert_indices: Tensor | None = None
    batch_size: int | None = None
    seq_len: int | None = None
    scoring_func: str = "sigmoid"


class RouterHookManager:
    """Captures router logits and routing weights via forward hooks."""

    def __init__(self) -> None:
        self._records: list[_HookRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        # Accumulated expert selection counts for bias update (detached).
        self._load_counts: dict[int, Tensor] = {}
        self._routers: list[nn.Module] = []

    def __repr__(self) -> str:
        return f"RouterHookManager(records={len(self._records)}, hooks={len(self._handles)})"

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def router_count(self) -> int:
        return len(self._routers)

    def add_record(self, record: _HookRecord) -> None:
        self._records.append(record)

    def clear(self) -> None:
        """Drop current-microbatch live records (frees autograd graph)."""
        self._records.clear()

    def clear_all(self) -> None:
        self.clear()
        self._load_counts.clear()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear_all()
        self._routers.clear()

    def accumulate_selection_counts(self) -> None:
        """Fold detached expert selections from current records into load counts."""
        for record in self._records:
            if record.expert_indices is None:
                continue
            indices = record.expert_indices.detach().reshape(-1)
            if indices.numel() == 0:
                continue
            n_experts = int(indices.max().item()) + 1
            if record.router_logits is not None:
                n_experts = max(n_experts, int(record.router_logits.shape[-1]))
            counts = torch.bincount(indices, minlength=n_experts).to(dtype=torch.float32)
            prev = self._load_counts.get(record.layer_index)
            if prev is None:
                self._load_counts[record.layer_index] = counts
            else:
                if prev.numel() < counts.numel():
                    padded = torch.zeros(counts.numel(), device=prev.device, dtype=prev.dtype)
                    padded[: prev.numel()] = prev
                    prev = padded
                    self._load_counts[record.layer_index] = prev
                if counts.numel() < prev.numel():
                    padded = torch.zeros_like(prev)
                    padded[: counts.numel()] = counts
                    counts = padded
                prev.add_(counts.to(device=prev.device))

    def get_stats(self) -> dict[int, RouterLayerStats]:
        """Build per-layer stats from accumulated load counts (preferred) or live records."""
        if self._load_counts:
            stats: dict[int, RouterLayerStats] = {}
            for layer_index, counts in self._load_counts.items():
                expert_load = counts / counts.sum().clamp_min(1.0)
                n_experts = expert_load.numel()
                target = 1.0 / n_experts
                stats[layer_index] = RouterLayerStats(
                    layer_index=layer_index,
                    expert_load=expert_load,
                    load_balance_loss=float((expert_load - target).abs().mean().item()),
                    max_expert_load=float(expert_load.max().item()),
                    dead_expert_count=int((expert_load < DEAD_EXPERT_THRESHOLD).sum().item()),
                )
            return stats

        stats = {}
        by_layer: dict[int, list[_HookRecord]] = {}
        for record in self._records:
            by_layer.setdefault(record.layer_index, []).append(record)
        for layer_index, records in by_layer.items():
            indices_list = [r.expert_indices for r in records if r.expert_indices is not None]
            if not indices_list:
                continue
            indices = torch.cat([idx.reshape(-1) for idx in indices_list], dim=0)
            n_experts = int(indices.max().item()) + 1
            counts = torch.bincount(indices, minlength=n_experts).float()
            expert_load = counts / counts.sum().clamp_min(1.0)
            target = 1.0 / n_experts
            stats[layer_index] = RouterLayerStats(
                layer_index=layer_index,
                expert_load=expert_load,
                load_balance_loss=float((expert_load - target).abs().mean().item()),
                max_expert_load=float(expert_load.max().item()),
                dead_expert_count=int((expert_load < DEAD_EXPERT_THRESHOLD).sum().item()),
            )
        return stats


def _is_router_module(module: nn.Module, name: str) -> bool:
    """Backward-compatible wrapper around the shared conservative classifier."""
    return is_router_module(module, name)


def _is_mimo_gate(module: nn.Module) -> bool:
    return type(module).__name__ == "MiMoV2MoEGate"


def _gate_scoring_func(module: nn.Module) -> str:
    scoring = getattr(module, "scoring_func", None)
    if scoring is None:
        config = getattr(module, "config", None)
        scoring = getattr(config, "scoring_func", "sigmoid") if config is not None else "sigmoid"
    return str(scoring).lower()


def _capture_mimo_gate_record(
    module: nn.Module,
    hidden_states: Tensor,
    output: Any,
    layer_index: int,
) -> _HookRecord:
    """Capture live logits (for losses) and detached routing outputs (for metrics)."""
    record = _HookRecord(layer_index=layer_index, scoring_func=_gate_scoring_func(module))
    if hidden_states.ndim == 3:
        record.batch_size = int(hidden_states.shape[0])
        record.seq_len = int(hidden_states.shape[1])
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    else:
        flat = hidden_states
    # Keep live graph so z-loss / soft-prob aux can train the gate weight.
    logits = F.linear(flat.type(torch.float32), module.weight.type(torch.float32), None)
    expected_experts = getattr(module, "n_routed_experts", None)
    if expected_experts is not None and logits.shape[-1] != int(expected_experts):
        raise RuntimeError(
            f"Router logits width {logits.shape[-1]} != n_routed_experts {expected_experts}"
        )
    record.router_logits = logits

    if isinstance(output, tuple) and len(output) >= 2:
        topk_idx, topk_weight = output[0], output[1]
        if isinstance(topk_idx, Tensor):
            record.expert_indices = topk_idx.detach()
        if isinstance(topk_weight, Tensor):
            record.routing_weights = topk_weight.detach()
    return record


def register_router_hooks(model: nn.Module) -> RouterHookManager:
    """Register forward hooks on router/gate modules."""
    manager = RouterHookManager()
    layer_counter = 0
    for name, module in model.named_modules():
        if not _is_router_module(module, name):
            continue
        layer_index = layer_counter
        layer_counter += 1
        manager._routers.append(module)

        if _is_mimo_gate(module):

            def make_mimo_hook(layer_idx: int):
                def hook(gate_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                    hidden = inputs[0] if inputs else None
                    if not isinstance(hidden, Tensor):
                        return
                    manager.add_record(_capture_mimo_gate_record(gate_module, hidden, output, layer_idx))

                return hook

            handle = module.register_forward_hook(make_mimo_hook(layer_index))
        else:

            def make_hook(layer_idx: int):
                def hook(gate_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
                    record = _HookRecord(layer_index=layer_idx, scoring_func=_gate_scoring_func(gate_module))
                    if inputs:
                        first = inputs[0]
                        if isinstance(first, Tensor):
                            # Prefer recomputing from weight when available so grads reach the gate.
                            if hasattr(gate_module, "weight") and isinstance(gate_module.weight, Tensor):
                                flat = first.reshape(-1, first.shape[-1]) if first.ndim >= 2 else first
                                record.router_logits = F.linear(
                                    flat.type(torch.float32),
                                    gate_module.weight.type(torch.float32),
                                    None,
                                )
                                if first.ndim == 3:
                                    record.batch_size = int(first.shape[0])
                                    record.seq_len = int(first.shape[1])
                            else:
                                record.router_logits = first
                    if isinstance(output, tuple):
                        if len(output) >= 2 and isinstance(output[0], Tensor) and isinstance(output[1], Tensor):
                            record.expert_indices = output[0].detach()
                            record.routing_weights = output[1].detach()
                        elif len(output) >= 1 and isinstance(output[0], Tensor):
                            record.routing_weights = output[0].detach()
                        if len(output) >= 2 and record.expert_indices is None and isinstance(output[1], Tensor):
                            record.expert_indices = output[1].detach()
                    elif isinstance(output, Tensor):
                        record.routing_weights = output.detach()
                    elif isinstance(output, dict):
                        if "router_logits" in output and isinstance(output["router_logits"], Tensor):
                            record.router_logits = output["router_logits"]
                        if "routing_weights" in output and isinstance(output["routing_weights"], Tensor):
                            record.routing_weights = output["routing_weights"].detach()
                        if "expert_indices" in output and isinstance(output["expert_indices"], Tensor):
                            record.expert_indices = output["expert_indices"].detach()
                    manager.add_record(record)

                return hook

            handle = module.register_forward_hook(make_hook(layer_index))
        manager._handles.append(handle)
    return manager


def list_router_modules(model: nn.Module) -> list[nn.Module]:
    return [module for name, module in model.named_modules() if _is_router_module(module, name)]


def _get_expert_bias_param(router: nn.Module) -> nn.Parameter | None:
    """Return the load-balancing bias parameter for a gate module."""
    if hasattr(router, "e_score_correction_bias") and isinstance(router.e_score_correction_bias, nn.Parameter):
        return router.e_score_correction_bias
    if hasattr(router, "expert_bias"):
        bias = router.expert_bias
        if isinstance(bias, nn.Parameter):
            return bias
    return None


def _soft_routing_probs(logits: Tensor, scoring_func: str) -> Tensor:
    """Convert live router logits to affinity scores (pre-normalization)."""
    flat = logits.float().reshape(-1, logits.shape[-1])
    if scoring_func == "softmax":
        return F.softmax(flat, dim=-1)
    # Default / MiMo: sigmoid affinity scores.
    return torch.sigmoid(flat)


def _normalized_routing_probs(logits: Tensor, scoring_func: str) -> Tensor:
    """DeepSeek-V3 ``s'``: per-token distribution over experts (eq. 19).

    Softmax scores are already normalized. Sigmoid affinities are divided by
    their sum across experts so ``P_i`` is a valid mean probability.
    """
    scores = _soft_routing_probs(logits, scoring_func)
    if scoring_func == "softmax":
        return scores
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)


def freeze_expert_load_biases(model: nn.Module) -> int:
    """Freeze aux-loss-free load-balancing biases (manual ±γ updates only)."""
    frozen = 0
    for router in list_router_modules(model):
        bias = _get_expert_bias_param(router)
        if bias is None:
            continue
        if bias.requires_grad:
            bias.requires_grad_(False)
            frozen += 1
    if frozen and (not dist.is_initialized() or dist.get_rank() == 0):
        log.info("Froze %d expert load-balancing bias parameter(s)", frozen)
    return frozen


@torch.no_grad()
def update_expert_biases(
    model: nn.Module,
    hook_manager: RouterHookManager,
    bias_update_speed: float = 0.001,
    target_load: float | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> float:
    """Aux-loss-free expert bias update using cross-rank synchronized load.

    Maps ``layer_index ->`` the same-index router (not zip of sparse stats).
    All-reduces selection counts so every rank applies the same ``±γ`` deltas.
    """
    if bias_update_speed <= 0:
        return 0.0

    routers = hook_manager._routers or list_router_modules(model)
    group = process_group
    if group is None and dist.is_initialized():
        group = dist.group.WORLD
    distributed = group is not None and dist.is_initialized() and dist.get_world_size(group) > 1
    if torch.cuda.is_available():
        reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        reduce_device = torch.device("cpu")

    # Prefer accumulated counts; fall back to one-shot stats from records.
    load_counts = dict(hook_manager._load_counts)
    if not load_counts:
        for layer_index, layer_stats in hook_manager.get_stats().items():
            load_counts[layer_index] = layer_stats.expert_load * float(layer_stats.expert_load.numel())

    # Never return before collectives: empty local counts must still all-reduce zeros.
    n_routers = len(routers)
    if distributed:
        n_routers_t = torch.tensor([n_routers], device=reduce_device, dtype=torch.long)
        dist.all_reduce(n_routers_t, op=dist.ReduceOp.MAX, group=group)
        n_routers = int(n_routers_t.item())
    if n_routers == 0:
        return 0.0

    total_delta = 0.0
    for layer_index in range(n_routers):
        router = routers[layer_index] if layer_index < len(routers) else None
        bias = _get_expert_bias_param(router) if router is not None else None
        counts = load_counts.get(layer_index)
        if counts is None:
            synced = torch.zeros(0, device=reduce_device, dtype=torch.float32)
        else:
            synced = counts.detach().float().reshape(-1).to(device=reduce_device)

        if distributed:
            n_local = torch.tensor([synced.numel()], device=reduce_device, dtype=torch.long)
            dist.all_reduce(n_local, op=dist.ReduceOp.MAX, group=group)
            n_experts = int(n_local.item())
            if synced.numel() < n_experts:
                padded = torch.zeros(n_experts, device=reduce_device, dtype=torch.float32)
                padded[: synced.numel()] = synced
                synced = padded
            elif n_experts > 0 and synced.numel() > n_experts:
                synced = synced[:n_experts]
            elif n_experts == 0:
                continue
            dist.all_reduce(synced, op=dist.ReduceOp.SUM, group=group)
        if bias is None or synced.numel() == 0:
            continue
        synced = synced.to(device=bias.device)

        n_experts = min(synced.numel(), bias.numel())
        if float(synced[:n_experts].sum().item()) <= 0:
            continue
        load = synced[:n_experts] / synced[:n_experts].sum().clamp_min(1.0)
        target = target_load if target_load is not None else 1.0 / n_experts
        # Vectorized DeepSeek-style ±γ update.
        delta = torch.where(
            load > target,
            -torch.full_like(load, bias_update_speed),
            torch.where(
                load < target,
                torch.full_like(load, bias_update_speed),
                torch.zeros_like(load),
            ),
        )
        if isinstance(bias, DTensor):
            local_shape, offset = compute_local_shape_and_global_offset(
                bias.shape, bias.device_mesh, bias.placements
            )
            local_n = int(local_shape[0])
            start = int(offset[0])
            local_bias = bias.to_local()
            local_delta = delta[start : start + local_n].to(local_bias.device)
            before = local_bias.clone()
            local_bias.add_(local_delta)
            local_bias.clamp_(-BIAS_CLAMP, BIAS_CLAMP)
            total_delta += float((local_bias - before).abs().sum().item())
        else:
            before = bias[:n_experts].clone()
            bias[:n_experts].add_(delta)
            bias[:n_experts].clamp_(-BIAS_CLAMP, BIAS_CLAMP)
            total_delta += float((bias[:n_experts] - before).abs().sum().item())

    hook_manager._load_counts.clear()
    if total_delta > 0 and (not dist.is_initialized() or dist.get_rank() == 0):
        log.debug("Expert bias update total_delta=%.6f", total_delta)
    return total_delta


def _seq_aux_f_and_p(
    scores: Tensor,
    expert_indices: Tensor | None,
    n_experts: int,
) -> tuple[Tensor, Tensor]:
    """DeepSeek-V3 sequence-wise ``f`` (eq. 18) and ``P`` (eq. 20) for one sequence.

    ``scores`` is ``s'`` with shape ``(T, E)``. ``f_i = N_r / (K_r · T) · count_i``
    so balanced routing yields ``f_i ≈ 1`` and ``Σ_i f_i P_i ≈ 1``.
    """
    tokens = int(scores.shape[0])
    p_i = scores.mean(dim=0)
    if expert_indices is None:
        # Soft proxy with the same balanced scale as hard f (mean ≈ 1).
        f_i = n_experts * p_i.detach()
        return f_i, p_i

    idx = expert_indices.detach().reshape(tokens, -1)
    top_k = int(idx.shape[-1])
    counts = torch.bincount(idx.reshape(-1), minlength=n_experts).to(
        device=p_i.device, dtype=p_i.dtype
    )
    # Eq. 18: f_i = N_r / (K_r * T) * count_i
    f_i = counts * (float(n_experts) / float(max(top_k * tokens, 1)))
    return f_i, p_i


def compute_seq_auxiliary_loss(
    hook_manager: RouterHookManager,
    alpha: float = 1e-4,
) -> Tensor:
    """DeepSeek-V3 complementary sequence-wise aux loss (eqs. 17–20).

    ``L = α · mean_seq(Σ_i f_i P_i)`` with normalized affinities ``s'`` and
    hard top-k counts scaled by ``N_r / (K_r · T)``.
    """
    if alpha <= 0 or not hook_manager._records:
        return torch.zeros((), device=_infer_device(hook_manager))

    losses: list[Tensor] = []
    for record in hook_manager._records:
        if record.router_logits is None:
            continue
        scores = _normalized_routing_probs(record.router_logits, record.scoring_func)  # (T, E)
        n_experts = int(scores.shape[-1])
        indices = record.expert_indices

        if record.batch_size and record.seq_len and scores.shape[0] == record.batch_size * record.seq_len:
            bsz, seqlen = record.batch_size, record.seq_len
            scores_seq = scores.view(bsz, seqlen, n_experts)
            for seq_idx in range(bsz):
                if indices is None:
                    seq_idx_tensor = None
                elif indices.ndim == 3:
                    seq_idx_tensor = indices[seq_idx]
                else:
                    seq_idx_tensor = indices.reshape(bsz, seqlen, -1)[seq_idx]
                f_i, p_i = _seq_aux_f_and_p(scores_seq[seq_idx], seq_idx_tensor, n_experts)
                losses.append((f_i * p_i).sum())
        else:
            f_i, p_i = _seq_aux_f_and_p(scores, indices, n_experts)
            losses.append((f_i * p_i).sum())

    if not losses:
        return torch.zeros((), device=_infer_device(hook_manager))
    return alpha * torch.stack(losses).mean()


def compute_z_loss(
    hook_manager: RouterHookManager,
    z_loss_coeff: float = 1e-4,
) -> Tensor:
    """ST-MoE router z-loss on live logits: mean(logsumexp(logits)^2)."""
    if z_loss_coeff <= 0:
        return torch.zeros((), device=_infer_device(hook_manager))
    logits_list = [record.router_logits for record in hook_manager._records if record.router_logits is not None]
    if not logits_list:
        return torch.zeros((), device=_infer_device(hook_manager))
    values: list[Tensor] = []
    for logits in logits_list:
        flat = logits.float().reshape(-1, logits.shape[-1])
        lse = torch.logsumexp(flat, dim=-1)
        values.append((lse**2).mean())
    return z_loss_coeff * torch.stack(values).mean()


def _infer_device(hook_manager: RouterHookManager) -> torch.device:
    for record in hook_manager._records:
        if record.router_logits is not None:
            return record.router_logits.device
        if record.routing_weights is not None:
            return record.routing_weights.device
    return torch.device("cpu")


def _param_category(name: str) -> str:
    lower = name.lower()
    if is_router_param_name(name):
        return "router"
    if "expert" in lower or "mlp" in lower:
        return "expert_ffn"
    if "attn" in lower or "attention" in lower:
        return "attention"
    if "embed" in lower or "lm_head" in lower:
        return "embed_lm_head"
    return "other"


def get_grad_norms(model: nn.Module, pg: ProcessGroups) -> dict[str, float]:
    """Compute FSDP-aware gradient norms per module category."""
    keys = ("router", "expert_ffn", "attention", "embed_lm_head", "other", "total")
    # FSDP CPU offload keeps parameters on CPU; NCCL all_reduce needs a CUDA tensor.
    if torch.cuda.is_available():
        reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        reduce_device = next(model.parameters()).device
    category_sq = torch.zeros(len(keys), dtype=torch.float64, device=reduce_device)
    key_to_index = {key: index for index, key in enumerate(keys)}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        local_grad = grad.to_local() if isinstance(grad, DTensor) else grad
        sq = local_grad.double().pow(2).sum().to(reduce_device)
        category = _param_category(name)
        category_sq[key_to_index[category]] += sq
        category_sq[key_to_index["total"]] += sq
    if dist.is_initialized() and dist.get_world_size(pg.ep_group) > 1:
        dist.all_reduce(category_sq, op=dist.ReduceOp.SUM, group=pg.ep_group)
    return {
        key: float(category_sq[index].clamp_min(0).sqrt().item())
        for index, key in enumerate(keys)
    }


def _clip_params(params: list[Tensor], max_norm: float, global_norm: float) -> float:
    clip_coef = min(1.0, max_norm / (global_norm + 1e-6))
    if clip_coef < 1.0:
        for param in params:
            if param.grad is not None:
                param.grad.mul_(clip_coef)
    return global_norm


def clip_grad_norms_per_module(
    model: nn.Module,
    pg: ProcessGroups,
    expert_max_norm: float = 1.0,
    attention_max_norm: float = 1.0,
    router_max_norm: float = 0.5,
) -> dict[str, float]:
    """Clip gradients separately per module type."""
    buckets: dict[str, list[Tensor]] = {
        "router": [],
        "expert_ffn": [],
        "attention": [],
        "embed_lm_head": [],
        "other": [],
    }
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        category = _param_category(name)
        buckets.setdefault(category, []).append(param)
    norms = get_grad_norms(model, pg)
    _clip_params(buckets["router"], router_max_norm, norms["router"])
    _clip_params(buckets["expert_ffn"], expert_max_norm, norms["expert_ffn"])
    _clip_params(buckets["attention"], attention_max_norm, norms["attention"])
    other_params = buckets["embed_lm_head"] + buckets["other"]
    other_norm = (norms["embed_lm_head"] ** 2 + norms["other"] ** 2) ** 0.5
    _clip_params(other_params, expert_max_norm, other_norm)
    return norms


def log_expert_utilization(
    hook_manager: RouterHookManager,
    step: int,
    writer: Any | None = None,
    summary: dict[str, float] | None = None,
) -> None:
    """Print a compact one-line expert utilization summary."""
    values = summary if summary is not None else router_stats_summary(hook_manager)
    dead_experts = values["dead_experts"]
    max_load = values["max_expert_load"]
    imbalance = values["imbalance"]
    message = f"step {step} | dead_experts={dead_experts} | max_load={max_load:.4f} | imbalance={imbalance:.4f}"
    log.info(message)
    if writer is not None and hasattr(writer, "add_scalar"):
        writer.add_scalar("moe/dead_experts", dead_experts, step)
        writer.add_scalar("moe/max_load", max_load, step)
        writer.add_scalar("moe/imbalance", imbalance, step)


def router_stats_summary(
    hook_manager: RouterHookManager,
    process_group: dist.ProcessGroup | None = None,
) -> dict[str, float]:
    """Return router stats, optionally from globally summed selection counts."""
    if (
        process_group is not None
        and dist.is_initialized()
        and dist.get_world_size(process_group) > 1
    ):
        # Match update_expert_biases collectives: every rank participates even
        # when local counts are empty, and pad to a common expert width.
        if torch.cuda.is_available():
            reduce_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            reduce_device = torch.device("cpu")
        n_routers = len(hook_manager._routers) if hook_manager._routers else 0
        n_t = torch.tensor([n_routers], device=reduce_device, dtype=torch.long)
        dist.all_reduce(n_t, op=dist.ReduceOp.MAX, group=process_group)
        n_routers = int(n_t.item())
        stats: dict[int, RouterLayerStats] = {}
        for layer_index in range(n_routers):
            counts = hook_manager._load_counts.get(layer_index)
            if counts is None:
                synced = torch.zeros(0, device=reduce_device, dtype=torch.float32)
            else:
                synced = counts.detach().float().reshape(-1).to(device=reduce_device)
            n_local = torch.tensor([synced.numel()], device=reduce_device, dtype=torch.long)
            dist.all_reduce(n_local, op=dist.ReduceOp.MAX, group=process_group)
            n_experts = int(n_local.item())
            if n_experts == 0:
                continue
            if synced.numel() < n_experts:
                padded = torch.zeros(n_experts, device=reduce_device, dtype=torch.float32)
                padded[: synced.numel()] = synced
                synced = padded
            elif synced.numel() > n_experts:
                synced = synced[:n_experts]
            dist.all_reduce(synced, op=dist.ReduceOp.SUM, group=process_group)
            load = synced / synced.sum().clamp_min(1.0)
            target = 1.0 / max(load.numel(), 1)
            stats[layer_index] = RouterLayerStats(
                layer_index=layer_index,
                expert_load=load,
                load_balance_loss=float((load - target).abs().mean().item()),
                max_expert_load=float(load.max().item()),
                dead_expert_count=int((load < DEAD_EXPERT_THRESHOLD).sum().item()),
            )
    else:
        stats = hook_manager.get_stats()
    if not stats:
        return {"dead_experts": 0.0, "max_expert_load": 0.0, "imbalance": 0.0}
    return {
        "dead_experts": float(sum(item.dead_expert_count for item in stats.values())),
        "max_expert_load": float(max(item.max_expert_load for item in stats.values())),
        "imbalance": float(sum(item.load_balance_loss for item in stats.values()) / len(stats)),
    }
