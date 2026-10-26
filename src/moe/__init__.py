"""MoE router hooks, losses, and expert-parallel dispatch."""

from src.moe.ep_dispatch import ExpertDispatcher
from src.moe.router import is_router_module, is_router_param_name
from src.moe.utils import (
    RouterLayerStats,
    clip_grad_norms_per_module,
    compute_seq_auxiliary_loss,
    compute_z_loss,
    freeze_expert_load_biases,
    log_expert_utilization,
    register_router_hooks,
    router_stats_summary,
    update_expert_biases,
)

__all__ = [
    "ExpertDispatcher",
    "RouterLayerStats",
    "clip_grad_norms_per_module",
    "compute_seq_auxiliary_loss",
    "compute_z_loss",
    "freeze_expert_load_biases",
    "is_router_module",
    "is_router_param_name",
    "log_expert_utilization",
    "register_router_hooks",
    "router_stats_summary",
    "update_expert_biases",
]
