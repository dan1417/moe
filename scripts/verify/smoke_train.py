#!/usr/bin/env python3
"""Smoke-test training wiring: gate patch, MoE losses, packing, bias updates."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test MoE training wiring")
    parser.add_argument("--config", default="configs/b300/train.yaml")
    parser.add_argument("--king-dir", default="./king")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--skip-torchrun", action="store_true", help="Run unit checks only (no GPU train)")
    return parser.parse_args()


def test_packing_mask() -> None:
    """Verify document attention mask blocks cross-doc attention."""
    import torch

    from src.data.packing import build_batch_document_attention_mask, find_document_boundaries

    eos_id = 0
    seq = torch.tensor([[1, 2, eos_id, 4, 5, eos_id, 7, 8] + [9] * 2040], dtype=torch.long)
    cu = find_document_boundaries(seq[0], eos_id)
    mask = build_batch_document_attention_mask(seq, eos_id)
    s1, e1 = int(cu[0]), int(cu[1])
    s2, e2 = int(cu[1]), int(cu[2])
    finfo_min = torch.finfo(mask.dtype).min
    assert mask[0, 0, s2, s1].item() == finfo_min, "cross-doc attention must be blocked"
    assert mask[0, 0, s1 + 1, s1].item() != finfo_min, "in-doc causal must allow prior tokens"
    log.info("packing mask test passed (cu_seqlens=%s)", cu.tolist())


def test_mimo_gate_hook() -> None:
    """Verify MiMo gate hook captures logits and correct output order."""
    import torch
    import torch.nn as nn

    from src.moe.utils import register_router_hooks

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            topk_idx = torch.zeros(hidden_states.shape[0], 2, dtype=torch.long)
            topk_weight = torch.ones(hidden_states.shape[0], 2)
            return topk_idx, topk_weight

    model = nn.Sequential(MiMoV2MoEGate())
    manager = register_router_hooks(model)
    hidden = torch.randn(2, 8)
    model(hidden)
    assert len(manager._records) == 1
    record = manager._records[0]
    assert record.router_logits is not None
    assert record.router_logits.shape == (2, 4)
    assert record.expert_indices is not None
    assert record.routing_weights is not None
    log.info("MiMo gate hook test passed")


def test_router_identity_and_optimizer_classification() -> None:
    """Expert gate projections must never be treated as MoE routers."""
    import torch
    import torch.nn as nn

    from src.moe.utils import register_router_hooks
    from src.optim.optimizer import _classify_param
    from src.moe.router import is_router_param_name

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(4))
            self.n_routed_experts = 4

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(x.shape[0], 2, dtype=torch.long), torch.ones(x.shape[0], 2)

    class Expert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)

    class TinyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = MiMoV2MoEGate()
            self.experts = nn.ModuleList([Expert()])
            self.norm = nn.LayerNorm(8)
            self.embed_tokens = nn.Embedding(32, 8)

    model = TinyMoE()
    manager = register_router_hooks(model)
    assert manager.router_count == 1
    assert not is_router_param_name("mlp.experts.0.gate_proj.weight")
    assert is_router_param_name("mlp.gate.weight")
    assert _classify_param(
        "mlp.experts.0.gate_proj.weight", model.experts[0].gate_proj.weight
    ) == "muon"
    assert _classify_param("mlp.gate.weight", model.gate.weight) == "router"
    assert (
        _classify_param("mlp.gate.e_score_correction_bias", model.gate.e_score_correction_bias)
        == "expert_bias"
    )
    assert _classify_param("model.norm.weight", model.norm.weight) == "no_decay"
    assert _classify_param("model.embed_tokens.weight", model.embed_tokens.weight) == "no_decay"
    model.gate(torch.randn(2, 8))
    assert manager.record_count == 1
    log.info("router identity and optimizer classification test passed")


def test_freeze_expert_load_biases() -> None:
    """Load-balancing biases must be frozen for aux-loss-free ±γ updates only."""
    import torch
    import torch.nn as nn

    from src.moe.utils import freeze_expert_load_biases, update_expert_biases, register_router_hooks

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(4))
            self.scoring_func = "sigmoid"

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = hidden_states.shape[0]
            idx = torch.zeros(n, 2, dtype=torch.long)
            idx[:6, :] = 0
            idx[6:, :] = 3
            return idx, torch.ones(n, 2)

    gate = MiMoV2MoEGate()
    assert gate.e_score_correction_bias.requires_grad
    assert freeze_expert_load_biases(gate) == 1
    assert not gate.e_score_correction_bias.requires_grad
    manager = register_router_hooks(gate)
    gate(torch.randn(8, 8))
    manager.accumulate_selection_counts()
    before = gate.e_score_correction_bias.clone()
    update_expert_biases(gate, manager, bias_update_speed=0.01)
    assert not torch.equal(before, gate.e_score_correction_bias), "frozen bias still gets ±γ"
    log.info("freeze expert load biases test passed")


def test_seq_aux_deepseek_scale() -> None:
    """Balanced top-k routing should yield seq-aux ≈ alpha (DeepSeek eqs. 17–20)."""
    import torch
    import torch.nn as nn

    from src.moe.utils import compute_seq_auxiliary_loss, register_router_hooks

    n_experts = 8
    top_k = 2
    seq_len = 16
    alpha = 1e-3

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Uniform logits → uniform sigmoid → uniform s'
            self.weight = nn.Parameter(torch.zeros(n_experts, 4))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(n_experts))
            self.scoring_func = "sigmoid"
            self.n_routed_experts = n_experts

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            bsz, seqlen, _ = hidden_states.shape
            # Perfectly round-robin top-k so f_i = 1 for every expert.
            flat = bsz * seqlen
            idx = torch.zeros(flat, top_k, dtype=torch.long)
            for t in range(flat):
                idx[t, 0] = (2 * t) % n_experts
                idx[t, 1] = (2 * t + 1) % n_experts
            return idx.view(bsz, seqlen, top_k), torch.ones(bsz, seqlen, top_k)

    gate = MiMoV2MoEGate()
    manager = register_router_hooks(gate)
    gate(torch.randn(1, seq_len, 4))
    aux = compute_seq_auxiliary_loss(manager, alpha=alpha)
    # Uniform s' → P_i = 1/E; balanced f_i = 1 → Σ fP = 1 → loss ≈ alpha.
    assert abs(float(aux.detach()) - alpha) < 0.05 * alpha, (
        f"balanced seq-aux should be ~alpha={alpha}, got {float(aux.detach())}"
    )
    assert aux.requires_grad
    log.info("seq-aux DeepSeek scale test passed (aux=%.6f alpha=%.6f)", float(aux.detach()), alpha)


def test_muon_flat_shard_apply() -> None:
    """Muon local-shard apply must update only the owned flat slice."""
    import torch
    import torch.nn as nn

    from src.optim.optimizer import _apply_muon_update, _fsdp_flat_shard_meta, _gather_matrix_grad

    param = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    update = torch.ones_like(param)
    # Simulate rank 0 of world=4 without distributed: direct apply path.
    _apply_muon_update(param, update, lr=1.0, weight_decay=0.0, group=None)
    assert torch.allclose(param.data, torch.arange(16, dtype=torch.float32).reshape(4, 4) - 1.0)

    # torch 2.14+ FSDP may flatten params; apply must still work via matrix_shape.
    flat_param = nn.Parameter(torch.arange(16, dtype=torch.float32))
    _apply_muon_update(
        flat_param,
        update,
        lr=1.0,
        weight_decay=0.0,
        group=None,
        matrix_shape=(4, 4),
    )
    assert torch.allclose(flat_param.data, torch.arange(16, dtype=torch.float32) - 1.0)

    shard_size, padded = _fsdp_flat_shard_meta(15, 4)
    assert shard_size == 4
    assert padded == 16

    # Flat 1D grad gather path (single-process fallback reshape).
    matrix_shape = (4, 4)
    flat_grad_param = nn.Parameter(torch.zeros(16, dtype=torch.float32))
    flat_grad_param.grad = torch.arange(16, dtype=torch.float32)
    gathered = _gather_matrix_grad(flat_grad_param, group=None, matrix_shape=matrix_shape)  # type: ignore[arg-type]
    assert gathered is not None
    full_grad, _shard_sizes = gathered
    assert tuple(full_grad.shape) == matrix_shape
    log.info("Muon flat shard helpers test passed")


def test_global_gradient_clipping() -> None:
    """A precomputed global norm must determine the clip coefficient."""
    import torch
    import torch.nn as nn

    from src.moe.utils import clip_grad_norms_per_module
    from src.dist.process_groups import ProcessGroups

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Linear(2, 2, bias=False)

    model = Tiny()
    model.attention.weight.grad = torch.full_like(model.attention.weight, 2.0)
    pg = ProcessGroups(
        ep_group=None,  # type: ignore[arg-type]
        dp_group=None,  # type: ignore[arg-type]
        ep_rank=0,
        dp_rank=0,
        ep_size=1,
        dp_size=1,
        local_rank=0,
        global_rank=0,
        world_size=1,
        node_rank=0,
        is_rank0=True,
    )
    norms = clip_grad_norms_per_module(model, pg, attention_max_norm=1.0)
    assert abs(norms["attention"] - 4.0) < 1e-6
    assert abs(float(model.attention.weight.grad.norm()) - 1.0) < 1e-5
    log.info("global gradient clipping test passed")


def test_packing_mode_fails_closed() -> None:
    """FlashAttention packing must be rejected until varlen wiring is implemented."""
    from src.train.cli import _validate_packing_mode

    model = type(
        "Model",
        (),
        {"config": type("Config", (), {"_attn_implementation": "flash_attention_4"})()},
    )()
    try:
        _validate_packing_mode(model, {"packing": {"enabled": True}})
        raise AssertionError("unsafe FlashAttention packing must raise")
    except RuntimeError as exc:
        assert "requires eager attention" in str(exc)
    _validate_packing_mode(
        type("Model", (), {"config": type("Config", (), {"_attn_implementation": "eager"})()})(),
        {"packing": {"enabled": True}},
    )
    log.info("packing fail-closed test passed")


def test_compile_noop_when_disabled() -> None:
    """Verify compile_decoder_layers is safe when compilation.enabled=false."""
    import torch.nn as nn

    from src.model import compile_decoder_layers
    from src.dist.process_groups import ProcessGroups

    class DummyLayer(nn.Module):
        def forward(self, x: nn.Module) -> nn.Module:
            return x

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": 1})()
            self.model = type("Inner", (), {"layers": nn.ModuleList([DummyLayer()])})()

    pg = ProcessGroups(
        ep_group=None,  # type: ignore[arg-type]
        dp_group=None,  # type: ignore[arg-type]
        ep_rank=0,
        dp_rank=0,
        ep_size=1,
        dp_size=1,
        local_rank=0,
        global_rank=0,
        world_size=1,
        node_rank=0,
        is_rank0=True,
    )
    model = DummyModel()
    compile_decoder_layers(model, pg, {"compilation": {"enabled": False}})
    log.info("compile noop test passed")


def test_expert_bias_param() -> None:
    """Verify update_expert_biases targets e_score_correction_bias."""
    import torch
    import torch.nn as nn

    from src.moe.utils import _get_expert_bias_param, register_router_hooks, update_expert_biases

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(4))
            self.scoring_func = "sigmoid"

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            n = hidden_states.shape[0]
            idx = torch.zeros(n, 2, dtype=torch.long)
            idx[:6, :] = 0
            idx[6:, :] = 3
            w = torch.ones(n, 2)
            return idx, w

    gate = MiMoV2MoEGate()
    assert _get_expert_bias_param(gate) is gate.e_score_correction_bias
    manager = register_router_hooks(gate)
    gate(torch.randn(8, 8))
    manager.accumulate_selection_counts()
    before = gate.e_score_correction_bias.clone()
    update_expert_biases(gate, manager, bias_update_speed=0.01)
    assert not torch.equal(before, gate.e_score_correction_bias), "bias should change"
    log.info("expert bias update test passed")


def test_moe_live_z_and_aux_loss() -> None:
    """Verify z-loss and seq-aux retain grad and update gate weights."""
    import torch
    import torch.nn as nn

    from src.moe.utils import compute_seq_auxiliary_loss, compute_z_loss, register_router_hooks

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 16))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(8))
            self.scoring_func = "sigmoid"

        def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # (B, S, H) -> topk indices/weights like MiMo
            bsz, seqlen, _ = hidden_states.shape
            flat = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = torch.nn.functional.linear(flat.float(), self.weight.float())
            idx = torch.topk(logits, k=2, dim=-1).indices
            w = torch.ones(flat.shape[0], 2)
            return idx.view(bsz, seqlen, 2), w.view(bsz, seqlen, 2)

    gate = MiMoV2MoEGate()
    manager = register_router_hooks(gate)
    hidden = torch.randn(2, 4, 16)
    gate(hidden)

    z = compute_z_loss(manager, z_loss_coeff=1e-3)
    aux = compute_seq_auxiliary_loss(manager, alpha=1e-3)
    assert z.requires_grad, "z-loss must retain grad into gate"
    assert aux.requires_grad, "seq-aux must retain grad into gate"

    before = gate.weight.detach().clone()
    (z + aux).backward()
    assert gate.weight.grad is not None and gate.weight.grad.abs().sum() > 0
    gate.weight.data.add_(gate.weight.grad, alpha=-0.01)
    assert not torch.equal(before, gate.weight), "gate weight should move from MoE losses"
    log.info("MoE live z-loss/seq-aux test passed (z=%.6f aux=%.6f)", float(z.detach()), float(aux.detach()))


def test_expert_bias_layer_pairing() -> None:
    """Verify bias updates map layer_index -> the matching gate (not zip-sparse)."""
    import torch
    import torch.nn as nn

    from src.moe.utils import RouterLayerStats, register_router_hooks, update_expert_biases

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(4))
            self.scoring_func = "sigmoid"

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(x.shape[0], 2, dtype=torch.long), torch.ones(x.shape[0], 2)

    model = nn.Sequential(MiMoV2MoEGate(), MiMoV2MoEGate(), MiMoV2MoEGate())
    manager = register_router_hooks(model)
    # Simulate missing middle-layer stats (only layers 0 and 2 observed).
    manager._load_counts = {
        0: torch.tensor([90.0, 10.0, 0.0, 0.0]),
        2: torch.tensor([0.0, 0.0, 90.0, 10.0]),
    }
    before = [g.e_score_correction_bias.clone() for g in model]
    update_expert_biases(model, manager, bias_update_speed=0.01)
    # Layer 0 overload on expert 0 -> bias0 decreases; layer 2 overload on expert 2.
    assert before[0][0] > model[0].e_score_correction_bias[0]
    assert torch.equal(before[1], model[1].e_score_correction_bias), "gate1 must be unchanged when no stats"
    assert before[2][2] > model[2].e_score_correction_bias[2]
    log.info("expert bias layer pairing test passed")


def test_ep_dispatcher_disabled() -> None:
    """Custom EP path must refuse to construct."""
    from src.moe.ep_dispatch import ExpertDispatcher

    try:
        ExpertDispatcher(n_experts=8, top_k=2, ep_group=None, hidden_dim=16)
        raise AssertionError("ExpertDispatcher must raise")
    except RuntimeError as exc:
        assert "disabled" in str(exc).lower()
    log.info("EP dispatcher disabled test passed")


def test_muon_rms_scaling() -> None:
    """Verify Moonlight RMS-matched update scaling produces correct magnitudes."""
    import math

    import torch
    import torch.nn as nn

    from src.optim.optimizer import DistributedMuon

    torch.manual_seed(42)
    init_weight = torch.randn(64, 128)
    x_input = torch.randn(4, 128)

    # ── Moonlight scaling (rms_scale=0.2) ──
    model1 = nn.Linear(128, 64, bias=False)
    model1.weight = nn.Parameter(init_weight.clone())
    opt_moon = DistributedMuon([model1.weight], lr=0.005, rms_scale=0.2)
    loss1 = (model1(x_input) ** 2).sum()
    loss1.backward()
    before1 = model1.weight.data.clone()
    opt_moon.step()
    moon_update = (model1.weight.data - before1).float()
    moon_rms = moon_update.norm() / math.sqrt(moon_update.numel())

    # ── Keller original scaling (rms_scale=None) ──
    model2 = nn.Linear(128, 64, bias=False)
    model2.weight = nn.Parameter(init_weight.clone())
    opt_keller = DistributedMuon([model2.weight], lr=0.005, rms_scale=None)
    loss2 = (model2(x_input) ** 2).sum()
    loss2.backward()
    before2 = model2.weight.data.clone()
    opt_keller.step()
    keller_update = (model2.weight.data - before2).float()
    keller_rms = keller_update.norm() / math.sqrt(keller_update.numel())

    m_dim, n_dim = 64, 128
    # Moonlight should scale by 0.2*sqrt(max(64,128)) = 0.2*sqrt(128) ≈ 2.26
    # Keller scales by max(1, 64/128)**0.5 = 1.0
    # So Moonlight RMS should be roughly 2.26× Keller RMS.
    expected_ratio = (0.2 * math.sqrt(max(m_dim, n_dim))) / max(1.0, m_dim / n_dim) ** 0.5
    actual_ratio = moon_rms / keller_rms if keller_rms > 0 else float("inf")
    assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.15, (
        f"RMS ratio mismatch: actual={actual_ratio:.3f}, expected={expected_ratio:.3f}"
    )
    log.info(
        "Muon RMS scaling test passed (moonlight_rms=%.6f, keller_rms=%.6f, ratio=%.3f, expected=%.3f)",
        moon_rms,
        keller_rms,
        actual_ratio,
        expected_ratio,
    )


def test_scheduler_updates_muon_lr() -> None:
    """Verify the WSD scheduler updates both Muon and AdamW LRs with correct ratios."""
    import torch
    import torch.nn as nn

    from src.optim.optimizer import BF16AdamW, CombinedOptimizer, DistributedMuon
    from src.optim.scheduler import WSDScheduler

    # Create simple params for each optimizer.
    muon_param = nn.Parameter(torch.randn(16, 16))
    adamw_param = nn.Parameter(torch.randn(4))
    router_param = nn.Parameter(torch.randn(8))

    muon_lr = 0.005
    adamw_lr = 1.5e-5
    router_lr = adamw_lr * 0.1
    lr_max = adamw_lr

    muon = DistributedMuon([muon_param], lr=muon_lr)
    adamw = BF16AdamW(
        [
            {"params": [adamw_param], "lr": adamw_lr, "weight_decay": 0.1},
            {"params": [router_param], "lr": router_lr, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.95),
    )
    combined = CombinedOptimizer(muon=muon, adamw=adamw, scheduler_lr_max=lr_max)

    scheduler = WSDScheduler(
        optimizer=combined,
        warmup_steps=10,
        stable_steps=80,
        decay_steps=10,
        lr_max=lr_max,
        lr_min=lr_max * 0.1,
    )

    # Verify initial LR ratios are stamped correctly.
    muon_ratio = combined.muon.param_groups[0]["lr_ratio"]
    adamw_ratio = combined.adamw.param_groups[0]["lr_ratio"]
    router_ratio = combined.adamw.param_groups[1]["lr_ratio"]
    assert abs(muon_ratio - muon_lr / lr_max) < 1e-6, f"muon lr_ratio wrong: {muon_ratio}"
    assert abs(adamw_ratio - 1.0) < 1e-6, f"adamw lr_ratio wrong: {adamw_ratio}"
    assert abs(router_ratio - 0.1) < 1e-6, f"router lr_ratio wrong: {router_ratio}"

    # Step 0 (warmup): all LRs should be near 0.
    scheduler.step()
    muon_actual = combined.muon.param_groups[0]["lr"]
    adamw_actual = combined.adamw.param_groups[0]["lr"]
    router_actual = combined.adamw.param_groups[1]["lr"]
    assert muon_actual < muon_lr, f"muon LR should be in warmup: {muon_actual}"
    assert adamw_actual < adamw_lr, f"adamw LR should be in warmup: {adamw_actual}"
    assert router_actual < router_lr, f"router LR should be in warmup: {router_actual}"

    # Step through to stable phase.
    for _ in range(14):
        scheduler.step()
    muon_stable = combined.muon.param_groups[0]["lr"]
    adamw_stable = combined.adamw.param_groups[0]["lr"]
    router_stable = combined.adamw.param_groups[1]["lr"]
    assert abs(muon_stable - muon_lr) / muon_lr < 0.01, f"muon LR not at max in stable: {muon_stable}"
    assert abs(adamw_stable - adamw_lr) / adamw_lr < 0.01, f"adamw LR not at max in stable: {adamw_stable}"
    assert abs(router_stable - router_lr) / router_lr < 0.01, f"router LR not at max in stable: {router_stable}"

    # Step through to decay phase and verify decay happens for all.
    for _ in range(85):
        scheduler.step()
    muon_decay = combined.muon.param_groups[0]["lr"]
    adamw_decay = combined.adamw.param_groups[0]["lr"]
    assert muon_decay < muon_lr, f"muon LR should have decayed: {muon_decay}"
    assert adamw_decay < adamw_lr, f"adamw LR should have decayed: {adamw_decay}"
    assert muon_decay > 0, "muon LR should not be zero mid-decay"
    log.info(
        "Scheduler Muon LR test passed (warmup→stable→decay verified for muon/adamw/router)"
    )


def test_muon_weight_decay() -> None:
    """Verify Muon weight decay actually shrinks parameter norms."""
    import torch
    import torch.nn as nn

    from src.optim.optimizer import DistributedMuon

    param = nn.Parameter(torch.randn(32, 32))
    norm_before = param.data.norm().item()

    opt = DistributedMuon([param], lr=0.001, weight_decay=0.05, rms_scale=0.2)
    # Give it a gradient and step.
    param.grad = torch.randn_like(param)
    opt.step()

    norm_after = param.data.norm().item()
    # Weight decay should reduce the norm (param *= 1 - lr*wd = 0.99995 per step),
    # but the update itself adds to the norm. Over multiple steps with zero grad,
    # pure decay dominates.
    opt.zero_grad(set_to_none=True)
    param.grad = torch.zeros_like(param)
    for _ in range(100):
        opt.step()
    norm_decayed = param.data.norm().item()
    assert norm_decayed < norm_after, (
        f"Weight decay should shrink norm over many zero-grad steps: "
        f"after_first={norm_after:.4f}, after_100_zero={norm_decayed:.4f}"
    )
    log.info("Muon weight decay test passed (norm %.4f → %.4f → %.4f)", norm_before, norm_after, norm_decayed)


def test_muon_momentum_is_local_shard() -> None:
    """Persistent Muon momentum must match the local parameter shard size."""
    import torch
    import torch.nn as nn

    from src.optim.optimizer import DistributedMuon

    param = nn.Parameter(torch.randn(64, 32))
    opt = DistributedMuon([param], lr=0.001, rms_scale=0.2)
    param.grad = torch.randn_like(param)
    opt.step()
    buf = opt.state[param]["momentum_buffer"]
    assert buf.numel() == param.numel(), (
        f"momentum must be local-sized: buf={buf.numel()} param={param.numel()}"
    )
    assert buf.dtype == torch.float32
    assert tuple(buf.shape) == (param.numel(),) or buf.shape == param.shape
    # Second step must not grow momentum to a replicated full copy.
    param.grad = torch.randn_like(param)
    opt.step()
    buf2 = opt.state[param]["momentum_buffer"]
    assert buf2.numel() == param.numel()
    log.info(
        "Muon local-shard momentum test passed (numel=%d dtype=%s)",
        buf2.numel(),
        buf2.dtype,
    )


def test_sequence_pool_resume_continuity() -> None:
    """Verify pool-mode sequence continuity across step boundaries.

    This unit test builds a tiny synthetic pool on disk and validates that:
      - `start_step` maps to the expected global sequence indices
      - switching num_workers (0 -> >0) does not change batch ordering
    """
    import json
    import tempfile

    import numpy as np

    from src.data import SEQUENCE_LENGTH, build_train_loader

    with tempfile.TemporaryDirectory() as td:
        pool_dir = Path(td)

        world_size = 2
        rank = 0
        micro_batch_size = 2
        grad_accum_steps = 2
        total_steps = 2

        sequences_per_step_per_rank = micro_batch_size * grad_accum_steps
        total_sequences_global = sequences_per_step_per_rank * world_size * total_steps

        sequences = np.zeros((total_sequences_global, SEQUENCE_LENGTH), dtype=np.uint32)
        source_ids = np.zeros((total_sequences_global,), dtype=np.uint16)

        for i in range(total_sequences_global):
            sequences[i].fill(i)  # allow simple “first token == global_seq_id” checks
            source_ids[i] = i % 2

        np.save(pool_dir / "pool_sequences.npy", sequences)
        np.save(pool_dir / "pool_source_ids.npy", source_ids)
        (pool_dir / "pool_meta.json").write_text(
            json.dumps(
                {
                    "sources": ["finewebedu", "automathtext-v2"],
                    "sequence_length": SEQUENCE_LENGTH,
                    "world_size": world_size,
                    "total_steps": total_steps,
                    "sequences_per_step_per_rank": sequences_per_step_per_rank,
                    "total_sequences_global": total_sequences_global,
                },
                indent=2,
            )
        )

        cfg = {
            "data": {
                "sequence_pool": {"enabled": True, "pool_dir": str(pool_dir)},
                "num_workers": 2,
                "pin_memory": False,
                "prefetch_factor": 2,
                "persistent_workers": False,
            },
            "training": {
                "micro_batch_size": micro_batch_size,
                "gradient_accumulation_steps": grad_accum_steps,
                "total_steps": total_steps,
            },
        }

        # “Resume” step 0: num_workers=0 for exact continuity.
        loader0 = build_train_loader(
            cfg,
            rank=rank,
            world_size=world_size,
            start_step=0,
            num_workers_override=0,
        )
        step0_tokens: list[int] = []
        for batch in loader0:
            # batch.input_ids is (B, T)
            step0_tokens.extend(int(x.item()) for x in batch.input_ids[:, 0])
            if len(step0_tokens) >= sequences_per_step_per_rank:
                step0_tokens = step0_tokens[:sequences_per_step_per_rank]
                break

        # “Resume” step 1: switch to configured num_workers.
        loader1 = build_train_loader(
            cfg,
            rank=rank,
            world_size=world_size,
            start_step=1,
            num_workers_override=None,
        )
        step1_tokens: list[int] = []
        for batch in loader1:
            step1_tokens.extend(int(x.item()) for x in batch.input_ids[:, 0])
            if len(step1_tokens) >= sequences_per_step_per_rank:
                step1_tokens = step1_tokens[:sequences_per_step_per_rank]
                break

        expected_step0 = [rank + local_idx * world_size for local_idx in range(sequences_per_step_per_rank)]
        expected_step1 = [
            rank + (sequences_per_step_per_rank + local_idx) * world_size
            for local_idx in range(sequences_per_step_per_rank)
        ]

        assert step0_tokens == expected_step0, f"step0 mismatch: got {step0_tokens} expected {expected_step0}"
        assert step1_tokens == expected_step1, f"step1 mismatch: got {step1_tokens} expected {expected_step1}"
        assert step1_tokens[0] == expected_step0[-1] + 2, "boundary continuity check failed"
        log.info("sequence_pool resume continuity test passed")


def test_low_precision_mode_resolve() -> None:
    """Verify low_precision aliases and legacy fp8_training mapping."""
    from src.model import (
        _capability_supports_low_precision,
        _layer_index_from_module_name,
        _should_enable_lp_layer,
        resolve_low_precision_mode,
    )
    import torch.nn as nn

    assert resolve_low_precision_mode({"training": {"low_precision": "mxfp8"}}) == "mxfp8"
    assert resolve_low_precision_mode({"training": {"low_precision": "fp4"}}) == "nvfp4"
    assert resolve_low_precision_mode({"training": {"low_precision": "none"}}) == "none"
    assert resolve_low_precision_mode({"training": {"fp8_training": True}}) == "mxfp8"
    assert resolve_low_precision_mode({"training": {}}) == "none"

    assert _capability_supports_low_precision("mxfp8", (10, 3))
    assert _capability_supports_low_precision("nvfp4", (10, 0))
    assert not _capability_supports_low_precision("mxfp8", (12, 0))
    assert not _capability_supports_low_precision("nvfp4", (9, 0))

    assert _layer_index_from_module_name("model.layers.47.mlp.experts.0.down_proj") == 47
    linear = nn.Linear(64, 64)
    assert _should_enable_lp_layer(
        "model.layers.0.mlp.down_proj",
        linear,
        n_layers=8,
        high_precision_tail=2,
        mode="nvfp4",
    )
    assert not _should_enable_lp_layer(
        "model.layers.7.mlp.down_proj",
        linear,
        n_layers=8,
        high_precision_tail=2,
        mode="nvfp4",
    )
    assert not _should_enable_lp_layer(
        "model.layers.0.mlp.gate",
        linear,
        n_layers=8,
        high_precision_tail=0,
        mode="mxfp8",
    )
    log.info("low_precision mode resolve test passed")


def test_selective_activation_checkpointing() -> None:
    """Verify selective AC wraps MLP/MoE children, not attention."""
    import torch.nn as nn

    from src.model import enable_gradient_checkpointing

    class DecLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Linear(8, 8)
            self.mlp = nn.Linear(8, 8)

    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([DecLayer(), DecLayer()])

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Inner()

    model = Model()
    enable_gradient_checkpointing(model, DecLayer, mode="selective")
    for layer in model.model.layers:
        assert "Checkpoint" in type(layer.mlp).__name__, "mlp must be checkpoint-wrapped"
        assert "Checkpoint" not in type(layer.self_attn).__name__, "attn must stay unwrapped"
    from src.model import _find_decoder_layer_class

    assert _find_decoder_layer_class(model) is DecLayer
    log.info("selective activation checkpointing test passed")


def test_lora_wraps_attention_not_expert_gate() -> None:
    """LoRA must wrap attention projections and leave expert gate_proj on Muon."""
    import torch
    import torch.nn as nn

    from src.model.lora import LoRALinear, apply_lora
    from src.optim.optimizer import _classify_param

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))

    class Expert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(8, 16, bias=False)

    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = nn.Linear(8, 24, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attn()
            self.gate = MiMoV2MoEGate()
            self.experts = nn.ModuleList([Expert()])

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = Layer()

    model = Model()
    wrapped = apply_lora(
        model,
        {
            "lora": {
                "enabled": True,
                "rank": 4,
                "alpha": 8,
                "freeze_base": True,
                "train_routers": True,
                "target_modules": ["qkv_proj", "o_proj"],
            }
        },
    )
    assert wrapped == 2
    assert isinstance(model.layer.self_attn.qkv_proj, LoRALinear)
    assert isinstance(model.layer.self_attn.o_proj, LoRALinear)
    assert isinstance(model.layer.experts[0].gate_proj, nn.Linear)
    assert not model.layer.self_attn.qkv_proj.base.weight.requires_grad
    assert model.layer.self_attn.qkv_proj.lora_A.requires_grad
    assert model.layer.gate.weight.requires_grad
    assert not model.layer.experts[0].gate_proj.weight.requires_grad
    assert _classify_param("layer.self_attn.qkv_proj.lora_A", model.layer.self_attn.qkv_proj.lora_A) == "muon"
    assert _classify_param("layer.gate.weight", model.layer.gate.weight) == "router"
    x = torch.randn(2, 8)
    out = model.layer.self_attn.qkv_proj(x)
    assert out.shape == (2, 24)
    log.info("LoRA wrap test passed")


def run_training_smoke(config: Path, steps: int) -> None:
    """Launch a short distributed training run via torchrun."""
    nproc = int(os.environ.get("NPROC", os.environ.get("LOCAL_WORLD_SIZE", "1")))
    env = os.environ.copy()
    env["TEUTONIC_ATTN"] = "flash_attention_4"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.setdefault("CUDA_TOOLKIT_PATH", "/usr/local/cuda-13.0")
    env.setdefault("CUDA_HOME", env["CUDA_TOOLKIT_PATH"])
    # FA4 first-step JIT under 8-way contention can take >10 min on some ranks.
    env.setdefault("NCCL_TIMEOUT", "3600")
    env.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    # Without this, every rank recompiles FA4 CuTe kernels from scratch (pathologically slow).
    env.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "1")
    cache_dir = str(ROOT / ".cache" / "fa4_cute")
    env.setdefault("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", cache_dir)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # Single-GPU prewarm fills the disk cache before 8 ranks contend on CuTe JIT.
    log.info("Prewarming FA4 MiMo kernels into %s", cache_dir)
    pre_env = {**env, "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES", "0")}
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from src.model import prewarm_fa4_mimo_kernels; "
                "prewarm_fa4_mimo_kernels(256)"
            ),
        ],
        cwd=str(ROOT),
        check=True,
        env=pre_env,
    )
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "-m",
        "src.train",
        "--config",
        str(config),
    ]
    log.info("Launching LoRA smoke train: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    test_packing_mask()
    test_mimo_gate_hook()
    test_router_identity_and_optimizer_classification()
    test_freeze_expert_load_biases()
    test_seq_aux_deepseek_scale()
    test_muon_flat_shard_apply()
    test_global_gradient_clipping()
    test_packing_mode_fails_closed()
    test_compile_noop_when_disabled()
    test_expert_bias_param()
    test_moe_live_z_and_aux_loss()
    test_expert_bias_layer_pairing()
    test_ep_dispatcher_disabled()
    test_muon_rms_scaling()
    test_scheduler_updates_muon_lr()
    test_muon_weight_decay()
    test_muon_momentum_is_local_shard()
    test_sequence_pool_resume_continuity()
    test_low_precision_mode_resolve()
    test_selective_activation_checkpointing()
    test_lora_wraps_attention_not_expert_gate()

    if args.skip_torchrun:
        log.info("Unit checks passed (--skip-torchrun)")
        return 0

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    import yaml

    cfg = yaml.safe_load(config_path.read_text())
    training = cfg.setdefault("training", {})
    training["smoke_steps"] = args.steps
    # Full 110B pre-training is not viable on 8× RTX PRO 6000. Smoke trains LoRA
    # adapters + routers only. Use FA4 (eager made the first step take forever).
    training["micro_batch_size"] = 1
    training["gradient_accumulation_steps"] = 1
    training["activation_checkpointing"] = True
    training["activation_checkpointing_mode"] = "selective"
    training["low_precision"] = "none"
    training["fp8_training"] = False
    training["attn_implementation"] = "flash_attention_4"
    training["log_every"] = 1
    # Short synthetic seq keeps smoke memory-safe while still exercising MoE/LoRA.
    training["smoke_seq_len"] = 256
    training["smoke_synthetic_data"] = True
    cfg.setdefault("parallelism", {})["fsdp_cpu_offload"] = False
    lora = cfg.setdefault("lora", {})
    lora["enabled"] = True
    lora.setdefault("rank", 8)
    lora.setdefault("alpha", 16)
    lora.setdefault("dropout", 0.0)
    lora["freeze_base"] = True
    lora["train_routers"] = True
    lora["save_full_checkpoint"] = False
    cfg.setdefault("checkpoint_averaging", {})["enabled"] = False
    cfg.setdefault("packing", {})["enabled"] = False
    cfg.setdefault("data", {})["num_workers"] = 0
    cfg.setdefault("data", {})["persistent_workers"] = False
    smoke_config = ROOT / "checkpoints" / "smoke_config.yaml"
    smoke_config.parent.mkdir(parents=True, exist_ok=True)
    smoke_config.write_text(yaml.dump(cfg))

    if not (ROOT / args.king_dir / "config.json").exists():
        log.warning("King not found at %s; skipping GPU smoke train", args.king_dir)
        log.info("Unit checks passed")
        return 0

    run_training_smoke(smoke_config, args.steps)
    log.info("Smoke training completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
