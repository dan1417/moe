#!/usr/bin/env python3
"""Verify MoE training pipeline wiring matches the MiMo-V2 / Teutonic design.

Checks (no king checkpoint required):
  1. Config design defaults
  2. Static source contracts (fail-closed paths, loss composition)
  3. Runtime unit suite from smoke_train.py
  4. Optimizer / MoE formula invariants for MiMo-V2 gates
"""

from __future__ import annotations

import argparse
import ast
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def check_config_design(cfg: dict) -> list[str]:
    """Assert train_config matches the intended MiMo-V2 CPT design."""
    failures: list[str] = []
    moe = cfg.get("moe", {})
    opt = cfg.get("optimizer", {})
    sched = cfg.get("scheduler", {})
    train = cfg.get("training", {})
    parallel = cfg.get("parallelism", {})
    packing = cfg.get("packing", {})
    sparse = cfg.get("sparse_loco", {})

    expectations = [
        (moe.get("dispatch_overlap") is False, "moe.dispatch_overlap must be false"),
        (moe.get("delay_wgrad") is False, "moe.delay_wgrad must be false"),
        (float(moe.get("bias_update_speed", -1)) == 0.001, "bias_update_speed should be 0.001"),
        (float(moe.get("z_loss_coeff", -1)) == 1e-4, "z_loss_coeff should be 1e-4"),
        (float(moe.get("seq_aux_alpha", -1)) == 1e-4, "seq_aux_alpha should be 1e-4"),
        (int(moe.get("node_limit", -1)) == 1, "node_limit must be 1"),
        (parallel.get("fsdp") is True, "parallelism.fsdp must be true"),
        (sched.get("type") == "wsd", "scheduler.type must be wsd"),
        (float(opt.get("router_lr_scale", 0)) == 0.1, "router_lr_scale should be 0.1"),
        (float(opt.get("gradient_clip", {}).get("router", 0)) == 0.5, "router clip should be 0.5"),
        (packing.get("enabled") is False, "packing.enabled should stay false until FA4 varlen"),
        (sparse.get("enabled") is False, "sparse_loco.enabled false on single-node default"),
        (train.get("activation_checkpointing_mode") == "selective", "AC mode should be selective"),
    ]
    for ok, msg in expectations:
        if not ok:
            failures.append(msg)
    failures.extend(check_profile_lrs(cfg))
    return failures


PROFILE_LRS = {
    "production": {"lr_max": 1.5e-5, "muon_lr": 0.005, "adamw_lr": 1.5e-5},
    # Depth-shrink 8.59B / toy mini CPT can use higher LRs for short patterned runs.
    "mini": {"lr_max": 3.0e-4, "muon_lr": 0.02, "adamw_lr": 3.0e-4},
}


def check_profile_lrs(cfg: dict) -> list[str]:
    """Validate LR knobs against training.profile (production|mini)."""
    train = cfg.get("training", {})
    profile = str(train.get("profile") or "").strip().lower()
    if not profile:
        # Infer from synthetic mini CPT flags when profile is omitted.
        if train.get("smoke_synthetic_data"):
            profile = "mini"
        else:
            profile = "production"
    expected = PROFILE_LRS.get(profile)
    if expected is None:
        return [f"unknown training.profile={profile!r}; use production|mini"]
    failures: list[str] = []
    sched = cfg.get("scheduler", {})
    opt = cfg.get("optimizer", {})
    checks = [
        (float(sched.get("lr_max", 0)), expected["lr_max"], "scheduler.lr_max"),
        (float(opt.get("muon_lr", 0)), expected["muon_lr"], "optimizer.muon_lr"),
        (float(opt.get("adamw_lr", 0)), expected["adamw_lr"], "optimizer.adamw_lr"),
    ]
    for got, want, name in checks:
        if abs(got - want) > 1e-12:
            failures.append(f"{name} for profile={profile} should be {want}, got {got}")
    return failures


def check_source_contracts() -> list[str]:
    """Static checks that train/model/moe/optimizer follow the design."""
    failures: list[str] = []
    train_src = (ROOT / "src" / "train" / "cli.py").read_text()
    model_src = (ROOT / "src" / "model" / "core.py").read_text()
    moe_src = (ROOT / "src" / "moe" / "utils.py").read_text()
    opt_src = (ROOT / "src" / "optim" / "optimizer.py").read_text()
    ep_src = (ROOT / "src" / "moe" / "ep_dispatch.py").read_text()

    required_snippets = [
        (train_src, "lm_loss + z_loss + aux_loss", "train loop must compose lm+z+aux"),
        (train_src, "update_expert_biases(", "train loop must run aux-loss-free bias update"),
        (train_src, "build_optimizer(", "train must use hybrid Muon+AdamW builder"),
        (train_src, "register_router_hooks(", "train must register MoE router hooks"),
        (train_src, "dispatch_overlap", "train must fail-close unsafe EP"),
        (train_src, "save_distributed_checkpoint", "train must use FSDP2/DCP checkpoints"),
        (model_src, "freeze_expert_load_biases(", "model load must freeze e_score_correction_bias"),
        (model_src, "_patch_mimo_gate_noaux_training(", "model load must patch noaux_tc gates"),
        (model_src, "fully_shard(", "model must wrap with FSDP2 fully_shard"),
        (model_src, "_patch_mimo_moe_fsdp_expert_alignment(", "model must align per-expert FSDP collectives"),
        (moe_src, "N_r / (K_r", "seq-aux must document DeepSeek f_i scaling"),
        (moe_src, "_normalized_routing_probs", "seq-aux must normalize sigmoid affinities"),
        (opt_src, "full_tensor()", "Muon must gather full matrix transiently for Newton-Schulz"),
        (opt_src, "_ensure_local_momentum", "Muon must persist local-shard momentum only"),
        (opt_src, "expert_bias", "optimizer must skip expert load biases"),
        (opt_src, 'weight_decay": 0.0', "AdamW specials/routers must use WD=0"),
        (model_src, "for expert in experts:", "FSDP2 must wrap routed experts individually"),
        (ep_src, "disabled", "ExpertDispatcher must remain disabled"),
    ]
    banned_snippets = [
        (train_src, "full_optim_state_dict", "train must not call FSDP1 full_optim_state_dict"),
        (train_src, "scatter_full_optim_state_dict", "train must not scatter FSDP1 optimizer state"),
        (model_src, "use_orig_params", "FSDP1 use_orig_params must not remain in model core"),
    ]
    for haystack, needle, msg in required_snippets:
        if needle not in haystack:
            failures.append(msg)
    for haystack, needle, msg in banned_snippets:
        if needle in haystack:
            failures.append(msg)

    # Ensure compute_seq_auxiliary_loss still uses N_r/(K*T) scaling in code.
    if "float(n_experts) / float(max(top_k * tokens, 1))" not in moe_src:
        failures.append("seq-aux f_i scaling expression missing from moe utils")

    # Parse train cli enough to ensure no silent bare except in the assembly file.
    tree = ast.parse(train_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            failures.append(f"bare except in src/train/cli.py at line {node.lineno}")
    return failures


def check_runtime_invariants() -> list[str]:
    """Runtime checks for MiMo-V2 MoE formula + optimizer classification."""
    import torch
    import torch.nn as nn

    from src.moe.utils import (
        compute_seq_auxiliary_loss,
        freeze_expert_load_biases,
        register_router_hooks,
        update_expert_biases,
    )
    from src.optim.optimizer import _classify_param, build_optimizer
    from src.dist.process_groups import ProcessGroups

    failures: list[str] = []

    class MiMoV2MoEGate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(8, 4))
            self.e_score_correction_bias = nn.Parameter(torch.zeros(8))
            self.scoring_func = "sigmoid"
            self.n_routed_experts = 8
            self.topk_method = "noaux_tc"

        def forward(self, hidden_states: torch.Tensor):
            bsz, seqlen, _ = hidden_states.shape
            flat = bsz * seqlen
            idx = torch.zeros(flat, 2, dtype=torch.long)
            for t in range(flat):
                idx[t, 0] = (2 * t) % 8
                idx[t, 1] = (2 * t + 1) % 8
            return idx.view(bsz, seqlen, 2), torch.ones(bsz, seqlen, 2)

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate = MiMoV2MoEGate()
            self.attn = nn.Linear(4, 4, bias=False)
            self.norm = nn.LayerNorm(4)
            self.embed_tokens = nn.Embedding(16, 4)

    model = Tiny()
    frozen = freeze_expert_load_biases(model)
    if frozen != 1 or model.gate.e_score_correction_bias.requires_grad:
        failures.append("freeze_expert_load_biases did not freeze MiMo gate bias")

    if _classify_param("gate.weight", model.gate.weight) != "router":
        failures.append("gate.weight must classify as router")
    if _classify_param("gate.e_score_correction_bias", model.gate.e_score_correction_bias) != "expert_bias":
        failures.append("e_score_correction_bias must classify as expert_bias")
    if _classify_param("attn.weight", model.attn.weight) != "muon":
        failures.append("attn weight must classify as muon")
    if _classify_param("norm.weight", model.norm.weight) != "no_decay":
        failures.append("norm.weight must classify as no_decay")

    manager = register_router_hooks(model)
    model.gate(torch.randn(1, 16, 4))
    aux = compute_seq_auxiliary_loss(manager, alpha=1e-3)
    if abs(float(aux.detach()) - 1e-3) > 5e-5:
        failures.append(f"balanced seq-aux should be ~1e-3 for MiMo sigmoid gate, got {float(aux)}")

    # Unbalanced counts so ±γ is nonzero (balanced round-robin leaves bias unchanged).
    manager.clear()
    manager._load_counts = {0: torch.tensor([90.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    before = model.gate.e_score_correction_bias.clone()
    update_expert_biases(model, manager, bias_update_speed=0.01)
    if torch.equal(before, model.gate.e_score_correction_bias):
        failures.append("bias ±γ update did not change frozen e_score_correction_bias")
    if not (model.gate.e_score_correction_bias[0] < before[0]):
        failures.append("overloaded expert 0 should get negative bias delta")

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
    cfg = {
        "optimizer": {
            "muon_lr": 0.005,
            "adamw_lr": 1.5e-5,
            "router_lr_scale": 0.1,
            "adamw_weight_decay": 0.1,
            "adamw_bf16_states": True,
        },
        "scheduler": {"lr_max": 1.5e-5},
    }
    opt = build_optimizer(model, cfg, pg)
    for group in opt.adamw.param_groups:
        if float(group.get("weight_decay", -1)) != 0.0:
            failures.append(f"AdamW group has unexpected WD={group.get('weight_decay')}")
    bias_ids = {id(model.gate.e_score_correction_bias)}
    for group in opt.param_groups:
        for param in group["params"]:
            if id(param) in bias_ids:
                failures.append("e_score_correction_bias must not be in any optimizer group")
    return failures


def run_smoke_units() -> None:
    import importlib.util

    smoke_path = ROOT / "scripts" / "verify" / "smoke_train.py"
    spec = importlib.util.spec_from_file_location("smoke_train", smoke_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {smoke_path}")
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)

    st.test_packing_mask()
    st.test_mimo_gate_hook()
    st.test_router_identity_and_optimizer_classification()
    st.test_freeze_expert_load_biases()
    st.test_seq_aux_deepseek_scale()
    st.test_muon_flat_shard_apply()
    st.test_global_gradient_clipping()
    st.test_packing_mode_fails_closed()
    st.test_compile_noop_when_disabled()
    st.test_expert_bias_param()
    st.test_moe_live_z_and_aux_loss()
    st.test_expert_bias_layer_pairing()
    st.test_ep_dispatcher_disabled()
    st.test_muon_rms_scaling()
    st.test_scheduler_updates_muon_lr()
    st.test_muon_weight_decay()
    st.test_sequence_pool_resume_continuity()
    st.test_low_precision_mode_resolve()
    st.test_selective_activation_checkpointing()
    st.test_lora_wraps_attention_not_expert_gate()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/b300/train.yaml")
    parser.add_argument("--king-dir", default=None, help="Optional king/pipeline model directory")
    parser.add_argument("--skip-units", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = _load_config(config_path)

    all_failures: list[str] = []
    log.info("Checking config design (%s)", config_path)
    all_failures.extend(check_config_design(cfg))
    log.info("Checking source contracts")
    all_failures.extend(check_source_contracts())
    log.info("Checking runtime MiMo-V2 invariants")
    all_failures.extend(check_runtime_invariants())

    if not args.skip_units:
        log.info("Running smoke unit suite")
        run_smoke_units()

    if all_failures:
        log.error("Design verification FAILED (%d issues):", len(all_failures))
        for item in all_failures:
            log.error("  - %s", item)
        return 1

    log.info("Design verification PASSED: config + source contracts + runtime invariants")
    king_parent = Path(args.king_dir) if args.king_dir else ROOT / "king"
    if not king_parent.is_absolute():
        king_parent = ROOT / king_parent
    king = king_parent / "config.json"
    if not king.exists():
        log.warning(
            "No king checkpoint at %s — full torchrun smoke/train not run. "
            "Run: python scripts/data/prepare_pipeline_model.py && bash scripts/train/run_pipeline_smoke.sh",
            king_parent,
        )
    else:
        log.info("King present at %s — run scripts/train/run_pipeline_smoke.sh for GPU smoke", king_parent)
    return 0


if __name__ == "__main__":
    sys.exit(main())
