#!/usr/bin/env python3
"""Hardware + stack verification for LoRA training on 8× RTX PRO 6000 (SM120)."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)
OUT_PATH = ROOT / "checkpoints" / "pro6000_verify.json"


def _record(name: str, ok: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {"name": name, "ok": ok, "detail": detail}
    if extra:
        row.update(extra)
    log.info("%s %s — %s", "PASS" if ok else "FAIL", name, detail)
    return row


def test_hardware() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return _record("hardware", False, "CUDA is not available")
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    mem = [round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 1) for i in range(n)]
    sm120 = all(cap[0] == 12 for cap in caps)
    pro6000 = all("PRO 6000" in name.upper() or "RTX PRO 6000" in name.upper() for name in names)
    ok = n == 8 and sm120 and pro6000
    return _record(
        "hardware",
        ok,
        f"{n} GPU(s), caps={caps}, ~{mem} GiB, names={names[0] if names else None}",
        extra={"gpu_count": n, "capabilities": [list(c) for c in caps], "memory_gib": mem, "gpu_name": names[0] if names else None},
    )


def test_unit_suite() -> dict[str, Any]:
    from scripts import smoke_train as st

    tests = [
        st.test_packing_mask,
        st.test_mimo_gate_hook,
        st.test_router_identity_and_optimizer_classification,
        st.test_global_gradient_clipping,
        st.test_packing_mode_fails_closed,
        st.test_compile_noop_when_disabled,
        st.test_expert_bias_param,
        st.test_moe_live_z_and_aux_loss,
        st.test_expert_bias_layer_pairing,
        st.test_ep_dispatcher_disabled,
        st.test_muon_rms_scaling,
        st.test_scheduler_updates_muon_lr,
        st.test_muon_weight_decay,
        st.test_sequence_pool_resume_continuity,
        st.test_low_precision_mode_resolve,
        st.test_selective_activation_checkpointing,
        st.test_lora_wraps_attention_not_expert_gate,
    ]
    failed: list[str] = []
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{fn.__name__}: {exc}")
    return _record(
        "unit_suite",
        not failed,
        "all 17 unit checks passed" if not failed else "; ".join(failed),
        extra={"passed": len(tests) - len(failed), "total": len(tests)},
    )


def test_config_moe_parallelism() -> dict[str, Any]:
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "pro6000.yaml").read_text())
    checks = {
        "packing_disabled": cfg.get("packing", {}).get("enabled") is False,
        "sparse_loco_disabled": cfg.get("sparse_loco", {}).get("enabled") is False,
        "dispatch_overlap_false": cfg.get("moe", {}).get("dispatch_overlap") is False,
        "delay_wgrad_false": cfg.get("moe", {}).get("delay_wgrad") is False,
        "node_limit_1": int(cfg.get("moe", {}).get("node_limit", 0)) == 1,
        "fsdp_true": cfg.get("parallelism", {}).get("fsdp") is True,
        "qk_norm_false": cfg.get("qk_norm", {}).get("enabled") is False,
        "low_precision_none": str(cfg.get("training", {}).get("low_precision")) == "none",
        "fp8_training_false": cfg.get("training", {}).get("fp8_training") is False,
        "z_loss_on": float(cfg.get("moe", {}).get("z_loss_coeff", 0)) > 0,
        "seq_aux_on": float(cfg.get("moe", {}).get("seq_aux_alpha", 0)) > 0,
        "wsd_scheduler": cfg.get("scheduler", {}).get("type") == "wsd",
        "compile_off": cfg.get("compilation", {}).get("enabled") is False,
        "single_node": int(cfg.get("hardware", {}).get("num_nodes", 0)) == 1,
        "eight_gpus": int(cfg.get("hardware", {}).get("gpus_per_node", 0)) == 8,
    }
    missing = [k for k, v in checks.items() if not v]
    return _record(
        "moe_parallelism_config",
        not missing,
        "pro6000.yaml fail-closed settings ok" if not missing else f"failed: {missing}",
        extra={"checks": checks},
    )


def test_scheduler_shape() -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from src.optim.optimizer import BF16AdamW, CombinedOptimizer, DistributedMuon
    from src.optim.scheduler import WSDScheduler, build_scheduler

    muon = DistributedMuon([nn.Parameter(torch.randn(8, 8))], lr=0.005)
    adamw = BF16AdamW(
        [
            {"params": [nn.Parameter(torch.randn(4))], "lr": 1.5e-5},
            {"params": [nn.Parameter(torch.randn(4))], "lr": 1.5e-6},
        ]
    )
    opt = CombinedOptimizer(muon=muon, adamw=adamw, scheduler_lr_max=1.5e-5)
    cfg = {
        "scheduler": {
            "type": "wsd",
            "warmup_steps": 200,
            "stable_steps": 4000,
            "decay_steps": 800,
            "lr_max": 1.5e-5,
            "lr_min": 1.5e-6,
        }
    }
    sched = build_scheduler(opt, cfg)
    assert isinstance(sched, WSDScheduler)
    lr0 = sched.get_lr(0)
    lr_warm = sched.get_lr(100)
    lr_stable = sched.get_lr(500)
    lr_decay = sched.get_lr(4900)
    ok = lr0 < lr_warm < lr_stable and abs(lr_stable - 1.5e-5) < 1e-10 and lr_decay < lr_stable
    return _record(
        "lr_scheduler",
        bool(ok),
        f"WSD LRs step0={lr0:.2e} warm100={lr_warm:.2e} stable={lr_stable:.2e} decay4900={lr_decay:.2e}",
        extra={"lr0": lr0, "warmup_100": lr_warm, "stable_500": lr_stable, "decay_4900": lr_decay},
    )


def test_data_sample() -> dict[str, Any]:
    import yaml

    from src.data import SEQUENCE_LENGTH, build_train_loader

    cfg = yaml.safe_load((ROOT / "configs" / "pro6000.yaml").read_text())
    cfg.setdefault("training", {})["micro_batch_size"] = 1
    cfg.setdefault("training", {})["gradient_accumulation_steps"] = 1
    cfg.setdefault("data", {})["num_workers"] = 0
    try:
        loader = build_train_loader(cfg, rank=0, world_size=1, num_workers_override=0)
        batch = next(iter(loader))
    except Exception as exc:  # noqa: BLE001
        return _record("data_sample", False, f"{type(exc).__name__}: {exc}")
    ids = batch.input_ids
    labels = batch.labels
    ok = (
        ids.ndim == 2
        and ids.shape[0] == 1
        and ids.shape[1] == SEQUENCE_LENGTH
        and labels.shape == ids.shape
        and int(ids.max()) >= 0
        and bool(batch.datasets)
    )
    return _record(
        "data_sample",
        bool(ok),
        f"batch input_ids={tuple(ids.shape)} labels={tuple(labels.shape)} dataset={batch.datasets[0]!r}",
        extra={
            "shape": list(ids.shape),
            "dataset": str(batch.datasets[0]),
            "vocab_max": int(ids.max()),
            "pad_or_eos_frac": float((labels < 0).float().mean()) if labels.numel() else 0.0,
        },
    )


def test_fa4() -> dict[str, Any]:
    import torch

    extra: dict[str, Any] = {}
    if not torch.cuda.is_available():
        return _record("flash_attention_4", False, "CUDA unavailable")
    try:
        from flash_attn.cute.interface import flash_attn_func
    except Exception as exc:  # noqa: BLE001
        return _record("flash_attention_4", False, f"import failed: {exc}")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    # MiMo-like: QK head_dim=192. Square V first, then asymmetric if the API allows.
    b, s, h, d = 1, 128, 4, 192
    q = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(b, s, h, d, device=device, dtype=torch.bfloat16, requires_grad=True)
    fwd_ok = False
    bwd_ok = False
    fwd_err = ""
    bwd_err = ""
    try:
        out = flash_attn_func(q, k, v, causal=True)
        if isinstance(out, tuple):
            out = out[0]
        torch.cuda.synchronize()
        fwd_ok = out.shape == q.shape
        extra["forward_shape"] = list(out.shape)
    except Exception as exc:  # noqa: BLE001
        fwd_err = f"{type(exc).__name__}: {exc}"
        extra["forward_error"] = fwd_err[:500]
    if fwd_ok:
        try:
            out.float().sum().backward()
            torch.cuda.synchronize()
            bwd_ok = q.grad is not None and float(q.grad.abs().sum()) > 0
            extra["q_grad_abs_sum"] = float(q.grad.abs().sum()) if q.grad is not None else 0.0
        except Exception as exc:  # noqa: BLE001
            bwd_err = f"{type(exc).__name__}: {exc}"
            extra["backward_error"] = bwd_err[:800]
    detail = (
        f"forward={'ok' if fwd_ok else 'FAIL'} backward={'ok' if bwd_ok else 'FAIL'}"
        + (f" fwd={fwd_err}" if fwd_err else "")
        + (f" bwd={bwd_err[:240]}" if bwd_err else "")
    )
    extra["forward"] = fwd_ok
    extra["backward"] = bwd_ok
    return _record("flash_attention_4", fwd_ok and bwd_ok, detail, extra=extra)


def test_fp8_te() -> dict[str, Any]:
    import torch

    extra: dict[str, Any] = {}
    if not torch.cuda.is_available():
        return _record("fp8_transformer_engine", False, "CUDA unavailable")
    cap = torch.cuda.get_device_capability(0)
    extra["capability"] = list(cap)
    extra["software_gate_allows_mxfp8"] = cap[0] == 10
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common import recipe as te_recipe
    except Exception as exc:  # noqa: BLE001
        return _record("fp8_transformer_engine", False, f"TE import failed: {exc}", extra=extra)

    torch.cuda.set_device(0)
    recipes: list[tuple[str, Any]] = []
    try:
        recipes.append(("MXFP8BlockScaling", te_recipe.MXFP8BlockScaling()))
    except Exception as exc:  # noqa: BLE001
        extra["mxfp8_recipe_error"] = str(exc)
    try:
        recipes.append(("DelayedScaling", te_recipe.DelayedScaling()))
    except Exception as exc:  # noqa: BLE001
        extra["delayed_recipe_error"] = str(exc)

    results: dict[str, str] = {}
    any_ok = False
    for name, rec in recipes:
        try:
            lin = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
            x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            ctx = te.autocast(enabled=True, recipe=rec) if hasattr(te, "autocast") else te.fp8_autocast(enabled=True, fp8_recipe=rec)
            with ctx:
                y = lin(x)
            y.float().sum().backward()
            torch.cuda.synchronize()
            results[name] = "ok"
            any_ok = True
        except Exception as exc:  # noqa: BLE001
            results[name] = f"{type(exc).__name__}: {str(exc)[:300]}"
    extra["recipes"] = results
    # Designed behavior: training stack must refuse MXFP8 on sm_120.
    from src.model import _capability_supports_low_precision

    extra["stack_refuses_mxfp8_on_sm120"] = not _capability_supports_low_precision("mxfp8", cap)
    detail = f"cap={cap} recipes={results} stack_gate_sm120={extra['stack_refuses_mxfp8_on_sm120']}"
    return _record("fp8_transformer_engine", any_ok, detail, extra=extra)


def test_lora_optimizer_groups_tiny() -> dict[str, Any]:
    import torch
    import torch.nn as nn

    from src.model.lora import LoRALinear, apply_lora
    from src.optim.optimizer import _classify_param
    from src.dist.process_groups import ProcessGroups
    from src.optim.optimizer import build_optimizer

    class Gate(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, 8))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    Gate.__name__ = "MiMoV2MoEGate"

    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv_proj = nn.Linear(8, 16, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attn()
            self.gate = Gate()

    model = Model()
    apply_lora(model, {"lora": {"enabled": True, "rank": 2, "alpha": 4, "target_modules": ["qkv_proj", "o_proj"]}})
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
    opt = build_optimizer(model, {"optimizer": {"muon_lr": 0.005, "adamw_lr": 1.5e-5}}, pg)
    muon_n = sum(p.numel() for g in opt.muon.param_groups for p in g["params"])
    router_n = sum(p.numel() for p in opt.adamw.param_groups[1]["params"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    ok = muon_n > 0 and router_n > 0 and frozen > trainable
    return _record(
        "lora_optimizer_groups",
        bool(ok),
        f"trainable={trainable} frozen={frozen} muon={muon_n} router={router_n} wraps_qkv={isinstance(model.self_attn.qkv_proj, LoRALinear)}",
        extra={"trainable": trainable, "frozen": frozen, "muon": muon_n, "router": router_n},
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    results: list[dict[str, Any]] = []
    for fn in (
        test_hardware,
        test_unit_suite,
        test_config_moe_parallelism,
        test_scheduler_shape,
        test_lora_optimizer_groups_tiny,
        test_data_sample,
        test_fa4,
        test_fp8_te,
    ):
        try:
            results.append(fn())
        except Exception as exc:  # noqa: BLE001
            results.append(_record(fn.__name__, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-400:]}"))
    payload = {"results": results, "passed": sum(1 for r in results if r["ok"]), "total": len(results)}
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s (%d/%d passed)", OUT_PATH, payload["passed"], payload["total"])
    print(json.dumps(payload, indent=2))
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
