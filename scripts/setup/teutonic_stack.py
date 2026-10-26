#!/usr/bin/env python3
"""Plan a Teutonic MoE / MiMo-V2 environment from GPU_check bootstrap policy.

Uses scripts/setup/bootstrap_ml.py inventory + stack selection, then applies
MoE-specific pins (FA4 wheel ↔ torch 2.11+cu130) and recommends which setup
script to run.

Examples:
  python scripts/setup/teutonic_stack.py --diagnose
  python scripts/setup/teutonic_stack.py --json
  python scripts/setup/teutonic_stack.py --dry-run
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = Path(__file__).resolve().parent / "bootstrap_ml.py"

# MoE FA4 Blackwell wheel is built against torch 2.11 + cu130 (see install_flash_attn.sh).
# bootstrap_ml prefers torch 2.13 on cu130 when the driver allows it; Teutonic CPT
# intentionally keeps 2.11 until a matching FA4 wheel for 2.13 is validated.
TEUTONIC_FA4_TORCH = {
    "index": "cu130",
    "torch": "2.11.0",
    "torchvision": None,
    "cuda": "13.0",
    "reason": "MoE FA4 wheel contract (scripts/setup/install_flash_attn.sh)",
}
TEUTONIC_4090_TORCH = {
    "index": "cu128",
    "torch": "2.11.0",
    "torchvision": None,
    "cuda": "12.8",
    "reason": "4090 dress rehearsal (eager OK; FA4 optional / may upgrade to cu130)",
}


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_ml", BOOTSTRAP)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BOOTSTRAP}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plan_teutonic(*, cuda: str = "auto", allow_busy: bool = False) -> dict:
    bm = _load_bootstrap()
    info = bm.inventory()
    caps = {g["cc"] for g in info["devices"]}
    names = [g["name"] for g in info["devices"]]

    # Hardware profile for MoE scripts
    if caps & {100, 103} or any("B300" in n or "B200" in n for n in names):
        profile = "b300"
        setup_script = "scripts/setup/setup_b300_venv.sh"
        extensions = ["fa4", "te"]
        torch_pin = dict(TEUTONIC_FA4_TORCH)
        attn = "flash_attention_4"
        low_precision = "mxfp8"
    elif caps <= {89} or any("4090" in n for n in names):
        profile = "4090"
        setup_script = "scripts/setup/setup_pipeline_venv.sh"
        extensions = []  # eager default; FA4 optional
        torch_pin = dict(TEUTONIC_4090_TORCH)
        attn = "eager"
        low_precision = "none"
    elif min(caps) >= 120:
        profile = "rtx_blackwell"
        setup_script = "scripts/setup/setup_pipeline_venv.sh"
        extensions = ["te"]  # FA4 datacenter-only in bootstrap policy
        torch_pin = dict(TEUTONIC_FA4_TORCH)
        attn = "flash_attention_4"  # MoE has sm_120 sink workaround
        low_precision = "none"  # stack refuses mxfp8 on sm_120
    else:
        # Fall back to bootstrap stack choice
        stack = bm.choose_stack(info, cuda)
        profile = "generic"
        setup_script = "scripts/setup/setup_b300_venv.sh" if stack["index"] == "cu130" else "scripts/setup/setup_pipeline_venv.sh"
        extensions = bm.extension_candidates(info, "auto")
        torch_pin = {
            "index": stack["index"],
            "torch": stack["torch"],
            "torchvision": stack.get("torchvision"),
            "cuda": stack["cuda"],
            "reason": "bootstrap_ml choose_stack",
        }
        attn = "flash_attention_4" if "fa4" in extensions else "eager"
        low_precision = "mxfp8" if "te" in extensions and min(caps) == 10 else "none"

    # Validate driver can run the chosen CUDA index via bootstrap policy when possible
    try:
        reviewed = bm.choose_stack(info, torch_pin["index"] if cuda == "auto" else cuda)
        driver_ok = True
        reviewed_stack = reviewed
    except bm.SetupError as exc:
        driver_ok = False
        reviewed_stack = {"error": str(exc)}

    processes = info["commands"].get("processes", {})
    busy = processes.get("returncode") == 0 and bool(processes.get("output", "").strip())

    return {
        "policy_date": getattr(bm, "POLICY_DATE", "unknown"),
        "profile": profile,
        "setup_script": setup_script,
        "gpu_count": len(info["devices"]),
        "devices": info["devices"],
        "driver": info.get("driver"),
        "driver_api": info.get("driver_api"),
        "torch_pin": torch_pin,
        "reviewed_bootstrap_stack": reviewed_stack,
        "driver_ok_for_index": driver_ok,
        "extensions": extensions,
        "recommended_env": {
            "TEUTONIC_ATTN": attn,
            "low_precision": low_precision,
            "GPUS_PER_NODE": len(info["devices"]),
        },
        "notes": bm.notes(info),
        "busy_gpus": busy,
        "allow_busy": allow_busy,
        "next_commands": [
            "python scripts/setup/optimize_ml_env.py --stage diagnose",
            "python scripts/setup/optimize_ml_env.py --stage plan",
            "python scripts/setup/optimize_ml_env.py --root ml-env-cpt --stage all --fast",
            f"bash {setup_script}   # legacy .venv path (fixed TE extras as of 2026-09-06)",
            "bash scripts/setup/check_env.sh",
        ],
        "te_install_note": (
            "Do not pip install bare transformer-engine (empty meta). "
            "Use transformer-engine[pytorch,core_cu13]==2.18.0 with pip nvcc/cccl, "
            "or scripts/setup/optimize_ml_env.py."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true", help="Full inventory JSON via bootstrap_ml")
    parser.add_argument("--dry-run", action="store_true", help="Alias for printing plan without install")
    parser.add_argument("--json", action="store_true", help="Print Teutonic plan as JSON")
    parser.add_argument("--cuda", default="auto", choices=["auto", "cu130", "cu128", "cu126"])
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--write", type=Path, help="Write plan JSON to this path")
    args = parser.parse_args(argv)

    bm = _load_bootstrap()
    if args.diagnose:
        info = bm.inventory()
        ns = argparse.Namespace(
            cuda=args.cuda,
            extensions="auto",
            profile="training",
            allow_source=False,
            skip_nccl=False,
            allow_busy=args.allow_busy,
        )
        plan = bm.installation_plan(info, ns)
        payload = {
            "inventory": info,
            "bootstrap_plan": plan,
            "teutonic": plan_teutonic(cuda=args.cuda, allow_busy=args.allow_busy),
        }
        text = json.dumps(payload, indent=2, default=str)
        print(text)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(text + "\n")
        return 0

    try:
        teutonic = plan_teutonic(cuda=args.cuda, allow_busy=args.allow_busy)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if teutonic["busy_gpus"] and not args.allow_busy:
        print("WARN: GPU compute processes detected; re-run with --allow-busy if intentional.", file=sys.stderr)

    if args.json or args.dry_run or args.write:
        text = json.dumps(teutonic, indent=2, default=str)
        print(text)
        if args.write:
            args.write.parent.mkdir(parents=True, exist_ok=True)
            args.write.write_text(text + "\n")
        return 0 if teutonic.get("driver_ok_for_index", True) else 2

    # Human summary
    print(f"profile={teutonic['profile']}  gpus={teutonic['gpu_count']}  driver={teutonic['driver']}")
    for d in teutonic["devices"]:
        print(f"  gpu[{d['index']}]={d['name']} cc={d['cc']} mem_gib={d['memory_bytes']/1024**3:.1f}")
    pin = teutonic["torch_pin"]
    print(f"torch_pin={pin['torch']}+{pin['index']}  ({pin['reason']})")
    print(f"extensions={teutonic['extensions']}")
    print(f"attn={teutonic['recommended_env']['TEUTONIC_ATTN']}  lp={teutonic['recommended_env']['low_precision']}")
    print(f"setup: bash {teutonic['setup_script']}")
    if not teutonic["driver_ok_for_index"]:
        print(f"WARN: bootstrap rejected index: {teutonic['reviewed_bootstrap_stack']}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
