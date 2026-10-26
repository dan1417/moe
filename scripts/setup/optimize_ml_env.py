#!/usr/bin/env python3
"""Workload-aware ML env optimizer for Teutonic / MiMo-V2 CPT on NVIDIA GPUs.

Policy research date: 2026-09-06.

One-command flow (fresh root only; never overwrites .venv or existing roots):

  python scripts/setup/optimize_ml_env.py --stage diagnose
  python scripts/setup/optimize_ml_env.py --stage plan
  python scripts/setup/optimize_ml_env.py --root ml-env-cpt --stage all --fast

Stages: diagnose → plan → install → validate → build-mini → bench → report

This is NOT a claim of universal optimality. Results apply only to the recorded
workload, hardware, candidates, and budgets in the generated report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request

POLICY_DATE = "2026-09-06"
REPO_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = Path(__file__).resolve().parent / "bootstrap_ml.py"

# Interview-locked defaults for this MoE CPT path (overridable via flags).
DEFAULT_WORKLOAD = {
    "name": "cpt-mimo",
    "goal": "continued_pretraining",
    "model_path": "mini_then_110B",
    "mini_dir": "pipeline_model",
    "king_repo": "dendriteholdings/teutonic-II-110B-genesis",
    "priority": "training_throughput",
    "seq_len_target": 2048,
    "precision_policy": "prefer_faster_if_stable",  # MXFP8 allowed
    "hardware_now": "1x_B300",
    "hardware_next": "8x_B300",
    "data_plan": "synthetic_then_short_real",
    "budgets": {
        "wall_clock_hours": 1.0,
        "disk_gib": 40,
        "gpu_mem_fraction": 0.80,
    },
}

# Candidates: same install stack for A/B (runtime config differs). C is separate.
CANDIDATES = {
    "A_fa4_bf16": {
        "title": "Baseline FA4 + BF16",
        "torch": "2.11.0",
        "cuda_index": "cu130",
        "attn": "flash_attention_4",
        "low_precision": "none",
        "config": "configs/b300/1gpu_speed_fa4.yaml",
        "why": "Matches validated FA4 wheel contract; correctness baseline for MiMo sinks.",
        "status_if_unrun": "compatible_untested",
    },
    "B_fa4_mxfp8": {
        "title": "Speed candidate FA4 + MXFP8",
        "torch": "2.11.0",
        "cuda_index": "cu130",
        "attn": "flash_attention_4",
        "low_precision": "mxfp8",
        "config": "configs/b300/1gpu_speed_mxfp8.yaml",
        "why": "B300 Tensor Cores + TE MXFP8 for throughput; needs working TE pytorch+core_cu13.",
        "status_if_unrun": "compatible_untested",
    },
    "C_eager_bf16": {
        "title": "Fallback eager + BF16",
        "torch": "2.11.0",
        "cuda_index": "cu130",
        "attn": "eager",
        "low_precision": "none",
        "config": "configs/b300/1gpu_speed_eager.yaml",
        "why": "Slow-path reference if FA4/TE misbehave; not expected to win throughput.",
        "status_if_unrun": "compatible_untested",
    },
}

FA2_WHEEL_URL = (
    "https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels/releases/download/"
    "FA.2.8.3-custom-linux-wheels-x86_64/"
    "flash_attn-2.8.3%2Bcu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc132-cp312-cp312-linux_x86_64.whl"
)
FA2_WHEEL_NAME = (
    "flash_attn-2.8.3+cu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc132-"
    "cp312-cp312-linux_x86_64.whl"
)

SOURCES = {
    "pytorch_wheels": "https://pytorch.org/get-started/previous-versions/",
    "te_install": "https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.18/user-guide/installation.html",
    "te_github": "https://github.com/NVIDIA/TransformerEngine",
    "fa4": "https://github.com/Dao-AILab/flash-attention",
    "cuda_compat": "https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html",
    "blackwell": "https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html",
}


class OptError(RuntimeError):
    pass


def load_bootstrap():
    spec = importlib.util.spec_from_file_location("bootstrap_ml", BOOTSTRAP)
    if spec is None or spec.loader is None:
        raise OptError(f"Cannot load {BOOTSTRAP}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def json_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def run(argv, *, timeout=None, env=None, check=True, log=None) -> dict:
    argv = [str(x) for x in argv]
    print("  $", shlex.join(argv), flush=True)
    try:
        proc = subprocess.run(
            argv, text=True, capture_output=True, timeout=timeout, env=env, check=False
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + (exc.stderr or "")
        if log:
            Path(log).write_text(out + "\nTIMEOUT\n")
        raise OptError(f"Timed out: {shlex.join(argv[:6])}") from exc
    out = (proc.stdout or "") + (proc.stderr or "")
    if log:
        Path(log).write_text(out)
    if check and proc.returncode:
        raise OptError(f"Command failed ({proc.returncode}). Log: {log}\n{out[-2500:]}")
    return {"returncode": proc.returncode, "output": out}


def version_tuple(value: str):
    import re

    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if not m:
        raise OptError(f"Cannot parse version: {value!r}")
    return tuple(int(v or 0) for v in m.groups())


def explain_cuda_layers(info: dict, plan: dict) -> dict:
    """Separate driver API vs PyTorch CUDA runtime vs local toolkit needs."""
    driver = info.get("driver")
    driver_api = info.get("driver_api")
    api_major = driver_api // 1000 if isinstance(driver_api, int) else None
    api_minor = (driver_api % 1000) // 10 if isinstance(driver_api, int) else None
    return {
        "driver_supports": {
            "driver_version": driver,
            "driver_api": driver_api,
            "driver_api_human": f"{api_major}.{api_minor}" if api_major is not None else None,
            "note": "Driver API is the ceiling for CUDA user-mode libraries, not a requirement that toolkit==driver.",
        },
        "pytorch_cuda_runtime": {
            "selected_index": plan["stack"]["cuda_index"],
            "torch": plan["stack"]["torch"],
            "runtime_cuda": plan["stack"]["cuda"],
            "note": "PyTorch cu130 wheels ship their own CUDA 13.0 runtime libs inside the venv.",
        },
        "local_toolkit_nvcc": {
            "needed_for": [
                "building transformer-engine-torch from sdist (no pip ABI-matched wheel for torch 2.11+cu130)",
                "optional FA2 custom builds",
            ],
            "not_needed_for": [
                "running PyTorch",
                "flash-attn-4 CuTeDSL JIT (uses cutlass-dsl; toolkit optional)",
            ],
            "strategy": "Install matching pip packages nvidia-cuda-nvcc + nvidia-cuda-cccl into the venv; do not apt-install system CUDA.",
        },
        "extensions": {
            "fa4": "flash-attn-4[cu13] pure/python JIT; MiMo requires FA4 for sinks + asymmetric QK/V",
            "fa2_wheel": "Optional Blackwell FA2 wheel pinned to torch 2.11+cu130 (dependency/legacy path)",
            "te": "Need transformer-engine-cu13 wheel + transformer-engine-torch built against THIS torch; empty meta package alone fails",
        },
        "sources": SOURCES,
    }


def build_plan(info: dict, args) -> dict:
    devices = info["devices"]
    caps = {g["cc"] for g in devices}
    names = [g["name"] for g in devices]
    if not devices:
        raise OptError("No CUDA-visible GPU.")
    if 103 not in caps and not any("B300" in n or "B200" in n for n in names):
        # Still allow plan generation, but mark caution.
        profile = "non_b300"
    else:
        profile = "b300"

    if version_tuple(info["driver"]) < version_tuple("580.65.06"):
        raise OptError(
            f"Driver {info['driver']} is below the conservative CUDA 13.0 floor 580.65.06. "
            "Ask an admin for a driver upgrade; this tool will not change drivers."
        )

    stack = {
        "torch": "2.11.0",
        "cuda_index": "cu130",
        "cuda": "13.0",
        "reason": "Teutonic FA4 wheel contract + verified TE source-build against torch 2.11+cu130 on 2026-09-06",
        "rejected_alternatives": [
            {
                "torch": "2.13.0+cu130 / 2.14.0+cu130",
                "why_deferred": "Newer is available, but MoE FA2 custom wheel and verified TE recipe are proven on 2.11; measure before switching.",
            },
            {
                "te": "NGC transformer_engine_torch cu13torch26.xx wheels",
                "why_rejected": "ABI mismatch with pip torch 2.11 (undefined c10::ValueError symbol) observed 2026-09-06.",
            },
            {
                "te": "pip install transformer-engine (no extras)",
                "why_rejected": "Installs empty meta package; MXFP8 probe fails.",
            },
        ],
    }

    plan = {
        "policy_date": POLICY_DATE,
        "workload": DEFAULT_WORKLOAD,
        "profile": profile,
        "gpu_count": len(devices),
        "devices": devices,
        "driver": info.get("driver"),
        "driver_api": info.get("driver_api"),
        "stack": stack,
        "candidates": CANDIDATES,
        "install_packages": {
            "torch": "torch==2.11.0 from https://download.pytorch.org/whl/cu130",
            "fa2": FA2_WHEEL_URL,
            "fa4": "flash-attn-4[cu13]==4.0.0b28 (pre)",
            "te": "transformer-engine[pytorch,core_cu13]==2.18.0 --no-build-isolation after pip nvcc/cccl",
            "ml": "requirements.txt minus flash-attn/transformer-engine lines",
        },
        "fast_mode": bool(args.fast),
        "bench_seq_lens": [256, 2048] if not args.fast else [256],
        "bench_steps": 3 if args.fast else 5,
        "notes": [
            "1×B300 now: no multi-GPU NCCL ranking; re-run validate on 8× before claiming multi-GPU readiness.",
            "Mini (~8.6B depth-shrunk) is a dress rehearsal; full 110B needs separate memory/throughput measurement.",
            "Installing FA4/TE does not prove the training loop uses them — bench configs must set attn/low_precision.",
        ],
    }
    plan["cuda_layers"] = explain_cuda_layers(info, plan)
    return plan


def stage_diagnose(args) -> int:
    bm = load_bootstrap()
    print("==> diagnose (read-only)", flush=True)
    info = bm.inventory()
    # Strip bulky existing package dumps from console; keep in file.
    slim = {k: v for k, v in info.items() if k not in ("existing_packages", "commands")}
    slim["command_keys"] = sorted((info.get("commands") or {}).keys())
    plan = build_plan(info, args)
    payload = {"inventory_summary": slim, "plan": plan, "inventory": info}
    text = json.dumps({"inventory_summary": slim, "plan": plan}, indent=2, default=str)
    print(text)
    out = Path(args.write) if args.write else REPO_ROOT / "logs" / "optimize_diagnose.json"
    json_write(out, payload)
    print(f"Wrote {out}", flush=True)
    return 0


def stage_plan(args) -> int:
    bm = load_bootstrap()
    info = bm.inventory()
    plan = build_plan(info, args)
    print(json.dumps(plan, indent=2, default=str))
    out = Path(args.write) if args.write else REPO_ROOT / "logs" / "optimize_plan.json"
    json_write(out, plan)
    print("\nRecommended install candidate set: A_fa4_bf16 (baseline) + B_fa4_mxfp8 (speed) + C_eager_bf16 (fallback).", flush=True)
    print("Same venv for A/B/C; configs differ. No install performed in --stage plan.", flush=True)
    print(f"Wrote {out}", flush=True)
    return 0


def ensure_fresh_root(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise OptError(f"{root} already exists. Choose a NEW --root. Nothing will be overwritten.")
    parent = root.parent
    if not parent.is_dir():
        raise OptError("Output parent must already exist.")
    free_gib = shutil.disk_usage(parent).free / 1024**3
    if free_gib < 25:
        raise OptError(f"Need ≥25 GiB free for a fast CPT env; found {free_gib:.1f} GiB.")


def venv_python(root: Path) -> Path:
    return root / "training" / "bin" / "python"


def pip_env(root: Path, extra: dict | None = None) -> dict:
    env = os.environ.copy()
    for key in ("PYTHONPATH", "PYTHONHOME", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        if env.get(key):
            raise OptError(f"{key} is set and could escape the venv. Unset it and rerun.")
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["PIP_NO_INPUT"] = "1"
    env["PIP_CACHE_DIR"] = str(root / "cache" / "pip")
    env["TORCH_EXTENSIONS_DIR"] = str(root / "cache" / "torch_extensions")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache" / "inductor")
    env["TRITON_CACHE_DIR"] = str(root / "cache" / "triton")
    env["HF_HOME"] = str(root / "cache" / "huggingface")
    if extra:
        env.update(extra)
    return env


def site_nvidia_roots(py: Path) -> dict:
    code = r"""
import json
from pathlib import Path
import site
roots = []
for sp in site.getsitepackages():
    n = Path(sp) / "nvidia"
    if n.is_dir():
        roots.append(str(n))
print(json.dumps(roots))
"""
    out = subprocess.check_output([str(py), "-c", code], text=True)
    nvidia_roots = [Path(p) for p in json.loads(out)]
    cu13 = None
    cudnn = None
    nccl = None
    for r in nvidia_roots:
        if (r / "cu13" / "bin" / "nvcc").is_file():
            cu13 = r / "cu13"
        if (r / "cudnn" / "include" / "cudnn.h").is_file():
            cudnn = r / "cudnn"
        if (r / "nccl" / "include" / "nccl.h").is_file():
            nccl = r / "nccl"
    return {"cu13": cu13, "cudnn": cudnn, "nccl": nccl}


def te_build_env(py: Path, base: dict) -> dict:
    roots = site_nvidia_roots(py)
    if roots["cu13"] is None:
        raise OptError("nvidia/cu13/nvcc missing after nvidia-cuda-nvcc install.")
    if roots["cudnn"] is None or roots["nccl"] is None:
        raise OptError("cuDNN/NCCL headers missing; torch cu130 packages incomplete.")
    env = base.copy()
    cuda_home = str(roots["cu13"])
    includes = f"{roots['cu13']}/include:{roots['cudnn']}/include:{roots['nccl']}/include"
    libs = f"{roots['cudnn']}/lib:{roots['nccl']}/lib:{roots['cu13']}/lib"
    env["CUDA_HOME"] = cuda_home
    env["CUDA_PATH"] = cuda_home
    env["CUDACXX"] = str(roots["cu13"] / "bin" / "nvcc")
    env["CUDNN_PATH"] = str(roots["cudnn"])
    env["CUDNN_HOME"] = str(roots["cudnn"])
    env["PATH"] = f"{py.parent}:{roots['cu13']}/bin:" + env.get("PATH", "")
    env["CPLUS_INCLUDE_PATH"] = includes
    env["C_INCLUDE_PATH"] = includes
    env["LIBRARY_PATH"] = libs
    env["LD_LIBRARY_PATH"] = libs + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    env["NVTE_FRAMEWORK"] = "pytorch"
    env["MAX_JOBS"] = env.get("MAX_JOBS", "4")
    env["CXXFLAGS"] = f"-I{roots['cu13']}/include -I{roots['cudnn']}/include -I{roots['nccl']}/include"
    return env


def runtime_lib_env(py: Path, base: dict) -> dict:
    roots = site_nvidia_roots(py)
    env = base.copy()
    libs = []
    for key in ("nccl", "cudnn", "cu13"):
        if roots[key] is not None:
            lib = roots[key] / "lib"
            if lib.is_dir():
                libs.append(str(lib))
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs) + ((":" + env["LD_LIBRARY_PATH"]) if env.get("LD_LIBRARY_PATH") else "")
    return env


def stage_install(args) -> int:
    root = Path(args.root).expanduser().absolute()
    ensure_fresh_root(root)
    bm = load_bootstrap()
    print("==> install (fresh env)", flush=True)
    info = bm.inventory()
    processes = info["commands"].get("processes", {})
    busy = processes.get("returncode") == 0 and bool(processes.get("output", "").strip())
    if busy and not args.allow_busy:
        raise OptError("GPU compute processes detected. Rerun with --allow-busy if intentional.")
    plan = build_plan(info, args)
    root.mkdir(mode=0o700)
    (root / "reports").mkdir()
    (root / "cache").mkdir()
    json_write(root / "inventory.json", info)
    json_write(root / "plan.json", plan)
    manifest = {
        "status": "INSTALLING",
        "policy_date": POLICY_DATE,
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": DEFAULT_WORKLOAD,
        "stages": {},
        "extension_status": {},
    }
    json_write(root / "manifest.json", manifest)

    py_sys = sys.executable
    run([py_sys, "-m", "venv", root / "training"], timeout=120, log=root / "reports" / "001-venv.log")
    py = venv_python(root)
    env = pip_env(root)

    run(
        [py, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "packaging", "ninja", "numpy"],
        timeout=600,
        env=env,
        log=root / "reports" / "002-pip-tools.log",
    )
    run(
        [py, "-m", "pip", "install", "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu130"],
        timeout=1800,
        env=env,
        log=root / "reports" / "003-torch.log",
    )

    # Toolkit pieces for TE torch extension build (venv-local; no apt/sudo).
    run(
        [py, "-m", "pip", "install", "nvidia-cuda-nvcc==13.0.88", "nvidia-cuda-cccl==13.0.85"],
        timeout=900,
        env=env,
        log=root / "reports" / "004-nvcc-cccl.log",
    )

    req = REPO_ROOT / "requirements.txt"
    filtered = root / "reports" / "requirements-filtered.txt"
    lines = []
    for line in req.read_text().splitlines():
        if line.strip().startswith(("flash-attn", "transformer-engine")):
            continue
        lines.append(line)
    filtered.write_text("\n".join(lines) + "\n")
    run(
        [py, "-m", "pip", "install", "-r", str(filtered)],
        timeout=1800,
        env=env,
        log=root / "reports" / "005-requirements.log",
    )

    # FA2 custom wheel (MiMo legacy/compat) + FA4.
    wheel_path = root / "cache" / FA2_WHEEL_NAME
    if not wheel_path.is_file():
        print("  downloading FA2 Blackwell wheel...", flush=True)
        urllib.request.urlretrieve(FA2_WHEEL_URL, wheel_path)
    run(
        [py, "-m", "pip", "install", str(wheel_path), "--no-deps"],
        timeout=600,
        env=env,
        log=root / "reports" / "006-fa2.log",
    )
    run(
        [py, "-m", "pip", "install", "--pre", "flash-attn-4[cu13]==4.0.0b28"],
        timeout=900,
        env=env,
        log=root / "reports" / "007-fa4.log",
    )
    manifest["extension_status"]["fa4"] = "INSTALLED"
    manifest["extension_status"]["fa2"] = "INSTALLED_WHEEL"

    # Transformer Engine (verified recipe 2026-09-06).
    te_env = te_build_env(py, env)
    try:
        run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "transformer-engine[pytorch,core_cu13]==2.18.0",
            ],
            timeout=2400,
            env=te_env,
            log=root / "reports" / "008-te.log",
        )
        manifest["extension_status"]["te"] = "INSTALLED_SOURCE_MATCHED_TORCH"
    except OptError as exc:
        manifest["extension_status"]["te"] = f"FAILED: {exc}"
        manifest["stages"]["te"] = "FAILED"
        json_write(root / "manifest.json", manifest)
        print("TE install failed; BF16/FA4 candidates may still work. See reports/008-te.log", flush=True)
        # Continue — A/C may still be useful.

    freeze = run([py, "-m", "pip", "freeze", "--all"], timeout=120, env=env, log=root / "reports" / "009-freeze.log")
    (root / "training.lock.txt").write_text(freeze["output"])
    run([py, "-m", "pip", "check"], timeout=120, env=env, check=False, log=root / "reports" / "010-pip-check.log")

    launcher = root / "run-training.sh"
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={shlex.quote(str(root))}
source "$ROOT/training/bin/activate"
export PYTHONNOUSERSITE=1
export HF_HOME="$ROOT/cache/huggingface"
export TORCH_EXTENSIONS_DIR="$ROOT/cache/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/inductor"
export TRITON_CACHE_DIR="$ROOT/cache/triton"
# Prefer venv NVIDIA libs for TE/NCCL.
SITE="$ROOT/training/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/nccl/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cu13/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
if [[ -z "${{CUDA_HOME:-}}" && -x "$SITE/nvidia/cu13/bin/nvcc" ]]; then
  export CUDA_HOME="$SITE/nvidia/cu13"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
export TEUTONIC_ATTN="${{TEUTONIC_ATTN:-flash_attention_4}}"
if [[ "$#" -eq 0 ]]; then exec python; fi
exec "$@"
"""
    )
    launcher.chmod(0o700)

    manifest["status"] = "INSTALLED"
    manifest["finished_install_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["stages"]["install"] = "PASSED"
    json_write(root / "manifest.json", manifest)
    print(f"Install complete: {root}", flush=True)
    print(f"Activate via: bash {launcher}  OR  source {root}/training/bin/activate", flush=True)
    return 0


def stage_validate(args) -> int:
    root = Path(args.root).expanduser().absolute()
    py = venv_python(root)
    if not py.is_file():
        raise OptError(f"Missing venv python at {py}. Run --stage install first.")
    print("==> validate", flush=True)
    env = runtime_lib_env(py, pip_env(root))
    results = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "checks": {}}

    # Core GPU matmul
    core = run(
        [
            py,
            "-c",
            "import torch; assert torch.cuda.is_available(); "
            "x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16,requires_grad=True); "
            "y=(x@x).square().mean(); y.backward(); torch.cuda.synchronize(); "
            "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))",
        ],
        timeout=120,
        env=env,
        log=root / "reports" / "020-core.log",
    )
    results["checks"]["core"] = {"status": "passed", "detail": core["output"].strip()}

    # FA4
    try:
        fa = run(
            [
                py,
                "-c",
                "import torch; from flash_attn.cute import flash_attn_func; "
                "q=torch.randn(1,8,128,64,device='cuda',dtype=torch.bfloat16); "
                "flash_attn_func(q,q,q); torch.cuda.synchronize(); print('FA4_OK')",
            ],
            timeout=300,
            env=env,
            log=root / "reports" / "021-fa4.log",
        )
        results["checks"]["fa4"] = {"status": "passed", "detail": fa["output"].strip()}
    except OptError as exc:
        results["checks"]["fa4"] = {"status": "failed", "detail": str(exc)[:500]}

    # TE / MXFP8
    te_probe = root / "reports" / "022-te-probe.py"
    te_probe.write_text(
        "import torch\n"
        "import transformer_engine.pytorch as te\n"
        "from transformer_engine.common import recipe\n"
        "layer = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()\n"
        "x = torch.randn(32, 256, device='cuda', dtype=torch.bfloat16, requires_grad=True)\n"
        "r = recipe.MXFP8BlockScaling()\n"
        "ctx = getattr(te, 'autocast', None) or te.fp8_autocast\n"
        "with ctx(enabled=True, recipe=r):\n"
        "    y = layer(x)\n"
        "    loss = y.float().square().mean()\n"
        "loss.backward()\n"
        "torch.cuda.synchronize()\n"
        "print('TE_MXFP8_OK')\n"
    )
    try:
        te = run(
            [py, str(te_probe)],
            timeout=300,
            env=env,
            log=root / "reports" / "022-te.log",
        )
        results["checks"]["te_mxfp8"] = {"status": "passed", "detail": te["output"].strip()}
    except OptError as exc:
        results["checks"]["te_mxfp8"] = {"status": "failed", "detail": str(exc)[:500]}

    # Multi-GPU NCCL only if >1 GPU
    bm = load_bootstrap()
    info = json.loads((root / "inventory.json").read_text()) if (root / "inventory.json").is_file() else bm.inventory()
    if len(info["devices"]) > 1 and not args.skip_nccl:
        results["checks"]["nccl"] = {"status": "skipped", "detail": "Use bootstrap_ml NCCL worker on 8× node; not run here."}
    else:
        results["checks"]["nccl"] = {
            "status": "skipped",
            "detail": "Single visible GPU — multi-GPU NCCL unverified (expected on this 1× allocation).",
        }

    results["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    json_write(root / "reports" / "validate.json", results)
    print(json.dumps(results, indent=2))
    failed = [k for k, v in results["checks"].items() if v["status"] == "failed"]
    if failed and not args.allow_validate_fail:
        raise OptError(f"Validation failed: {failed}")
    return 0


def stage_build_mini(args) -> int:
    root = Path(args.root).expanduser().absolute()
    py = venv_python(root)
    if not py.is_file():
        raise OptError("Install stage required before build-mini.")
    print("==> build local mini model (~8.6B depth-shrunk)", flush=True)
    env = runtime_lib_env(py, pip_env(root))
    env["PYTHONPATH"] = str(REPO_ROOT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else "")
    out_dir = REPO_ROOT / DEFAULT_WORKLOAD["mini_dir"]
    run(
        [
            py,
            str(REPO_ROOT / "scripts/data/prepare_pipeline_model.py"),
            "--output",
            str(out_dir),
            "--num-layers",
            "4",
        ],
        timeout=1800,
        env=env,
        log=root / "reports" / "030-build-mini.log",
    )
    meta = {"mini_dir": str(out_dir), "num_layers": 4, "built_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    json_write(root / "reports" / "mini_model.json", meta)
    print(f"Mini model at {out_dir}", flush=True)
    return 0


def stage_bench(args) -> int:
    root = Path(args.root).expanduser().absolute()
    py = venv_python(root)
    if not py.is_file():
        raise OptError("Install stage required before bench.")
    print("==> bench (synthetic mini CPT; ask before long/real runs)", flush=True)
    launcher = root / "run-training.sh"
    seqs = [256] if args.fast else [256, 2048]
    steps = 3 if args.fast else 5
    results = {
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": DEFAULT_WORKLOAD,
        "candidates": {},
        "disclaimer": "Synthetic mini steps only; not full 110B or convergence certification.",
    }
    for key, cand in CANDIDATES.items():
        # Skip MXFP8 candidate if TE validate failed
        if cand["low_precision"] == "mxfp8":
            val = root / "reports" / "validate.json"
            if val.is_file():
                v = json.loads(val.read_text())
                if v.get("checks", {}).get("te_mxfp8", {}).get("status") != "passed":
                    results["candidates"][key] = {
                        "status": "skipped",
                        "reason": "TE/MXFP8 validate did not pass",
                    }
                    continue
        entry = {"title": cand["title"], "config": cand["config"], "seq_results": []}
        for seq in seqs:
            log = root / "reports" / f"bench-{key}-seq{seq}.log"
            # Prefer existing launcher configs; override smoke_seq_len / steps via env when supported.
            cmd = [
                "bash",
                str(REPO_ROOT / "scripts/train/run_b300_1gpu.sh"),
                "--config",
                cand["config"],
            ]
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = str(root / "training")
            env["PATH"] = f"{root / 'training' / 'bin'}:" + env.get("PATH", "")
            env["PYTHON"] = str(py)
            env["KING_DIR"] = str(REPO_ROOT / DEFAULT_WORKLOAD["mini_dir"])
            env["TEUTONIC_ATTN"] = cand["attn"]
            # Runtime YAML overrides are handled inside run_b300_1gpu.sh; keep bench short by
            # copying a temp config when seq/steps differ from yaml defaults.
            tmp_cfg = root / "reports" / f"runtime-{key}-seq{seq}.yaml"
            cfg_text = (REPO_ROOT / cand["config"]).read_text()
            # Minimal surgical overrides
            import re

            cfg_text = re.sub(r"smoke_seq_len:\s*\d+", f"smoke_seq_len: {seq}", cfg_text)
            cfg_text = re.sub(r"total_steps:\s*\d+", f"total_steps: {steps}", cfg_text)
            cfg_text = re.sub(
                r"king_dir:\s*.*",
                f"king_dir: {REPO_ROOT / DEFAULT_WORKLOAD['mini_dir']}",
                cfg_text,
            )
            cfg_text = re.sub(
                r"output_dir:\s*.*",
                f"output_dir: {root / 'bench_ckpts' / key / f'seq{seq}'}",
                cfg_text,
            )
            tmp_cfg.write_text(cfg_text)
            cmd = ["bash", str(REPO_ROOT / "scripts/train/run_b300_1gpu.sh"), "--config", str(tmp_cfg)]
            t0 = time.perf_counter()
            try:
                out = run(cmd, timeout=3600 if args.fast else 7200, env=env, log=log, check=True)
                elapsed = time.perf_counter() - t0
                # Parse tokens/s if present
                toks = None
                for line in out["output"].splitlines():
                    if "tokens/s" in line or "tokens/sec" in line:
                        toks = line.strip()
                entry["seq_results"].append(
                    {
                        "seq_len": seq,
                        "status": "passed_smoke",
                        "wall_seconds": round(elapsed, 2),
                        "throughput_line": toks,
                    }
                )
            except OptError as exc:
                entry["seq_results"].append(
                    {
                        "seq_len": seq,
                        "status": "failed",
                        "error": str(exc)[:1000],
                    }
                )
                break
        # Summarize candidate status
        if any(r.get("status") == "passed_smoke" for r in entry["seq_results"]):
            entry["status"] = "passed_smoke"
        elif all(r.get("status") == "skipped" for r in entry["seq_results"]):
            entry["status"] = "skipped"
        else:
            entry["status"] = "failed"
        results["candidates"][key] = entry

    results["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    json_write(root / "reports" / "bench.json", results)
    print(json.dumps(results, indent=2, default=str))
    return 0


def stage_report(args) -> int:
    root = Path(args.root).expanduser().absolute()
    print("==> report", flush=True)
    plan = {}
    if (root / "plan.json").is_file():
        plan = json.loads((root / "plan.json").read_text())
    validate = {}
    if (root / "reports" / "validate.json").is_file():
        validate = json.loads((root / "reports" / "validate.json").read_text())
    bench = {}
    if (root / "reports" / "bench.json").is_file():
        bench = json.loads((root / "reports" / "bench.json").read_text())

    winner = None
    winner_reason = "No successful bench yet — treat package installs as compatible/untested or smoke-only."
    best_score = None  # higher tokens/s is better; fall back to lower wall time

    def parse_toks(line):
        if not line:
            return None
        import re

        m = re.search(r"tokens/s=([0-9.]+)", line)
        return float(m.group(1)) if m else None

    for key, entry in (bench.get("candidates") or {}).items():
        for sr in entry.get("seq_results") or []:
            if sr.get("status") != "passed_smoke":
                continue
            toks = parse_toks(sr.get("throughput_line"))
            wall = sr.get("wall_seconds")
            prefer_target = sr.get("seq_len") == DEFAULT_WORKLOAD["seq_len_target"]
            # Primary: tokens/s (steady-step metric). Wall clock includes FA4 JIT warmup.
            if toks is not None:
                score = ("toks", toks, prefer_target)
            elif wall is not None:
                score = ("wall", -float(wall), prefer_target)
            else:
                continue
            better = False
            if winner is None:
                better = True
            else:
                # Prefer target seq length when available
                if prefer_target and not winner.get("prefer_target"):
                    better = True
                elif prefer_target == winner.get("prefer_target"):
                    if score[0] == best_score[0]:
                        better = score[1] > best_score[1]
                    elif score[0] == "toks" and best_score[0] == "wall":
                        better = True
            if better:
                winner = {
                    "candidate": key,
                    "title": entry.get("title"),
                    "seq_len": sr.get("seq_len"),
                    "tokens_per_sec": toks,
                    "wall_seconds": wall,
                    "prefer_target": prefer_target,
                    "status": "best_measured_among_tested_smoke",
                }
                best_score = score
                if toks is not None:
                    winner_reason = (
                        f"Highest tokens/s among passed smoke runs at seq={sr.get('seq_len')} "
                        f"(wall clock includes JIT warmup; not globally optimal)."
                    )
                else:
                    winner_reason = (
                        f"Lowest wall time among passed smoke runs at seq={sr.get('seq_len')} "
                        f"(tokens/s unavailable)."
                    )

    mxfp8 = (bench.get("candidates") or {}).get("B_fa4_mxfp8") or {}
    mxfp8_note = None
    if mxfp8.get("status") == "failed":
        mxfp8_note = (
            "MXFP8 candidate failed model integration smoke: TE requires leading dims product "
            "divisible by 8 (saw dims like [7, 3072]). TE Linear smoke passed, but MiMo CPT path "
            "needs padding/alignment fixes before MXFP8 can be ranked. Keep FA4+BF16 as working baseline."
        )

    report = {
        "policy_date": POLICY_DATE,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "workload": DEFAULT_WORKLOAD,
        "hardware_note": "Validated path targets 1×B300 now; 8×B300 requires re-bench + NCCL.",
        "cuda_layers": plan.get("cuda_layers"),
        "validation": validate,
        "bench": bench,
        "selection": {
            "best_measured_among_tested": winner,
            "reason": winner_reason,
            "mxfp8_note": mxfp8_note,
            "labels": {
                "compatible_untested": "Resolves/installs but no successful model smoke yet",
                "passed_smoke": "Short synthetic mini CPT steps finished without hard failure",
                "best_measured": "Best tokens/s (or wall time) among candidates that passed smoke on this machine",
            },
        },
        "how_to_rerun_after_upgrades": [
            "Use a NEW --root (never reuse a partial/failed directory).",
            "python scripts/setup/optimize_ml_env.py --stage diagnose",
            "python scripts/setup/optimize_ml_env.py --stage plan",
            f"python scripts/setup/optimize_ml_env.py --root <new-root> --stage all --fast",
            "Compare reports/*/bench.json; keep BF16 FA4 as correctness reference when changing precision.",
        ],
        "sources": SOURCES,
        "limitations": [
            "Mini dress rehearsal ≠ full 110B throughput or quality.",
            "Synthetic patterned data ≠ production shard mix.",
            "1× results do not certify 8× scaling or IB/NVLink behavior.",
            "FA4 CuTeDSL may warn about AuxData JIT args; treat as warning unless smoke fails.",
            "Never call this setup universally optimal.",
        ],
    }
    out = root / "REPORT.json"
    json_write(out, report)
    md = root / "REPORT.md"
    lines = [
        f"# CPT optimize report ({POLICY_DATE})",
        "",
        f"Workload: continued pretraining · mini→110B · seq {DEFAULT_WORKLOAD['seq_len_target']} · prefer faster precision if stable.",
        "",
        "## Selection",
        "",
        f"- Best measured among tested: `{json.dumps(winner)}`",
        f"- Reason: {winner_reason}",
        "",
        "## Validation summary",
        "",
        "```json",
        json.dumps(validate.get("checks", {}), indent=2),
        "```",
        "",
        "## Commands",
        "",
        f"- Launcher: `bash {root}/run-training.sh`",
        f"- Full JSON: `{out}`",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {x}" for x in report["limitations"]]
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print(f"Wrote {out} and {md}", flush=True)
    return 0


def stage_all(args) -> int:
    # install → validate → build-mini → bench → report
    stage_install(args)
    stage_validate(args)
    stage_build_mini(args)
    if args.skip_bench:
        print("Skipping bench (--skip-bench). Run --stage bench when ready.", flush=True)
    else:
        stage_bench(args)
    stage_report(args)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--stage",
        choices=["diagnose", "plan", "install", "validate", "build-mini", "bench", "report", "all"],
        default="diagnose",
    )
    p.add_argument("--root", type=Path, default=Path("ml-env-cpt"), help="NEW directory for isolated env")
    p.add_argument("--fast", action="store_true", help="Shorter benches (seq 256, 3 steps)")
    p.add_argument("--skip-bench", action="store_true", help="Install/validate/mini only")
    p.add_argument("--skip-nccl", action="store_true")
    p.add_argument("--allow-busy", action="store_true")
    p.add_argument("--allow-validate-fail", action="store_true")
    p.add_argument("--write", type=Path, help="Optional JSON output path for diagnose/plan")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if platform.system() != "Linux":
        raise OptError("Linux x86_64 only for this CPT recipe.")
    os.chdir(REPO_ROOT)
    (REPO_ROOT / "logs").mkdir(exist_ok=True)
    stages = {
        "diagnose": stage_diagnose,
        "plan": stage_plan,
        "install": stage_install,
        "validate": stage_validate,
        "build-mini": stage_build_mini,
        "bench": stage_bench,
        "report": stage_report,
        "all": stage_all,
    }
    return stages[args.stage](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OptError, OSError) as exc:
        print(f"\nSTOP: {exc}", file=sys.stderr)
        raise SystemExit(1)
