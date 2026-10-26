#!/usr/bin/env bash
# Verify the active venv is ready for Teutonic MoE / MiMo-V2 CPT.
# Adapted from GPU_check/check.sh with FA4 + TE/MXFP8 + MoE imports.
#
# Usage (after activate):
#   bash scripts/setup/check_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" && -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
REQUIRE_FA4="${REQUIRE_FA4:-0}"
REQUIRE_TE="${REQUIRE_TE:-0}"
REQUIRE_MXFP8="${REQUIRE_MXFP8:-0}"

# Auto-raise requirements for datacenter Blackwell when possible
if "$PYTHON" - <<'PY' 2>/dev/null
import torch, sys
sys.exit(0 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10 else 1)
PY
then
  REQUIRE_FA4=1
  REQUIRE_TE=1
  REQUIRE_MXFP8=1
fi

echo "==> pip check"
"$PYTHON" -m pip check || true

echo "==> nvidia-smi"
nvidia-smi || echo "WARN: nvidia-smi failed"

echo "==> package / CUDA / MoE probes"
REQUIRE_FA4="$REQUIRE_FA4" REQUIRE_TE="$REQUIRE_TE" REQUIRE_MXFP8="$REQUIRE_MXFP8" "$PYTHON" - <<'PY'
import os
import sys
from importlib.metadata import PackageNotFoundError, version

print("\nPython executable:", sys.executable)
print("Python version:", sys.version.split()[0])

for package in [
    "torch",
    "transformers",
    "accelerate",
    "triton",
    "flash-attn",
    "flash-attn-4",
    "transformer-engine",
    "safetensors",
    "pyyaml",
]:
    try:
        print(f"{package}: {version(package)}")
    except PackageNotFoundError:
        print(f"{package}: not installed")

try:
    import torch
except ImportError:
    print("\nPyTorch cannot be imported.")
    raise SystemExit(1)

print("\nPyTorch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())

if not torch.cuda.is_available():
    print("ERROR: CUDA unavailable")
    raise SystemExit(1)

print("Compiled architecture targets:", torch.cuda.get_arch_list())
caps = []
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    caps.append((p.major, p.minor))
    print(
        f"GPU {i}: {p.name}, "
        f"compute capability {p.major}.{p.minor}, "
        f"VRAM {p.total_memory / 1024**3:.1f} GiB"
    )

# FA4
req_fa4 = os.environ.get("REQUIRE_FA4") == "1"
try:
    from transformers.utils import is_flash_attn_4_available
    fa4 = bool(is_flash_attn_4_available())
except Exception as exc:  # noqa: BLE001
    fa4 = False
    print(f"FA4 registry check failed: {exc}")
print(f"FA4 available={fa4}")
if fa4:
    from flash_attn.cute import flash_attn_func

    q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    flash_attn_func(q, q, q)
    torch.cuda.synchronize()
    print("FA4 GPU smoke OK")
elif req_fa4:
    raise SystemExit("FA4 required but not available")

# TE / MXFP8
req_te = os.environ.get("REQUIRE_TE") == "1"
req_mx = os.environ.get("REQUIRE_MXFP8") == "1"
te_ok = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    from src.model import _capability_supports_low_precision

    major, minor = caps[0]
    gate = _capability_supports_low_precision("mxfp8", (major, minor))
    print(f"TE import OK; stack_gate_mxfp8={gate} cap=({major},{minor})")
    if req_mx or gate:
        rec = te_recipe.MXFP8BlockScaling()
        lin = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
        x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        ctx = te.autocast(enabled=True, recipe=rec) if hasattr(te, "autocast") else te.fp8_autocast(enabled=True, fp8_recipe=rec)
        with ctx:
            y = lin(x)
        y.float().sum().backward()
        torch.cuda.synchronize()
        print("TE MXFP8 Linear smoke OK")
        if req_mx and not gate:
            raise SystemExit("MXFP8 required but stack gate refused this capability")
    te_ok = True
except Exception as exc:  # noqa: BLE001
    print(f"TE/MXFP8 probe: {type(exc).__name__}: {exc}")
    if req_te or req_mx:
        raise SystemExit("TE/MXFP8 required but probe failed") from exc

# MoE package import
import src.train  # noqa: F401
import src.model  # noqa: F401
print("MoE src packages import OK")
print("\ncheck_env: PASS")
PY
