#!/usr/bin/env bash
set -euo pipefail

# Install Flash Attention for MiMo-V2 on RTX PRO 6000 / cu130.
#
# MiMo-V2 needs FA4 (sinks + asymmetric QK/V). FA2 alone breaks CE (~8).
# This script installs:
#   1) flash-attn 2.8.3 Blackwell wheel (dependency / legacy FA2 path)
#   2) flash-attn-4[cu13] pre-release (correct backend for this model)
#
# Usage:
#   bash scripts/setup/install_flash_attn.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

WHEEL_URL="https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels/releases/download/FA.2.8.3-custom-linux-wheels-x86_64/flash_attn-2.8.3%2Bcu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc132-cp312-cp312-linux_x86_64.whl"
WHEEL_NAME="flash_attn-2.8.3+cu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc132-cp312-cp312-linux_x86_64.whl"
WHEEL_PATH="/tmp/${WHEEL_NAME}"

if [[ ! -f "$WHEEL_PATH" ]]; then
  echo "Downloading Blackwell flash-attn wheel..."
  wget -q "$WHEEL_URL" -O "$WHEEL_PATH"
fi

echo "Installing torch 2.11+cu130 (required by flash-attn wheel)..."
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130
pip install ninja packaging psutil einops
pip install "$WHEEL_PATH" --no-deps

echo "Installing flash-attn-4 (MiMo-V2 sinks / asymmetric heads)..."
pip install --pre "flash-attn-4[cu13]"

bash "$ROOT/scripts/setup/fix_nvml.sh" 2>/dev/null || true

python - <<'PY'
import torch
from transformers.utils import is_flash_attn_2_available, is_flash_attn_4_available

print(f"torch {torch.__version__}")
print(f"FA2 available={is_flash_attn_2_available()} | FA4 available={is_flash_attn_4_available()}")
if not is_flash_attn_4_available():
    raise SystemExit("flash-attn-4 did not register with transformers")
major, minor = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
print(f"GPU arch sm_{major}{minor}")
if torch.cuda.is_available():
    from flash_attn.cute import flash_attn_func

    q = torch.randn(1, 8, 128, 64, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(8, device="cuda", dtype=torch.bfloat16)
    # On sm_120, native learnable_sink is broken until we patch; smoke without sink here.
    flash_attn_func(q, q, q)
    print(f"FA4 GPU smoke OK on {torch.cuda.get_device_name(0)}")
    if major >= 12:
        print("Note: sm_120 uses Teutonic LSE sink workaround (src/model/core.py) for MiMo sinks")
PY

echo "flash-attn-4 ready. Default TEUTONIC_ATTN=auto → flash_attention_4 when installed."
