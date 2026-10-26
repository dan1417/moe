#!/usr/bin/env bash
# 1×B300 (and 8×B300) production-style venv: cu130 + FA4 + Transformer Engine.
# Reuse this image/venv on 8× — only change GPUS_PER_NODE.
#
# Usage:
#   bash scripts/setup/setup_b300_venv.sh
#   source .venv/bin/activate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

echo "==> 0) Teutonic stack preflight (GPU_check policy)"
mkdir -p logs
if ! python scripts/setup/teutonic_stack.py --json --write logs/teutonic_env_plan.json; then
  echo "WARN: teutonic_stack preflight returned non-zero; continuing if CUDA still usable" >&2
fi
python scripts/setup/teutonic_stack.py 2>/dev/null || true

echo "==> 0) Prefer workload optimizer for CPT (does not touch .venv)"
echo "    python scripts/setup/optimize_ml_env.py --stage plan"
echo "    python scripts/setup/optimize_ml_env.py --root ml-env-cpt --stage all --fast"
echo ""
PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"
SKIP_MINI_BUILD="${SKIP_MINI_BUILD:-0}"
SKIP_TE="${SKIP_TE:-0}"

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -U pip setuptools wheel

echo "==> Installing torch 2.11 + cu130 (FA4 wheel contract)"
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu130

echo "==> Installing requirements (excluding flash-attn / TE; installed next)"
pip install -r <(grep -vE '^(flash-attn|transformer-engine)' requirements.txt)

echo "==> Installing FA4 (required for MiMo-V2 on B300)"
bash scripts/setup/install_flash_attn.sh

if [[ "$SKIP_TE" != "1" ]]; then
  echo "==> Installing Transformer Engine (MXFP8) — pytorch + core_cu13"
  # Bare `transformer-engine` is an empty meta package and fails check_env.
  # Official NGC TE torch wheels target container torch tags and ABI-mismatch
  # pip torch 2.11+cu130. Verified 2026-09-06: build transformer-engine-torch
  # from sdist against this torch using pip nvidia-cuda-nvcc + cccl.
  pip install "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-cccl==13.0.85" ninja
  SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
  export CUDA_HOME="${CUDA_HOME:-$SITE/nvidia/cu13}"
  export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
  export CUDNN_PATH="${CUDNN_PATH:-$SITE/nvidia/cudnn}"
  export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
  export C_INCLUDE_PATH="$CPLUS_INCLUDE_PATH"
  export LIBRARY_PATH="$SITE/nvidia/cudnn/lib:$SITE/nvidia/nccl/lib:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export LD_LIBRARY_PATH="$SITE/nvidia/nccl/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export PATH="$CUDA_HOME/bin:$PATH"
  export NVTE_FRAMEWORK=pytorch
  pip install --no-build-isolation "transformer-engine[pytorch,core_cu13]==2.18.0" || {
    echo "WARN: TE build failed; BF16/FA4 still usable. Prefer: python scripts/setup/optimize_ml_env.py" >&2
  }
fi

# FA4 CuTe JIT / TE: prefer venv toolkit, else host toolkit.
if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  SITE="$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
  if [[ -x "$SITE/nvidia/cu13/bin/nvcc" ]]; then
    export CUDA_TOOLKIT_PATH="$SITE/nvidia/cu13"
    export CUDA_HOME="$SITE/nvidia/cu13"
  else
    for _cuda in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
      if [[ -x "${_cuda}/bin/nvcc" ]]; then
        export CUDA_TOOLKIT_PATH="${_cuda}"
        export CUDA_HOME="${_cuda}"
        break
      fi
    done
  fi
fi

echo "==> Capability / FA4 / TE smoke + check_env"
bash scripts/setup/check_env.sh

if [[ "$SKIP_MINI_BUILD" != "1" ]]; then
  echo "==> Building default 4-layer shrink king (~8.59B)"
  python scripts/data/prepare_pipeline_model.py --output "$ROOT/pipeline_model" --num-layers 4
  python scripts/verify/verify_pipeline_design.py \
    --config configs/b300/1gpu_speed_mxfp8.yaml \
    --king-dir pipeline_model || true
fi

echo ""
echo "B300 venv ready. Next:"
echo "  source .venv/bin/activate"
echo "  # background 110B download:"
echo "  python scripts/data/download_king.py --output ./king --workers 8 > logs/download_king.log 2>&1 &"
echo "  # run gates:"
echo "  bash scripts/train/run_b300_1gpu_tests.sh"
echo "  See docs/TESTPLAN.md"
