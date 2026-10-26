#!/usr/bin/env bash
# Launch helper for 1×B300 (also works with GPUS_PER_NODE>1 later).
#
# Usage:
#   bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_speed_mxfp8.yaml
#   KING_DIR=./pipeline_model_L3 bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_speed_mxfp8.yaml
#   bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_soup.yaml --resume checkpoints/b300_1gpu_soup/step_0000009 --trigger-decay
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

# Prefer an already-selected env (e.g. ml-env-cpt). Only fall back to .venv.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  :
elif [[ -n "${PYTHON:-}" && -x "${PYTHON}" ]]; then
  :
elif [[ -x "$ROOT/ml-env-cpt/training/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/ml-env-cpt/training/bin/activate"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python}"
# Ensure TE/NCCL libs resolve when using ml-env-cpt / venv NVIDIA wheels.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  _site="${VIRTUAL_ENV}/lib/python3.12/site-packages"
  if [[ -d "$_site/nvidia/nccl/lib" ]]; then
    export LD_LIBRARY_PATH="$_site/nvidia/nccl/lib:$_site/nvidia/cudnn/lib:$_site/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  if [[ -z "${CUDA_HOME:-}" && -x "$_site/nvidia/cu13/bin/nvcc" ]]; then
    export CUDA_HOME="$_site/nvidia/cu13"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
fi
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CONFIG=""
EXTRA_ARGS=()
KING_DIR_OVERRIDE="${KING_DIR:-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  echo "Usage: bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_....yaml [train.py args...]" >&2
  exit 2
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG" >&2
  exit 1
fi

mkdir -p logs checkpoints results .cache/fa4_cute

# Production-style env (safe defaults for 1×; override for IB on multi-node later).
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TEUTONIC_NODES=1
export GPUS_PER_NODE
export TEUTONIC_DIST_BACKEND="${TEUTONIC_DIST_BACKEND:-nccl}"
export TEUTONIC_ATTN="${TEUTONIC_ATTN:-flash_attention_4}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED="${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}"
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR="${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-$ROOT/.cache/fa4_cute}"

if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  for _cuda in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
    if [[ -x "${_cuda}/bin/nvcc" ]]; then
      export CUDA_TOOLKIT_PATH="${_cuda}"
      export CUDA_HOME="${_cuda}"
      break
    fi
  done
fi

RUNTIME_CFG="$ROOT/checkpoints/_runtime/b300_1gpu_runtime.yaml"
mkdir -p "$(dirname "$RUNTIME_CFG")"
"$PYTHON" - <<PY
import copy
import yaml
from pathlib import Path

root = Path("$ROOT")
cfg = yaml.safe_load((root / "$CONFIG").read_text())
cfg = copy.deepcopy(cfg)
cfg.setdefault("hardware", {})["gpus_per_node"] = int("$GPUS_PER_NODE")
train = cfg.setdefault("training", {})
king = "${KING_DIR_OVERRIDE}"
out = "${OUTPUT_DIR_OVERRIDE}"
if king:
    train["king_dir"] = king
if out:
    train["output_dir"] = out
Path("$RUNTIME_CFG").write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"runtime_config={Path('$RUNTIME_CFG')} king={train.get('king_dir')} out={train.get('output_dir')}")
PY

echo "launch: nproc=$GPUS_PER_NODE config=$CONFIG attn=$TEUTONIC_ATTN backend=$TEUTONIC_DIST_BACKEND"
CMD=(
  "$PYTHON" -m torch.distributed.run
  --standalone
  --nproc_per_node="$GPUS_PER_NODE"
  -m
  src.train
  --config
  "$RUNTIME_CFG"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi
exec "${CMD[@]}"
