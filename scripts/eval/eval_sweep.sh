#!/usr/bin/env bash
set -euo pipefail

# Fast multi-challenger eval: one FSDP shell, swap weights per path.
#
# Usage:
#   bash scripts/eval/eval_sweep.sh ./merge_eval_results/aoe_a0.*
#   bash scripts/eval/eval_sweep.sh --eval-n 48 ./merge_eval_results/aoe_a0.2 ./merge_eval_results/aoe_a0.5
#   EVAL_N=48 bash scripts/eval/eval_sweep.sh ./merge_eval_results/aoe_a*

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/setup/detect_gpus.sh"
detect_gpus

CONFIG=${CONFIG:-configs/b300/train.yaml}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DETECTED_GPUS}
MASTER_PORT=${MASTER_PORT:-29505}
EVAL_N=${EVAL_N:-}
EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR:-./merge_eval_results}

export CONFIG
export GPUS_PER_NODE
export MASTER_PORT
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Prefer FA4 when installed (correct MiMo sinks); else eager. Never default FA2.
export TEUTONIC_ATTN=${TEUTONIC_ATTN:-auto}

WEIGHTS=()
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-n)
      EVAL_N="$2"
      shift 2
      ;;
    --eval-output-dir)
      EVAL_OUTPUT_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    -*)
      EXTRA+=("$1")
      shift
      ;;
    *)
      WEIGHTS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#WEIGHTS[@]} -eq 0 ]]; then
  echo "Usage: bash scripts/eval/eval_sweep.sh [--eval-n N] PATH [PATH...]" >&2
  exit 2
fi

mkdir -p "$EVAL_OUTPUT_DIR"
echo "eval_sweep: ${#WEIGHTS[@]} challenger(s) on ${GPUS_PER_NODE} GPU(s), attn=${TEUTONIC_ATTN}"

ARGS=(--eval-only --weights-sweep "${WEIGHTS[@]}" --eval-output "${EVAL_OUTPUT_DIR}/sweep_placeholder.json")
if [[ -n "$EVAL_N" ]]; then
  ARGS+=(--eval-n "$EVAL_N")
fi
ARGS+=("${EXTRA[@]}")

bash "$ROOT/scripts/train/launch.sh" "${ARGS[@]}"
