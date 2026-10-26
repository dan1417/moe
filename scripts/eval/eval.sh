#!/usr/bin/env bash
set -euo pipefail

# Offline paired dual eval (validator-style king vs challenger on identical sequences).
#
# 110B MoE does not fit under DDP (full replica per GPU). Eval uses FSDP
# FULL_SHARD across all visible GPUs so every rank participates in forwards.
#
# Usage:
#   bash scripts/eval/eval.sh --weights ./merged_aoe --eval-output ./results/merged_aoe_eval.json
#   bash scripts/eval/eval.sh --checkpoint ./checkpoints/step_0003000
#   GPUS_PER_NODE=4 bash scripts/eval/eval.sh --weights ./merged_aoe

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
# Dedicated port so eval does not collide with a training job on 29500.
MASTER_PORT=${MASTER_PORT:-29502}

export CONFIG
export GPUS_PER_NODE
export MASTER_PORT
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
# Dataset build / dual-model load can exceed the default 10m store timeout.
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

echo "eval: ${GPUS_PER_NODE} GPU(s) via FSDP (detected=${DETECTED_GPUS})"

# Prefer FA4 when installed (MiMo sinks); SM120 uses LSE sink workaround in src/model.py.
export TEUTONIC_ATTN=${TEUTONIC_ATTN:-auto}

bash "$ROOT/scripts/train/launch.sh" --eval-only "$@"
