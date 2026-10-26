#!/usr/bin/env bash
set -euo pipefail

# Fast distributed merge on 8× GPU (Pro 6000 / similar).
# Usage:
#   bash scripts/merge/launch_merge.sh --model-a ./king --model-b ./other_king --strategy aoe --output ./merged_aoe
#   bash scripts/merge/launch_merge.sh --model-a ./king --model-b ./ckpt --strategy linear --alpha 0.5 --output ./merged_linear

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/setup/detect_gpus.sh"
detect_gpus

NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29501}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DETECTED_GPUS}

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE

echo "merge: nproc_per_node=${GPUS_PER_NODE} (detected=${DETECTED_GPUS})"

torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  scripts/merge/merge_models.py \
  --distributed \
  --gpus-per-node "$GPUS_PER_NODE" \
  "$@"
