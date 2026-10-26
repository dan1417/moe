#!/usr/bin/env bash
set -euo pipefail

# Auto-detects single vs multi-node from NODE_RANK and MASTER_ADDR env vars.
# Usage:
#   Single node (Pro 6000):  bash scripts/train/launch.sh
#   Node 0 of 2: NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh
#   Node 1 of 2: NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh
#   B300 + InfiniBand cluster: NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=bond0 bash scripts/train/launch.sh
#   Override GPU count: GPUS_PER_NODE=4 bash scripts/train/launch.sh

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
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DETECTED_GPUS}
CONFIG=${CONFIG:-"configs/pro6000/train.yaml"}

echo "launch: nnodes=${NUM_NODES} nproc_per_node=${GPUS_PER_NODE} (detected=${DETECTED_GPUS}) config=${CONFIG}"

# apt CUDA packages can install NVML ahead of the kernel driver; breaks NCCL barriers.
if ! nvidia-smi >/dev/null 2>&1; then
  bash "$ROOT/scripts/setup/fix_nvml.sh" 2>/dev/null || true
fi

# Default: single-node Pro 6000 (NVLink, no IB). Override for B300/IB multi-node.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
if [[ "${NCCL_IB_DISABLE}" == "0" ]]; then
  export NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-5}
  export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
  export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}
  export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
fi
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
# FA4 first-step JIT can exceed the PyTorch new_group default (600s).
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-3600}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}
# Persist FA4 CuTe-DSL kernels across ranks/runs (default is off → every process JITs from scratch).
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=${FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED:-1}
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=${FLASH_ATTENTION_CUTE_DSL_CACHE_DIR:-"$PWD/.cache/fa4_cute"}
mkdir -p "$FLASH_ATTENTION_CUTE_DSL_CACHE_DIR"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TEUTONIC_NODES=$NUM_NODES
export GPUS_PER_NODE
# FlashAttention-4 is the default fast path (MiMo QK=192/V=128 works on SM120).
export TEUTONIC_ATTN=${TEUTONIC_ATTN:-flash_attention_4}
if [[ -z "${CUDA_TOOLKIT_PATH:-}" ]]; then
  for _cuda in /usr/local/cuda-13.0 /usr/local/cuda; do
    if [[ -x "${_cuda}/bin/nvcc" ]]; then
      export CUDA_TOOLKIT_PATH="${_cuda}"
      export CUDA_HOME="${_cuda}"
      break
    fi
  done
fi

torchrun \
  --nnodes="$NUM_NODES" \
  --nproc_per_node="$GPUS_PER_NODE" \
  --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m src.train --config "$CONFIG" "$@"
