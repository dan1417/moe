#!/usr/bin/env bash
# Launch src.cpt.train on this 8xB300 pod with the node environment (/workspace/teutonic/env.sh).
#   bash scripts/train/node_train.sh configs/cpt/node/line_a_upper.yaml [extra trainer args]
#   bash scripts/train/node_train.sh configs/cpt/node/gate_full.yaml --max-steps 6
# Logs: /workspace/teutonic/logs/train_<config-stem>_<utc>.log (tail -f it). Runs in the foreground;
# wrap with nohup/tmux for long lines.
set -euo pipefail
CONFIG=${1:?config yaml}; shift || true
source /workspace/teutonic/env.sh
cd "$REPO_ROOT"
NPROC=${NPROC:-8}
STEM=$(basename "$CONFIG" .yaml)
LOG=/workspace/teutonic/logs/train_${STEM}_$(date -u +%Y%m%dT%H%M%SZ).log
ln -sfn "$LOG" /workspace/teutonic/logs/train_${STEM}_latest.log
echo "==> $CONFIG  (log: $LOG)"
free -g | head -2; df -h /dev/shm /mnt/ram | tail -2
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m src.cpt.train --config "$CONFIG" "$@" 2>&1 | tee "$LOG"
