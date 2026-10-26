#!/usr/bin/env bash
# Run a training line under pm2 (no auto-restart: a crashed 110B run must be looked at, not relaunched).
#   bash scripts/train/pm2_line.sh line_a configs/cpt/node/line_a_upper_mb8.yaml [extra trainer args]
#   pm2 logs line_a --lines 50        # live log
#   pm2 stop line_a                   # hard stop (prefer: touch /workspace/teutonic/runs/line_a/STOP)
#   python scripts/train/progress.py  # compact progress summary of every run
set -euo pipefail
NAME=${1:?pm2 process name}; CONFIG=${2:?config yaml}; shift 2
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
pm2 delete "$NAME" >/dev/null 2>&1 || true
pm2 start bash --name "$NAME" --no-autorestart --time --merge-logs \
  --output "/workspace/teutonic/logs/pm2_${NAME}.out.log" --error "/workspace/teutonic/logs/pm2_${NAME}.err.log" \
  -- scripts/train/node_train.sh "$CONFIG" "$@"
pm2 save >/dev/null 2>&1 || true
pm2 ls
