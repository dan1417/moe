#!/usr/bin/env bash
# Judge a submission candidate on this node (GPUs must be free: STOP/pause the training line first).
#   bash scripts/eval/judge_candidate.sh <export_dir> [limit=8000]      # limit 30000 = full samples A and B
# 1. validator checks vs king #30 (contract files, tensors, dtypes, no MTP, differs from king)
# 2. parity eval vs reign:30 on samples A and B (validator math, FA4, 8 single-GPU replicas)
# 3. summary: mu_hat / lcb / robust_lcb per sample + acceptance probability from the val-set estimate
set -euo pipefail
CAND=${1:?export dir}; LIMIT=${2:-8000}
source /workspace/teutonic/env.sh
cd "$REPO_ROOT"
LOG=/workspace/teutonic/logs/judge_$(basename "$CAND")_$(date -u +%H%M%S).log
{
echo "== validator checks: $CAND"
python scripts/verify/verify_submission.py --model-dir "$CAND" --king-dir /dev/shm/teutonic/king30 | tail -6
echo "== parity eval vs reign:30 (limit $LIMIT)"
ARGS=(--attn flash_attention_4 --batch-size 16 --replicas 8)
[[ "$LIMIT" != "30000" ]] && ARGS+=(--limit "$LIMIT")
python scripts/eval/parity_eval.py "${ARGS[@]}" duel --king reign:30 --challenger "$CAND" --sample tag:A --sample tag:B 2>&1 \
  | grep -v -E "FutureWarning|warnings.warn|JitArgument|exe_args"
} 2>&1 | tee "$LOG"
echo "log: $LOG"
