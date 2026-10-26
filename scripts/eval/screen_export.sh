#!/usr/bin/env bash
# Score a checkpoint dir (export or start body) against king #30 with the validator-parity eval.
#   bash scripts/eval/screen_export.sh <model_dir_or_recipe> [limit] [samples...]
#   limit: first N sequences per sample (default 8000 = quick screen; empty/30000 = full, needed to submit)
# Uses the real FA4 kernels (the validator's production backend). GPUs must be free (stop training first).
set -euo pipefail
MODEL=${1:?model dir or recipe json}
LIMIT=${2:-8000}
shift 2 || shift $# 
SAMPLES=${*:-"tag:A tag:B"}
source /workspace/teutonic/env.sh
cd "$REPO_ROOT"
ARGS=(--attn flash_attention_4 --batch-size 16 --replicas ${REPLICAS:-8})
[[ -n "$LIMIT" && "$LIMIT" != "30000" ]] && ARGS+=(--limit "$LIMIT")
SAMPLE_ARGS=(); for s in $SAMPLES; do SAMPLE_ARGS+=(--sample "$s"); done
LOG=/workspace/teutonic/logs/screen_$(basename "$MODEL")_$(date -u +%Y%m%dT%H%M%SZ).log
python scripts/eval/parity_eval.py "${ARGS[@]}" duel --king reign:30 --challenger "$MODEL" "${SAMPLE_ARGS[@]}" 2>&1 | tee "$LOG"
