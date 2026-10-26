#!/usr/bin/env bash
set -euo pipefail

# End-to-end AoE + linear merge and eval (auto-detects GPU count).
#
# Prerequisites:
#   - ./king downloaded (safetensors)
#   - ./other_model downloaded (second king or checkpoint dir)
#   - pip deps installed; GPUs visible
#
# Usage:
#   MODEL_B=./other_king bash scripts/merge/run_merge_eval_pro6000.sh
#   MODEL_B=./checkpoints/step_0003000 bash scripts/merge/run_merge_eval_pro6000.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
# shellcheck disable=SC1091
source "$ROOT/scripts/setup/detect_gpus.sh"
detect_gpus

set -a
MODEL_A=${MODEL_A:-./king}
MODEL_B=${MODEL_B:?Set MODEL_B to the second model path}
ALPHA=${ALPHA:-0.5}
GPUS_PER_NODE=${GPUS_PER_NODE:-$DETECTED_GPUS}
CONFIG=${CONFIG:-configs/pro6000/train.yaml}
RESULTS_DIR=${RESULTS_DIR:-./merge_eval_results}
set +a

mkdir -p "$RESULTS_DIR"
echo "merge+eval: ${GPUS_PER_NODE} GPU(s) (detected=${DETECTED_GPUS})"

echo "=== AoE merge (FSDP/${GPUS_PER_NODE}-GPU distributed) ==="
bash scripts/merge/launch_merge.sh \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --strategy aoe \
  --alpha "$ALPHA" \
  --output "$RESULTS_DIR/merged_aoe" \
  --contract-dir "$MODEL_A"

echo "=== Linear merge (FSDP/${GPUS_PER_NODE}-GPU distributed) ==="
bash scripts/merge/launch_merge.sh \
  --model-a "$MODEL_A" \
  --model-b "$MODEL_B" \
  --strategy linear \
  --alpha "$ALPHA" \
  --output "$RESULTS_DIR/merged_linear" \
  --contract-dir "$MODEL_A"

echo "=== Eval merged AoE ==="
bash scripts/eval/eval.sh \
  --weights "$RESULTS_DIR/merged_aoe" \
  --eval-output "$RESULTS_DIR/merged_aoe_eval.json"

echo "=== Eval merged linear ==="
bash scripts/eval/eval.sh \
  --weights "$RESULTS_DIR/merged_linear" \
  --eval-output "$RESULTS_DIR/merged_linear_eval.json"

echo "Done. Results in $RESULTS_DIR"
