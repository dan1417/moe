#!/usr/bin/env bash
set -euo pipefail

# AoE merge: aoe_a0.2 (A) + aoe_am0.1 (B), alpha=0.5 on expert groups.
# Run after eval frees GPUs:
#   bash scripts/merge/merge_a0.2_x_am0.1.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
bash scripts/setup/fix_nvml.sh 2>/dev/null || true

OUT=./merge_eval_results/aoe_a0.2_x_am0.1
LOG=./merge_eval_results/aoe_a0.2_x_am0.1_merge.log

echo "=== AoE merge aoe_a0.2 + aoe_am0.1 -> $OUT ===" | tee "$LOG"
rm -rf "$OUT"
MASTER_PORT=29550 bash scripts/merge/launch_merge.sh \
  --model-a ./merge_eval_results/aoe_a0.2 \
  --model-b ./merge_eval_results/aoe_am0.1 \
  --strategy aoe \
  --alpha 0.5 \
  --output "$OUT" \
  --contract-dir ./king 2>&1 | tee -a "$LOG"

test -f "$OUT/model.safetensors.index.json"
python3 -c "import json; print(json.load(open('$OUT/merge_report.json')))"
echo "Done: $OUT"
