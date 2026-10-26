#!/usr/bin/env bash
set -euo pipefail

# Negative-alpha AoE merges (task-arithmetic extrapolation) + smoke eval.
# Naming: alpha=-0.1 -> merge_eval_results/aoe_am0.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
bash scripts/setup/fix_nvml.sh 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

merge_one() {
  local alpha="$1"
  local port="$2"
  local tag="aoe_am${alpha#-}"
  local out="./merge_eval_results/${tag}"

  echo "=== AoE merge alpha=${alpha} -> ${out} ==="
  rm -rf "$out"
  MASTER_PORT="$port" bash scripts/merge/launch_merge.sh \
    --model-a ./king \
    --model-b ./previous_king \
    --strategy aoe \
    --alpha "$alpha" \
    --output "$out" \
    --contract-dir ./king 2>&1 | tee "./merge_eval_results/${tag}_merge.log"

  test -f "$out/model.safetensors.index.json" || {
    echo "FAIL incomplete $out" >&2
    exit 1
  }
  python3 - <<PY
import json
r=json.load(open("$out/merge_report.json"))
print(f"OK $out alpha={r.get('alpha')} blended={r.get('n_blended')}")
PY
}

merge_one -0.1 29541
merge_one -0.2 29542
merge_one -1.0 29543

echo "=== All negative-alpha merges done; starting eval sweep ==="
TEUTONIC_ATTN="${TEUTONIC_ATTN:-flash_attention_4}" bash scripts/eval/eval_sweep.sh --eval-n 48 \
  ./merge_eval_results/aoe_am0.1 \
  ./merge_eval_results/aoe_am0.2 \
  ./merge_eval_results/aoe_am1.0

echo "=== Results ==="
for tag in aoe_am0.1 aoe_am0.2 aoe_am1.0; do
  f="./merge_eval_results/${tag}_eval.json"
  # sweep may write under different names; also check sweep outputs
  if [[ -f "$f" ]]; then
    python3 -c "import json; r=json.load(open('$f')); print('$tag', f\"mu_hat={r.get('mu_hat'):+.4f} lcb={r.get('lcb'):+.4f}\")"
  fi
done
ls -1 ./merge_eval_results/*am*_eval.json ./merge_eval_results/sweep*.json 2>/dev/null || true
echo "=== Done ==="
