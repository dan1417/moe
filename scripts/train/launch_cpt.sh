#!/usr/bin/env bash
# End-to-end CPT on an 8×B300 node: current king -> pools -> N training lines.
# Then evaluate each line's EMA export with parity_eval.py and soup the winners
# (configs/merge/*.json recipe + scripts/merge/write_merge.py). See docs/CPT.md.
#
#   DATA=/data LINES="a b" TOKENS_PER_LINE_ROUNDS=10 bash scripts/train/launch_cpt.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source .venv-cpt/bin/activate

DATA=${DATA:-/data}
LINES=${LINES:-"a b"}
ROUNDS=${TOKENS_PER_LINE_ROUNDS:-10}   # 1 round ≈ 61M tokens at --scale 1 (×SCALE)
SCALE=${SCALE:-2}
BASE_CONFIG=${BASE_CONFIG:-configs/cpt/b300.yaml}
NPROC=${NPROC:-8}
mkdir -p "$DATA/logs"

if [[ ! -f "$DATA/king/model.safetensors.index.json" ]]; then
  python scripts/data/download_king.py --output "$DATA/king" --workers 32 2>&1 | tee "$DATA/logs/download_king.log"
fi
# Optional: start from a merged body instead of the king (e.g. START_RECIPE=configs/merge/transplant26_b0.25.json).
# Its reign:/digest: terms are downloaded into $DATA/models and the body is written with write_merge.py
# (the exact code path the parity eval scored). Training "vs_king" numbers are then relative to this start.
START_DIR="$DATA/king"
if [[ -n "${START_RECIPE:-}" ]]; then
  export TEUTONIC_MODELS_DIRS="$DATA/models"
  for digest in $(python - "$START_RECIPE" <<'PY'
import json, sys, urllib.request
recipe = json.load(open(sys.argv[1]))
chain = {int(r["reign_number"]): r["model_digest"] for r in json.load(urllib.request.urlopen(
    urllib.request.Request("https://teutonic.ai/dashboard.json", headers={"User-Agent": "cpt"}), timeout=60))["king_chain"]}
for term in recipe["terms"]:
    model = term["model"]
    print(chain[int(model.split(":")[1])] if model.startswith("reign:") else model.split(":", 1)[1])
PY
  ); do
    [[ -f "$DATA/models/$digest/model.safetensors.index.json" ]] || \
      python scripts/data/download_king.py --digest "$digest" --output "$DATA/models/$digest" --workers 32
  done
  # START_UNPERMUTE="<digest>:<perm json>" maps a hidden-dim-permuted king (e.g. #30) back into the
  # lineage basis before merging; the permuted original is kept as <digest>.permuted.
  if [[ -n "${START_UNPERMUTE:-}" ]]; then
    digest="${START_UNPERMUTE%%:*}"; permfile="${START_UNPERMUTE#*:}"
    if [[ ! -f "$DATA/models/$digest/UNPERMUTED.json" ]]; then
      python scripts/merge/unpermute_checkpoint.py --src "$DATA/models/$digest" --perm "$permfile" --out "$DATA/models/$digest.unpermuted" \
        && mv "$DATA/models/$digest" "$DATA/models/$digest.permuted" && mv "$DATA/models/$digest.unpermuted" "$DATA/models/$digest"
    fi
  fi
  START_DIR="$DATA/start_$(basename "$START_RECIPE" .json)"
  [[ -f "$START_DIR/model.safetensors.index.json" ]] || \
    python scripts/merge/write_merge.py --recipe "$START_RECIPE" --out "$START_DIR"
fi
[[ -d "$DATA/pools/val" ]] || python scripts/data/build_train_pool.py --out "$DATA/pools/val" --rounds 1 --scale 0.05 --tag val
for line in $LINES; do
  [[ -d "$DATA/pools/line_$line" ]] || \
    python scripts/data/build_train_pool.py --out "$DATA/pools/line_$line" --rounds "$ROUNDS" --scale "$SCALE" --tag "line_$line"
done

for line in $LINES; do
  cfg="$DATA/runs/line_$line.yaml"
  mkdir -p "$DATA/runs"
  python - "$BASE_CONFIG" "$cfg" "$DATA" "$line" "$START_DIR" <<'PY'
import sys, yaml
base, out, data, line, start = sys.argv[1:]
cfg = yaml.safe_load(open(base))
cfg["king_dir"] = start
cfg["output_dir"] = f"{data}/runs/line_{line}"
cfg["seed"] = sum(map(ord, line))
cfg["data"]["pool_dir"] = f"{data}/pools/line_{line}"
cfg["data"]["val_dir"] = f"{data}/pools/val"
cfg["data"]["seed"] = cfg["seed"]
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
PY
  echo "==> training line $line ($cfg)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m torch.distributed.run --standalone --nproc_per_node="$NPROC" -m src.cpt.train --config "$cfg" \
    2>&1 | tee "$DATA/logs/train_line_$line.log"
done
echo "done: exports under $DATA/runs/line_*/exports (evaluate *_ema with scripts/eval/parity_eval.py duel)"
