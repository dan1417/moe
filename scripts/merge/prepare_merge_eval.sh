#!/usr/bin/env bash
set -euo pipefail

# Prepare environment for MoE model merging and offline evaluation.
#
# Downloads king + genesis contract files, prefetches eval shard cache,
# and verifies GPU / Python dependencies.
#
# Usage:
#   bash scripts/merge/prepare_merge_eval.sh
#   bash scripts/merge/prepare_merge_eval.sh --skip-king        # skip 205 GB king download
#   bash scripts/merge/prepare_merge_eval.sh --skip-eval-cache  # skip eval shard prefetch
#   bash scripts/merge/prepare_merge_eval.sh --sequential       # king then eval (old behavior)
#   KING_DIR=./king CONFIG=configs/pro6000/train.yaml KING_WORKERS=8 bash scripts/merge/prepare_merge_eval.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

CONFIG=${CONFIG:-configs/pro6000/train.yaml}
KING_DIR=${KING_DIR:-./king}
GENESIS_DIR=${GENESIS_DIR:-./genesis}
GENESIS_REVISION=${GENESIS_REVISION:-8f971f030f5d801f812540007b396058f69cc08a}
RESULTS_DIR=${RESULTS_DIR:-./merge_eval_results}
# shellcheck disable=SC1091
source "$ROOT/scripts/setup/detect_gpus.sh"
detect_gpus
GPUS_PER_NODE=${GPUS_PER_NODE:-$DETECTED_GPUS}
KING_WORKERS=${KING_WORKERS:-8}
SKIP_KING=0
SKIP_EVAL_CACHE=0
PARALLEL=1

for arg in "$@"; do
  case "$arg" in
    --skip-king) SKIP_KING=1 ;;
    --skip-eval-cache) SKIP_EVAL_CACHE=1 ;;
    --sequential) PARALLEL=0 ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

echo "=== MoE merge/eval preparation ==="
echo "Config:      $CONFIG"
echo "King dir:    $KING_DIR"
echo "Genesis dir: $GENESIS_DIR"
echo "Results dir: $RESULTS_DIR"
echo "GPUs/node:   $GPUS_PER_NODE"
echo

echo "=== [1/5] Python dependencies ==="
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [[ -f setup.sh ]]; then
  bash setup.sh
else
  grep -v '^flash-attn' requirements.txt | pip install -r /dev/stdin
  if [[ -f scripts/setup/install_flash_attn.sh ]]; then
    bash scripts/setup/install_flash_attn.sh || echo "flash-attn optional, skipping"
  else
    pip install flash-attn --no-build-isolation || echo "flash-attn optional, skipping"
  fi
  pip install huggingface_hub
fi

python - <<'PY'
import torch, transformers, safetensors, numpy, yaml, requests
print(
    f"torch {torch.__version__} | cuda={torch.cuda.is_available()} | "
    f"gpus={torch.cuda.device_count()} | transformers {transformers.__version__}"
)
PY

VISIBLE="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_NODE - 1)))}"
export CUDA_VISIBLE_DEVICES="$VISIBLE"
ACTUAL=$(python -c "import torch; print(torch.cuda.device_count())")
if (( ACTUAL < GPUS_PER_NODE )); then
  echo "WARNING: requested $GPUS_PER_NODE GPUs but only $ACTUAL visible (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)" >&2
else
  echo "GPUs OK: $ACTUAL visible"
fi
echo

echo "=== [2/5] Output directories ==="
mkdir -p "$RESULTS_DIR" "$KING_DIR" "$GENESIS_DIR" ./shard_cache ./checkpoints
echo "Created: $RESULTS_DIR, $KING_DIR, $GENESIS_DIR, ./shard_cache, ./checkpoints"
echo

echo "=== [3/5] Genesis contract files ==="
if [[ -f "$GENESIS_DIR/config.json" && -f "$GENESIS_DIR/modeling_mimo_v2.py" ]]; then
  echo "Genesis contract already present in $GENESIS_DIR"
else
  hf download dendriteholdings/teutonic-II-110B-genesis \
    --revision "$GENESIS_REVISION" \
    --local-dir "$GENESIS_DIR" \
    config.json chat_template.jinja.txt configuration_mimo_v2.py modeling_mimo_v2.py \
    tokenizer.json tokenizer_config.json
fi
echo

echo "=== [4+5/5] King download + eval cache (parallel by default) ==="
KING_PID=""
EVAL_PID=""
PREP_EXIT=0

_run_king_download() {
  if (( SKIP_KING )); then
    echo "[king] Skipped (--skip-king). Run: python scripts/data/download_king.py --output $KING_DIR --workers $KING_WORKERS"
    return 0
  fi
  if [[ -f "$KING_DIR/king_info.json" ]]; then
    echo "[king] Verifying existing king in $KING_DIR..."
    python scripts/data/download_king.py --output "$KING_DIR" --verify-only --workers "$KING_WORKERS"
  else
    echo "[king] Downloading current king to $KING_DIR ($KING_WORKERS workers)..."
    python scripts/data/download_king.py --output "$KING_DIR" --workers "$KING_WORKERS"
  fi
}

_run_eval_cache() {
  if (( SKIP_EVAL_CACHE )); then
    echo "[eval] Skipped (--skip-eval-cache). Run: python scripts/data/prepare_eval_cache.py --config $CONFIG"
    return 0
  fi
  echo "[eval] Prefetching eval shard cache (2000 sequences, seed=99999)..."
  python scripts/data/prepare_eval_cache.py --config "$CONFIG"
}

if (( PARALLEL && ! SKIP_KING && ! SKIP_EVAL_CACHE )); then
  echo "Running king download and eval cache prefetch concurrently."
  _run_king_download > >(sed 's/^/[king] /') 2>&1 &
  KING_PID=$!
  _run_eval_cache > >(sed 's/^/[eval] /') 2>&1 &
  EVAL_PID=$!
  if ! wait "$KING_PID"; then
    echo "King download failed" >&2
    PREP_EXIT=1
  fi
  if ! wait "$EVAL_PID"; then
    echo "Eval cache prefetch failed" >&2
    PREP_EXIT=1
  fi
  (( PREP_EXIT == 0 )) || exit "$PREP_EXIT"
elif (( PARALLEL && ! SKIP_KING && SKIP_EVAL_CACHE )); then
  _run_king_download
elif (( PARALLEL && SKIP_KING && ! SKIP_EVAL_CACHE )); then
  _run_eval_cache
else
  _run_king_download
  echo
  _run_eval_cache
fi
echo

echo "=== Ready for merge + eval ==="
echo "  AoE merge:    bash scripts/merge/launch_merge.sh --model-a $KING_DIR --model-b <MODEL_B> --strategy aoe --output $RESULTS_DIR/merged_aoe --contract-dir $KING_DIR"
echo "  Linear merge: bash scripts/merge/launch_merge.sh --model-a $KING_DIR --model-b <MODEL_B> --strategy linear --alpha 0.5 --output $RESULTS_DIR/merged_linear --contract-dir $KING_DIR"
echo "  Eval:         bash scripts/eval/eval.sh --weights $RESULTS_DIR/merged_aoe --eval-output $RESULTS_DIR/merged_aoe_eval.json"
echo "  Full pipeline: MODEL_B=<path> bash scripts/merge/run_merge_eval_pro6000.sh"
