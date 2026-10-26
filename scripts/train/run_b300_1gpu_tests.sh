#!/usr/bin/env bash
# Ordered 1×B300 TESTPLAN gates (short). Skip with SKIP_*=1 or GATE=name.
#
# Usage:
#   bash scripts/train/run_b300_1gpu_tests.sh              # A-smoke through E (not G download)
#   GATE=C bash scripts/train/run_b300_1gpu_tests.sh       # speed only
#   GATE=D bash scripts/train/run_b300_1gpu_tests.sh
#   GATE=E bash scripts/train/run_b300_1gpu_tests.sh
#   GATE=F bash scripts/train/run_b300_1gpu_tests.sh       # layer ladder 3 vs 4
#   GATE=G bash scripts/train/run_b300_1gpu_tests.sh       # eval (king must exist)
#   GATE=download bash scripts/train/run_b300_1gpu_tests.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  :
elif [[ -x "$ROOT/ml-env-cpt/training/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/ml-env-cpt/training/bin/activate"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

mkdir -p logs checkpoints results sequence_pool_b300_1gpu sequence_pool_b300_soup

GATE="${GATE:-all}"
export GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
export TEUTONIC_ATTN="${TEUTONIC_ATTN:-flash_attention_4}"

run_train() {
  local cfg="$1"
  shift
  bash scripts/train/run_b300_1gpu.sh --config "$cfg" "$@"
}

ensure_mini() {
  if [[ ! -f pipeline_model/config.json ]]; then
    echo "==> Building 4-layer pipeline_model"
    python scripts/data/prepare_pipeline_model.py --output ./pipeline_model --num-layers 4
  fi
}

gate_A_smoke() {
  echo "==> A) Env smoke (torch/FA4/TE/capability)"
  python - <<'PY'
import torch
from transformers.utils import is_flash_attn_4_available
from src.model import _capability_supports_low_precision
assert torch.cuda.is_available(), "no CUDA"
cap = torch.cuda.get_device_capability(0)
print("gpu", torch.cuda.get_device_name(0), "cap", cap)
assert is_flash_attn_4_available(), "FA4 missing"
assert _capability_supports_low_precision("mxfp8", cap), f"MXFP8 gate failed for {cap}"
import transformer_engine  # noqa: F401
print("A_smoke PASS")
PY
}

gate_B() {
  echo "==> B) Shrink king + verifier"
  ensure_mini
  python scripts/verify/verify_pipeline_design.py \
    --config configs/b300/1gpu_speed_mxfp8.yaml \
    --king-dir pipeline_model
  python scripts/verify/smoke_train.py --skip-torchrun
  echo "B PASS"
}

gate_C() {
  echo "==> C) Speed path (eager / FA4 / MXFP8) — 5 steps each"
  ensure_mini
  TEUTONIC_ATTN=eager run_train configs/b300/1gpu_speed_eager.yaml \
    2>&1 | tee logs/b300_1gpu_C1_eager.log
  TEUTONIC_ATTN=flash_attention_4 run_train configs/b300/1gpu_speed_fa4.yaml \
    2>&1 | tee logs/b300_1gpu_C2_fa4.log
  TEUTONIC_ATTN=flash_attention_4 run_train configs/b300/1gpu_speed_mxfp8.yaml \
    2>&1 | tee logs/b300_1gpu_C3_mxfp8.log
  echo "C PASS — compare tok/s in logs/b300_1gpu_C*.log"
}

gate_D() {
  echo "==> D) Real Teutonic mini CPT (20 steps)"
  ensure_mini
  python scripts/data/prepare_train_sequence_pool.py \
    --config configs/b300/1gpu_mini_real.yaml --world-size 1
  run_train configs/b300/1gpu_mini_real.yaml 2>&1 | tee logs/b300_1gpu_D_real.log
  echo "D PASS"
}

gate_E() {
  echo "==> E1) Soup short train + trigger-decay"
  ensure_mini
  python scripts/data/prepare_train_sequence_pool.py \
    --config configs/b300/1gpu_soup.yaml --world-size 1
  run_train configs/b300/1gpu_soup.yaml 2>&1 | tee logs/b300_1gpu_E1_soup_train.log
  LATEST=$(ls -d checkpoints/b300_1gpu_soup/step_* 2>/dev/null | sort | tail -1 || true)
  if [[ -z "${LATEST}" ]]; then
    echo "ERROR: no soup checkpoints under checkpoints/b300_1gpu_soup" >&2
    exit 1
  fi
  run_train configs/b300/1gpu_soup.yaml --resume "$LATEST" --trigger-decay \
    2>&1 | tee logs/b300_1gpu_E1_soup_decay.log
  grep -E "Averaging|pre-decay soup|triggered decay|No stable" logs/b300_1gpu_E1_soup_decay.log | tail -20 || true

  echo "==> E2) LoRA smoke"
  run_train configs/b300/1gpu_lora.yaml 2>&1 | tee logs/b300_1gpu_E2_lora.log
  grep -Ei "lora|lm_loss|step=" logs/b300_1gpu_E2_lora.log | tail -20 || true

  echo "==> E3) DCP resume from real CPT (if present)"
  REAL_CKPT=$(ls -d checkpoints/b300_1gpu_mini_real/step_* 2>/dev/null | sort | head -1 || true)
  if [[ -n "${REAL_CKPT}" ]]; then
    run_train configs/b300/1gpu_mini_real.yaml --resume "$REAL_CKPT" \
      2>&1 | tee logs/b300_1gpu_E3_resume.log
  else
    echo "SKIP E3: no b300_1gpu_mini_real checkpoint (run GATE=D first)"
  fi
  echo "E PASS"
}

gate_F() {
  echo "==> F) Layer ladder (3 → 4 → 6 → 8), FA4+MXFP8 speed"
  # Allowed depths in prepare_pipeline_model.py: 3,4,6,8
  local layers_list="${LAYERS_LIST:-3 4 6 8}"
  for L in $layers_list; do
    echo "==> F) Building pipeline_model_L${L}"
    python scripts/data/prepare_pipeline_model.py \
      --output "./pipeline_model_L${L}" --num-layers "${L}"
    echo "==> F) Train L${L} (FA4+MXFP8)"
    KING_DIR="./pipeline_model_L${L}" OUTPUT_DIR="./checkpoints/b300_1gpu_L${L}" \
      TEUTONIC_ATTN=flash_attention_4 \
      run_train configs/b300/1gpu_speed_mxfp8.yaml \
      2>&1 | tee "logs/b300_1gpu_F_L${L}.log"
    rg -n 'tokens/s=|param_count|num_hidden_layers|Saved|Error|ChildFailed' \
      "logs/b300_1gpu_F_L${L}.log" | tail -30 || true
  done
  echo "F PASS — fill TEST.md layer table from logs/b300_1gpu_F_L*.log"
}

gate_download() {
  echo "==> G download) 110B king → ./king (foreground; use & yourself for bg)"
  mkdir -p logs
  python scripts/data/download_king.py --output ./king --workers "${DOWNLOAD_WORKERS:-8}" \
    2>&1 | tee logs/download_king.log
}

gate_G() {
  echo "==> G) 110B eval smoke"
  if [[ ! -f king/config.json && ! -f king/model.safetensors && ! -f king/model.safetensors.index.json ]]; then
    echo "ERROR: ./king missing — run GATE=download first" >&2
    exit 1
  fi
  python scripts/data/prepare_eval_cache.py --config configs/b300/1gpu_eval.yaml --workers 8 \
    2>&1 | tee logs/b300_1gpu_G_prefetch.log || true
  mkdir -p results
  run_train configs/b300/1gpu_eval.yaml --eval-only --eval-n "${EVAL_N:-48}" \
    --eval-output results/b300_110b_eval_smoke48.json \
    2>&1 | tee logs/b300_1gpu_G_eval.log
  echo "G PASS — see results/b300_110b_eval_smoke48.json"
}

case "$GATE" in
  all)
    gate_A_smoke
    gate_B
    gate_C
    gate_D
    gate_E
    gate_F
    echo "DONE core gates A–F. Run GATE=download then GATE=G for 110B eval."
    ;;
  A|A_smoke) gate_A_smoke ;;
  B) gate_B ;;
  C) gate_C ;;
  D) gate_D ;;
  E) gate_E ;;
  F) gate_F ;;
  download) gate_download ;;
  G) gate_G ;;
  *)
    echo "Unknown GATE=$GATE (all|A|B|C|D|E|F|download|G)" >&2
    exit 2
    ;;
esac
