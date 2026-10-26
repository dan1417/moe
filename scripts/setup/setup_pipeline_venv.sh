#!/usr/bin/env bash
# Minimal venv for depth-shrunk full-width MiMo-V2 pipeline tests
# (1-GPU smoke + 8×4090 full CPT on ~8.6B king).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
VENV="$ROOT/.venv"

if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv at $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Installing PyTorch (CUDA 12.8 wheels)"
pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing mini-pipeline dependencies"
pip install -r <(grep -vE '^(flash-attn|transformer-engine)' requirements.txt)

echo "==> Optional: flash-attn / FA4 (skip on failure)"
if [[ -f scripts/setup/install_flash_attn.sh ]]; then
  bash scripts/setup/install_flash_attn.sh || echo "flash-attn optional, skipping"
fi

echo "==> Sanity check imports + env probe"
REQUIRE_FA4=0 REQUIRE_TE=0 REQUIRE_MXFP8=0 bash scripts/setup/check_env.sh || {
  echo "WARN: check_env soft-failed on pipeline (4090) profile; verifying basic imports"
  python - <<'PY'
import torch, transformers, safetensors, yaml, tokenizers
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} ngpu={torch.cuda.device_count()}")
PY
}

echo "==> Building depth-shrink full-width MiMo-V2 king (~8.6B)"
python scripts/data/prepare_pipeline_model.py --output "$ROOT/pipeline_model"

echo "==> Running design verifier"
python scripts/verify/verify_pipeline_design.py \
  --config configs/smoke/pipeline.yaml \
  --king-dir pipeline_model

echo ""
echo "Venv ready. Activate with:"
echo "  source $VENV/bin/activate"
echo ""
echo "1-GPU LoRA smoke (gloo OK if NVML broken):"
echo "  bash scripts/train/run_pipeline_smoke.sh"
echo ""
echo "8-GPU mini full CPT (needs healthy nvidia-smi / NCCL):"
echo "  BRINGUP=1 bash scripts/train/run_4090_mini_cpt.sh   # 5-step NCCL/FSDP check"
echo "  bash scripts/train/run_4090_mini_cpt.sh             # 1000-step patterned CPT"
echo "  See docs/TEST.md for baseline results + pass criteria."
