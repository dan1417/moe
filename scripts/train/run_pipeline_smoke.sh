#!/usr/bin/env bash
# Build the depth-shrunk full-width MiMo-V2 king (~8.6B, 4 layers × 110B width)
# and run a short LoRA smoke train (1-GPU baseline).
# For 8×4090 full CPT (NCCL + FSDP shard + LoRA off), use:
#   bash scripts/train/run_4090_mini_cpt.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -z "$PYTHON" && -x "/root/workspace/SN3Teutonic/training_v4/.venv/bin/python" ]]; then
  PYTHON="/root/workspace/SN3Teutonic/training_v4/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

NPROC="${NPROC:-1}"
# Cap by training.total_steps in pipeline_smoke.yaml (default 10).
STEPS="${STEPS:-10}"

echo "==> Building depth-shrink full-width pipeline mini model"
"$PYTHON" scripts/data/prepare_pipeline_model.py --output "$ROOT/pipeline_model"

echo "==> Design verifier (unit checks)"
"$PYTHON" scripts/verify/verify_pipeline_design.py --config configs/smoke/pipeline.yaml --king-dir pipeline_model

echo "==> LoRA smoke train (request smoke_steps=$STEPS, nproc=$NPROC)"
export NPROC
export TEUTONIC_ATTN="${TEUTONIC_ATTN:-eager}"
# Single-GPU smoke: gloo avoids NCCL/NVML issues. Multi-GPU: prefer run_4090_mini_cpt.sh
export TEUTONIC_DIST_BACKEND="${TEUTONIC_DIST_BACKEND:-gloo}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CFG="$ROOT/checkpoints/pipeline_smoke/runtime_config.yaml"
mkdir -p "$ROOT/checkpoints/pipeline_smoke"
"$PYTHON" - <<PY
import yaml
from pathlib import Path
root = Path("$ROOT")
cfg = yaml.safe_load((root / "configs/smoke/pipeline.yaml").read_text())
cfg["training"]["smoke_steps"] = int("$STEPS")
# Ensure total_steps does not silently cap below smoke_steps
cfg["training"]["total_steps"] = max(int(cfg["training"].get("total_steps", 10)), int("$STEPS"))
(root / "checkpoints/pipeline_smoke/runtime_config.yaml").write_text(yaml.dump(cfg))
print(f"effective total_steps={cfg['training']['total_steps']} smoke_steps={cfg['training']['smoke_steps']}")
PY

"$PYTHON" -m torch.distributed.run --nproc_per_node="$NPROC" -m src.train --config "$CFG"

echo "==> Pipeline smoke complete (see TEST.md). Next: bash scripts/train/run_4090_mini_cpt.sh"
