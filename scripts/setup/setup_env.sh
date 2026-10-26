#!/usr/bin/env bash
# Hardware-aware Teutonic MoE environment setup.
#
# 1) Diagnose GPUs via GPU_check bootstrap_ml policy
# 2) Plan Teutonic stack (FA4/TE pins for MiMo-V2)
# 3) Run the matching setup_*_venv.sh
# 4) Verify with check_env.sh
#
# Usage:
#   bash scripts/setup/setup_env.sh
#   bash scripts/setup/setup_env.sh --diagnose-only
#   bash scripts/setup/setup_env.sh --profile b300
#   ALLOW_BUSY=1 bash scripts/setup/setup_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

PROFILE="${PROFILE:-auto}"
DIAGNOSE_ONLY=0
ALLOW_BUSY="${ALLOW_BUSY:-0}"
SKIP_MINI_BUILD="${SKIP_MINI_BUILD:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --diagnose-only|--diagnose)
      DIAGNOSE_ONLY=1
      shift
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --allow-busy)
      ALLOW_BUSY=1
      shift
      ;;
    --skip-mini-build)
      SKIP_MINI_BUILD=1
      shift
      ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p logs reports
PYTHON="${PYTHON:-python3}"

echo "==> 1) Hardware diagnose (bootstrap_ml + Teutonic plan)"
DIAG_ARGS=(scripts/setup/teutonic_stack.py --diagnose --write logs/teutonic_env_diagnose.json)
PLAN_ARGS=(scripts/setup/teutonic_stack.py --json --write logs/teutonic_env_plan.json)
if [[ "$ALLOW_BUSY" == "1" ]]; then
  DIAG_ARGS+=(--allow-busy)
  PLAN_ARGS+=(--allow-busy)
fi
"$PYTHON" "${DIAG_ARGS[@]}"
"$PYTHON" "${PLAN_ARGS[@]}"

if [[ "$DIAGNOSE_ONLY" == "1" ]]; then
  echo "Diagnose-only done. See logs/teutonic_env_diagnose.json and logs/teutonic_env_plan.json"
  exit 0
fi

SETUP_SCRIPT=$(
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
plan = json.loads(Path("logs/teutonic_env_plan.json").read_text())
print(plan["setup_script"])
PY
)
echo "    planned profile from logs/teutonic_env_plan.json"

if [[ "$PROFILE" == "b300" ]]; then
  SETUP_SCRIPT="scripts/setup/setup_b300_venv.sh"
elif [[ "$PROFILE" == "4090" || "$PROFILE" == "pipeline" ]]; then
  SETUP_SCRIPT="scripts/setup/setup_pipeline_venv.sh"
elif [[ "$PROFILE" != "auto" ]]; then
  echo "Unknown --profile $PROFILE (auto|b300|4090|pipeline)" >&2
  exit 2
fi

echo "==> 2) Installing via $SETUP_SCRIPT"
export SKIP_MINI_BUILD
if [[ "$ALLOW_BUSY" == "1" ]]; then
  export ALLOW_BUSY=1
fi
bash "$SETUP_SCRIPT"

echo "==> 3) Post-install check_env"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
bash scripts/setup/check_env.sh | tee logs/check_env.log

echo ""
echo "Environment ready."
echo "  activate: source .venv/bin/activate"
echo "  plan:     logs/teutonic_env_plan.json"
echo "  guide:    docs/ML_ENV_GUIDE.md"
echo "  next:     bash scripts/train/run_b300_1gpu_tests.sh   # or 4090 runners"
