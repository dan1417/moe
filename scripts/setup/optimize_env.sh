#!/usr/bin/env bash
# Thin wrapper: workload-aware CPT optimizer (see docs/ML_ENV_GUIDE.md).
#
# Usage:
#   bash scripts/setup/optimize_env.sh diagnose
#   bash scripts/setup/optimize_env.sh plan
#   bash scripts/setup/optimize_env.sh all --fast
#   bash scripts/setup/optimize_env.sh all --fast --root ml-env-cpt2
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
cd "$REPO_ROOT"

STAGE="${1:-diagnose}"
shift || true
exec python3 scripts/setup/optimize_ml_env.py --stage "$STAGE" "$@"
