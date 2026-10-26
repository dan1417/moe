#!/usr/bin/env bash
# Parity-eval environment: the validator's own package + its transformers fork.
#
#   bash scripts/setup/setup_eval_venv.sh        # creates .venv-eval
#
# Reuses the system torch (needs a CUDA build with torch._grouped_mm, i.e. >= 2.8;
# tested with 2.12.0+cu130 on RTX 5090). Pins must match src/parity/upstream.py.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
cd "$REPO_ROOT"

VALIDATOR_COMMIT=5118c326a10fde4bdf485e1963c0f09f9c759645
TRANSFORMERS_FORK="git+https://github.com/tony-dendrite/transformers.git@9e8a1817d44502602b5b50e09b2a5ab0271bdb39"

python3 -c "import torch; assert hasattr(torch, '_grouped_mm'), 'torch too old'; print('torch', torch.__version__)"
python3 -m venv --system-site-packages .venv-eval
# shellcheck disable=SC1091
source .venv-eval/bin/activate
pip install -q --upgrade pip
pip install -q "transformers @ ${TRANSFORMERS_FORK}" accelerate safetensors requests pyyaml tqdm httpx fastapi pydantic
pip install -q --no-deps "git+https://github.com/unarbos/teutonic.git@${VALIDATOR_COMMIT}"
python - <<'PY'
import transformers
from src.parity import upstream
print("transformers", transformers.__version__, "| validator", upstream.UPSTREAM_COMMIT[:12], "| ok")
PY
