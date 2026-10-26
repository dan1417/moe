#!/usr/bin/env bash
# One-shot environment for src/cpt on a rented 8×B300 node (bf16 CPT; no Transformer Engine).
# Pins match the validator's evaluator: transformers fork + flash-attn-4 4.0.0b28.
#
#   bash scripts/setup/setup_cpt_b300.sh          # creates .venv-cpt
#   source .venv-cpt/bin/activate
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
cd "$REPO_ROOT"

VALIDATOR_COMMIT=5479810a8c8b8635ede24e5be93515ffa9db4c6f
TRANSFORMERS_FORK="git+https://github.com/tony-dendrite/transformers.git@9e8a1817d44502602b5b50e09b2a5ab0271bdb39"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"

python3 -m venv .venv-cpt
# shellcheck disable=SC1091
source .venv-cpt/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu130
pip install -q "transformers @ ${TRANSFORMERS_FORK}" "flash-attn-4==4.0.0b28" \
    accelerate safetensors pyyaml requests tqdm numpy httpx fastapi pydantic
pip install -q --no-deps "git+https://github.com/unarbos/teutonic.git@${VALIDATOR_COMMIT}"

python - <<'PY'
import torch, transformers
from transformers.utils import is_flash_attn_4_available
assert torch.cuda.is_available(), "no CUDA"
major, minor = torch.cuda.get_device_capability(0)
print(f"torch {torch.__version__} | transformers {transformers.__version__} | sm_{major}{minor} x{torch.cuda.device_count()}")
assert hasattr(torch, "_grouped_mm"), "torch._grouped_mm missing"
assert is_flash_attn_4_available(), "flash-attn-4 not visible to transformers"
from flash_attn.cute import flash_attn_func
q = torch.randn(1, 256, 48, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(1, 256, 8, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(1, 256, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
sink = torch.randn(48, device="cuda", dtype=torch.float32)
out = flash_attn_func(q, k, v, causal=True, window_size=(127, 127), learnable_sink=sink)
out = out[0] if isinstance(out, tuple) else out
out.float().sum().backward()
print("FA4 fwd+bwd with sinks + asymmetric V: OK")
PY
echo "ready: source .venv-cpt/bin/activate; see docs/CPT.md (runbook) and scripts/train/launch_cpt.sh"
