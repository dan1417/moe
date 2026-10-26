# Node environment for the 8xB300 pod: copy to /workspace/teutonic/env.sh (sourced by scripts/train/*.sh, scripts/eval/*.sh).
source /workspace/teutonic/.venv-cpt/bin/activate
export REPO_ROOT=/root/workspace/MoE
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TEUTONIC_EVAL_CACHE=/workspace/teutonic/eval_cache
export TEUTONIC_MODELS_DIRS=/dev/shm/teutonic/models
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=/workspace/teutonic/fa4_cache
export HF_HOME=/workspace/hf
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$TEUTONIC_EVAL_CACHE" "$TEUTONIC_MODELS_DIRS" "$FLASH_ATTENTION_CUTE_DSL_CACHE_DIR" /workspace/teutonic/logs /workspace/teutonic/runs /workspace/teutonic/pools
