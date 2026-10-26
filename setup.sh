#!/usr/bin/env bash
# setup.sh — prepare a fresh GPU box for Teutonic merging and training (run once; safe to re-run).
#
#   HF_TOKEN=hf_xxx bash setup.sh
#
# Then:  bash download.sh 10      (latest 10 public kings + the reigning ones, keeps watching for new kings)
#        bash merge.sh            (merge campaign over everything downloaded; finds the best model per competition)
#        bash train.sh <merge label | checkpoint dir | model name> <main|math|code|text>
#        bash dashboard.sh        (http://<box>:8787)
#
# What it does:
#   1 repo at /root/workspace/MoE (symlinked if cloned elsewhere: the helper scripts use that path)
#   2 .venv-eval: CUDA 13 torch (the image's if new enough, else from the cu130 index) + the validator's transformers fork
#     and package (pins from scripts/setup/setup_eval_venv.sh) + flash-attn-4 + bitsandbytes + hf_transfer
#   3 /data/teutonic layout + env.sh, helper scripts copied to /data/teutonic/logs (training launcher, guard, watchers)
#   4 HF token (/data/teutonic/hf_token; uploads go to HF_UPLOAD_OWNER, default Valhalla-ai) and pm2
# Env: DATA (default /data/teutonic), HF_UPLOAD_OWNER, SKIP_VENV=1.
# The full hardware-aware stack (FA4 / Transformer Engine builds, diagnostics) is scripts/setup/setup_env.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
D=${DATA:-/data/teutonic}
log() { echo "$(date -u +%H:%M:%S) [setup] $*"; }

# 1 repo path ------------------------------------------------------------------------------------------------------
if [[ $ROOT != /root/workspace/MoE ]]; then
  if [[ ! -e /root/workspace/MoE ]]; then mkdir -p /root/workspace && ln -s "$ROOT" /root/workspace/MoE && log "linked /root/workspace/MoE -> $ROOT"
  else log "WARNING: /root/workspace/MoE exists and is not this checkout ($ROOT); the helper scripts use /root/workspace/MoE"; fi
fi

# 2 venv -----------------------------------------------------------------------------------------------------------
if [[ ${SKIP_VENV:-0} != 1 ]] && ! .venv-eval/bin/python -c "import torch, transformers, flash_attn.cute, teutonic" 2>/dev/null; then
  log "creating .venv-eval"
  VALIDATOR_COMMIT=$(grep -o 'VALIDATOR_COMMIT=[0-9a-f]*' scripts/setup/setup_eval_venv.sh | cut -d= -f2)
  TRANSFORMERS_FORK=$(grep -o 'TRANSFORMERS_FORK="[^"]*"' scripts/setup/setup_eval_venv.sh | cut -d'"' -f2)
  if python3 -c "import torch; assert hasattr(torch, '_grouped_mm') and torch.version.cuda and int(torch.version.cuda.split('.')[0]) >= 13" 2>/dev/null; then
    python3 -m venv --system-site-packages .venv-eval          # reuse the image's CUDA 13 torch
    source .venv-eval/bin/activate
  else
    python3 -m venv .venv-eval
    source .venv-eval/bin/activate
    pip install -q -U pip wheel
    pip install -q torch --index-url https://download.pytorch.org/whl/cu130   # B300 = sm_103 needs CUDA 13
  fi
  pip install -q -U pip
  pip install -q "transformers @ ${TRANSFORMERS_FORK}" accelerate safetensors requests pyyaml tqdm httpx fastapi pydantic numpy
  pip install -q --no-deps "git+https://github.com/unarbos/teutonic.git@${VALIDATOR_COMMIT}"
  pip install -q --pre "flash-attn-4[cu13]"
  pip install -q bitsandbytes hf_transfer huggingface_hub
fi
.venv-eval/bin/python -c "import torch, transformers; from transformers.utils import is_flash_attn_4_available as fa4; print('torch', torch.__version__, '| transformers', transformers.__version__, '| FA4', fa4(), '| GPUs', torch.cuda.device_count())"

# 3 layout, env.sh, helper scripts ------------------------------------------------------------------------------------
mkdir -p "$D"/{models,ckpt,pools,logs,eval_cache,submit,search,vec,www,hf_home,cache/triton,cache/inductor}
if [[ ! -f $D/env.sh ]]; then
  cat > "$D/env.sh" <<EOF
cd /root/workspace/MoE && source .venv-eval/bin/activate
export TEUTONIC_EVAL_CACHE=$D/eval_cache TEUTONIC_MODELS_DIRS=$D/models PYTHONUNBUFFERED=1 HF_HOME=$D/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export T1=$D/models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0
export TRITON_CACHE_DIR=$D/cache/triton TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor
EOF
  log "wrote $D/env.sh"
fi
for f in text_run.sh main_watch.sh ckpt_retention.sh train_guard.py king_args.py render_dashboard.sh; do
  cp -f "scripts/box/$f" "$D/logs/$f"     # the training launcher / guard / watcher read their own copies from here
done
[[ -f configs/merge/k41/models.json ]] || echo '{}' > configs/merge/k41/models.json

# 4 token, upload target, pm2 --------------------------------------------------------------------------------------
if [[ -n ${HF_TOKEN:-} ]]; then umask 077; echo -n "$HF_TOKEN" > "$D/hf_token"; umask 022; log "HF token saved"; fi
[[ -s $D/hf_token ]] || log "WARNING: no HF token (HF_TOKEN=hf_... bash setup.sh): gated checkpoints and uploads will fail"
[[ -f $D/hf_upload_target.json ]] || echo "{\"owner\": \"${HF_UPLOAD_OWNER:-Valhalla-ai}\"}" > "$D/hf_upload_target.json"
command -v pm2 >/dev/null || { command -v npm >/dev/null || { apt-get update -qq && apt-get install -y -qq nodejs npm; }; npm install -g pm2; }
log "done. Next: bash download.sh 10   then   bash merge.sh   /   bash train.sh <label> <competition>   /   bash dashboard.sh"
