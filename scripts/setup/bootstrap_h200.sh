#!/usr/bin/env bash
# Fresh 8-GPU box (8x H200 141 GB, also fine on 8x H100/B200) -> same TEXT LoRA campaign as the H100 box, continued from
# the best checkpoint on Hugging Face. Idempotent: re-running skips finished steps.
#
#   echo -n hf_xxx > /data/teutonic/hf_token && chmod 600 /data/teutonic/hf_token        # HF token with read access
#   START_REPO=yamalies/<repo> bash scripts/setup/bootstrap_h200.sh                         # repo printed by the H100 box
#
# Env: START_REPO (required), MODE (full_ema = B300/B200 EMA recipe of 2026-10-09 | full_all = B200/B300 whole model |
# full = H200 upper-half experts | lora, default full), RUN, BATCHES (per-GPU batch x accum, same global batch for
# every entry; the driver falls back down the list on OOM/slow), FIRST_POOL (default 3: v1/v2 were used on
# the H100 box and are rebuilt here only so their shards are excluded).
set -euo pipefail
START_REPO=${START_REPO:?set START_REPO=<hf repo id of the start checkpoint>}
MODE=${MODE:-full}
if [[ $MODE == full_ema ]]; then FIRST_POOL=${FIRST_POOL:-10}; else FIRST_POOL=${FIRST_POOL:-3}; fi
# MODE=full: full-parameter training of the routed experts of layers 23-44 with fp32 master weights (what the kings'
#   own steps look like; ~121 GB/GPU estimated: fp32 weights 56 + grads of the trained layers 27 + 8-bit Adam 13 + activations).
# MODE=lora: the H100 LoRA recipe with bigger per-GPU batches (faster, but it saturated near +0.0021 on the H100 box).
D=/data/teutonic
if [[ $MODE == full_ema ]]; then
  # B300/B200 (2026-10-09 recipe, 42.md "text_full_ema_a"): every parameter except the routers, fp32 masters, 8-bit AdamW
  # with states offloaded to host RAM, experts lr 1.5e-6 / dense 3e-6 (constant after 20 warmup steps), 2 048 sequences =
  # 4.2M tokens per step (gradient noise scale is ~10M tokens: smaller steps are mostly noise), fp32 weight EMA 0.98/step
  # in host RAM (val + checkpoints = EMA weights; raw weights jitter ±0.0003 on the screen), checkpoint + screen every
  # 25 steps, 200 steps over two fresh pools. Needs ~1.1 TB host RAM (EMA 440 GB + Adam 210 GB pinned + loading).
  RUN=${RUN:-text_full_ema_b1}; BATCHES=${BATCHES:-16x16,8x32,4x64}  # B300 288 GB: 16 fits (B200 measured 152 GiB allocated at 16 + offload)
  EXTRA="--lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 1.5e-6 --lr-dense-frac 2.0 --warmup 20 --min-lr-frac 1.0 --ema 0.98 --save-every 25 --eval-every 25 --steps 200 --pool $D/pools/text_cpt_v$FIRST_POOL,$D/pools/text_cpt_v$((FIRST_POOL + 1))"
  MIN_TOK=30  # a bigger per-GPU batch must beat 30k tok/s at step 1 (B200 batch 4 + offload: 26k), else fall back
elif [[ $MODE == full_all ]]; then
  # B200/B300 (>= 180 GB per GPU): every parameter except the routers, fp32 master weights, 8-bit AdamW, lr 1e-5.
  # ~140 GB/GPU for weights 56 + grads 56 + optimizer 28, plus activations; host RAM >= 600 GB (rank 0 loads one fp32 copy).
  RUN=${RUN:-text_full_b1}; BATCHES=${BATCHES:-4x32,2x64,1x128}
  EXTRA="--lora-r 0 --dtype fp32 --optim adamw8bit --lr 1e-5 --warmup 20"
  MIN_TOK=0
elif [[ $MODE == full ]]; then
  RUN=${RUN:-text_full_h1}; BATCHES=${BATCHES:-4x32,2x64,1x128}
  EXTRA="--lora-r 0 --dtype fp32 --optim adamw8bit --lr 1e-5 --warmup 20 --train-only \\.layers\\.(2[3-9]|3[0-9]|4[0-4])\\.mlp\\.experts\\."
  MIN_TOK=0
else
  RUN=${RUN:-text_lora_h1}; BATCHES=${BATCHES:-16x8,8x16,4x32}; EXTRA=""; MIN_TOK=17
fi
T1_DIGEST=95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO=$PWD; T1=$D/models/$T1_DIGEST; START=$D/ckpt/start_${START_REPO##*/}
step() { echo "=== $(TZ=Asia/Tokyo date +%H:%M) JST: $*"; }
[[ -s $D/hf_token ]] || { echo "missing $D/hf_token"; exit 1; }
mkdir -p $D/{models,ckpt,pools,logs,eval_cache/manifests_split,www,hf_home}

step "environment"
python3 -c "import torch; assert hasattr(torch, '_grouped_mm')" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu130
[[ -x .venv-eval/bin/python ]] || bash scripts/setup/setup_eval_venv.sh
.venv-eval/bin/pip install -q bitsandbytes huggingface_hub hf_transfer >/dev/null
command -v pm2 >/dev/null || npm install -g pm2
cat > $D/env.sh <<ENV
cd $REPO && source .venv-eval/bin/activate
export TEUTONIC_EVAL_CACHE=$D/eval_cache TEUTONIC_MODELS_DIRS=$D/models PYTHONUNBUFFERED=1 HF_HOME=$D/hf_home
export T1=$T1 START=$START TRITON_CACHE_DIR=$D/cache/triton TORCHINDUCTOR_CACHE_DIR=$D/cache/inductor
ENV
source $D/env.sh
python -c "import torch;a=torch.arange(10,device='cuda:0');assert torch.equal(a.to('cuda:1').cpu(),a.cpu()),'P2P broken';print(torch.cuda.device_count(),'GPUs,',torch.cuda.get_device_name(0))"

step "start checkpoint $START_REPO"
[[ -f $START/model.safetensors.index.json ]] || python - "$START_REPO" "$START" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], local_dir=sys.argv[2], max_workers=16, token=open("/data/teutonic/hf_token").read().strip()))
PY

step "eval samples + split manifests"
python scripts/eval/parity_eval.py --competition mix sample --sample tag:A
python scripts/eval/parity_eval.py --competition text sample --sample tag:A --sample tag:B
for s in math code text; do [[ -s $D/eval_cache/manifests_split/$s.json ]] || \
  curl -s -A teutonic-cpt-pool/1.0 -o $D/eval_cache/manifests_split/$s.json "https://pub-d923bc4e8fcb45f6b703bc750bcf8aa6.r2.dev/splits/$s/manifest.json"; done

step "pools (v1 + val rebuilt deterministically; earlier pools so the chain excludes their shards)"
[[ -f $D/pools/text_cpt_v1/meta.json ]] || bash scripts/data/build_text_pool.sh text_cpt_v1
[[ -f $D/pools/text_val_v1/meta.json ]] || .venv-eval/bin/python scripts/data/build_diverse_pool.py --out $D/pools/text_val_v1 \
    --n 2048 --block 1 --tag text_val_v1 --competition text --workers 48 --exclude-pool $D/pools/text_cpt_v1
if [[ $MODE == full_ema ]]; then
  # only the two pools this run reads (fresh tags; a random 32-sequence block per shard makes overlap with the old box's
  # pools negligible, so v2..v9 are not rebuilt) - ~4 min each
  for k in $FIRST_POOL $((FIRST_POOL + 1)); do
    [[ -f $D/pools/text_cpt_v$k/meta.json ]] || bash scripts/data/build_text_pool.sh text_cpt_v$k $(ls -d $D/pools/text_cpt_v* | grep -v "_v$k$") $D/pools/text_val_v1
  done
else
  for k in $(seq 2 $((FIRST_POOL - 1))); do
    [[ -f $D/pools/text_cpt_v$k/meta.json ]] || bash scripts/data/build_text_pool.sh text_cpt_v$k $(ls -d $D/pools/text_cpt_v* | grep -v "_v$k$") $D/pools/text_val_v1
  done
fi

step "T1 (TEXT king) + its screen losses"
[[ -f $T1/model.safetensors.index.json ]] || python scripts/data/download_king.py --digest $T1_DIGEST --output $T1 --workers 32
python scripts/eval/parity_eval.py --competition text --limit 2400 score --model $T1 --sample tag:A

step "dashboard (localhost:8787) + chain driver ($RUN from $START, batches $BATCHES)"
cp scripts/train/render_dashboard.sh $D/logs/render_dashboard.sh
sed -i -e "s|^RUN=.*|RUN=$D/ckpt/$RUN|" -e "s|^cd .*|cd $REPO|" $D/logs/render_dashboard.sh
pm2 delete dashboard_render dashboard_http >/dev/null 2>&1 || true
pm2 start $D/logs/render_dashboard.sh --name dashboard_render --time --merge-logs --output $D/logs/pm2_dashboard_render.log
# tolerant server: a noisy port-forward glued \x09+NUL bytes before request lines and `http.server` answered 400/501
pm2 start $REPO/.venv-eval/bin/python3 --name dashboard_http --time -- $REPO/scripts/train/dashboard_server.py --dir $D/www --port 8787
pm2 delete chain_text >/dev/null 2>&1 || true
pm2 start bash --name chain_text --no-autorestart --time --merge-logs --output $D/logs/pm2_chain_text.log -- \
  -c "source $D/env.sh && exec python scripts/train/chain_runs.py --start $RUN --init-ckpt $START --first-pool $FIRST_POOL --batches $BATCHES --min-tok-s $MIN_TOK --extra-args '$EXTRA' --king $T1"
pm2 save
step "running: pm2 logs chain_text | pm2 logs $RUN | dashboard http://localhost:8787"
