#!/usr/bin/env bash
# User (04:37): "once the best CODE model is evaluated and the best trainable merge is prepared, start training; train until
# my model beats the current MAIN king (#43, bar 0.003)". Usage: main_train.sh <recipe json>
#   1) write the start checkpoint from the recipe (CPU, bit-identical to the scored merge)
#   2) chain_runs --no-scorer (every checkpoint backed up to HF) with c1's recipe on the MAIN pools, 600 steps
set -euo pipefail
RECIPE=${1:?recipe json}
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
K43=/data/teutonic/models/9b202f1afae9c3575cdf6a57a37326aeb0a8ab8704658e3b64224bee72c537da
RUN=main_upper_ema_m1; START=$D/ckpt/start_m1
if [[ ! -f $START/model.safetensors.index.json ]]; then  # written while the last pools build
  echo "$(date -u +%H:%M:%S) writing the start checkpoint from $RECIPE"
  nice -n 5 python scripts/merge/write_merge.py --recipe "$RECIPE" --out $START --gpus cpu --workers 32
fi
until grep -q "MORE POOLS DONE" $D/logs/pm2_build_main_pools_more.log; do echo "$(date -u +%H:%M:%S) waiting for the MAIN pools"; sleep 20; done
POOLS=$(ls -d $D/pools/main_cpt_v* | sort -V | paste -sd,)
EXTRA="--freeze \\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 1.5e-6 --lr-dense-frac 2.0 --warmup 20 --min-lr-frac 1.0 --ema 0.98 --val-raw 0 --contract-dir $T1 --save-every 25 --eval-every 25 --steps 600 --pool $POOLS --val $D/pools/main_val_v1"
echo "$(date -u +%H:%M:%S) starting $RUN from $START on $POOLS"
exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool 10 --batches 16x16 \
  --min-tok-s 0 --extra-args "$EXTRA" --king $K43 --max-runs 1
