#!/usr/bin/env bash
# User (12:30): 3x LR probe for MAIN. main_upper_ema_m1 plateaued on MAIN vs #43 (+0.002364 @100, +0.002397 @150,
# +0.002408 @175). Resume from m1/step-175 with experts 4.5e-6 / dense 9e-6 (3x), short warmup (fresh optimizer state),
# same 4.2M-token steps and EMA 0.98, continuing m1's data order (seed 0, skip the 175 x 2 048 sequences already used).
set -euo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
K43=/data/teutonic/models/9b202f1afae9c3575cdf6a57a37326aeb0a8ab8704658e3b64224bee72c537da
RUN=main_upper_ema_m2; START=$D/ckpt/main_upper_ema_m1/step-175
POOLS=$(ls -d $D/pools/main_cpt_v* | sort -V | paste -sd,)
EXTRA="--freeze \\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 4.5e-6 --lr-dense-frac 2.0 --warmup 10 --min-lr-frac 1.0 --ema 0.98 --val-raw 0 --contract-dir $T1 --save-every 25 --eval-every 25 --steps 425 --pool $POOLS --val $D/pools/main_val_v1 --seed 0 --skip-seqs 358400"
echo "$(date -u +%H:%M:%S) starting $RUN from $START (3x LR)"
exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool 10 --batches 16x16 \
  --min-tok-s 0 --extra-args "$EXTRA" --king $K43 --max-runs 1
