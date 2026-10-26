#!/usr/bin/env bash
# User (14:35): after the failed 3x LR probe (m2 step 50 MAIN +0.002380 vs start +0.002408), restart MAIN training from
# m1/step-175 at the old LR (1.5e-6 / 3e-6) with the weakest MAIN sources vs #43 upweighted in the loss: code-reasoning
# (+0.0010) and sft-reasoning (+0.0013) x2; same 4.2M-token steps, EMA 0.98, warmup 10 (fresh optimizer), m1 data order.
set -euo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
K43=/data/teutonic/models/9b202f1afae9c3575cdf6a57a37326aeb0a8ab8704658e3b64224bee72c537da
RUN=main_upper_ema_m3; START=$D/ckpt/main_upper_ema_m1/step-175
POOLS=$(ls -d $D/pools/main_cpt_v* | sort -V | paste -sd,)
EXTRA="--freeze \\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 1.5e-6 --lr-dense-frac 2.0 --warmup 10 --min-lr-frac 1.0 --ema 0.98 --val-raw 0 --contract-dir $T1 --save-every 25 --eval-every 25 --steps 425 --pool $POOLS --val $D/pools/main_val_v1 --seed 0 --skip-seqs 358400 --loss-weights code-reasoning=2,sft-reasoning=2"
echo "$(date -u +%H:%M:%S) starting $RUN from $START (old LR, code-reasoning + sft-reasoning x2)"
exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool 10 --batches 16x16 \
  --min-tok-s 0 --extra-args "$EXTRA" --king $K43 --max-runs 1
