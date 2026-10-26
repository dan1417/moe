#!/usr/bin/env bash
# B300 box, 2026-10-09: TEXT campaign, MODE=full_ema (docs/START_TRAINING_H200.md) from yamalies/a7745d17833c
# (= B200 text_full_1e6b/step-50, +0.002161 vs T1 on the pre-masking screen). Same as the last step of
# scripts/setup/bootstrap_h200.sh; the trainer now also uses --doc-mask 1 (validator 14a9bb2 scoring) by default.
set -euo pipefail
D=/data/teutonic
source $D/env.sh
# upper-layer split of the fast B300 lines (A 40k, C1 72k tok/s; the kings' own steps): embeddings, layers 0-15 and
# routers frozen -> no backward below layer 16, no grads/optimizer state for 38B params -> bigger micro-batches fit
# measured here: upper batch 16 = 34.0k tok/s (135.8 GB), batch 32 = 30.9k, 64 = 17.1k (memory ceiling); full-param 16 = 32.1k
RUN=${RUN:-text_upper_ema_c1}; FIRST_POOL=10; BATCHES=${BATCHES:-16x16}; MIN_TOK=0
EXTRA="--freeze \\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 1.5e-6 --lr-dense-frac 2.0 --warmup 20 --min-lr-frac 1.0 --ema 0.98 --val-raw 0 --contract-dir $T1 --save-every 25 --eval-every 25 --steps 200 --pool $D/pools/text_cpt_v10,$D/pools/text_cpt_v11"
pm2 delete chain_text >/dev/null 2>&1 || true
pm2 start bash --name chain_text --no-autorestart --time --merge-logs --output $D/logs/pm2_chain_text.log -- \
  -c "source $D/env.sh && exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool $FIRST_POOL --batches $BATCHES --min-tok-s $MIN_TOK --extra-args '$EXTRA' --king $T1"
pm2 save >/dev/null
echo "chain_text started: pm2 logs chain_text | pm2 logs $RUN | dashboard http://127.0.0.1:8787"
