#!/usr/bin/env bash
# MATH training data for after the TEXT milestone: math_cpt_v1 (200k) is 48 steps at 4 096 seqs; four more 200k pools (config
# abfd40f3) from shards no MATH pool / the MATH val set uses. Waits for the TEXT pool build (shared R2 rate limit).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
until grep -q "TEXT POOLS 18-27 DONE" /data/teutonic/logs/pm2_build_text_pools4.log 2>/dev/null; do sleep 60; done
P=/data/teutonic/pools; EX="--exclude-pool $P/math_val_v1 --exclude-pool $P/math_cpt_v1"
for v in 2 3 4 5; do
  nice -n 10 python scripts/data/build_diverse_pool.py --competition math --n 200000 --block 32 --tag math_cpt_v$v --out $P/math_cpt_v$v $EX || exit 1
  EX="$EX --exclude-pool $P/math_cpt_v$v"
done
echo "MATH POOLS 2-5 DONE"
