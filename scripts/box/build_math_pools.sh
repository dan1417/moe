#!/usr/bin/env bash
# MATH training data (MATH crown vacant: challengers face #43 on the MATH sample, bar 0.002), same sizes as the CODE pools.

set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
nice -n 10 python scripts/data/build_diverse_pool.py --competition math --n 200000 --block 32 --tag math_cpt_v1 \
  --out /data/teutonic/pools/math_cpt_v1 && \
nice -n 10 python scripts/data/build_diverse_pool.py --competition math --n 2048 --block 1 --tag math_val_v1 \
  --out /data/teutonic/pools/math_val_v1 --exclude-pool /data/teutonic/pools/math_cpt_v1
echo "POOLS DONE rc=$?"
