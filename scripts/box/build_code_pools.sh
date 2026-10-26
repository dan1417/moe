#!/usr/bin/env bash
# CODE training data in case merges cannot beat KC2: CODE-competition mix (code 70 / math 15 / text 15), 200k x 2048
# tokens (= text_cpt_v1's size), plus a 2048-sequence validation pool disjoint from it.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
nice -n 10 python scripts/data/build_diverse_pool.py --competition code --n 200000 --block 32 --tag code_cpt_v1 \
  --out /data/teutonic/pools/code_cpt_v1 && \
nice -n 10 python scripts/data/build_diverse_pool.py --competition code --n 2048 --block 1 --tag code_val_v1 \
  --out /data/teutonic/pools/code_val_v1 --exclude-pool /data/teutonic/pools/code_cpt_v1
echo "POOLS DONE rc=$?"
