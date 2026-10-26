#!/usr/bin/env bash
# More MAIN training data for a long run (user: train until we beat #43 on MAIN): main_cpt_v2..v6, 200k x 2048 tokens each,
# disjoint from each other and from main_cpt_v1 / main_val_v1 (--exclude-pool), two builds at a time.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
P=/data/teutonic/pools
until grep -q "POOLS DONE" /data/teutonic/logs/pm2_build_main_pools.log; do sleep 10; done
build() { local v=$1; shift; nice -n 10 python scripts/data/build_diverse_pool.py --competition main --n 200000 --block 32 \
  --tag main_cpt_v$v --out $P/main_cpt_v$v --exclude-pool $P/main_cpt_v1 --exclude-pool $P/main_val_v1 "$@"; }
build 2 & build 3 & wait
build 4 --exclude-pool $P/main_cpt_v2 --exclude-pool $P/main_cpt_v3 & build 5 --exclude-pool $P/main_cpt_v2 --exclude-pool $P/main_cpt_v3 & wait
build 6 --exclude-pool $P/main_cpt_v2 --exclude-pool $P/main_cpt_v3 --exclude-pool $P/main_cpt_v4 --exclude-pool $P/main_cpt_v5
ls -d $P/main_cpt_v* ; echo "MORE POOLS DONE"
