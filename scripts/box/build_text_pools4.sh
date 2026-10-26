#!/usr/bin/env bash
# More fresh TEXT data for continued TEXT training (user 01:05 JST Oct 12: keep training TEXT after c6). c6 ends near
# sequence 604 160 of v1 + v12-v14 (800 000); three more 200k pools from shards no TEXT pool or the val set uses.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
P=/data/teutonic/pools
EX="--exclude-pool $P/text_val_v1"; for v in 1 10 11 12 13 14 15 16 17; do EX="$EX --exclude-pool $P/text_cpt_v$v"; done
for v in 18 19 20 21 22 23 24 25 26 27; do
  nice -n 10 python scripts/data/build_diverse_pool.py --competition text --n 200000 --block 32 --tag text_cpt_v$v --out $P/text_cpt_v$v $EX || exit 1
  EX="$EX --exclude-pool $P/text_cpt_v$v"
done
echo "TEXT POOLS 18-27 DONE"
