#!/usr/bin/env bash
# Fresh TEXT training data (user 08:00 JST: "train for TEXT and MAIN"): c1 already used text_cpt_v10 + v11 in full, so three
# new 200k-sequence pools from shards not used by any TEXT pool or the val set (TEXT config 0a2840c9: text / math / code).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
P=/data/teutonic/pools
EX="--exclude-pool $P/text_val_v1 --exclude-pool $P/text_cpt_v1 --exclude-pool $P/text_cpt_v10 --exclude-pool $P/text_cpt_v11"
for v in 12 13 14; do
  nice -n 10 python scripts/data/build_diverse_pool.py --competition text --n 200000 --block 32 --tag text_cpt_v$v --out $P/text_cpt_v$v $EX || exit 1
  EX="$EX --exclude-pool $P/text_cpt_v$v"
done
echo "TEXT POOLS DONE"
