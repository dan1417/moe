#!/usr/bin/env bash
# TEXT training pools for the autopilot (text / math / code 70/15/15 of the splits): four 200k-row window pools -> one
# long pool (4 consecutive rows joined = 8 192-token documents) -> one mix (half long rows, half 4-window rows).
#   bash scripts/box/build_text_pools.sh 1      # text_cpt_v1..v4 -> text_long_v1 -> text_mix_v1 (also text_val_v1)
#   bash scripts/box/build_text_pools.sh 2      # text_cpt_v5..v8 -> text_long_v2 -> text_mix_v2
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." && source /data/teutonic/env.sh
N=${1:-1}; P=/data/teutonic/pools
[[ -f $P/text_val_v1/meta.json ]] || python scripts/data/build_diverse_pool.py --competition text --n 2048 --block 1 --tag text_val_v1 --out $P/text_val_v1
EX=(--exclude-pool $P/text_val_v1)
for d in $P/text_cpt_v*; do [[ -f $d/meta.json ]] && EX+=(--exclude-pool $d); done
CSV=""
for i in $(seq $(( (N-1)*4 + 1 )) $(( N*4 ))); do
  if [[ ! -f $P/text_cpt_v$i/meta.json ]]; then
    python scripts/data/build_diverse_pool.py --competition text --n 200000 --block 32 --tag text_cpt_v$i --out $P/text_cpt_v$i "${EX[@]}"
    EX+=(--exclude-pool $P/text_cpt_v$i)
  fi
  CSV+="${CSV:+,}$P/text_cpt_v$i"
done
[[ -f $P/text_long_v$N/meta.json ]] || python scripts/data/build_long_pool.py --pools $CSV --out $P/text_long_v$N --join 4
[[ -f $P/text_mix_v$N/meta.json ]] || python scripts/data/build_mix_pool.py --long $P/text_long_v$N --windows $CSV --out $P/text_mix_v$N --n 200000
echo "TEXT POOLS $N DONE"
