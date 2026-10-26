#!/usr/bin/env bash
# MAIN training pools for the autopilot (the five eval sources at their live proportions): four 200k-row window pools ->
# one long pool (4 consecutive rows joined = 8 192-token documents) -> one mix (half long rows, half 4-window rows).
#   bash scripts/box/build_main_pools.sh 1      # main_cpt_v1..v4 -> main_long_v1 -> main_mix_v1 (also main_val_v1)
#   bash scripts/box/build_main_pools.sh 2      # main_cpt_v5..v8 -> main_long_v2 -> main_mix_v2
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." && source /data/teutonic/env.sh
N=${1:-1}; P=/data/teutonic/pools
[[ -f $P/main_val_v1/meta.json ]] || python scripts/data/build_diverse_pool.py --competition main --n 2048 --block 1 --tag main_val_v1 --out $P/main_val_v1
EX=(--exclude-pool $P/main_val_v1)
for d in $P/main_cpt_v*; do [[ -f $d/meta.json ]] && EX+=(--exclude-pool $d); done
CSV=""
for i in $(seq $(( (N-1)*4 + 1 )) $(( N*4 ))); do
  if [[ ! -f $P/main_cpt_v$i/meta.json ]]; then
    python scripts/data/build_diverse_pool.py --competition main --n 200000 --block 32 --tag main_cpt_v$i --out $P/main_cpt_v$i "${EX[@]}"
    EX+=(--exclude-pool $P/main_cpt_v$i)
  fi
  CSV+="${CSV:+,}$P/main_cpt_v$i"
done
[[ -f $P/main_long_v$N/meta.json ]] || python scripts/data/build_long_pool.py --pools $CSV --out $P/main_long_v$N --join 4
[[ -f $P/main_mix_v$N/meta.json ]] || python scripts/data/build_mix_pool.py --long $P/main_long_v$N --windows $CSV --out $P/main_mix_v$N --n 200000
echo "MAIN POOLS $N DONE"
