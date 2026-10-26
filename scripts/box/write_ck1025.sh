#!/usr/bin/env bash
# Write k43_o42_ck1_025 (#43 + our lineage since #42 + 0.25 x CK1 delta) on CPU: elementwise fp32, every coefficient a power of two, bit-identical to the
# GPU path the campaign scored. The recipe file lands next to (not inside) the output directory.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
nice -n 5 python scripts/merge/write_merge.py --recipe configs/merge/k43/k43_o42_ck1_025.json \
  --out /data/teutonic/submit/k43_o42_ck1_025 --gpus cpu --workers 32
echo "WRITE DONE rc=$?"
