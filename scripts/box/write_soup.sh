#!/usr/bin/env bash
# Write soup_k43_c1_050 (0.5 x #43 + 0.5 x our c1 final) on CPU: elementwise fp32 average -> bf16, bit-identical to the
# GPU path the campaign scored. The recipe file lands next to (not inside) the output directory.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
nice -n 5 python scripts/merge/write_merge.py --recipe configs/merge/k43/soup_k43_c1_050.json \
  --out /data/teutonic/submit/soup_k43_c1_050 --gpus cpu --workers 32
echo "WRITE DONE rc=$?"
