#!/usr/bin/env bash
# Wave 3 (MAIN/MATH only, 4 800 sequences): after wave 2 leaves the GPUs, rescore wave 1's leaders on the full 4 800
# (they had 2 400; only the missing half is scored) and interpolate between k43_dc1_100 (best MAIN) and k43_o42_100
# (best MATH).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
while pgrep -f "[c]ampaign43.py --recipe configs/merge/k43/k43_o42_125" >/dev/null; do
  echo "$(date -u +%H:%M:%S) waiting for wave 2"; sleep 30; done
python scripts/merge/recipes43_w2.py
R=(); for f in slerp_k43_c1_040 slerp_k43_c1_060 soup_k43_c1_050 slerp_k43_c1_030 slerp_k43_c1s_050 k43_dc1_100 k43_o42_100 k43_dc1_100_cs025 k43_dc1_100_cs050 k43_o42_d100_e050 k43_o42_d125_e050 k43_dc1_050; do R+=(--recipe configs/merge/k43/$f.json); done
exec python scripts/merge/campaign43.py "${R[@]}" --limit 4800 --comps main,math --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
