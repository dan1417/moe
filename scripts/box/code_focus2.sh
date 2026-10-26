#!/usr/bin/env bash
# User (04:28): evaluate soup_k43_c1_040_ck1_025 (CODE leader vs KC2, +0.002122) on MAIN and MATH too.
# Waits for code_focus's CODE stage to end, takes over: the new CODE recipes on all four kings (that one first), then the
# TEXT full verification and every recipe vs all four kings (code_focus's remaining steps).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
# user (04:33): evaluate soup_k43_c1_050_dc1_050_ck1_025 on MAIN and the others now (training start candidate): stop the
# focused CODE stage (its remaining KC2 items are covered by the final all-kings pass)
pm2 stop code_focus >/dev/null 2>&1; pm2 delete code_focus >/dev/null 2>&1; sleep 8
K=configs/merge
FIRST="$K/code/soup_k43_c1_050_dc1_050_ck1_025.json $K/code/soup_k43_c1_040_ck1_025.json \
$K/code/soup_k43_c1_040_dc1_050_ck1_025.json $K/code/soup_k43_c1_050_dc1_100_ck1_025.json $K/code/soup_k43_c1_040_dc1_100_ck1_025.json \
$K/code/soup_k43_c1_050_ck1_050.json $K/code/soup_k43_c1_050_ck1_025.json $K/code/soup_k43_c1_040.json $K/code/soup_k43kc2_c1_050_ck1_025.json"
R=(); for f in $FIRST; do R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) CODE recipes vs all four kings"
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) TEXT full verification vs KT2 (A, B, C)"
python scripts/merge/campaign43.py --recipe $K/k43/k43_o42_ck1_025.json --recipe $K/k43/soup_k43_c1_050.json \
  --recipe $K/k43/slerp_k43_c1_040.json --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) every recipe vs all four kings"
R=(); for f in $K/code/*.json $K/kc2/*.json $K/kt2/*.json $K/k43/*.json; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done"
