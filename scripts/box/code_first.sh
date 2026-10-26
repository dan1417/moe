#!/usr/bin/env bash
# User (04:10): "let's make the best model for CODE" -> CODE campaign vs the new CODE king KC2 (5DALwjE4-99, a light
# fine-tune of #43: distance^2 0.68) FIRST, then the paused TEXT full verification, then every recipe vs all four kings.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
K=configs/merge
FIRST="$K/kc2/kc2_raw.json $K/kc2/soup_kc2_c1_050.json $K/kc2/slerp_kc2_c1_040.json $K/k43/soup_k43_c1_050.json \
$K/k43/slerp_k43_c1_040.json $K/kc2/slerp_kc2_c1_050.json $K/kc2/soup_kc2_c1_050_ck1_025.json $K/kc2/kc2_mom050.json \
$K/kc2/kc2_mom100.json $K/kc2/soup_kc2mom050_c1_050.json $K/kc2/kc2_o42_100.json $K/kc2/orth_kc2_o42_100.json \
$K/kc2/k43_o42_kc2half.json $K/kc2/kc2_o42_050.json $K/kc2/kc2_dc1_100.json $K/kc2/kc2_dc1_050.json"
R=(); for f in $FIRST; do R+=(--recipe "$f"); done
for f in $K/k43/*.json $K/kt2/*.json; do [[ " $FIRST " == *" $f "* ]] || R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) CODE campaign vs KC2 (${#R[@]} args)"
python scripts/merge/campaign43.py "${R[@]}" --comps code --king code=KC2:0.002 --campaign kc2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) TEXT full verification vs KT2 (A, B, C)"
python scripts/merge/campaign43.py --recipe $K/k43/k43_o42_ck1_025.json --recipe $K/k43/soup_k43_c1_050.json \
  --recipe $K/k43/slerp_k43_c1_040.json --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) every recipe vs all four kings"
R=(); for f in $K/kc2/*.json $K/kt2/*.json $K/k43/*.json; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done"
