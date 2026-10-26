#!/usr/bin/env bash
# User (03:16): evaluate ALL merged models against every current king: MAIN #43 (0.003), MATH -> #43 (vacant; 0.002),
# CODE CK1 (0.002; KC2 once crowned), TEXT KT2 (0.002). Screen = first 4 800 sequences of each competition's sample A;
# recipe losses are cached per sample, so only never-scored parts cost GPU time.
#   1) the ~20 most promising recipes on all four   2) full TEXT confirmation (A, B, C) of the 2 best vs KT2
#   3) every remaining recipe on all four
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
until grep -q "TEXT campaign done" /data/teutonic/logs/pm2_text_kt2.log; do echo "$(date -u +%H:%M:%S) waiting for the TEXT screen"; sleep 15; done
sleep 5; pm2 stop campaign43_w3 >/dev/null 2>&1; sleep 5
K=/root/workspace/MoE/configs/merge
CODE_KING=$(python3 -c "import json; m=json.load(open('configs/merge/k41/models.json')); print('KC2' if 'KC2' in m else 'CK1')")
KINGS=(--king text=KT2:0.002 --king code=$CODE_KING:0.002)
TOP="kt2/kt2_o42_100 kt2/orth_kt2_o42_100 kt2/k43_o42_kt2half k43/slerp_k43_c1_040 k43/soup_k43_c1_050 k43/slerp_k43_c1_030 \
k43/k43_dc1_100_cs050 k43/k43_dc1_100_cs025 k43/k43_dc1_100 k43/k43_o42_100 k43/orth_k43_o42_100 k43/k43_o42_ck1_025 \
k43/slerp_k43_c1_050 k43/tiestrim_k43_o42_05 k43/k43_o42_100_mn k43/k43_o42_d100_e050 k43/k43_o42_050 k43/slerp_k43_c1_060 \
k43/k43_dc1_050 k43/k43_dc1_150"
R=(); for f in $TOP; do R+=(--recipe $K/$f.json); done
echo "$(date -u +%H:%M:%S) stage 1: top recipes vs all four kings (code king $CODE_KING)"
python scripts/merge/campaign43.py "${R[@]}" "${KINGS[@]}" --comps main,math,code,text --campaign allkings \
  --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
TOP2=$(python3 - <<'PY'
import json, glob
from pathlib import Path
best = {}
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json")):
    d = json.load(open(f))
    if d.get("campaign") not in ("kt2", "allkings"):
        continue
    for r in d["results"]:
        if "8864c279" not in json.dumps(r["king"]):
            continue
        label = r["challenger"].get("label")
        if label and label != "kt2_raw":
            best[label] = r["verdict"]["mu_hat"]
top = sorted(best, key=best.get, reverse=True)[:2]
print(" ".join(str(next(iter(Path("configs/merge").glob(f"*/{l}.json")))) for l in top))
PY
)
echo "$(date -u +%H:%M:%S) stage 2: full TEXT confirmation vs KT2 of: $TOP2"
R=(); for f in $TOP2; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) stage 3: every remaining recipe vs all four kings"
R=(); for f in $K/kt2/*.json $K/k43/*.json; do
  l=$(basename $f .json); [[ " $TOP " == *"/$l "* ]] && continue; R+=(--recipe $f); done
python scripts/merge/campaign43.py "${R[@]}" "${KINGS[@]}" --comps main,math,code,text --campaign allkings \
  --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done"
