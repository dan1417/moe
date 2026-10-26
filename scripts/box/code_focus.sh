#!/usr/bin/env bash
# CODE vs KC2, focused (user 04:20: "why don't you eval slerp_k43_c1_040?" -> its CODE verdict had been computed but lost:
# result files were second-stamped and the cached reports of one second overwrote each other; fixed in save_result).
#   0) let code_first finish the item it is scoring, then stop it
#   1) re-save every recipe whose CODE losses are cached (no GPU)
#   2) the new #43 soup + CODE-king-delta recipes, the KC2 recipes still pending, a few CODE-relevant others
#   3) TEXT full verification vs KT2 (A, B, C) of k43_o42_ck1_025 / soup_k43_c1_050 / slerp_k43_c1_040
#   4) every recipe vs all four kings
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
until grep -q "=== kc2_mom100" /data/teutonic/logs/pm2_code_first.log; do sleep 5; done
sleep 3; pm2 stop code_first >/dev/null 2>&1; pm2 delete code_first >/dev/null 2>&1; sleep 8
python scripts/merge/recipes_code.py >/dev/null
K=configs/merge
CACHED=$(.venv-eval/bin/python - <<'PY'
import sys, glob, numpy as np
sys.path.insert(0, "scripts/eval")
import parity_eval as pe
from src.parity.cache import LossCache, scorer_signature
from src.parity.sample import load_live_config
c, sig = LossCache(), scorer_signature("flash_attention_4", 1024, True)
s = pe.resolve_sample("tag:A", load_live_config(competition="code"))
out = []
for f in sorted(glob.glob("configs/merge/k43/*.json") + glob.glob("configs/merge/kt2/*.json") + glob.glob("configs/merge/kc2/*.json")):
    x = c.load(s.sample_id, pe.resolve_recipe(f).identity(), sig, s.n)
    if np.isfinite(x[:4800]).all():
        out.append(f)
print(" ".join(out))
PY
)
R=(); for f in $CACHED; do R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) re-saving ${#R[@]} cached CODE verdicts vs KC2"
python scripts/merge/campaign43.py "${R[@]}" --comps code --king code=KC2:0.002 --campaign kc2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
FOCUS="$K/code/soup_k43_c1_050_ck1_025.json $K/code/soup_k43_c1_040_ck1_025.json $K/code/soup_k43_c1_050_ck1_050.json \
$K/code/soup_k43_c1_040.json $K/code/soup_k43kc2_c1_050_ck1_025.json $K/code/soup_k43_c1_050_dc1_050_ck1_025.json \
$K/kc2/soup_kc2mom050_c1_050.json $K/kc2/kc2_o42_100.json $K/kc2/orth_kc2_o42_100.json $K/kc2/k43_o42_kc2half.json \
$K/k43/k43_ck1_075.json $K/kc2/kc2_o42_050.json $K/kc2/kc2_dc1_100.json $K/kc2/kc2_dc1_050.json"
R=(); for f in $FOCUS; do R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) focused CODE campaign vs KC2"
python scripts/merge/campaign43.py "${R[@]}" --comps code --king code=KC2:0.002 --campaign kc2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) TEXT full verification vs KT2 (A, B, C)"
python scripts/merge/campaign43.py --recipe $K/k43/k43_o42_ck1_025.json --recipe $K/k43/soup_k43_c1_050.json \
  --recipe $K/k43/slerp_k43_c1_040.json --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) every recipe vs all four kings"
R=(); for f in $K/code/*.json $K/kc2/*.json $K/kt2/*.json $K/k43/*.json; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done"
