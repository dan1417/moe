#!/usr/bin/env bash
# User (03:45): new CODE king KC2 (5DALwjE4-99, fa428e5e..., crowned 03:42:38) -> merge it and find the best CODE model.
# Runs after the soup's full TEXT check (soup_full) so the TEXT submission is not delayed:
#   1) KC2 ingested (code_kc2.sh downloads it)  2) KC2-based recipes (parent detection)
#   3) CODE screen vs KC2 of the KC2 recipes + every merged model (CODE losses are cached for most)
#   4) every recipe vs all four kings (MAIN/MATH #43, CODE KC2, TEXT KT2)
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
until grep -q "soup full B/C done" /data/teutonic/logs/pm2_soup_full.log; do echo "$(date -u +%H:%M:%S) waiting for the soup TEXT check"; sleep 20; done
sleep 3; pm2 stop all_kings >/dev/null 2>&1; pm2 delete all_kings >/dev/null 2>&1; sleep 8
# the TEXT submission t005 (soup_k43_c1_050 with a random hidden permutation): same losses as the soup it was made from?
echo "$(date -u +%H:%M:%S) t005 equivalence check (TEXT A, first 1 200 sequences)"
python scripts/merge/campaign43.py --recipe /data/teutonic/submit/t005 --comps text --king text=KT2:0.002 --campaign check \
  --limit 1200 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
.venv-eval/bin/python - <<'PY'
import sys; sys.path.insert(0, "scripts/eval")
import numpy as np, parity_eval as pe
from src.parity.cache import LossCache, scorer_signature
from src.parity.sample import load_live_config
c, sig = LossCache(), scorer_signature("flash_attention_4", 1024, True)
s = pe.resolve_sample("tag:A", load_live_config(competition="text"))
a = c.load(s.sample_id, pe.resolve_recipe("configs/merge/k43/soup_k43_c1_050.json").identity(), sig, s.n)[:1200]
b = c.load(s.sample_id, pe.resolve_recipe("/data/teutonic/submit/t005").identity(), sig, s.n)[:1200]
d = np.abs(a - b); print(f"T005 CHECK: max |dloss| {d.max():.2e}, mean {d.mean():.2e}, mean loss soup {a.mean():.6f} t005 {b.mean():.6f}, diff of means {b.mean() - a.mean():+.2e}")
PY
# user (04:00): verify the TEXT candidates on the validator's size -- full TEXT A, B, C vs KT2 (soup B/C already scored)
echo "$(date -u +%H:%M:%S) TEXT full verification vs KT2 (A, B, C)"
python scripts/merge/campaign43.py --recipe configs/merge/k43/k43_o42_ck1_025.json --recipe configs/merge/k43/soup_k43_c1_050.json \
  --recipe configs/merge/k43/slerp_k43_c1_040.json --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) TEXT full verification done"
until python3 -c "import json,sys; sys.exit(0 if 'KC2' in json.load(open('configs/merge/k41/models.json')) else 1)"; do
  echo "$(date -u +%H:%M:%S) waiting for the KC2 ingest"; sleep 20; done
echo "$(date -u +%H:%M:%S) KC2 ingested"
pm2 delete code_kc2 >/dev/null 2>&1  # its remaining steps are done here
RECIPES=$(.venv-eval/bin/python scripts/merge/recipes_kt2.py --king KC2) || { echo "RECIPES FAILED"; exit 1; }
echo "KC2 recipes: $RECIPES"
K=configs/merge
LEAD="$K/k43/soup_k43_c1_050.json $K/k43/slerp_k43_c1_040.json $K/k43/slerp_k43_c1_030.json $K/k43/slerp_k43_c1_050.json"
R=(); for f in $RECIPES $LEAD $K/kt2/*.json $K/k43/*.json; do R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) CODE screen vs KC2"
python scripts/merge/campaign43.py "${R[@]}" --comps code --king code=KC2:0.002 --campaign kc2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) every recipe vs all four kings"
R=(); for f in $K/kc2/*.json $K/kt2/*.json $K/k43/*.json; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done"
