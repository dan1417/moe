#!/usr/bin/env bash
# User (03:30): a new CODE king is about to be crowned (uid 239, 5Ci6YAhr..., model fa428e5e..., ~+0.0032 vs CK1) -> prepare.
# 1) ingest the model as KC2 the moment R2 publishes it (manifest first, shards ~1 min later: retries)
# 2) wait for the crown (exit quietly if uid 239 is rejected)
# 3) pre-empt all_kings (its stage 1 results stay cached) and run the CODE campaign vs KC2: KC2-based recipes + the leaders
# 4) the TEXT full confirmation vs KT2 (all_kings stage 2), then every remaining recipe vs all four kings (CODE = KC2)
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8 DL_WORKERS=64
DIGEST=fa428e5e3565110a037a68d7ecde4b561909478f5936bd5af17e93c90c3ff61b
URL=https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/$DIGEST/manifest.json
code_state() { curl -s -A teutonic-miner/1.0 https://teutonic.ai/competitions/code/dashboard.json | python3 -c "
import json,sys; d=json.load(sys.stdin); k=(d.get('king') or {}).get('king_digest') or ''
rej=any(h.get('uid')==239 and h.get('verdict') in ('rejected','error') and h.get('timestamp','')>'2026-10-10T03' for h in d.get('history') or [])
print('crowned' if k.startswith('$DIGEST'[:8]) else 'rejected' if rej else 'pending')" 2>/dev/null || echo pending; }
if ! python3 -c "import json,sys; sys.exit(0 if 'KC2' in json.load(open('configs/merge/k41/models.json')) else 1)"; then
  until [[ $(curl -s -o /dev/null -w "%{http_code}" -A teutonic-miner/1.0 "$URL") == 200 ]]; do
    [[ $(code_state) == rejected ]] && { echo "$(date -u +%H:%M:%S) uid 239 rejected: nothing to do"; exit 0; }
    echo "$(date -u +%H:%M:%S) waiting for KC2 ($DIGEST) to be published"; sleep 20; done
  echo "$(date -u +%H:%M:%S) KC2 published: ingesting"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    bash scripts/merge/ingest_king41.sh "$DIGEST" KC2 && break
    echo "$(date -u +%H:%M:%S) ingest attempt $attempt failed; retrying in 30 s"; sleep 30
  done
  python3 -c "import json,sys; sys.exit(0 if 'KC2' in json.load(open('configs/merge/k41/models.json')) else 1)" || { echo "INGEST FAILED"; exit 1; }
fi
echo "$(date -u +%H:%M:%S) KC2 ingested"
until [[ $(code_state) == crowned ]]; do echo "$(date -u +%H:%M:%S) waiting for the CODE crown"; sleep 30; done
echo "$(date -u +%H:%M:%S) KC2 crowned"
RECIPES=$(.venv-eval/bin/python scripts/merge/recipes_kt2.py --king KC2) || { echo "RECIPES FAILED"; exit 1; }
echo "recipes: $RECIPES"
until grep -q "stage 2\|all kings done" /data/teutonic/logs/pm2_all_kings.log; do echo "$(date -u +%H:%M:%S) waiting for all_kings stage 1"; sleep 20; done
pm2 stop all_kings >/dev/null 2>&1; sleep 8   # replicas die with their parent (PR_SET_PDEATHSIG)
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
K=configs/merge
LEAD="$K/k43/orth_k43_o42_100.json $K/k43/k43_o42_100.json $K/kt2/k43_o42_kt2half.json $K/k43/k43_o42_ck1_025.json $K/kt2/kt2_o42_100.json $K/kt2/orth_kt2_o42_100.json"
R=(); for f in $RECIPES $LEAD; do R+=(--recipe "$f"); done
echo "$(date -u +%H:%M:%S) CODE campaign vs KC2"
python scripts/merge/campaign43.py "${R[@]}" --comps code --king code=KC2:0.002 --campaign kc2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
TOP2=$(python3 - <<'PY'
import json, glob
from pathlib import Path
best = {}
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json")):
    d = json.load(open(f))
    if d.get("campaign") not in ("kt2", "allkings"):
        continue
    for r in d["results"]:
        if "8864c279" in json.dumps(r["king"]) and r["challenger"].get("label") not in (None, "kt2_raw"):
            best[r["challenger"]["label"]] = r["verdict"]["mu_hat"]
top = sorted(best, key=best.get, reverse=True)[:2]
print(" ".join(str(next(iter(Path("configs/merge").glob(f"*/{l}.json")))) for l in top))
PY
)
echo "$(date -u +%H:%M:%S) TEXT full confirmation vs KT2 of: $TOP2"
R=(); for f in $TOP2; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) every recipe vs all four kings (CODE = KC2)"
R=(); for f in $K/kt2/*.json $K/kc2/*.json $K/k43/*.json; do R+=(--recipe $f); done
python scripts/merge/campaign43.py "${R[@]}" --king text=KT2:0.002 --king code=KC2:0.002 --comps main,math,code,text \
  --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) all kings done (CODE = KC2)"
