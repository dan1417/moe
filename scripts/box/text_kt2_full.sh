#!/usr/bin/env bash
# After the TEXT screen vs KT2 (text_kt2.sh): confirm the two best recipes against KT2 on FULL TEXT samples A, B, C
# (31.6k sequences each, the validator's size; best first), then hand the GPUs back to wave 3 (MAIN/MATH).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
until grep -q "TEXT campaign done" /data/teutonic/logs/pm2_text_kt2.log; do echo "$(date -u +%H:%M:%S) waiting for the TEXT screen"; sleep 20; done
sleep 5
pm2 stop campaign43_w3 >/dev/null 2>&1; sleep 5  # replicas die with their parent (PR_SET_PDEATHSIG)
TOP=$(python3 - <<'PY'
import json, glob
from pathlib import Path
best = {}
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json")):
    d = json.load(open(f))
    if d.get("campaign") != "kt2":
        continue
    for r in d["results"]:
        label = r["challenger"].get("label")
        if label and label != "kt2_raw":
            best[label] = r["verdict"]["mu_hat"]
top = sorted(best, key=best.get, reverse=True)[:2]
paths = [next(iter(Path("configs/merge").glob(f"*/{l}.json"))) for l in top]
print(" ".join(str(p) for p in paths))
PY
)
echo "$(date -u +%H:%M:%S) full confirmation of: $TOP"
R=(); for f in $TOP; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --comps text --king text=KT2:0.002 --campaign kt2full \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) full confirmation done: wave 3 resumes"
pm2 restart campaign43_w3 >/dev/null 2>&1
