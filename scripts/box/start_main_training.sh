#!/usr/bin/env bash
# User (04:44): evaluate up to the 5 models, then start MAIN training from the best. Stops the evaluation after model 5,
# picks the start by MAIN mu vs #43 among the 4 training-step soups, writes it, launches the run and the step-50 watcher.
set -uo pipefail
D=/data/teutonic; cd /root/workspace/MoE
until grep -q "=== soup_k43_c1_040_dc1_100_ck1_025" $D/logs/pm2_code_focus2.log; do sleep 10; done
sleep 8   # let the report finish (TEXT line + saved)
pm2 stop code_focus2 >/dev/null 2>&1; pm2 delete code_focus2 >/dev/null 2>&1; sleep 8
echo "$(date -u +%H:%M:%S) evaluation stopped after model 5; MAIN/MATH parameter sweep"
export PARITY_MERGE_SLOTS=8; source $D/env.sh
R=(); for f in configs/merge/code/sw_*.json; do R+=(--recipe "$f"); done
for attempt in 1 2 3; do  # r2.dev throttles while the MAIN pools build: retry the whole (cached) sweep if it fails
  python scripts/merge/campaign43.py "${R[@]}" --comps main,math --campaign allkings --limit 4800 --sample tag:A \
    --replicas 4 --batch-size 32 --pingpong 1 2>&1 | grep -E "^=== |  (MAIN|MATH) |Traceback|Error"
  [[ ${PIPESTATUS[0]} == 0 ]] && break
  echo "$(date -u +%H:%M:%S) sweep attempt $attempt failed; retrying in 60 s"; sleep 60
done
echo "$(date -u +%H:%M:%S) sweep done"
BEST=$(python3 - <<'PY'
import json, glob
cands = ["soup_k43_c1_050_dc1_050_ck1_025", "soup_k43_c1_040_dc1_050_ck1_025", "soup_k43_c1_050_dc1_100_ck1_025",
         "soup_k43_c1_040_dc1_100_ck1_025"] + [p.split("/")[-1][:-5] for p in glob.glob("configs/merge/code/sw_*.json")]
main = {}
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json")):
    d = json.load(open(f))
    for r in d.get("results", []):
        if r["challenger"].get("label") in cands and r["sample"].startswith("e8ef9ae1") and r["verdict"].get("n_sequences", 0) >= 4800:
            main[r["challenger"]["label"]] = r["verdict"]["mu_hat"]
print(" ".join(f"{k}={v:+.6f}" for k, v in sorted(main.items(), key=lambda x: -x[1])), file=__import__("sys").stderr)
print(f"configs/merge/code/{max(main, key=main.get)}.json")
PY
)
echo "$(date -u +%H:%M:%S) training start: $BEST"
pm2 start /data/teutonic/logs/main_watch.sh --name main_watch --no-autorestart --output $D/logs/pm2_main_watch.log --error $D/logs/pm2_main_watch.log >/dev/null
exec bash /data/teutonic/logs/main_train.sh "$BEST"
