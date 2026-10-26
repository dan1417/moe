#!/usr/bin/env bash
# After the joint merge search (merge_search2.py, MAIN / MATH / CODE / TEXT, linear + TIES / DARE / breadcrumbs / per-module):
#  1. full samples of each competition's best vs its king (MAIN A, C; MATH / CODE / TEXT A, B, C; ~31.8k seqs each)
#  2. train from the MAIN best (user: "once merge can't improve, you must run the training"): write it, switch.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
L=/data/teutonic/gpu_screen.lock
ST=/data/teutonic/search/all1.json
until grep -q "SEARCH DONE" /data/teutonic/logs/pm2_search_all.log || ! pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='search_all' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 60; done
echo "$(date -u +%H:%M:%S) search finished"
best() { python3 -c "import json; print(json.load(open('$ST'))['best']['$1']['recipe'])"; }
declare -A KING=([main]=KMAIN_c7ef5487:0.003 [math]=KMATH_ddd9e626:0.002 [code]=T007:0.002 [text]=KTEXT_9a39afee:0.002)
declare -A SAMP=([main]=tag:A,tag:C [math]=tag:A,tag:B,tag:C [code]=tag:A,tag:B,tag:C [text]=tag:A,tag:B,tag:C)
while [[ -e $L ]]; do sleep 30; done
echo after_search > $L; trap 'rm -f $L' EXIT
for c in main math code text; do
  R=$(best $c) || continue
  echo "$(date -u +%H:%M:%S) full $c: $(python3 -c "import json; print(json.load(open('$R'))['description'])")"
  PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py --recipe "$R" --king $c=${KING[$c]} --comps $c --campaign ${c}full \
    --sample ${SAMP[$c]} --limit 40000 --replicas 4 --batch-size 32 --pingpong 1 2>&1 | grep -E "^=== |  (MAIN|MATH|CODE|TEXT)|Traceback|Error"
done
rm -f $L
R=$(best main); LABEL=$(basename "$R" .json); OUT=/data/teutonic/ckpt/start_$LABEL
echo "$(date -u +%H:%M:%S) writing the MAIN best $LABEL -> $OUT"
[[ -f $OUT/model.safetensors.index.json ]] || python scripts/merge/write_merge.py --recipe "$R" --out "$OUT" 2>&1 | tail -3
bash /data/teutonic/logs/switch_training.sh "$OUT" "MAIN merge-search best $LABEL ($(python3 -c "import json; print(json.load(open('$R'))['description'])"))"
echo "$(date -u +%H:%M:%S) AFTER_SEARCH DONE"
