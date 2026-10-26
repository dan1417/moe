#!/usr/bin/env bash
# CODE gate (user 09:50): full CODE samples (~31.5k sequences) vs the CODE king KC2, bar 0.002 (pass = LCB above the bar):
#   A -> (if A passed) B -> (if B passed) C; a sample that two checkpoints already passed is skipped (counted as passed);
#   A, B and C passed -> "ALL PASSED" (the user submits). Usage: code_gate.sh <checkpoint dir> <step>; the caller holds
#   the trainer's PAUSE. State: <run>/code_gate.json = {"A": [steps passed], "B": [...], "C": [...]}.
set -uo pipefail
CK=$1; STEP=$2
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
STATE=$D/ckpt/main_upper_ema_m1/code_gate.json
[[ -f $STATE ]] || echo '{"A": [], "B": [], "C": []}' > $STATE
passes() { python3 -c "import json; print(len(json.load(open('$STATE'))['$1']))"; }
record() { python3 -c "
import json; s = json.load(open('$STATE')); s['$1'].append($STEP); json.dump(s, open('$STATE', 'w'))"; }
for T in A B C; do
  if [[ $T != C ]] && (( $(passes $T) >= 2 )); then echo "$(date -u +%H:%M:%S) step $STEP: CODE $T skipped (passed twice)"; continue; fi
  until python3 -c "
import json, glob, sys
sys.exit(0 if any(json.load(open(m)).get('label') == 'tag:$T' and json.load(open(m))['config_version'].startswith('7dcbbb5f')
                  for m in glob.glob('$D/eval_cache/samples/*/meta.json')) else 1)"; do
    echo "$(date -u +%H:%M:%S) waiting for CODE sample $T"; sleep 30; done
  echo "$(date -u +%H:%M:%S) step $STEP: CODE $T full sample vs KC2"
  OUT=$(python scripts/merge/campaign43.py --recipe "$CK" --comps code --king code=KC2:0.002 --campaign kc2full --limit 40000 \
        --sample tag:$T --replicas 4 --batch-size 32 --pingpong 0 2>&1 | grep -E "^=== |  CODE |Traceback|Error")
  echo "$OUT"
  LCB=$(echo "$OUT" | grep -oP "  CODE\s+vs \S+\s+mu [+-][\d.]+\s+lcb \K[+-][\d.]+" | tail -1)
  MU=$(echo "$OUT" | grep -oP "  CODE\s+vs \S+\s+mu \K[+-][\d.]+" | tail -1)
  if [[ -n $LCB ]] && python3 -c "import sys; sys.exit(0 if float('$LCB') > 0.002 else 1)"; then
    record $T; echo "$(date -u +%H:%M:%S) step $STEP: CODE $T PASS (mu $MU, lcb $LCB)"
  else
    echo "$(date -u +%H:%M:%S) step $STEP: CODE $T FAIL (mu ${MU:-?}, lcb ${LCB:-?}) -- gate stops for this checkpoint"; exit 0
  fi
done
echo "$(date -u +%H:%M:%S) ALL PASSED: step $STEP passed CODE A, B and C vs KC2 -> ready for the user's submission"
