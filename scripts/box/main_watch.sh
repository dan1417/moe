#!/usr/bin/env bash
# Every 50 steps of main_upper_ema_m1: pause the trainer, screen the checkpoint vs #43 on MAIN (bar 0.003) and MATH (vacant
# crown, bar 0.002) and vs the CODE king KC2 (bar 0.002; user 09:39), 4 800 sequences each, one replica set (a paused
# trainer keeps ~85 GB per GPU), resume (trap). Usage: main_watch.sh [first step, default 50]
set -uo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
RUN=${RUN:-$D/ckpt/main_upper_ema_m1}
COMPS=${COMPS:-main,math}  # competitions screened every 25 steps (TEXT run: text,main)
OWN=0  # remove the shared PAUSE only if this script created it (a restarted watcher's trap un-paused another's eval)
resume() { [[ $OWN == 1 ]] && rm -f "$RUN/PAUSE"; OWN=0; }
trap resume EXIT
EVERY=${EVERY:-25}  # screen interval in steps (= the run's save interval)
# total steps of this run (its last checkpoint is `final`): STEPS, else the last --steps of the run's launcher script
# (written a few minutes after the launch; a guard resume runs the remaining steps under its own numbering)
TOTAL=${STEPS:-}
until [[ -n $TOTAL ]]; do
  TOTAL=$(grep -o -- '--steps [0-9]*' $D/logs/run_$(basename $RUN).sh 2>/dev/null | tail -1 | tr -dc 0-9); [[ -n $TOTAL ]] || sleep 30
done
for N in $(seq ${1:-50} $EVERY $TOTAL) $(( TOTAL % EVERY ? TOTAL : 0 )); do  # every checkpoint, then final
  (( N > 0 )) || continue
  CK=$RUN/step-$N; [[ $N == $TOTAL ]] && CK=$RUN/final
  until [[ -f $CK/SAVED ]]; do sleep 30; [[ -d $CK || -f $RUN/step-$((N + EVERY))/SAVED ]] || continue; [[ -f $RUN/step-$((N + EVERY))/SAVED && ! -f $CK/SAVED ]] && break; done
  [[ -f $CK/SAVED ]] || { echo "$(date -u +%H:%M:%S) step $N: no complete checkpoint, skipped"; continue; }
  until [[ ! -f $RUN/PAUSE ]]; do sleep 10; done; touch "$RUN/PAUSE"; OWN=1
  until [[ -f $RUN/PAUSED ]] && [[ $RUN/PAUSED -nt $RUN/PAUSE ]]; do sleep 5; done
  # the reigning kings of the run's competitions; king_args.py exits 2 (empty stdout) when one is not ingested yet:
  # skip the screen rather than score against campaign43's default kings
  if ! KARGS=$(python3 /data/teutonic/logs/king_args.py $COMPS) || [[ -z $KARGS ]]; then
    echo "$(date -u +%H:%M:%S) step $N: screen skipped: king not ready ($COMPS)"
    resume; echo "$(date -u +%H:%M:%S) step $N: training resumed"; continue
  fi
  echo "$(date -u +%H:%M:%S) step $N: trainer paused, screening $COMPS vs the reigning kings ($KARGS)"
  EXTRA_CK=""  # SR=1: the run also writes a stochastically rounded twin sr-<ckpt> of the same weights; screen both (A/B)
  if [[ ${SR:-0} == 1 ]]; then
    TW=$RUN/sr-$(basename $CK); for i in $(seq 1 60); do [[ -f $TW/SAVED ]] && break; sleep 15; done
    [[ -f $TW/SAVED ]] && EXTRA_CK="--recipe $TW"
  fi
  python scripts/merge/campaign43.py --recipe "$CK" $EXTRA_CK --comps $COMPS $KARGS --campaign m1 --limit 4800 \
    --sample tag:A --replicas 4 --batch-size 32 --pingpong 0 2>&1 | grep -E "^=== |  (MAIN|MATH|CODE|TEXT) |Traceback|Error" || true
  # TEXT win check (user 02:05 JST Oct 12: "train until my model beats the TEXT king"): a checkpoint whose TEXT screen LCB clears
  # the bar gets the full TEXT samples A/B/C (~31.8k seqs each) right away, still in this pause
  if [[ ,$COMPS, == *,text,* ]]; then
    for C2 in "$CK" ${EXTRA_CK#--recipe }; do
      LCB=$(python3 - "$C2" <<'PY'
import glob, json, os, sys
best = None
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json"), key=os.path.getmtime)[-40:]:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        if len(t) == 1 and t[0].get("model", "").rstrip("/") == sys.argv[1].rstrip("/") and r.get("sample", "").startswith("0a2840c9"):
            best = (r.get("verdict") or {}).get("lcb")
print(best if best is not None else -1)
PY
)
      if python3 -c "import sys; sys.exit(0 if float('$LCB') > 0.002 else 1)"; then
        echo "$(date -u +%H:%M:%S) step $N: $(basename $C2) TEXT screen LCB $LCB > 0.002 -> full TEXT samples A/B/C"
        if ! TARGS=$(python3 /data/teutonic/logs/king_args.py text) || [[ -z $TARGS ]]; then
          echo "$(date -u +%H:%M:%S) step $N: full TEXT check skipped: king not ready"; continue
        fi
        python scripts/merge/campaign43.py --recipe "$C2" --comps text $TARGS --campaign textfull \
          --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 0 2>&1 | grep -E "^=== |  TEXT|Traceback|Error" || true
        echo "$(date -u +%H:%M:%S) TEXT FULL CHECK DONE for $C2"
      fi
    done
  fi
  # CODE gate off (user 12:20: target MAIN; step-150 already passed full CODE A/B/C vs KC2, t007 is staged)
  resume; echo "$(date -u +%H:%M:%S) step $N: training resumed"
done
