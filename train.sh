#!/usr/bin/env bash
# train.sh — train one competition until a checkpoint beats a king (scripts/train/autopilot.py under pm2).
#
#   bash train.sh <main|math|code|text> [start]   start the autopilot (or adopt its lineage after a restart)
#        start (optional, first run only): a checkpoint dir, a models.json name (e.g. MA2) or a merge label
#        (configs/merge/*/<label>.json, written first). Default: search/train_inits.json (merge.sh <comp>), else the king.
#   bash train.sh status            kings, every autopilot's phase / run / LR / best screens / SUBMIT-READY list
#   bash train.sh stop <comp>       stop the autopilot and its trainer (checkpoint + exact state first)
#   bash train.sh <start> <comp>    the old argument order works too
#   DRY=1 bash train.sh math ...    print the plan, change nothing
#   LEGACY=1 bash train.sh <start> <comp>   the previous fixed-recipe launcher (scripts/box/train_legacy.sh)
#
# What the autopilot does (details: docs/AUTOPILOT.md): exact-resume training with live LR control, screens on all four
# competitions every SCREEN_EVERY steps, full B/C confirmation + staging of any winner (submission stays manual), and a
# merge round (scripts/merge/merge4train.py) whenever the competition's king (or the MAIN king) changes.
# Overrides (env): STEPS LR WD DENSE WARMUP BATCHES FREEZE EMA SCREEN_EVERY SAVE_EVERY STATE_EVERY SCALE_MAX FOLLOW_MAIN POLICY DECAY_FRAC ROLLBACK_RISE EVAL_EVERY VAL_N SCREEN_LIMIT FORCE=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
[[ ${LEGACY:-0} == 1 ]] && exec bash scripts/box/train_legacy.sh "$@"
D=/data/teutonic
source $D/env.sh
die() { echo "train.sh: $*" >&2; exit 1; }
comp_of() { case $(echo "$1" | tr 'A-Z' 'a-z') in 1|main) echo main;; 2|math) echo math;; 3|code) echo code;; 4|text) echo text;; *) echo "";; esac; }

case ${1:-} in
  ""|-h|--help) sed -n '2,19p' "$0"; exit 0;;
  status)
    python3 scripts/box/kings.py || true
    for f in $D/autopilot/*/status.json; do
      [[ -f $f ]] || continue
      python3 - "$f" <<'PY'
import json, sys, time
s = json.load(open(sys.argv[1]))
lr = s.get("lr") or {}
print(f"\n== autopilot {s['comp']}: phase {s.get('phase')}, run {s.get('run')} (updated {time.strftime('%H:%M:%S', time.gmtime(s['t']))} UTC)")
print(f"   start {s.get('start')}")
print(f"   LR scale {lr.get('scale')} (ceiling {lr.get('ceiling')}), last decision: {(lr.get('last') or {}).get('act')} - {(lr.get('last') or {}).get('why')}")
for c, b in (s.get("best") or {}).items():
    print(f"   best {c.upper():5s} {b['mu']:+.6f} (lcb {b['lcb']:+.6f}) vs {b['king']}: {b['run']}/{b['ckpt']}")
for r in s.get("ready") or []:
    print(f"   SUBMIT-READY {r['comp'].upper()}: {r.get('staged_dir')} (log {r.get('stage_log')})")
for e in (s.get("events") or [])[-6:]:
    print(f"   {time.strftime('%H:%M', time.gmtime(e['t']))} {e['kind']}: " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ('t', 'kind'))[:150])
PY
    done
    exit 0;;
  stop)
    COMP=$(comp_of "${2:-}"); [[ -n $COMP ]] || die "usage: bash train.sh stop <main|math|code|text>"
    pm2 stop autopilot_$COMP >/dev/null 2>&1 && echo "autopilot_$COMP stopped" || echo "autopilot_$COMP was not running"
    RUN=$(python3 -c "import json; s=json.load(open('$D/autopilot/$COMP/state.json')); print(s['segments'][-1]['run'])" 2>/dev/null || true)
    if [[ -n $RUN ]] && pgrep -f "[s]cripts/train/cpt_fsdp.py .*--out $D/ckpt/$RUN( |$)" >/dev/null; then
      python3 - "$D/ckpt/$RUN/CONTROL.json" <<'PY'
import json, os, sys
p = sys.argv[1]; c = json.load(open(p)) if os.path.exists(p) else {}
c.update(id=int(c.get("id", 0)) + 1, stop=True, save=False, state=False, by="train.sh stop")
open(p + ".tmp", "w").write(json.dumps(c)); os.replace(p + ".tmp", p)
PY
      echo "trainer $RUN: stop requested (it writes a checkpoint + exact state after the current step, then exits; pm2 logs $RUN)"
    fi
    exit 0;;
esac

# bash train.sh <comp> [start]   or   bash train.sh <start> <comp>
COMP=$(comp_of "$1"); START=${2:-}
if [[ -z $COMP ]]; then COMP=$(comp_of "${2:-}"); START=$1; fi
[[ -n $COMP ]] || die "usage: bash train.sh <main|math|code|text> [start]   (bash train.sh --help)"

# a merge label: write it first (GPU write = exactly the scored model, also for stochastic recipes)
if [[ -n $START && ! -f $START/model.safetensors.index.json ]] && ! python3 -c "import json,sys; sys.exit(0 if sys.argv[1] in json.load(open('configs/merge/k41/models.json')) else 1)" "$START"; then
  RECIPE=$(ls -t configs/merge/*/"$START".json 2>/dev/null | head -1 || true)
  [[ -n $RECIPE ]] || die "'$START' is neither a checkpoint dir, a models.json name nor a merge label (configs/merge/*/$START.json)"
  OUT=$D/ckpt/start_$START
  if [[ ! -f $OUT/SAVED ]] || [[ $(python3 -c "import json; print(json.load(open('$OUT/SAVED')).get('recipe', ''))" 2>/dev/null) != "$RECIPE" ]]; then
    pgrep -f "[s]cripts/train/cpt_fsdp.py" >/dev/null && die "writing $START needs the GPUs, but a trainer is running (bash train.sh stop <comp>)"
    echo "writing the merge $START -> $OUT"
    python scripts/merge/write_merge.py --recipe "$RECIPE" --out "$OUT" --force 2>&1 | tail -2
    python3 -c "import json,time; json.dump({'recipe': '$RECIPE', 't': time.time()}, open('$OUT/SAVED', 'w'))"
  fi
  START=$OUT
fi

ARGS=(--comp "$COMP")
[[ -n $START ]] && ARGS+=(--start "$START")
[[ -n ${STEPS:-} ]] && ARGS+=(--steps "$STEPS");             [[ -n ${LR:-} ]] && ARGS+=(--lr "$LR")
[[ -n ${DENSE:-} ]] && ARGS+=(--dense-frac "$DENSE");        [[ -n ${WARMUP:-} ]] && ARGS+=(--warmup "$WARMUP")
[[ -n ${BATCHES:-} ]] && ARGS+=(--batches "$BATCHES");       [[ -n ${FREEZE:-} ]] && ARGS+=(--freeze "$FREEZE")
[[ -n ${EMA:-} ]] && ARGS+=(--ema "$EMA");                   [[ -n ${SCREEN_EVERY:-} ]] && ARGS+=(--screen-every "$SCREEN_EVERY")
[[ -n ${SAVE_EVERY:-} ]] && ARGS+=(--save-every "$SAVE_EVERY"); [[ -n ${STATE_EVERY:-} ]] && ARGS+=(--state-every "$STATE_EVERY")
[[ -n ${SCALE_MAX:-} ]] && ARGS+=(--scale-max "$SCALE_MAX"); [[ -n ${FOLLOW_MAIN:-} ]] && ARGS+=(--follow-main "$FOLLOW_MAIN")
[[ -n ${WD:-} ]] && ARGS+=(--wd "$WD");                      [[ -n ${POLICY:-} ]] && ARGS+=(--policy "$POLICY")
[[ -n ${DECAY_FRAC:-} ]] && ARGS+=(--decay-frac "$DECAY_FRAC"); [[ -n ${ROLLBACK_RISE:-} ]] && ARGS+=(--rollback-rise "$ROLLBACK_RISE")
[[ -n ${EVAL_EVERY:-} ]] && ARGS+=(--eval-every "$EVAL_EVERY"); [[ -n ${VAL_N:-} ]] && ARGS+=(--val-n "$VAL_N")
[[ -n ${SCREEN_LIMIT:-} ]] && ARGS+=(--screen-limit "$SCREEN_LIMIT")
[[ ${FORCE:-0} == 1 ]] && ARGS+=(--force)

if [[ ${DRY:-0} == 1 ]]; then exec python scripts/train/autopilot.py "${ARGS[@]}" --dry; fi
# the old per-run jobs would fight the autopilot over the same GPUs / checkpoints
for name in train_guard main_watch ckpt_retention public_merge; do pm2 stop "$name" >/dev/null 2>&1 || true; done
if pgrep -f "[s]cripts/train/cpt_fsdp.py" >/dev/null && [[ ! -f $D/autopilot/$COMP/state.json ]] && [[ ${FORCE:-0} != 1 ]]; then
  die "a trainer that is not this autopilot's is on the GPUs (pm2 ls); FORCE=1 bash train.sh $COMP ... replaces it"
fi
printf -v CMD '%q ' python scripts/train/autopilot.py "${ARGS[@]}"
pm2 delete autopilot_$COMP >/dev/null 2>&1 || true
pm2 start bash --name autopilot_$COMP --restart-delay 60 --max-restarts 50 --time --output $D/logs/autopilot_$COMP.log \
  --error $D/logs/autopilot_$COMP.log -- -c "source $D/env.sh && exec $CMD" >/dev/null
pm2 save >/dev/null 2>&1 || true
echo "autopilot_$COMP started: pm2 logs autopilot_$COMP | bash train.sh status | bash dashboard.sh"
