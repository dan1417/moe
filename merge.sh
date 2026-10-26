#!/usr/bin/env bash
# merge.sh — run the merge campaign: every downloaded public king (and our checkpoints) combined with every merge method,
# each candidate scored on MAIN / MATH / CODE / TEXT vs that competition's reigning king, to find the best model per
# competition (scripts/merge/public_merge.py).
#
#   bash merge.sh                 # anchor = our checkpoint C10_S50 (fetched from HF if missing), else the reigning TEXT king
#   bash merge.sh TX5             # any model name from configs/merge/k41/models.json as the anchor
#   bash merge.sh stop            # stop the campaign (its scorer finishes the recipe it is on)
#
# Needs download.sh to have fetched the four reigning kings (it waits for them). Runs under pm2 as "public_merge"
# (pm2 logs public_merge). Results: /data/teutonic/search/public.json ("best" per competition, with the recipe path and the
# merge label you pass to train.sh), the dashboard's merge panel, and search/public_ready.json for any merge that passes
# the full validator-size check (written to /data/teutonic/submit/<comp>_<label> and uploaded to HF; submitting is manual).
#
# MERGE MODE for training (scripts/merge/merge4train.py) — the best START for training against one competition's king:
#   bash merge.sh <main|math|code|text>          if that competition's autopilot runs: it gets a merge round at its next
#                                                check (it stops its trainer, merges, continues); otherwise the merge round
#                                                runs here (needs the 8 GPUs) and prints the chosen start
#   TRAIN=1 bash merge.sh math                   ... then starts the autopilot from the chosen start (bash train.sh math <it>)
#   OURS="<ckpt>::<its start>::<label>,..." bash merge.sh math   extra checkpoints of ours to merge (default: the autopilot's)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
D=/data/teutonic
if [[ ${1:-} == stop ]]; then pm2 stop public_merge && echo "public_merge stopped"; exit 0; fi
case ${1:-} in
  main|math|code|text)
    COMP=$1
    if pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='autopilot_$COMP' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; then
      mkdir -p $D/autopilot/$COMP && touch $D/autopilot/$COMP/REQUEST_MERGE
      echo "autopilot_$COMP is running: merge round requested (it starts within ~5 min; pm2 logs autopilot_$COMP)"; exit 0
    fi
    source $D/env.sh
    EXTRA=(); IFS=',' read -ra OL <<< "${OURS:-}"; for o in "${OL[@]}"; do [[ -n $o ]] && EXTRA+=(--ours "$o"); done
    OUTJ=$D/search/merge4train_$COMP.json
    python scripts/merge/merge4train.py --comp $COMP --out $OUTJ "${EXTRA[@]}"
    CHOICE=$(python3 -c "import json; print(json.load(open('$OUTJ'))['choice']['path'] or '')")
    echo "merge mode $COMP: chosen start $CHOICE (details $OUTJ)"
    [[ ${TRAIN:-0} == 1 && -n $CHOICE ]] && exec bash train.sh $COMP "$CHOICE"
    exit 0;;
esac
# GPU-busy guard: the campaign scores on all 8 GPUs; never start it next to a live trainer (FORCE=1 bash merge.sh overrides)
if [[ ${FORCE:-0} != 1 ]] && TRAINER=$(pgrep -af '[s]cripts/train/cpt_fsdp.py'); then
  OUT=$(grep -o -- '--out [^ ]*' <<<"$TRAINER" | head -1 | cut -d' ' -f2)
  echo "merge.sh: a trainer is running on the GPUs (scripts/train/cpt_fsdp.py, pids $(pgrep -f '[s]cripts/train/cpt_fsdp.py' | tr '\n' ' '))"
  echo "  run dir: ${OUT:-?}"
  echo "  stop it cleanly (checkpoint + exact state, then it exits): python3 -c 'import json,os,time; p=\"${OUT:-<run dir>}/CONTROL.json\"; c=json.load(open(p)) if os.path.exists(p) else {}; c.update(id=int(time.time()), stop=True); json.dump(c, open(p, \"w\"))'"
  echo "  then stop its pm2 jobs so nothing restarts it (pm2 ls: the run, its launch_*, main_watch, train_guard, ckpt_retention)"
  echo "  or start the campaign anyway: FORCE=1 bash merge.sh ${*:-}"
  exit 1
fi
source $D/env.sh
ANCHOR=${1:-}
# our best checkpoint (TEXT c10/step-50, on HF) joins every campaign as an ingredient and the default anchor
C10=$D/ckpt/text_upper_ema_c10/step-50
if [[ ! -f $C10/SAVED && -s $D/hf_token ]]; then
  echo "fetching our checkpoint ${OURS_REPO:-Valhalla-ai/0ea860bbc41a} -> $C10 in the background"
  mkdir -p "$C10" "$D/logs"
  (
    python - "${OURS_REPO:-Valhalla-ai/0ea860bbc41a}" "$C10" "$D/hf_token" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], token=Path(sys.argv[3]).read_text().strip(), local_dir=sys.argv[2], max_workers=32)
Path(sys.argv[2], "SAVED").touch()
PY
  ) >> "$D/logs/fetch_c10.log" 2>&1 &
  echo "c10 fetch pid $!"
fi
if [[ -z $ANCHOR ]]; then
  if [[ -f $D/ckpt/text_upper_ema_c10/step-50/SAVED ]]; then ANCHOR=C10_S50
  else ANCHOR=$(python3 -c "import json; d=json.load(open('$D/search/public_kings.json')); print(d.get('text') or d.get('main') or '')" 2>/dev/null || true); fi
fi
[[ -n $ANCHOR ]] || ANCHOR=TX6
[[ -n $ANCHOR ]] || { echo "no anchor yet: run download.sh first (or pass a model name)"; exit 1; }
if (( $(df -BG --output=avail /root | tail -1 | tr -dc 0-9) >= 5000 )) && ! pm2 describe cold_archiver >/dev/null 2>&1; then
  pm2 start --name cold_archiver --restart-delay 60 bash -- -c "source $D/env.sh && exec python scripts/merge/cold_archiver.py" >/dev/null
  echo "cold_archiver started (probed old kings move to /root/teutonic_cold instead of being deleted)"
fi
pm2 delete public_merge >/dev/null 2>&1 || true
pm2 start --name public_merge --restart-delay 60 --max-restarts 10 bash -- -c "source $D/env.sh && until python - <<'PY'
import json, sys
sys.path.insert(0, 'scripts/merge')
import public_merge as pm
av = pm.available()
k = json.load(open('$D/search/public_kings.json')) if __import__('os').path.exists('$D/search/public_kings.json') else {}
missing = sorted(({'$ANCHOR'} | set(k.values())) - set(av)) + sorted({'main', 'math', 'code', 'text'} - set(k))
print('public_merge waits for:', missing or 'nothing', flush=True)
sys.exit(1 if missing else 0)
PY
do sleep 120; done; exec python scripts/merge/public_merge.py --anchor $ANCHOR --batch 64 --score-chunk 48 --focus 1 --delete-old 1" >/dev/null
echo "public_merge started (anchor $ANCHOR): pm2 logs public_merge ; best so far: python3 -m json.tool $D/search/public.json | head -40"
