#!/usr/bin/env bash
# Hand the 8 GPUs from the c11 TEXT run to the public-model merge campaign (user 11:30 JST Oct 12: "use all of the public
# models for merging ... use the 8 B300 efficiently"). Waits for c11's step-25 checkpoint (a new merge ingredient) and its
# 4 800-seq TEXT / MAIN screen by main_watch, then stops the run's jobs (trainer, launcher, guard, watcher, retention),
# frees the GPUs and starts scripts/merge/public_merge.py under pm2. c11 resumes later from step-25 or the best merge.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=/data/teutonic/ckpt/text_upper_ema_c11
log() { echo "$(date -u +%H:%M:%S) $*"; }
until [[ -f $RUN/step-25/SAVED ]]; do sleep 20; done
log "step-25 saved"
# main_watch pauses the trainer for the step-25 screen; wait until the trainer has logged a step after 25 (screen done)
until python3 -c "
import json,sys
L=[json.loads(l) for l in open('$RUN/train_log.jsonl')]
sys.exit(0 if any(x.get('step',0)>=26 and 'train' in x for x in L) else 1)" && [[ ! -f $RUN/PAUSE ]]; do sleep 20; done
log "step-25 screen done; stopping c11"
pm2 stop train_guard launch_text_upper_ema_c11 text_upper_ema_c11 main_watch ckpt_retention >/dev/null 2>&1
pkill -f "scripts/train/cpt_fsdp.py" ; sleep 5
for i in $(seq 1 60); do
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] && break
  [[ $i -gt 6 ]] && pkill -9 -f "scripts/train/cpt_fsdp.py"
  sleep 5
done
log "GPUs free: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
rm -f /data/teutonic/gpu_screen.lock
# anchor = c11/step-25 if its TEXT screen (4 800 seqs) beats c10/step-50's +0.001327, else c10/step-50
ANCHOR=$(python3 -c "
import glob, json, os
t0 = os.path.getmtime('$RUN/step-25/SAVED'); best = None
for f in sorted(glob.glob('/data/teutonic/eval_cache/results/*_duel.json'), key=os.path.getmtime):
    if os.path.getmtime(f) < t0: continue
    for r in json.load(open(f)).get('results', []):
        if (r.get('challenger') or {}).get('label') == 'step-25' and (r.get('king') or {}).get('label', '').startswith('9a39afee'):
            best = (r.get('verdict') or {}).get('mu_hat')
print('C11_S25' if best is not None and best > 0.001327 else 'C10_S50')
")
log "anchor: $ANCHOR"
pm2 delete public_merge >/dev/null 2>&1
pm2 start --name public_merge --restart-delay 60 --max-restarts 5 bash -- -c \
  "cd $(pwd) && source /data/teutonic/env.sh && exec python scripts/merge/public_merge.py --anchor $ANCHOR --batch 16"
log "public_merge started"
