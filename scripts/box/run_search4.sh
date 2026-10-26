#!/usr/bin/env bash
# User 05:50 JST Oct 12: merge the new TEXT king (uid 130) with our latest checkpoints and the top models, every merge method.
# Waits for KTEXT2_6d6713a5 in models.json and for c8's step-40 checkpoint + screen (at most 45 min), stops c8 (all 8 GPUs for
# the merges), runs merge_search4 (round 1: 31 candidates - soups / SLERP / task arithmetic / TIES / DARE / DELLA / EMR /
# breadcrumbs / Model Stock / per-part scales - then per-competition refinement rounds).
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE
log() { echo "$(date -u +%H:%M:%S) $*"; }
until python3 -c "import json,sys; sys.exit(0 if 'KTEXT2_6d6713a5' in json.load(open('configs/merge/k41/models.json')) else 1)"; do sleep 30; done
log "new TEXT king registered: $(python3 -c "import json; print(json.load(open('configs/merge/k41/models.json'))['KTEXT2_6d6713a5'])")"
for i in $(seq 1 90); do
  [[ -f $D/ckpt/text_upper_ema_c8/step-40/SAVED ]] && ! [[ -f $D/ckpt/text_upper_ema_c8/PAUSE ]] && tail -n 20 $L/pm2_main_watch.log | grep -q "step 40: training resumed" && break
  sleep 30
done
log "stopping c8 for the merges (latest: $(ls -d $D/ckpt/text_upper_ema_c8/step-* | sort -V | tail -1))"
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c8 text_upper_ema_c8; do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c8/PAUSE $D/gpu_screen.lock
log "running merge_search4"
exec .venv-eval/bin/python scripts/merge/merge_search4.py --name all4 --max-rounds 4
