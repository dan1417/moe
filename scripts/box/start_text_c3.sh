#!/usr/bin/env bash
# TEXT run c3 (12:40 JST): back to the only lr that lowered the loss steadily (c2: experts 1.5e-6, dense 3e-6: val -0.00023
# in 50 steps; 3e-6 / 7.5e-6 were flat, 7.5e-6 raw val +0.00036) and unfreeze layers 8-15 + embeddings, which 5DAL's TEXT step
# changed as much as the upper layers (we froze 0-15 + embeddings). From c2_r1/step-25 (abs 75, val 1.22686, TEXT +0.000784),
# only data no run has seen (skip 286 720), 150 steps (decay from 100), raw val every 25 steps.
set -uo pipefail
D=/data/teutonic; L=$D/logs
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c2_r2 text_upper_ema_c2_r2; do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c2_r2/PAUSE
RUN=text_upper_ema_c3; START=$D/ckpt/text_upper_ema_c2_r1/step-25
FREEZE='\.mlp\.gate\.|^model\.layers\.[0-7]\.'
python3 - "$FREEZE" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c3", "start": "/data/teutonic/ckpt/text_upper_ema_c2_r1/step-25", "total": 150, "skip0": 286720,
           "decay_abs": 100, "lr": 1.5e-6, "dense_frac": 2.0, "min_frac": 0.1, "warmup": 10, "seqs_per_step": 2048, "max_resumes": 6,
           "launcher": "text_run.sh", "comps": "text,main", "val_raw": 1, "freeze": sys.argv[1],
           "why": "c2 lr (1.5e-6 / dense 3e-6, the only steady val drop) + layers 8-15 and embeddings trainable (5DAL trains them); "
                  "from c2_r1/step-25 (abs 75)"}, open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_STEPS=150 M_SKIP=286720 M_DECAY_START=100 M_LR=1.5e-6 M_DENSE_FRAC=2.0 M_VAL_RAW=1 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 30
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
echo "$(date -u +%H:%M:%S) C3 LAUNCHED"
