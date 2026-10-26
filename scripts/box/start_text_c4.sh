#!/usr/bin/env bash
# TEXT run c4 (13:30 JST): c3 (layers 8-15 + embeddings trainable) made val worse after 25 steps (EMA +0.00008, raw +0.00012, math +0.00039)
# -> back to c2 exactly (experts 1.5e-6, dense 3e-6, routers / embeddings / layers 0-15 frozen: the only steady val drop) and run to the end
# without further restarts (each one cost ~25 steps of optimizer state). From c2_r1/step-25 (abs 75, val 1.22686),
# only data no run has seen (skip 348 160), 150 steps (decay from 100), raw val every 25 steps.
set -uo pipefail
D=/data/teutonic; L=$D/logs
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c3 text_upper_ema_c3; do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c3/PAUSE
RUN=text_upper_ema_c4; START=$D/ckpt/text_upper_ema_c2_r1/step-25
FREEZE='\.mlp\.gate\.|^model\.embed_tokens\.|^model\.layers\.([0-9]|1[0-5])\.'
python3 - "$FREEZE" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c4", "start": "/data/teutonic/ckpt/text_upper_ema_c2_r1/step-25", "total": 150, "skip0": 348160,
           "decay_abs": 100, "lr": 1.5e-6, "dense_frac": 2.0, "min_frac": 0.1, "warmup": 10, "seqs_per_step": 2048, "max_resumes": 6,
           "launcher": "text_run.sh", "comps": "text,main", "val_raw": 1, "freeze": sys.argv[1],
           "why": "c2 recipe exactly (the only steady val drop), run to the end; c3 unfreeze of layers 8-15 + embeddings made val worse; "
                  "from c2_r1/step-25 (abs 75)"}, open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_STEPS=150 M_SKIP=348160 M_DECAY_START=100 M_LR=1.5e-6 M_DENSE_FRAC=2.0 M_VAL_RAW=1 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 30
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
echo "$(date -u +%H:%M:%S) C4 LAUNCHED"
