#!/usr/bin/env bash
# c11 (Oct 12): the long TEXT run after the mixed-pool LR sweep from c9/step-50 (TEXT screens at step 50: 1e-5 +0.001249,
# 6e-6 +0.001327, 3e-6 see 42.md). Continues the winner's step-50 checkpoint at its LR on text_mix_v1, skipping the 12 800 rows
# (50 x 256) the sweep already used, constant LR, 400 steps, screened every 25 (TEXT + MAIN); retention keeps the two newest.
#   bash start_c11.sh <start ckpt dir> <lr>
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
START=$1; LR=$2; START_VAL=${3:-}
[[ -f $START/model.safetensors.index.json ]] || { log "no checkpoint at $START"; exit 1; }
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c10c text_upper_ema_c10c; do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
RUN=text_upper_ema_c11; FREEZE='\.mlp\.gate\.|^model\.layers\.[0-7]\.'
python3 - "$START" "$FREEZE" "$LR" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c11", "start": sys.argv[1], "total": 400, "skip0": 12800, "decay_abs": 1000, "lr": float(sys.argv[3]),
           "dense_frac": 1.0, "min_frac": 0.1, "warmup": 20, "seqs_per_step": 256, "seq_len": 8192, "eval_every": 25, "max_resumes": 6,
           "launcher": "text_run.sh", "comps": "text,main", "val_raw": 1, "pools": "/data/teutonic/pools/text_mix_v1",
           "val": "/data/teutonic/pools/text_val_v1", "batches": "4x8", "save_every": 25, "ema": 0.98, "save_sr": 0, "freeze": sys.argv[2],
           "why": f"c11: long TEXT run from the LR sweep winner {sys.argv[1]} at lr {sys.argv[3]}, mixed pool, fresh rows"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START ${START_VAL:+M_START_VAL=$START_VAL }M_STEPS=400 M_SKIP=12800 M_DECAY_START=1000 M_LR=$LR M_DENSE_FRAC=1.0 M_WARMUP=20 M_EVAL_EVERY=25 M_VAL_RAW=1 M_POOLS=$D/pools/text_mix_v1 M_VAL=$D/pools/text_val_v1 M_BATCHES=4x8 M_SAVE_EVERY=25 M_EMA=0.98 M_SAVE_SR=0 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main EVERY=25 SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "c11 LAUNCHED from $START (mixed pool, lr $LR, 400 steps, skip 12 800)"
