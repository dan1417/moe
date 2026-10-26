#!/usr/bin/env bash
# c10 (07:50 JST Oct 12): c9 (lr 1e-5, 8 192-token contiguous rows) grew the TEXT long-document component (+0.0072 -> +0.0084 in
# 50 steps) but kept eroding the 2,048-token windows (+0.00077 -> +0.00064; MAIN +0.00131 -> +0.00125) - 2 048-token raw val
# +0.0021. c10 continues from c9/step-50 on text_mix_v1 (half contiguous 8 192-token rows, half four EOS-separated 2 048-token
# windows = cold starts like the validator's windows), lr 6e-6 (all trainable groups), warmup 20, 250 steps (decay from 200).
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
START=$D/ckpt/text_upper_ema_c9/step-50
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c9 text_upper_ema_c9; do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c9/PAUSE
RUN=text_upper_ema_c10; FREEZE='\.mlp\.gate\.|^model\.layers\.[0-7]\.'
python3 - "$START" "$FREEZE" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c10", "start": sys.argv[1], "total": 250, "skip0": 0, "decay_abs": 200, "lr": 6e-6, "dense_frac": 1.0,
           "min_frac": 0.1, "warmup": 20, "seqs_per_step": 256, "seq_len": 8192, "eval_every": 25, "max_resumes": 6, "launcher": "text_run.sh",
           "comps": "text,main", "val_raw": 1, "pools": "/data/teutonic/pools/text_mix_v1", "val": "/data/teutonic/pools/text_val_v1",
           "batches": "4x8", "save_every": 25, "ema": 0.98, "save_sr": 0, "freeze": sys.argv[2],
           "why": "c10: mixed long rows + EOS-separated 2 048-token windows (windows were eroding), lr 6e-6, from c9/step-50"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_START_VAL=1.22658 M_STEPS=250 M_SKIP=0 M_DECAY_START=200 M_LR=6.0e-6 M_DENSE_FRAC=1.0 M_WARMUP=20 M_EVAL_EVERY=25 M_VAL_RAW=1 M_POOLS=$D/pools/text_mix_v1 M_VAL=$D/pools/text_val_v1 M_BATCHES=4x8 M_SAVE_EVERY=25 M_EMA=0.98 M_SAVE_SR=0 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main EVERY=25 SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "c10 LAUNCHED from $START (mixed pool, lr 6e-6, 250 steps)"
