#!/usr/bin/env bash
# User 22:50 JST Oct 11: "stop merging; train TEXT first, then MATH, then MAIN". Each stage: write its start model (the ES
# merge best for that competition) if missing, wait for disk, launch the run (text_run.sh: c2 recipe = experts lr 1.5e-6,
# dense 3e-6, routers / embeddings / layers 0-15 frozen, EMA 0.98, raw val, WSD), the 25-step screener for the stage's
# competitions, checkpoint retention and the crash guard; wait for the final checkpoint; next stage.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
free_gb() { df -BG --output=avail $D | tail -1 | tr -dc 0-9; }
log() { echo "$(date -u +%H:%M:%S) $*"; }
stage() {  # run start recipe steps decay skip pools val comps
  local RUN=$1 START=$2 RECIPE=$3 STEPS=$4 DECAY=$5 SKIP=$6 POOLS=$7 VAL=$8 COMPS=$9
  if [[ ! -f $START/model.safetensors.index.json ]]; then
    until (( $(free_gb) >= 1710 )); do log "$RUN: waiting for 1710 GB free to write the start (now $(free_gb))"; sleep 300; done
    log "$RUN: writing the start $START from $RECIPE"
    python scripts/merge/write_merge.py --recipe $RECIPE --out $START 2>&1 | tail -2 || return 1
  fi
  until (( $(free_gb) >= 1500 )); do log "$RUN: waiting for 1500 GB free for the launcher (now $(free_gb))"; sleep 300; done
  for name in train_guard main_watch ckpt_retention; do pm2 delete $name >/dev/null 2>&1; done
  python3 - "$RUN" "$START" "$STEPS" "$DECAY" "$SKIP" "$POOLS" "$VAL" "$COMPS" <<'PY'
import json, sys
run, start, steps, decay, skip, pools, val, comps = sys.argv[1:]
json.dump({"base": run, "start": start, "total": int(steps), "skip0": int(skip), "decay_abs": int(decay), "lr": 1.5e-6, "dense_frac": 2.0,
           "min_frac": 0.1, "warmup": 10, "seqs_per_step": 2048, "max_resumes": 6, "launcher": "text_run.sh", "comps": comps, "val_raw": 1,
           "pools": pools, "val": val, "why": f"train sequence TEXT -> MATH -> MAIN (user 22:50 JST Oct 11): {run} from the ES merge best"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
  rm -f $D/ckpt/$RUN.lineage.json
  pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
    -- -c "M_RUN=$RUN M_START=$START M_STEPS=$STEPS M_SKIP=$SKIP M_DECAY_START=$DECAY M_LR=1.5e-6 M_DENSE_FRAC=2.0 M_VAL_RAW=1 M_POOLS=$POOLS M_VAL=$VAL exec bash $L/text_run.sh" >/dev/null
  pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=$COMPS RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
  pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
  sleep 60
  pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
  log "$RUN launched ($STEPS steps, decay from $DECAY, skip $SKIP, pools $POOLS, screens $COMPS)"
  until ls $D/ckpt/${RUN}*/final/SAVED >/dev/null 2>&1; do
    sleep 120
    if grep -q "PREFLIGHT FAILED" $L/pm2_launch_$RUN.log 2>/dev/null; then log "$RUN: $(grep PREFLIGHT $L/pm2_launch_$RUN.log | tail -1)"; return 1; fi
    if grep -q "GIVING UP\|nothing to resume" $L/pm2_train_guard.log 2>/dev/null && ! pgrep -f "[c]pt_fsdp.py" >/dev/null; then log "$RUN: guard gave up"; return 1; fi
  done
  log "$RUN FINISHED: $(ls -d $D/ckpt/${RUN}*/final)"
  for name in main_watch ckpt_retention; do pm2 delete $name >/dev/null 2>&1; done
}
P=$D/pools
stage text_upper_ema_c5 $D/ckpt/start_es1_tm configs/merge/search3/s3_d09ffd2ace.json 125 85 348160 \
  $P/text_cpt_v1,$P/text_cpt_v12,$P/text_cpt_v13,$P/text_cpt_v14 $P/text_val_v1 text,main || exit 1
stage math_upper_ema_a1 $D/ckpt/start_es1_math configs/merge/search3/s3_542b8f0299.json 95 65 0 \
  $P/math_cpt_v1 $P/math_val_v1 math,main || exit 1
stage main_upper_ema_m5 $D/ckpt/start_es1_tm configs/merge/search3/s3_d09ffd2ace.json 150 110 491520 \
  $P/main_cpt_v1,$P/main_cpt_v2,$P/main_cpt_v3,$P/main_cpt_v4,$P/main_cpt_v5,$P/main_cpt_v6 $P/main_val_v1 main,math || exit 1
log "SEQUENCE DONE"
