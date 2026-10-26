#!/usr/bin/env bash
# v4 (user 01:45 JST Oct 12: "increase bs ... make training fast, reliable and stably converging every step"): c6 (1 024 seqs,
# lr 1e-6) was flat - Adam moves each weight ~lr per step, far below bf16 resolution (~0.4 %), so the computing model barely
# changed. 4 096 seqs/step (16x32) with lr scaled to it (experts 3e-6, dense 3e-6: sqrt scaling of 1.5e-6 at 2 048), warmup 15,
# EMA 0.96, val (raw + EMA) every 10 steps, checkpoints + screens every 20, stochastic-rounding A/B saves.
# v3 (user 01:05 JST Oct 12: "after the 200 steps, keep training for TEXT so we beat the TEXT king completely; don't ask"):
# after c6, screen its final, continue TEXT as c7 from the best TEXT checkpoint on fresh pools v15-v17, every checkpoint saved
# twice (nearest bf16 rounding + stochastic rounding sr-<ckpt>, --save-sr 2) and both screened: c6's checkpoints did not
# improve because sub-ulp updates round away; stochastic rounding keeps them in expectation. Then MATH, then MAIN.
# v2 (user 23:20 JST Oct 11: "after step 25, halve the batch and reduce the lr so that training is stable"): 1 024 seqs/step
# (16x8), experts lr 1.0e-6 / dense 2.0e-6, EMA 0.99 (same data window as 0.98 at 2 048), save / val / screens every 50 steps.
# TEXT continues from c5/step-25 as c6. (v1:) User 22:50 JST Oct 11: "stop merging; train TEXT first, then MATH, then MAIN". Each stage: write its start model (the ES
# merge best for that competition) if missing, wait for disk, launch the run (text_run.sh: c2 recipe = experts lr 1.5e-6,
# dense 3e-6, routers / embeddings / layers 0-15 frozen, EMA 0.98, raw val, WSD), the 25-step screener for the stage's
# competitions, checkpoint retention and the crash guard; wait for the final checkpoint; next stage.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
free_gb() { df -BG --output=avail $D | tail -1 | tr -dc 0-9; }
log() { echo "$(date -u +%H:%M:%S) $*"; }
LR=3.0e-6; BATCHES=16x32; SEQS=4096; EVERY=20; EMA=0.96; SAVE_SR=2; EVAL_EVERY=10; DENSE=1.0; WARM=15
stage() {  # run start recipe steps decay skip pools val comps
  local RUN=$1 START=$2 RECIPE=$3 STEPS=$4 DECAY=$5 SKIP=$6 POOLS=$7 VAL=$8 COMPS=$9
  if [[ ! -f $START/model.safetensors.index.json ]]; then
    until (( $(free_gb) >= 1710 )); do log "$RUN: waiting for 1710 GB free to write the start (now $(free_gb))"; sleep 300; done
    log "$RUN: writing the start $START from $RECIPE"
    python scripts/merge/write_merge.py --recipe $RECIPE --out $START 2>&1 | tail -2 || return 1
  fi
  until (( $(free_gb) >= 1500 )); do log "$RUN: waiting for 1500 GB free for the launcher (now $(free_gb))"; sleep 300; done
  for p in ${POOLS//,/ } $VAL; do until [[ -f $p/meta.json ]]; do log "$RUN: waiting for pool $p"; sleep 60; done; done
  for name in train_guard main_watch ckpt_retention; do pm2 delete $name >/dev/null 2>&1; done
  python3 - "$RUN" "$START" "$STEPS" "$DECAY" "$SKIP" "$POOLS" "$VAL" "$COMPS" "$LR" "$BATCHES" "$SEQS" "$EVERY" "$EMA" "$SAVE_SR" <<'PY'
import json, sys
run, start, steps, decay, skip, pools, val, comps, lr, batches, seqs, every, ema, save_sr = sys.argv[1:]
json.dump({"base": run, "start": start, "total": int(steps), "skip0": int(skip), "decay_abs": int(decay), "lr": float(lr), "dense_frac": 1.0,
           "min_frac": 0.1, "warmup": 15, "seqs_per_step": int(seqs), "eval_every": 10, "max_resumes": 6, "launcher": "text_run.sh", "comps": comps, "val_raw": 1,
           "batches": batches, "save_every": int(every), "ema": float(ema), "save_sr": int(save_sr),
           "pools": pools, "val": val, "why": f"train sequence TEXT -> MATH -> MAIN (user 22:50 JST Oct 11): {run} from the ES merge best"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
  rm -f $D/ckpt/$RUN.lineage.json
  pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
    -- -c "M_RUN=$RUN M_START=$START M_STEPS=$STEPS M_SKIP=$SKIP M_DECAY_START=$DECAY M_LR=$LR M_DENSE_FRAC=$DENSE M_WARMUP=$WARM M_EVAL_EVERY=$EVAL_EVERY M_VAL_RAW=1 M_POOLS=$POOLS M_VAL=$VAL M_BATCHES=$BATCHES M_SAVE_EVERY=$EVERY M_EMA=$EMA M_SAVE_SR=$SAVE_SR exec bash $L/text_run.sh" >/dev/null
  pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=$COMPS EVERY=$EVERY SR=$([[ $SAVE_SR == 2 ]] && echo 1 || echo 0) RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh $EVERY" >/dev/null
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
# 1. c6 is flat (TEXT +0.001233 / +0.001229 / +0.001227 at c5-25 / c6-50 / c6-100): stop it now
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c6 text_upper_ema_c6; do pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c6/PAUSE
log "c6 stopped at step ~110 (flat); c7 with the larger batch next"
# 2. the best TEXT checkpoint of c5 / c6 (4 800-seq screens vs the TEXT king) starts c7
BEST=$(python3 - <<'PY'
import glob, json
best = (-1.0, "/data/teutonic/ckpt/text_upper_ema_c5/step-25")
for f in glob.glob("/data/teutonic/eval_cache/results/*_duel.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        terms = (r.get("challenger") or {}).get("terms") or []
        m = terms[0].get("model", "") if len(terms) == 1 else ""
        v = r.get("verdict") or {}
        if ("text_upper_ema_c5/" in m or "text_upper_ema_c6/" in m) and r.get("sample", "").startswith("0a2840c9") and v.get("n_sequences") == 4800:
            if v["mu_hat"] > best[0]:
                best = (v["mu_hat"], m)
print(best[1])
PY
)
log "c7 starts from $BEST (best TEXT screen of c5 / c6)"
stage text_upper_ema_c7 $BEST none 120 90 0 $P/text_cpt_v15,$P/text_cpt_v16,$P/text_cpt_v17 $P/text_val_v1 text,main || exit 1
stage math_upper_ema_a1 $D/ckpt/start_es1_math configs/merge/search3/s3_542b8f0299.json 45 32 0 \
  $P/math_cpt_v1 $P/math_val_v1 math,main || exit 1
stage main_upper_ema_m5 $D/ckpt/start_es1_tm configs/merge/search3/s3_d09ffd2ace.json 150 115 491520 \
  $P/main_cpt_v1,$P/main_cpt_v2,$P/main_cpt_v3,$P/main_cpt_v4,$P/main_cpt_v5,$P/main_cpt_v6 $P/main_val_v1 main,math || exit 1
log "SEQUENCE DONE"
