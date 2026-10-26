#!/usr/bin/env bash
# User 06:10 JST Oct 12: "others train and beat the king; why can't you?" Their steps move routed experts ~2e-3 relative (32 % of
# entries) and train layers 8-15 + embeddings; ours moved ~1e-5 with layers 0-15 / embeddings frozen and ~10 restarts. After the
# new-king merge search: one committed run (c9) from the best TEXT merge - experts lr 5e-6 / dense 3e-6, warmup 50, routers +
# layers 0-7 frozen (8-15 and embeddings trained, like 5DAL), 4 096 x 2 048-token-equivalent per step on the 8 192-token pool
# (1 024 rows), EMA 0.98, 300 steps with decay over the last 60, screens every 20 - no restarts on early noise.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
until grep -q "SEARCH DONE" $L/pm2_search4.log 2>/dev/null || ! pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='search4' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do sleep 60; done
REC=$(python3 - <<'PY'
import json
s = json.load(open("/data/teutonic/search/all4.json"))
print(s["best"]["text"]["recipe"])
PY
)
log "best TEXT merge: $REC ($(python3 -c "import json; print(json.load(open('$REC'))['description'])"))"
START=$D/ckpt/start_c9_$(basename $REC .json)
[[ -f $START/model.safetensors.index.json ]] || python scripts/merge/write_merge.py --recipe "$REC" --out "$START" 2>&1 | tail -2
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
RUN=text_upper_ema_c9; FREEZE='\.mlp\.gate\.|^model\.layers\.[0-7]\.'
python3 - "$START" "$FREEZE" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c9", "start": sys.argv[1], "total": 300, "skip0": 0, "decay_abs": 240, "lr": 5e-6, "dense_frac": 0.6,
           "min_frac": 0.1, "warmup": 50, "seqs_per_step": 1024, "seq_len": 8192, "eval_every": 10, "max_resumes": 6, "launcher": "text_run.sh",
           "comps": "text,main", "val_raw": 1, "pools": "/data/teutonic/pools/text_long_v1", "val": "/data/teutonic/pools/text_val_v1",
           "batches": "4x32", "save_every": 20, "ema": 0.98, "save_sr": 0, "freeze": sys.argv[2],
           "why": "committed run matching the kings' update size: experts 5e-6, layers 8-15 + embeddings trained, 300 steps, no restarts"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_STEPS=300 M_SKIP=0 M_DECAY_START=240 M_LR=5.0e-6 M_DENSE_FRAC=0.6 M_WARMUP=50 M_EVAL_EVERY=10 M_VAL_RAW=1 M_POOLS=$D/pools/text_long_v1 M_VAL=$D/pools/text_val_v1 M_BATCHES=4x32 M_SAVE_EVERY=20 M_EMA=0.98 M_SAVE_SR=0 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main EVERY=20 SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 20" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "c9 LAUNCHED from $START (experts 5e-6, layers 8-15 + embeddings trained, 8 192-token rows, 300 steps)"
