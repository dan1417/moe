#!/usr/bin/env bash
# User 06:10 JST Oct 12: "others train and beat the king; why can't you?" Their steps move routed experts ~2e-3 relative (32 % of
# entries) and train layers 8-15 + embeddings; ours moved ~1e-5 with layers 0-15 / embeddings frozen and ~10 restarts. After the
# new-king merge search: one committed run (c9) from the best TEXT merge - experts lr 5e-6 / dense 3e-6, warmup 50, routers +
# layers 0-7 frozen (8-15 and embeddings trained, like 5DAL), 4 096 x 2 048-token-equivalent per step on the 8 192-token pool
# (1 024 rows), EMA 0.98, 300 steps with decay over the last 60, screens every 20 - no restarts on early noise.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
# v2 (07:00 JST Oct 12, user: "analyze what the top miners do and try that way"): 5DAL's TEXT step (KTEXT - KMAIN, made in ~2.5 h)
# and 5Ek5Ko's CODE step are full-rank fine-tunes that move 40-85 % of the weights by 1-2 bf16 ulps (rel 2.5e-3 - 1e-2), train
# layers 8+ / embeddings / lm_head / shared experts and leave the routers alone; ours moved 2.5-9 % (rel 3e-5 - 2e-4). c9 mirrors
# them: lr 1e-5 (all trainable groups), warmup 30, 256 x 8 192 tokens per step (4x the updates per hour), routers + layers 0-7
# frozen, 300 steps with decay over the last 60, screens every 25 steps.
# uid 130's TEXT duel ended at LCB +0.001998 (< 0.002): no new king, no merge round. c9 starts after c8's step-40 screen from the
# best TEXT checkpoint so far (4 800-seq screens vs the TEXT king; nearest rounding)
for i in $(seq 1 120); do
  [[ -f $D/ckpt/text_upper_ema_c8/step-40/SAVED ]] && tail -n 30 $L/pm2_main_watch.log | grep -q "step 40: training resumed" && break
  sleep 30
done
BEST=$(python3 - <<'PY'
import glob, json
best = (-1.0, "/data/teutonic/ckpt/text_upper_ema_c8/step-20")
for f in glob.glob("/data/teutonic/eval_cache/results/*_duel.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        m = t[0].get("model", "") if len(t) == 1 else ""
        v = r.get("verdict") or {}
        if any(x in m for x in ("/text_upper_ema_c5/", "/text_upper_ema_c7_r1/", "/text_upper_ema_c8/")) and "/sr-" not in m \
           and r.get("sample", "").startswith("0a2840c9") and v.get("n_sequences") == 4800 and v["mu_hat"] > best[0]:
            best = (v["mu_hat"], m)
print(best[1])
PY
)
log "c9 starts from $BEST (best TEXT screen)"
for name in train_guard main_watch ckpt_retention launch_text_upper_ema_c8 text_upper_ema_c8; do pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c8/PAUSE
START=$BEST
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
RUN=text_upper_ema_c9; FREEZE='\.mlp\.gate\.|^model\.layers\.[0-7]\.'
python3 - "$START" "$FREEZE" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c9", "start": sys.argv[1], "total": 300, "skip0": 0, "decay_abs": 240, "lr": 1e-5, "dense_frac": 1.0,
           "min_frac": 0.1, "warmup": 30, "seqs_per_step": 256, "seq_len": 8192, "eval_every": 25, "max_resumes": 6, "launcher": "text_run.sh",
           "comps": "text,main", "val_raw": 1, "pools": "/data/teutonic/pools/text_long_v1", "val": "/data/teutonic/pools/text_val_v1",
           "batches": "4x8", "save_every": 25, "ema": 0.98, "save_sr": 0, "freeze": sys.argv[2],
           "why": "mirror the top miners: lr 1e-5, 256 x 8192 per step, layers 8+ / embeddings / lm_head trained, routers frozen, 300 steps"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_STEPS=300 M_SKIP=0 M_DECAY_START=240 M_LR=1.0e-5 M_DENSE_FRAC=1.0 M_WARMUP=30 M_EVAL_EVERY=25 M_VAL_RAW=1 M_POOLS=$D/pools/text_long_v1 M_VAL=$D/pools/text_val_v1 M_BATCHES=4x8 M_SAVE_EVERY=25 M_EMA=0.98 M_SAVE_SR=0 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main EVERY=25 SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "c9 LAUNCHED from $START (lr 1e-5, 256 x 8 192 per step, layers 8+ / embeddings / lm_head trained, 300 steps)"
