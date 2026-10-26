#!/usr/bin/env bash
# Start the TEXT run text_upper_ema_c2 (user 08:00 JST: "no merge helpful, train for TEXT and MAIN; reduce the loss consistently
# and stably") once the fresh pools and the start model exist. c1's proven-stable recipe (val -0.002 in 175 steps, no spikes):
# lr 1.5e-6 (x2 for dense), EMA 0.98, frozen routers / embeddings / layers 0-15, 2 048 seqs x 2 048 tokens per step; plus a
# warmup-stable-decay tail (constant to 180, cosine to 0.1x by 240) for a final annealing drop. Screens every 25 steps:
# TEXT vs KTEXT (bar 0.002) and MAIN vs KMAIN (0.003). Crash / hang -> train_guard resumes (same data order and LR).
set -uo pipefail
D=/data/teutonic; L=$D/logs
until grep -q "TEXT POOLS DONE" $L/pm2_build_text_pools2.log && grep -q "WRITE DONE" $L/pm2_write_text_start.log; do sleep 20; done
START=$D/ckpt/start_text_s2_33af3cb46e
[[ -f $START/model.safetensors.index.json ]] || { echo "start missing"; exit 1; }
for name in main_watch ckpt_retention train_guard main_train3; do pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 30); do [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) == 0 ]] && break; sleep 5; done
RUN=text_upper_ema_c2
cat > $D/ckpt/current_plan.json <<JSON
{"base": "$RUN", "start": "$START", "total": 240, "skip0": 0, "decay_abs": 180, "lr": 1.5e-6, "min_frac": 0.1, "warmup": 10,
 "seqs_per_step": 2048, "max_resumes": 3, "launcher": "text_run.sh", "comps": "text,main",
 "why": "TEXT run from the joint-search TEXT best s2_33af3cb46e (TEXT +0.000697 vs KTEXT at 1 200 seqs)"}
JSON
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$START M_STEPS=240 M_SKIP=0 M_DECAY_START=180 exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 30
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
echo "$(date -u +%H:%M:%S) TEXT RUN LAUNCHED: $RUN from $START"
