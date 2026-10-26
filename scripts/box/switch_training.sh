#!/usr/bin/env bash
# Restart MAIN training from a better start (user 15:55: "once merging cannot improve any more, run the training" -- and
# when a merge beats the running training, train from the merge). Usage: switch_training.sh <start checkpoint dir> <why>
# Stops the current lineage (guard first, so it does not "resume" it), continues the data order where it stopped, writes
# /data/teutonic/ckpt/current_plan.json for the guard, launches main_upper_ema_m<N+1> with the m3 recipe + watcher +
# retention + guard.
set -uo pipefail
START=${1:?start dir}; WHY=${2:-}
D=/data/teutonic; cd /root/workspace/MoE
[[ -f $START/model.safetensors.index.json ]] || { echo "start $START incomplete"; exit 1; }
plan=$D/ckpt/current_plan.json
base=$(python3 -c "import json,os; print(json.load(open('$plan'))['base'] if os.path.exists('$plan') else 'main_upper_ema_m3')")
skip0=$(python3 -c "import json,os; print(json.load(open('$plan'))['skip0'] if os.path.exists('$plan') else 358400)")
lineage=$D/ckpt/$base.lineage.json
# trained steps of the lineage so far (offset of its last run + that run's last logged step)
done=$(python3 - "$lineage" "$base" <<'PY'
import json, os, sys
lin, base = sys.argv[1], sys.argv[2]
runs = json.load(open(lin))["runs"] if os.path.exists(lin) else [{"run": base, "offset": 0}]
cur = runs[-1]; steps = 0
log = f"/data/teutonic/ckpt/{cur['run']}/train_log.jsonl"
if os.path.exists(log):
    for line in open(log):
        try: steps = max(steps, int(json.loads(line).get("step", 0)))
        except Exception: pass
print(cur["offset"] + steps)
PY
)
skip=$(( skip0 + done * 2048 ))
n=$(ls -d $D/ckpt/main_upper_ema_m* 2>/dev/null | grep -oE "_m[0-9]+$" | tr -dc '0-9\n' | sort -n | tail -1)
new=main_upper_ema_m$(( n + 1 ))
echo "$(date -u +%H:%M:%S) switch: $base lineage trained $done steps -> $new from $START (skip $skip). $WHY"
pm2 delete train_guard >/dev/null 2>&1
for name in main_watch ckpt_retention main_train3 $(pm2 jlist 2>/dev/null | python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(sys.stdin) if p['name'].startswith(('launch_main_upper', 'main_upper_ema_m'))))"); do
  pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 36); do [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l) == 0 ]] && break; [[ $i -gt 6 ]] && nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9; sleep 5; done
rm -f $D/ckpt/*/PAUSE
cat > $plan <<JSON
{"base": "$new", "start": "$START", "total": 425, "skip0": $skip, "decay_abs": 340, "lr": 1.5e-6, "min_frac": 0.1, "warmup": 10,
 "seqs_per_step": 2048, "max_resumes": 3, "why": "$WHY"}
JSON
rm -f $D/ckpt/$new.lineage.json
pm2 start bash --name launch_$new --no-autorestart --time --merge-logs --output $D/logs/pm2_launch_$new.log --error $D/logs/pm2_launch_$new.log \
  -- -c "M_RUN=$new M_START=$START M_SKIP=$skip exec bash $D/logs/main_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $D/logs/pm2_main_watch.log --error $D/logs/pm2_main_watch.log -- -c "RUN=$D/ckpt/$new exec bash $D/logs/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $D/logs/pm2_ckpt_retention.log --error $D/logs/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$new KEEP=none exec bash $D/logs/ckpt_retention.sh" >/dev/null
sleep 30
pm2 start $D/logs/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $D/logs/pm2_train_guard.log --error $D/logs/pm2_train_guard.log >/dev/null
echo "$(date -u +%H:%M:%S) switched to $new"
