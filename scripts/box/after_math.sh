#!/usr/bin/env bash
# When the MATH full check finishes (GPUs free): launch MAIN run m3 + its watcher (screens every 25 steps) + retention.
set -uo pipefail
cd /root/workspace/MoE
until grep -q "MATH full done" /data/teutonic/logs/pm2_math_full.log; do sleep 15; done
sleep 10
pm2 start bash --name main_train3 --no-autorestart --time --merge-logs --output /data/teutonic/logs/pm2_main_train3.log --error /data/teutonic/logs/pm2_main_train3.log -- -c "exec bash /data/teutonic/logs/main_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output /data/teutonic/logs/pm2_main_watch.log --error /data/teutonic/logs/pm2_main_watch.log -- -c "RUN=/data/teutonic/ckpt/main_upper_ema_m3 exec bash /data/teutonic/logs/main_watch.sh 25" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output /data/teutonic/logs/pm2_ckpt_retention.log --error /data/teutonic/logs/pm2_ckpt_retention.log -- -c "RUN=/data/teutonic/ckpt/main_upper_ema_m3 KEEP=none exec bash /data/teutonic/logs/ckpt_retention.sh" >/dev/null
echo "$(date -u +%H:%M:%S) launched main_upper_ema_m3 + watcher + retention"
