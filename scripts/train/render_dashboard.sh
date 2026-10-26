#!/usr/bin/env bash
# Re-render the dashboard every 10 s (index.html + data.json, which the open page polls); one log line per render.
# Recipe, steps, batch, pool and val set are read from /data/teutonic/logs/run_<run>.sh by dashboard.py itself.
cd /root/workspace/MoE
RUN=/data/teutonic/ckpt/text_full_ema_a
ARGS=(--run "$RUN" --name "Teutonic TEXT full-parameter run (8x B200)")
while true; do
  if out=$(timeout -s KILL 60 python3 scripts/train/dashboard.py "${ARGS[@]}" --out /data/teutonic/www/index.html 2>&1); then
    echo "rendered: ${out##*: }"
  else
    echo "RENDER FAILED: $(echo "$out" | tail -2)"
  fi
  sleep 10
done
