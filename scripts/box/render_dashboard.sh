#!/usr/bin/env bash
# Re-render the dashboard every 10 s (index.html + data.json, which the open page polls); one log line per render.
# Follows the MAIN run (main_upper_ema_m1) once it exists, else the last TEXT run (text_upper_ema_c1).
cd /root/workspace/MoE
while true; do
  # the newest MAIN run (m3, m4, ... and guard resumes m3_r1, ...), else the last TEXT run
  # the newest MAIN / TEXT run that has trained (a launched-but-stopped run without a train log shows nothing)
  # newest by train log (a directory's mtime also moves when checkpoints are deleted from it)
  RUN=$(ls -t /data/teutonic/ckpt/*/train_log.jsonl 2>/dev/null | while read f; do [[ -s $f ]] && { dirname $f; break; }; done)
  KIND=MAIN; case $(basename "$RUN") in text*) KIND=TEXT;; math*) KIND=MATH;; code*) KIND=CODE;; esac
  if [[ -n $RUN ]]; then NAME="Teutonic $KIND training ($(basename $RUN)) + merge campaign (8x B300)"
  else RUN=/data/teutonic/ckpt/text_upper_ema_c1; NAME="Teutonic training + merge campaign (8x B300)"; fi
  # the public-model merge campaign holds the GPUs (training paused): say so in the title; its panel moves to the top
  if pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name']=='public_merge' and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; then
    NAME="Teutonic merge campaign over every public model (8x B300) · training paused: $(basename $RUN) stopped at its last checkpoint"
  fi
  if out=$(timeout -s KILL 60 python3 scripts/train/dashboard.py --run "$RUN" --name "$NAME" --out /data/teutonic/www/index.html 2>&1); then
    echo "rendered: ${out##*: }"
  else
    echo "RENDER FAILED: $(echo "$out" | tail -2)"
  fi
  sleep 10
done
