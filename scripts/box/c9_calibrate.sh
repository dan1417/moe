#!/usr/bin/env bash
# c9 calibration (07:00 JST Oct 12): at c9's step-25 and step-50 checkpoints, profile the weight change vs its start the way the
# top miners' steps were profiled (fraction of entries changed, size in bf16 ulps, spectrum). Target: their ~40 % changed at the
# end of a run; single digits after 50 steps means the lr is still too low.
cd /root/workspace/MoE; source /data/teutonic/env.sh
START=$(python3 -c "import json; print(json.load(open('/data/teutonic/ckpt/current_plan.json'))['start'])")
for S in 25 50; do
  until ls /data/teutonic/ckpt/text_upper_ema_c9*/step-$S/SAVED >/dev/null 2>&1; do sleep 60; done
  CK=$(ls -d /data/teutonic/ckpt/text_upper_ema_c9*/step-$S | head -1)
  echo "$(date -u +%H:%M:%S) CALIBRATION step-$S vs $START"
  nice -n 15 .venv-eval/bin/python scripts/merge/step_profile.py "$CK" "$START" 2>&1 | grep -v Warn
done
