#!/usr/bin/env bash
# Keep merge screens off the GPUs while the TEXT run trains (user 08:00 JST: "no merge helpful, let's train"): hold the GPU screen
# lock king_merge waits on (refreshed every 10 min; king_merge treats a lock older than 4 h as stale). Released when the run's
# trainer is gone, so the new CODE king's merges run right after with all 8 GPUs.
RUN=${1:-text_upper_ema_c4}
L=/data/teutonic/gpu_screen.lock
sleep 120
while pm2 jlist 2>/dev/null | python3 -c "import json,sys; sys.exit(0 if any(p['name'].startswith('$RUN') and p['pm2_env']['status']=='online' for p in json.load(sys.stdin)) else 1)"; do
  echo "training $RUN" > $L; sleep 600
done
rm -f $L; echo "$(date -u +%H:%M:%S) $RUN no longer training: GPU screen lock released"
