#!/usr/bin/env bash
# MAIN gains slowed (+0.00003 from step 100 to 150): is the run under-stepping? Pause the trainer and screen
# start + beta x (step-150 - start) for beta 2, 3 on MAIN and MATH (one replica set next to the paused trainer).
set -uo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
RUN=$D/ckpt/main_upper_ema_m1
resume() { rm -f "$RUN/PAUSE"; echo "$(date -u +%H:%M:%S) training resumed"; }
trap resume EXIT
until [[ ! -f $RUN/PAUSE ]]; do sleep 10; done   # main_watch's own pause first
touch "$RUN/PAUSE"
until [[ -f $RUN/PAUSED ]] && [[ $RUN/PAUSED -nt $RUN/PAUSE ]]; do sleep 5; done
echo "$(date -u +%H:%M:%S) trainer paused; screening extrapolations"
python scripts/merge/campaign43.py --recipe configs/merge/m1/m1_s150_x2.json --recipe configs/merge/m1/m1_s150_x3.json \
  --comps main,math --campaign m1 --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 0 2>&1 \
  | grep -E "^=== |  (MAIN|MATH) |Traceback|Error"
