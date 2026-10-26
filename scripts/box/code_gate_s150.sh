#!/usr/bin/env bash
# One-off: the CODE gate for step-150 now (pauses the trainer; PAUSE ownership as in main_watch.sh).
set -uo pipefail
RUN=/data/teutonic/ckpt/main_upper_ema_m1
OWN=0
resume() { [[ $OWN == 1 ]] && { rm -f "$RUN/PAUSE"; echo "$(date -u +%H:%M:%S) training resumed"; }; OWN=0; }
trap resume EXIT
until [[ ! -f $RUN/PAUSE ]]; do sleep 10; done
touch "$RUN/PAUSE"; OWN=1
until [[ -f $RUN/PAUSED ]] && [[ $RUN/PAUSED -nt $RUN/PAUSE ]]; do sleep 5; done
echo "$(date -u +%H:%M:%S) trainer paused; CODE gate for step-150"
bash /data/teutonic/logs/code_gate.sh $RUN/step-150 150
