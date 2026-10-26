#!/usr/bin/env bash
# One-off (user 12:20: "evaluate the model with MAIN, our target is still MAIN"): pause the trainer and screen the newest
# complete checkpoint vs #43 on MAIN / MATH and vs KC2 on CODE (4 800 each). Usage: m1_screen_now.sh <step>
set -uo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
RUN=$D/ckpt/main_upper_ema_m1; CK=$RUN/step-$1
OWN=0
resume() { [[ $OWN == 1 ]] && { rm -f "$RUN/PAUSE"; echo "$(date -u +%H:%M:%S) training resumed"; }; OWN=0; }
trap resume EXIT
until [[ ! -f $RUN/PAUSE ]]; do sleep 10; done
touch "$RUN/PAUSE"; OWN=1
until [[ -f $RUN/PAUSED ]] && [[ $RUN/PAUSED -nt $RUN/PAUSE ]]; do sleep 5; done
echo "$(date -u +%H:%M:%S) step $1: trainer paused, screening vs #43 on MAIN and MATH and vs KC2 on CODE"
python scripts/merge/campaign43.py --recipe "$CK" --comps main,math,code --king code=KC2:0.002 --campaign m1 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 0 2>&1 | grep -E "^=== |  (MAIN|MATH|CODE) |Traceback|Error"
