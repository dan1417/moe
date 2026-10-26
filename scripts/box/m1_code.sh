#!/usr/bin/env bash
# User (09:39): evaluate CODE too (vs KC2) to compete for the CODE crown. After the extrapolation screen: pause the trainer,
# score step-150 and its x2 / x3 extrapolations on CODE vs KC2.
set -uo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
RUN=$D/ckpt/main_upper_ema_m1
OWN=0  # remove the shared PAUSE only if this script created it
resume() { [[ $OWN == 1 ]] && { rm -f "$RUN/PAUSE"; echo "$(date -u +%H:%M:%S) training resumed"; }; OWN=0; }
trap resume EXIT
until [[ ! -f $RUN/PAUSE ]]; do sleep 10; done
touch "$RUN/PAUSE"; OWN=1
until [[ -f $RUN/PAUSED ]] && [[ $RUN/PAUSED -nt $RUN/PAUSE ]]; do sleep 5; done
echo "$(date -u +%H:%M:%S) trainer paused; CODE vs KC2"
python scripts/merge/campaign43.py --recipe $RUN/step-150 \
  --comps code --king code=KC2:0.002 --campaign m1 --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 0 2>&1 \
  | grep -E "^=== |  CODE |Traceback|Error"
