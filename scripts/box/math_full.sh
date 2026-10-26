#!/usr/bin/env bash
# User (14:35): MATH gate -- the MATH crown is vacant (challengers face the MAIN king #43 on the MATH sample, bar 0.002).
# Full MATH A, B, C (~31.8k sequences each) vs #43 for the best MATH screen (m2/step-50: +0.002557), GPUs free (probe stopped).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
python scripts/merge/campaign43.py --recipe /data/teutonic/ckpt/main_upper_ema_m2/step-50 --comps math --campaign mathfull \
  --sample tag:A,tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1 2>&1 | grep -E "^=== |  MATH|Traceback|Error"
echo "$(date -u +%H:%M:%S) MATH full done"
