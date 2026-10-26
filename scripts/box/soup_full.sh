#!/usr/bin/env bash
# User (03:44): submit soup_k43_c1_050 to TEXT. Full-sample check vs KT2 on TEXT B and C (A was screened at 4 800:
# +0.003832) while the submission is written / permuted / uploaded; then all_kings resumes (stage 1 is cached).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
python scripts/merge/campaign43.py --recipe configs/merge/k43/soup_k43_c1_050.json --comps text --king text=KT2:0.002 \
  --campaign kt2full --sample tag:B,tag:C --limit 40000 --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) soup full B/C done: resuming all_kings"
pm2 restart all_kings >/dev/null 2>&1
