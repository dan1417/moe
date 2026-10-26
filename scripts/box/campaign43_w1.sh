#!/usr/bin/env bash
# Wave 1 of the #43 merge campaign: wait for the K43 ingest, write the recipes, screen each on MAIN/MATH/CODE/TEXT.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
until python3 -c "import json,sys; sys.exit(0 if 'K43' in json.load(open('configs/merge/k41/models.json')) else 1)"; do
  echo "$(date -u +%H:%M:%S) waiting for the K43 ingest"; sleep 30; done
python scripts/merge/recipes43.py
R=(); for f in k43_raw c1_final k43_dc1_050 k43_dc1_100 c1_t43_050 c1_t43_100 soup_k43_c1 k43_ours025; do R+=(--recipe configs/merge/k43/$f.json); done
exec python scripts/merge/campaign43.py "${R[@]}" --limit 2400 --sample tag:A --replicas 4 --batch-size 32
