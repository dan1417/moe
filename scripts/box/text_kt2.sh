#!/usr/bin/env bash
# User (02:37): "NEW TEXT king will appear! merge it! and eval TEXT with merged model vs new king!"
# 1) poll R2 for uid 83's model (8864c279...) and ingest it as KT2 the moment it is published (download + un-permute)
# 2) pre-empt the MAIN/MATH queue (stop campaign43_w2 / _w3 and their replica processes)
# 3) recipes_kt2.py (parent detection + transplants) -> TEXT campaign vs KT2 on 4 800 sequences of TEXT sample A
# 4) give the GPUs back to wave 3 (MAIN/MATH)
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8
DIGEST=8864c2796e4935f781d65a32aa46de4865adebbb3b51b7faa19a2cda101e173e
URL=https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/$DIGEST/manifest.json
if ! python3 -c "import json,sys; sys.exit(0 if 'KT2' in json.load(open('configs/merge/k41/models.json')) else 1)"; then
  until [[ $(curl -s -o /dev/null -w "%{http_code}" -A teutonic-miner/1.0 "$URL") == 200 ]]; do
    echo "$(date -u +%H:%M:%S) waiting for KT2 ($DIGEST) to be published"; sleep 20; done
  echo "$(date -u +%H:%M:%S) KT2 published: ingesting"
  # the manifest goes public before the shards finish copying (404s): retry; download_king.py resumes finished ranges
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    bash scripts/merge/ingest_king41.sh "$DIGEST" KT2 && break
    echo "$(date -u +%H:%M:%S) ingest attempt $attempt failed; retrying in 30 s"; sleep 30
  done
  python3 -c "import json,sys; sys.exit(0 if 'KT2' in json.load(open('configs/merge/k41/models.json')) else 1)" || { echo "INGEST FAILED"; exit 1; }
fi
echo "$(date -u +%H:%M:%S) KT2 ingested: $(python3 -c "import json; print(json.load(open('configs/merge/k41/models.json'))['KT2'])")"
RECIPES=$(.venv-eval/bin/python scripts/merge/recipes_kt2.py) || { echo "RECIPES FAILED"; exit 1; }
echo "recipes: $RECIPES"
pm2 stop campaign43_w3 >/dev/null 2>&1
for job in campaign43_w2; do
  P=$(pgrep -f "[c]ampaign43.py --recipe configs/merge/k43/k43_o42_125"); KIDS=$( [[ -n "$P" ]] && pgrep -P "$P" | tr '\n' ' ')
  pm2 stop $job >/dev/null 2>&1; sleep 3; for k in $P $KIDS; do kill -9 $k 2>/dev/null; done
done
P=$(pgrep -f "[c]ampaign43.py --recipe configs/merge/k43/k43_dc1_100.json"); [[ -n "$P" ]] && { KIDS=$(pgrep -P "$P" | tr '\n' ' '); kill -9 $P $KIDS 2>/dev/null; }
sleep 8; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
R=(); for f in $RECIPES; do R+=(--recipe "$f"); done
python scripts/merge/campaign43.py "${R[@]}" --comps text --king text=KT2:0.002 --campaign kt2 --limit 4800 \
  --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
echo "$(date -u +%H:%M:%S) TEXT campaign done: handing the GPUs back to wave 3"
pm2 restart campaign43_w3 >/dev/null 2>&1
