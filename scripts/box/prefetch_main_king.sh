#!/usr/bin/env bash
# User goal MAIN: uid 107 (5ERPUKwf...) took the MAIN crown (16:46, +0.003805 vs #43). king_merge is busy with the MATH
# king's waves, so download + ingest the new MAIN king now (CPU / network only) under king_merge's name KMAIN_<8>:
# king_merge then finds it registered and goes straight to the MAIN merge waves.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
until D=$(curl -s -A teutonic-miner/1.0 https://teutonic.ai/dashboard.json | python3 -c "
import json,sys; k=(json.load(sys.stdin).get('king') or {}).get('king_digest') or ''
sys.exit(1) if (not k or k.startswith('9b202f1a')) else print(k)"); do echo "$(date -u +%H:%M:%S) waiting for the MAIN promotion"; sleep 30; done
NAME=KMAIN_${D:0:8}
echo "$(date -u +%H:%M:%S) new MAIN king $D -> $NAME"
until [[ $(curl -s -o /dev/null -w "%{http_code}" -A teutonic-miner/1.0 https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/$D/manifest.json) == 200 ]]; do sleep 20; done
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  DL_WORKERS=64 nice -n 5 bash scripts/merge/ingest_king41.sh "$D" "$NAME"
  python3 -c "import json,sys; m=json.load(open('configs/merge/k41/models.json')); sys.exit(0 if '$NAME' in m else 1)" && break
  echo "$(date -u +%H:%M:%S) ingest attempt $attempt failed; retrying in 30 s"; sleep 30
done
python3 -c "import json; m=json.load(open('configs/merge/k41/models.json')); print('registered', '$NAME', '->', m.get('$NAME'))"
