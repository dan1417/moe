#!/usr/bin/env bash
# User 05:50 JST Oct 12: "a new TEXT king is coming: download it and merge it with my latest checkpoints and the top models, every
# merge method". uid 130 (coldkey 5Fxtier6, digest 6d6713a5...) leads its TEXT duel (provisional LCB +0.00204 at 76 %). Download +
# un-permute (ingest_king41.sh) as soon as R2 publishes its manifest (at or after promotion) -> models.json KTEXT2_<8>.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
D=6d6713a5879b40c0cc3e0a2e2b3382387fc3bb09ebb910ceb71fcd13cf767183; NAME=KTEXT2_${D:0:8}
R2=https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/$D/manifest.json
until [[ $(curl -s -o /dev/null -w "%{http_code}" -A teutonic-miner/1.0 $R2) == 200 ]]; do
  K=$(curl -s -A teutonic-miner/1.0 https://teutonic.ai/competitions/text/dashboard.json | python3 -c "import json,sys; print(((json.load(sys.stdin).get('king') or {}).get('king_digest') or '')[:8])" 2>/dev/null)
  echo "$(date -u +%H:%M:%S) waiting for the manifest (TEXT king now $K)"; sleep 30
done
echo "$(date -u +%H:%M:%S) manifest published; ingesting $D as $NAME"
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  DL_WORKERS=64 nice -n 5 bash scripts/merge/ingest_king41.sh "$D" "$NAME"
  python3 -c "import json,sys; m=json.load(open('configs/merge/k41/models.json')); sys.exit(0 if '$NAME' in m else 1)" && break
  echo "$(date -u +%H:%M:%S) ingest attempt $attempt failed; retrying in 30 s"; sleep 30
done
python3 -c "import json; m=json.load(open('configs/merge/k41/models.json')); print('registered', '$NAME', '->', m.get('$NAME'))"
