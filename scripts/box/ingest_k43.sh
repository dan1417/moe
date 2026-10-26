#!/usr/bin/env bash
# New MAIN king by 5DALwjE4 (uid 236, accepted 2026-10-10 00:23:48 UTC, mu +0.003266 vs #42): wait for its promotion
# (digest published on the MAIN dashboard), then download + un-permute against T1 + register as K43.
set -uo pipefail
cd /root/workspace/MoE
until DIGEST=$(curl -s -A teutonic-miner/1.0 https://teutonic.ai/dashboard.json | python3 -c "
import json,sys; k=json.load(sys.stdin).get('king') or {}; d=k.get('king_digest') or k.get('model_digest') or ''
sys.exit(1) if d.startswith('67643e86') or not d else print(d)"); do
  echo "$(date -u +%H:%M:%S) waiting for the promotion of the new MAIN king"; sleep 60; done
echo "$(date -u +%H:%M:%S) new MAIN king digest $DIGEST"
exec nice -n 5 bash scripts/merge/ingest_king41.sh "$DIGEST" K43
