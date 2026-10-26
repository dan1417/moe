#!/usr/bin/env bash
# User (2026-10-10 02:00): 5DAL is about to take the TEXT and CODE crowns -> evaluate against the NEW kings once crowned.
# Watches both dashboards; when a king's digest changes, downloads + un-permutes it into the T1 basis
# (scripts/merge/ingest_king41.sh) and registers it in configs/merge/k41/models.json as KT2 (TEXT) / KC2 (CODE).
# A crown taken by a model we already hold (e.g. #43 itself) is only registered under the new name.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
declare -A OLD=([text]=95ef7a26 [code]=7e4aa823) NAME=([text]=KT2 [code]=KC2) DONE=()
while (( ${#DONE[@]} < 2 )); do
  for c in text code; do
    [[ -n "${DONE[$c]:-}" ]] && continue
    D=$(curl -s -A teutonic-miner/1.0 "https://teutonic.ai/competitions/$c/dashboard.json" | python3 -c "
import json,sys; k=json.load(sys.stdin).get('king') or {}; print(k.get('king_digest') or '')" 2>/dev/null)
    if [[ -n "$D" && "${D:0:8}" != "${OLD[$c]}" ]]; then
      echo "$(date -u +%H:%M:%S) new $c king $D"
      HAVE=$(python3 -c "
import json,sys; m=json.load(open('configs/merge/k41/models.json')); print(next((k for k,v in m.items() if '$D' in v), ''))")
      if [[ -n "$HAVE" ]]; then
        python3 - "${NAME[$c]}" "$HAVE" <<'PY'
import json, sys
from pathlib import Path
p = Path("configs/merge/k41/models.json"); m = json.loads(p.read_text()); m[sys.argv[1]] = m[sys.argv[2]]
p.write_text(json.dumps(m, indent=1)); print(f"registered {sys.argv[1]} = {sys.argv[2]} (already on disk)")
PY
      else
        nice -n 5 bash scripts/merge/ingest_king41.sh "$D" "${NAME[$c]}" || { echo "ingest failed for $c"; continue; }
      fi
      DONE[$c]=1
    fi
  done
  (( ${#DONE[@]} < 2 )) && { echo "$(date -u +%H:%M:%S) waiting (text ${DONE[text]:+done} code ${DONE[code]:+done})"; sleep 60; }
done
echo "$(date -u +%H:%M:%S) both new kings ingested"
