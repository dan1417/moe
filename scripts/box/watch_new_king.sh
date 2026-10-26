#!/usr/bin/env bash
# Wait for a competition's accepted challenger to be promoted (its identity is hidden until then), then ingest it into the
# merge basis and make it that competition's king in the public-model campaign (search/public_kings.json).
#   bash scripts/box/watch_new_king.sh main M45 c7ef5487      (competition, our name for it, the current king's digest prefix)
set -uo pipefail
cd "$(dirname "$0")/../.."
COMP=$1; NAME=$2; OLD=$3
URL=https://teutonic.ai/dashboard.json; [[ $COMP != main ]] && URL=https://teutonic.ai/competitions/$COMP/dashboard.json
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "waiting for the new $COMP king (current ${OLD})"
while true; do
  K=$(curl -s -m 60 -A teutonic-miner/1.0 "$URL" | python3 -c "import json,sys; k=json.load(sys.stdin).get('king') or {}; print(json.dumps(k))" 2>/dev/null)
  D=$(echo "$K" | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_digest') or '')" 2>/dev/null)
  [[ -n $D && $D != ${OLD}* ]] && break
  sleep 60
done
log "new $COMP king: $K"
python3 - "$COMP" "$NAME" "$K" <<'PY'
import json, sys
comp, name, k = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
p = "configs/merge/public/public_models.json"; rows = json.load(open(p))
if not any(r["name"] == name for r in rows):
    rows.append({"name": name, "digest": k["model_digest"], "comp": comp, "reign": k.get("reign_number"), "repo": k.get("model_repo"),
                 "coldkey": k.get("coldkey"), "uid": k.get("uid"), "crowned_at": k.get("crowned_at"), "ended_at": None})
    json.dump(rows, open(p, "w"), indent=1)
PY
DL_PROCS=8 bash scripts/merge/ingest_king_fast.sh "$D" "$NAME" || { log "ingest failed"; exit 1; }
python3 - "$COMP" "$NAME" <<'PY'
import json, sys
from pathlib import Path
f = Path("/data/teutonic/search/public_kings.json")
k = json.loads(f.read_text()) if f.exists() else {}
k[sys.argv[1]] = sys.argv[2]
f.write_text(json.dumps(k))
print("public_kings.json:", k)
PY
log "$NAME ingested; the campaign scores $COMP against it from its next round"
