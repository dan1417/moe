#!/usr/bin/env bash
# Endless new-king watcher for one competition (unattended boxes): whenever the king's digest differs from the last one
# seen, ingest it (8-process download, hidden-perm recovery, un-permute, register), add it to public_models.json and make
# it the campaign's king for that competition (search/public_kings.json); then wait for the next one under the next name.
#   bash scripts/box/watch_kings_loop.sh main M46 414bcb8a     (OLD=none ingests the current king right away)
set -uo pipefail
cd "$(dirname "$0")/../.."
COMP=$1; NAME=$2; OLD=$3
URL=https://teutonic.ai/dashboard.json; [[ $COMP != main ]] && URL=https://teutonic.ai/competitions/$COMP/dashboard.json
log() { echo "$(date -u +%H:%M:%S) $*"; }
next_name() { python3 -c "import re,sys; p,n=re.match(r'([A-Z]+)(\d+)$',sys.argv[1]).groups(); print(f'{p}{int(n)+1:0{len(n)}d}')" "$1"; }
while true; do
  log "waiting for the next $COMP king (current ${OLD}); it will be $NAME"
  while true; do
    K=$(curl -s -m 60 -A teutonic-miner/1.0 "$URL" | python3 -c "import json,sys; print(json.dumps(json.load(sys.stdin).get('king') or {}))" 2>/dev/null)
    D=$(echo "$K" | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_digest') or '')" 2>/dev/null)
    [[ -n $D && $D != ${OLD}* ]] && break
    sleep 60
  done
  log "new $COMP king $NAME: $(echo "$K" | python3 -c "import json,sys; k=json.load(sys.stdin); print(k.get('model_repo'), k.get('model_digest','')[:12], k.get('crowned_at'))")"
  python3 - "$COMP" "$NAME" "$K" <<'PY'
import json, sys
comp, name, k = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
p = "configs/merge/public/public_models.json"; rows = json.load(open(p))
if not any(r["digest"] == k["model_digest"] for r in rows):
    rows.append({"name": name, "digest": k["model_digest"], "comp": comp, "reign": k.get("reign_number"), "repo": k.get("model_repo"),
                 "coldkey": k.get("coldkey"), "uid": k.get("uid"), "crowned_at": k.get("crowned_at"), "ended_at": None})
    json.dump(rows, open(p, "w"), indent=1)
PY
  if DL_PROCS=8 bash scripts/merge/ingest_king_fast.sh "$D" "$NAME"; then
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
    OLD=${D:0:8}; NAME=$(next_name "$NAME")
  else
    log "ingest of $NAME failed; retry in 10 min"; sleep 600
  fi
done
