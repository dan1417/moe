#!/usr/bin/env bash
# Ingest a newly published model (any competition's king) into the k41 campaign basis.
#
#   bash scripts/merge/ingest_king41.sh <sha256 digest> <short name>      e.g. ... d81e83b9... CK2
#
# download_king.py -> /data/teutonic/models/<digest>; model.safetensors.index.json written if absent;
# hidden permutation recovered against T1's embedding (perm_from_embed.py); if not the identity the checkpoint is
# re-indexed into our basis at /data/teutonic/models/<name>_b41 (unpermute_checkpoint.py) and the raw download removed;
# finally configs/merge/k41/models.json gets "<name>": <path>.
set -euo pipefail
cd "$(dirname "$0")/../.."
source /data/teutonic/env.sh
DIGEST=$1; NAME=$2
DST=/data/teutonic/models/$DIGEST
echo "=== download $(date -u +%H:%M:%S)"
python scripts/data/fast_download.py --digest "$DIGEST" --output "$DST" --procs "${DL_PROCS:-2}" --workers "${DL_WORKERS:-4}"
if [[ ! -f "$DST/model.safetensors.index.json" ]]; then
  python - "$DST" <<'EOF'
import json, struct, sys
from pathlib import Path
root = Path(sys.argv[1]); wm, total = {}, 0
for shard in sorted(root.glob("*.safetensors")):
    with open(shard, "rb") as h:
        (n,) = struct.unpack("<Q", h.read(8)); header = json.loads(h.read(n))
    for name, e in header.items():
        if name == "__metadata__": continue
        wm[name] = shard.name; total += e["data_offsets"][1] - e["data_offsets"][0]
(root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(wm.items()))}, indent=1))
print("index written:", len(wm), "tensors")
EOF
fi
echo "=== permutation vs T1 $(date -u +%H:%M:%S)"
mkdir -p configs/merge/k41/perm
CUDA_VISIBLE_DEVICES="" python scripts/merge/perm_from_embed.py --permuted "$DST" --reference "$T1" --out "configs/merge/k41/perm/perm_${NAME}.json"
if python -c "import json,sys; p=json.load(open('configs/merge/k41/perm/perm_${NAME}.json'))['perm']; sys.exit(0 if p==list(range(len(p))) else 1)"; then
  echo "identity permutation: already in our basis"; FINAL=$DST
else
  FINAL=/data/teutonic/models/${NAME}_b41
  echo "=== unpermute -> $FINAL $(date -u +%H:%M:%S)"
  python scripts/merge/unpermute_checkpoint.py --src "$DST" --perm "configs/merge/k41/perm/perm_${NAME}.json" --out "$FINAL" --workers 4
  rm -rf "$DST"
fi
flock /data/teutonic/models_json.lock python - "$NAME" "$FINAL" <<'EOF'
import json, sys
p = "configs/merge/k41/models.json"; m = json.load(open(p)); m[sys.argv[1]] = sys.argv[2]; json.dump(m, open(p, "w"), indent=1)
print("registered", sys.argv[1], "->", sys.argv[2])
EOF
touch "$FINAL/INGESTED"   # completion marker (ingest_public.py: config.json alone appears early in a download)
echo "=== done $(date -u +%H:%M:%S)"
