#!/usr/bin/env bash
# MATH submission (15:20): m2/step-50 passed full MATH A/B/C vs #43 (vacant crown); random hidden permutation, verify.
# as other kings ship), no metadata beyond {"format": "pt"}, no recipe / provenance files in the directory.
# The permutation stays private (/data/teutonic/submit/private, never uploaded).
set -euo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
SRC=/data/teutonic/ckpt/main_upper_ema_m2/step-50; OUT=/data/teutonic/submit/t008; PERM=/data/teutonic/submit/private/t008.perm.pt


.venv-eval/bin/python - "$SRC" "$PERM" <<'PY'
import json, os, sys, torch
from pathlib import Path
hidden = json.loads((Path(sys.argv[1]) / "config.json").read_text())["hidden_size"]
g = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
perm = torch.randperm(hidden, generator=g)
torch.save({"perm": perm}, sys.argv[2]); os.chmod(sys.argv[2], 0o600)
print("random hidden permutation:", hidden, "fixed points", int((perm == torch.arange(hidden)).sum()))
PY
nice -n 5 .venv-eval/bin/python scripts/merge/unpermute_checkpoint.py --src "$SRC" --perm "$PERM" --out "$OUT" --workers 16
rm -f "$OUT/UNPERMUTED.json"
echo "files:"; ls "$OUT" | grep -v "^model-[0-9]*\.safetensors$" ; echo "shards: $(ls $OUT | grep -c '^model-[0-9]*\.safetensors$')"
.venv-eval/bin/python - "$OUT" <<'PY'
import json, struct, sys
from pathlib import Path
out = Path(sys.argv[1]); meta = set()
for f in sorted(out.glob("*.safetensors")):
    with open(f, "rb") as h:
        (n,) = struct.unpack("<Q", h.read(8)); meta.add(json.dumps(json.loads(h.read(n)).get("__metadata__")))
print("safetensors metadata:", meta)
PY
.venv-eval/bin/python scripts/verify/verify_submission.py --model-dir "$OUT" --king-dir /data/teutonic/models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0 | tail -4
echo "PERMUTE DONE"
