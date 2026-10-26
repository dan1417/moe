#!/usr/bin/env bash
# User (05:12): back up the best work to HF so it survives losing this box. Low priority (nice 19 / ionice idle class 7)
# so the training run is not slowed. Gated (manual approval) public repos under random names; the permutation keys of the
# staged submissions go to one private repo (a few KB).
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
for CK in /data/teutonic/ckpt/start_m1 /data/teutonic/submit/t005; do
  echo "$(date -u +%H:%M:%S) uploading $CK"
  nice -n 19 ionice -c2 -n7 python scripts/submit/hf_upload_ckpt.py --ckpt "$CK" --workers 8 || echo "UPLOAD FAILED: $CK"
done
nice -n 19 python - <<'PY'
import os, secrets
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi(token=Path("/data/teutonic/hf_token").read_text().strip())
user = api.whoami()["name"]
repo = f"{user}/{secrets.token_hex(6)}"
api.create_repo(repo, repo_type="model", private=True, exist_ok=False)
files = [p for p in Path("/data/teutonic/submit/private").glob("*.perm.pt")] + list(Path("/data/teutonic/submit").glob("*.recipe.json"))
for p in files:
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo, repo_type="model")
print(f"KEYS UPLOADED to private {repo}: {[p.name for p in files]}")
Path("/data/teutonic/submit/private/keys_repo.txt").write_text(repo + "\n")
PY
echo "$(date -u +%H:%M:%S) ALL UPLOADS DONE"
