#!/usr/bin/env bash
# Share a checkpoint with the team (user 01:45 JST Oct 12: "upload the checkpoints to share; others will share theirs; merge
# them and train the best"). Uploads <checkpoint dir> to a PRIVATE repo <token's account>/<name> with the shared token in
# /data/teutonic/hf_token_share (written by the user; never printed, never committed). Usage: share_upload.sh <ckpt dir> <name>
set -euo pipefail
CK=${1:?checkpoint dir}; NAME=${2:?repo name}
[[ -f $CK/model.safetensors.index.json ]] || { echo "not a checkpoint: $CK"; exit 1; }
cd /root/workspace/MoE
exec .venv-eval/bin/python - "$CK" "$NAME" <<'PY'
import sys, time
from huggingface_hub import HfApi
ck, name = sys.argv[1:]
tok = open("/data/teutonic/hf_token_share").read().strip()
api = HfApi(token=tok)
owner = api.whoami()["name"]
repo = f"{owner}/{name}"
api.create_repo(repo, private=True, exist_ok=True)
t0 = time.time()
api.upload_folder(repo_id=repo, folder_path=ck, ignore_patterns=["SAVED", "*.lock"], commit_message=f"checkpoint {ck}")
print(f"SHARED {ck} -> {repo} (private) in {time.time() - t0:.0f}s", flush=True)
PY
