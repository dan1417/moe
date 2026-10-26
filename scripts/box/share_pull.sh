#!/usr/bin/env bash
# List the team's shared checkpoints (the shared token's account and its orgs) and download the ones not on the box yet into
# /data/teutonic/models/shared/<repo name>, registered in configs/merge/k41/models.json as SH_<name> for merging.
# Usage: share_pull.sh [--list]
set -euo pipefail
cd /root/workspace/MoE
exec .venv-eval/bin/python - "${1:-}" <<'PY'
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
tok = open("/data/teutonic/hf_token_share").read().strip()
api = HfApi(token=tok)
me = api.whoami()
owners = [me["name"]] + [o["name"] for o in me.get("orgs", [])]
models_json = Path("configs/merge/k41/models.json"); M = json.loads(models_json.read_text())
dest = Path("/data/teutonic/models/shared")
for owner in owners:
    for m in api.list_models(author=owner, limit=200):
        local = dest / m.id.replace("/", "__")
        have = (local / "model.safetensors.index.json").exists()
        print(f"{m.id}  {'on box' if have else 'not downloaded'}", flush=True)
        if sys.argv[1] == "--list" or have:
            continue
        files = api.list_repo_files(m.id)
        if "model.safetensors.index.json" not in files:
            print(f"  skip {m.id}: no safetensors index", flush=True)
            continue
        snapshot_download(m.id, local_dir=str(local), token=tok, max_workers=32)
        M["SH_" + m.id.split("/")[-1]] = str(local)
        models_json.write_text(json.dumps(M, indent=1) + "\n")
        print(f"  downloaded {m.id} -> {local} (models.json SH_{m.id.split('/')[-1]})", flush=True)
PY
