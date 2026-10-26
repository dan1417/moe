#!/usr/bin/env bash
# Publish the `rt_smoke` candidate (= ck3_d41_025_pmom015 + 6 router steps, TEXT +0.00200 vs T1) to Hugging Face as a
# PUBLIC, GATED (manual approval) model repo yamalies/rt_smoke.  Token: HF_TOKEN env var, or the file passed as $1.
#
#   HF_TOKEN=hf_... bash scripts/submit/hf_upload_rt_smoke.sh
#   bash scripts/submit/hf_upload_rt_smoke.sh /path/to/token_file
#
# Steps: 1) write the checkpoint (CPU, ~30-40 min, 220 GB) if not already on disk; 2) create the repo + gating;
#        3) resumable multi-commit upload with hf_transfer (re-run the script to resume); 4) print the URL.
set -euo pipefail
cd "$(dirname "$0")/../.."
export REPO_ID=${REPO_ID:-yamalies/rt_smoke}
export LABEL=${LABEL:-rt_smoke}
export OUT=/data/teutonic/submit/$LABEL
[[ -n "${1:-}" ]] && export HF_TOKEN=$(cat "$1")
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN not set"; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER=1

if [[ ! -f $OUT/model.safetensors.index.json ]]; then
  echo "=== writing $OUT $(date -u +%H:%M:%S)"
  source /data/teutonic/env.sh
  python scripts/merge/write_merge.py --recipe configs/merge/k41/$LABEL.json --out "$OUT" --gpus cpu --workers 32 --force
fi
cat > "$OUT/README.md" <<EOF
---
license: other
library_name: transformers
tags: [teutonic, sn3, moe, merge]
---
# $LABEL
Teutonic-II-110B-A7B candidate built by weight merging in our basis (see 41.md / 42.md in the MoE repo):
$(python3 -c "import json; print(json.load(open('configs/merge/k41/$LABEL.json'))['description'])")

Screen vs the TEXT king T1 (2400-seq mix, 2026-10-08): TEXT +0.00200, CODE +0.00018, MATH -0.00015.
Load with \`trust_remote_code=True\` (MiMo-V2 architecture); shards + index are plain bf16 safetensors.
EOF
echo "=== repo $REPO_ID $(date -u +%H:%M:%S)"
.venv-eval/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("REPO_ID", "yamalies/rt_smoke")
print("create:", api.create_repo(repo, repo_type="model", private=False, exist_ok=True))
api.update_repo_settings(repo, gated="manual")
info = api.repo_info(repo); print("gated:", info.gated, "private:", info.private)
PY
echo "=== upload $(date -u +%H:%M:%S) (resumable; re-run to continue)"
.venv-eval/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"]); repo = os.environ.get("REPO_ID", "yamalies/rt_smoke")
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=os.environ["OUT"], num_workers=16, print_report=True)
print("done: https://huggingface.co/" + repo)
PY
