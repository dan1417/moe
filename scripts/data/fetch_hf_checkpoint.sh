#!/usr/bin/env bash
# Download the user's start checkpoint (merged body) from HuggingFace straight onto the RAM disk and
# run every validator-shaped check against king #30 (contract files, tensor set, dtypes, no MTP).
#   bash scripts/data/fetch_hf_checkpoint.sh <repo_id> [dest=/dev/shm/teutonic/start] [revision]
# Needs HF_TOKEN in the environment for a private repo.
set -euo pipefail
REPO_ID=${1:?huggingface repo id, e.g. user/teutonic-start-body}
DEST=${2:-/dev/shm/teutonic/start}
REV=${3:-main}
source /workspace/teutonic/env.sh
cd "$REPO_ROOT"
mkdir -p "$DEST"
echo "==> free RAM-disk: $(df -h /dev/shm | tail -1 | awk '{print $4}') on /dev/shm; downloading $REPO_ID@$REV -> $DEST"
# hf_transfer: many parallel streams; --local-dir writes files in place (no cache duplicate on /)
hf download "$REPO_ID" --revision "$REV" --local-dir "$DEST" --max-workers 32 \
  --exclude "*.md" --exclude ".gitattributes" 2>&1 | tail -5
rm -rf "$DEST/.cache"
echo "==> files:"; ls -la "$DEST" | head -70
echo "==> verifying against king #30 (all checks must PASS; 'weights differ from king' is expected)"
python scripts/verify/verify_submission.py --model-dir "$DEST" --king-dir /dev/shm/teutonic/king30
