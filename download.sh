#!/usr/bin/env bash
# download.sh — keep the latest public Teutonic kings on disk (in our merge basis) and fetch every new king as it is crowned.
#
#   bash download.sh            # the 10 most recently crowned kings + every reigning king + T1, then keep watching
#   bash download.sh 20         # the latest 20
#   bash download.sh 10 --once  # one pass in the foreground, no watching
#   bash download.sh 10 --dry   # what one pass would download / publish; downloads and writes nothing
#
# Runs under pm2 as "download_kings" (pm2 logs download_kings). One model at a time, 8 connections, and only
# after the eval sample download has finished.
# Writes configs/merge/public/public_models.json (every public king), /data/teutonic/search/public_kings.json
# ({competition: name} of every live king that is ingested), which merge.sh and train.sh use, and
# search/public_kings_live.json (live digests, "pending" kings not ingested yet). Failed ingests: search/ingest_failed.json
# (a king outside T1's hidden basis is skipped for good; network failures are retried with backoff).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
TOP=${1:-10}
[[ $TOP =~ ^[0-9]+$ ]] || { echo "usage: bash download.sh [N] [--once|--dry]"; exit 1; }
source /data/teutonic/env.sh
if [[ ${2:-} == --dry ]]; then
  exec python scripts/data/download_kings.py --top "$TOP" --dry
fi
if [[ ${2:-} == --once ]]; then
  exec python scripts/data/download_kings.py --top "$TOP"
fi
pm2 delete download_kings >/dev/null 2>&1 || true
pm2 start --name download_kings --restart-delay 60 bash -- -c \
  "source /data/teutonic/env.sh && exec python scripts/data/download_kings.py --top $TOP --loop --poll 300" >/dev/null
echo "download_kings started (latest $TOP kings + reigning + T1, checks for new kings every 5 min): pm2 logs download_kings"
