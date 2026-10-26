#!/usr/bin/env bash
# Superseded by the root scripts (setup.sh, download.sh, merge.sh, dashboard.sh); kept as the one-shot equivalent:
#   HF_TOKEN=hf_... bash scripts/box/bootstrap_merge_box.sh [N latest kings, default 10]
set -euo pipefail
cd "$(dirname "$0")/../.."
bash setup.sh
bash download.sh "${1:-10}"
bash merge.sh
bash dashboard.sh
