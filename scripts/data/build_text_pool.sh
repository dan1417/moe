#!/usr/bin/env bash
# build_pool.sh <tag> [exclude pool dirs...]: 200k-seq text-competition pool, eval-sample shards always excluded
set -euo pipefail
cd /root/workspace/MoE
export TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache PYTHONUNBUFFERED=1
TAG=$1; shift
EXC=(); for p in "$@"; do EXC+=(--exclude-pool "$p"); done
exec .venv-eval/bin/python scripts/data/build_diverse_pool.py --out /data/teutonic/pools/$TAG --n 200000 --block 32 --tag $TAG \
  --competition text --workers 48 "${EXC[@]}"
