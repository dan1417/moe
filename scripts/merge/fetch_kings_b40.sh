#!/usr/bin/env bash
# Download public kings and map each into king #40's hidden basis (function-preserving re-index).
#
#   bash scripts/merge/fetch_kings_b40.sh 39 36 35 33 32
#
# For each reign: download_king.py -> $DL/rN, perm_from_embed.py vs #40 -> configs/merge/k40/perm_rN.json,
# unpermute_checkpoint.py -> $OUT/rN_b40, then the raw download is deleted. Kings already in #40's basis
# (perm == identity) are moved instead of rewritten.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv-eval/bin/activate
K40=${K40:-/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c}
DL=${DL:-/root/teutonic_dl}
OUT=${OUT:-/opt/teutonic/hf}
mkdir -p "$DL" "$OUT" configs/merge/k40
for r in "$@"; do
  dst="$OUT/r${r}_b40"
  if [[ -f "$dst/UNPERMUTED.json" || -f "$dst/.b40_done" ]]; then echo "r$r already in basis"; continue; fi
  echo "=== reign $r: download $(date -u +%H:%M:%S)"
  python scripts/data/download_king.py --reign "$r" --output "$DL/r$r" --workers 16
  echo "=== reign $r: permutation vs #40 $(date -u +%H:%M:%S)"
  python scripts/merge/perm_from_embed.py --permuted "$DL/r$r" --reference "$K40" --out "configs/merge/k40/perm_r${r}.json"
  if python -c "import json,sys; p=json.load(open('configs/merge/k40/perm_r${r}.json'))['perm']; sys.exit(0 if p==list(range(len(p))) else 1)"; then
    echo "reign $r already in #40 basis; moving"
    if [[ -e "$dst" ]]; then  # existing dir or symlink to one: move the contents, never nest
      find "$DL/r$r" -mindepth 1 -maxdepth 1 -exec mv -t "$dst/" {} + && rmdir "$DL/r$r"
    else
      mv "$DL/r$r" "$dst"
    fi
    touch "$dst/.b40_done"
  else
    echo "=== reign $r: unpermute $(date -u +%H:%M:%S)"
    python scripts/merge/unpermute_checkpoint.py --src "$DL/r$r" --perm "configs/merge/k40/perm_r${r}.json" --out "$dst" --workers ${UNPERM_WORKERS:-2}
    rm -rf "$DL/r$r"
  fi
  echo "=== reign $r: done $(date -u +%H:%M:%S) -> $dst"
done
