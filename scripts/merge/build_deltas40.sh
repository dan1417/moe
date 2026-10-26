#!/usr/bin/env bash
# Build int8 task-vector checkpoints for king steps, one after another, wherever there is disk room.
#   bash scripts/merge/build_deltas40.sh 37:36 38:37 39:38 40:39 27:24
# Output: /opt/teutonic/vec/dA_Bq if the overlay has > 260 GB free, else /root/teutonic_vec/dA_Bq if > 130 GB.
cd "$(dirname "$0")/../.."
source .venv-eval/bin/activate
for pair in "$@"; do
  a=${pair%%:*}; b=${pair##*:}; name="d${a}_${b}q"
  if [[ -f /opt/teutonic/vec/$name/DELTA.json || -f /root/teutonic_vec/$name/DELTA.json ]]; then echo "$name exists"; continue; fi
  while [[ -n "$(pgrep -f '^python scripts/merge/remote_delta40.py')" ]]; do sleep 30; done
  while true; do
    ov=$(df --output=avail -B1G /opt | tail -1 | tr -d ' '); rt=$(df --output=avail -B1G /root | tail -1 | tr -d ' ')
    if (( ov > 260 )); then out=/opt/teutonic/vec/$name; break; fi
    if (( rt > 130 )); then out=/root/teutonic_vec/$name; break; fi
    echo "$(date -u +%H:%M) waiting for disk (overlay ${ov}G, root ${rt}G)"; sleep 120
  done
  mkdir -p "$(dirname "$out")"
  echo "=== $name -> $out $(date -u +%H:%M:%S)"
  python scripts/merge/remote_delta40.py --a "$a" --b "$b" --out "$out" --int8 --workers 32 || echo "FAILED $name"
done
