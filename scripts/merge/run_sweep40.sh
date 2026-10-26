#!/usr/bin/env bash
# Supervisor for sweep40.py: restarts it after crashes and after a PAUSE (GPU handed to grad40.py) ends.
#   nohup bash scripts/merge/run_sweep40.sh > /opt/teutonic/logs/sweep40.log 2>&1 &
# Pause:  touch configs/merge/k40/queue/PAUSE   (server exits after its current item)
# Resume: rm configs/merge/k40/queue/PAUSE      Stop for good: touch configs/merge/k40/queue/STOP
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
source .venv-eval/bin/activate
export TEUTONIC_MODELS_DIRS=/opt/teutonic/models:/opt/teutonic/hf
Q=configs/merge/k40/queue
while [[ ! -f $Q/STOP ]]; do
  while [[ -f $Q/PAUSE ]]; do sleep 10; done
  python scripts/merge/sweep40.py "$@"
  echo "$(date -u +%H:%M:%S) sweep40 exited ($?)"
  sleep 5
done
