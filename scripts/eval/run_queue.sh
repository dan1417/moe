#!/usr/bin/env bash
# Sequential GPU job queue for parity evals: each line of QUEUE_FILE is one
# parity_eval.py argument string; a job whose model files are not downloaded yet is
# retried after the others. Lines starting with # are ignored. Finished jobs are
# recorded in DONE_FILE so the queue can be edited and restarted safely.
#
#   QUEUE_FILE=eval_cache/queue.txt bash scripts/eval/run_queue.sh
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"
QUEUE_FILE=${QUEUE_FILE:-eval_cache/queue.txt}
DONE_FILE=${DONE_FILE:-eval_cache/queue.done}
LOG_DIR=${LOG_DIR:-eval_cache/queue_logs}
PY=${PY:-.venv-eval/bin/python}
mkdir -p "$LOG_DIR"
touch "$DONE_FILE"
echo $$ > "${PID_FILE:-eval_cache/queue.pid}"  # scripts/eval/queue_ctl.sh uses this, never pgrep

while true; do
  progressed=0
  pending=0
  while IFS= read -r job; do
    [[ -z "$job" || "$job" == \#* ]] && continue
    grep -qxF -- "$job" "$DONE_FILE" && continue
    pending=1
    # touch eval_cache/queue.pause to hold the GPUs for something else between jobs
    while [[ -e "${PAUSE_FILE:-eval_cache/queue.pause}" ]]; do sleep 15; done
    name=$(echo "$job" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-120)
    log="$LOG_DIR/$(date -u +%Y%m%dT%H%M%S)_$name.log"
    echo "[$(date -u +%FT%TZ)] START $job" | tee -a "$LOG_DIR/queue.log"
    # shellcheck disable=SC2086
    $PY scripts/eval/parity_eval.py $job > "$log" 2>&1
    status=$?
    if [[ $status -eq 0 ]]; then
      echo "$job" >> "$DONE_FILE"
      progressed=1
      echo "[$(date -u +%FT%TZ)] DONE  $job" | tee -a "$LOG_DIR/queue.log"
    elif grep -q "not downloaded under" "$log"; then
      echo "[$(date -u +%FT%TZ)] WAIT  $job (model not downloaded yet)" | tee -a "$LOG_DIR/queue.log"
    else
      echo "$job" >> "$DONE_FILE"
      echo "[$(date -u +%FT%TZ)] FAIL  $job (see $log)" | tee -a "$LOG_DIR/queue.log"
    fi
  done < "$QUEUE_FILE"
  [[ $pending -eq 0 ]] && { echo "[$(date -u +%FT%TZ)] queue empty" | tee -a "$LOG_DIR/queue.log"; sleep 60; continue; }
  [[ $progressed -eq 0 ]] && sleep 120
done
