#!/usr/bin/env bash
# Control the parity eval queue by PID file (pattern-matching kills can hit the caller).
#   queue_ctl.sh start|restart|stop|pause|resume|status
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PID_FILE=eval_cache/queue.pid
PAUSE_FILE=eval_cache/queue.pause
export TEUTONIC_MODELS_DIRS=${TEUTONIC_MODELS_DIRS:-/opt/teutonic/models:/root/teutonic_models}
export PYTHONUNBUFFERED=1

running() { [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
stop() {
  if running; then
    pid=$(cat "$PID_FILE")
    pkill -P "$pid" 2>/dev/null   # the current job (child of the runner)
    kill "$pid" 2>/dev/null
    sleep 2
  fi
  rm -f "$PID_FILE"
}
start() {
  running && { echo "already running (pid $(cat "$PID_FILE"))"; return; }
  nohup bash scripts/eval/run_queue.sh >> eval_cache/queue_logs/runner.out 2>&1 &
  sleep 1
  echo "started runner pid $(cat "$PID_FILE" 2>/dev/null)"
}
case "${1:-status}" in
  start) start ;;
  stop) stop; echo stopped ;;
  restart) stop; rm -f "$PAUSE_FILE"; start ;;
  pause) touch "$PAUSE_FILE"; echo "paused (current job finishes first)" ;;
  resume) rm -f "$PAUSE_FILE"; echo resumed ;;
  status) running && echo "running pid $(cat "$PID_FILE")" || echo "not running"; [[ -e "$PAUSE_FILE" ]] && echo paused; tail -3 eval_cache/queue_logs/queue.log ;;
  *) echo "usage: $0 start|restart|stop|pause|resume|status"; exit 2 ;;
esac
