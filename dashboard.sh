#!/usr/bin/env bash
# dashboard.sh — the training + merge dashboard (http://<box>:8787; re-rendered every 10 s, the page polls data.json).
#
#   bash dashboard.sh            # start (or restart) it
#   bash dashboard.sh stop
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
D=/data/teutonic
if [[ ${1:-} == stop ]]; then pm2 stop dashboard_http dashboard_render >/dev/null 2>&1 || true; echo "dashboard stopped"; exit 0; fi
mkdir -p $D/www $D/logs
cp -f scripts/box/render_dashboard.sh $D/logs/render_dashboard.sh
for name in dashboard_http dashboard_render; do pm2 delete $name >/dev/null 2>&1 || true; done
pm2 start --name dashboard_http --restart-delay 10 bash -- -c "source $D/env.sh && exec python scripts/train/dashboard_server.py --dir $D/www --port 8787 --bind 0.0.0.0" >/dev/null
pm2 start --name dashboard_render --restart-delay 10 bash -- -c "exec bash $D/logs/render_dashboard.sh" >/dev/null
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "dashboard: http://${IP:-localhost}:8787  (or ssh -L 8787:localhost:8787 <box> and open http://localhost:8787)"
