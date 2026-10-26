#!/usr/bin/env bash
# User (13:40): submit t007 (MAIN-run step-150, passed full CODE A/B/C vs KC2) to CODE with sn3auto006.
# final (hotkey file + finalized registration), then submit_ckpt.py: Ed25519 + registration checks, verify READY, auth, empty
# low priority: the MAIN run keeps training), prefix check, king re-check (KC2 fa428e5e), ready.
set -uo pipefail
cd /root/workspace/MoE
until .venv-miner/bin/teutonic-miner register --wallet-name autotao --hotkey-name sn3auto006 --network finney --netuid 3 \
      --check-only >/tmp/reg006.out 2>&1; do
  echo "$(date -u +%H:%M:%S) sn3auto006: $(tail -1 /tmp/reg006.out | cut -c1-100)"; sleep 30; done
echo "$(date -u +%H:%M:%S) sn3auto006 registered: submitting t007 to CODE"
exec nice -n 10 ionice -c2 -n7 .venv-miner/bin/python /data/teutonic/logs/submit_ckpt.py --hotkey sn3auto006 \
  --model-dir /data/teutonic/submit/t007 --name teutonic-II-110B-A7B-006 --competition code --ready
