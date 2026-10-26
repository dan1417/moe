#!/usr/bin/env bash
# Stop a running king_merge screen without risking the paused trainer: kill campaign43 and ALL its descendants (the
# spawned replica processes outlive their parent), wait until only the trainer holds GPU memory, then stop king_merge
# (its finally removes PAUSE and the trainer resumes). 18:06: killing only the parent left 4 replicas (107 GB / GPU)
# next to a resuming trainer for 30 s.
desc() { local p; for p in $(pgrep -P "$1"); do echo "$p"; desc "$p"; done; }
for c in $(pgrep -f "[s]cripts/merge/campaign43.py"); do PIDS="$PIDS $c $(desc $c)"; done
[[ -n ${PIDS// } ]] && kill -9 $PIDS 2>/dev/null
TRAINER=$(pgrep -f "cpt_fsdp.py" | tr '\n' '|' | sed 's/|$//')
for i in $(seq 1 60); do
  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Ev "^(${TRAINER:-none})$" | sort -u)
  [[ -z $left ]] && break
  [[ $i -gt 5 ]] && kill -9 $left 2>/dev/null
  sleep 2
done
pm2 stop king_merge >/dev/null 2>&1
echo "screen stopped; GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | wc -l)"
