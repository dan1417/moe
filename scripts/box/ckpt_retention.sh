#!/usr/bin/env bash
# Disk guard for the MAIN run (11:45: the step-200 save hit "No space left on device"; 220 GB per checkpoint, 25-step saves).
# Every 5 min: delete step-N checkpoints that are backed up to HF (uploads.json), are not among the newest two complete
# ones, and are not kept on purpose (step-150 = the CODE submission's source). Warn below 500 GB free.
set -uo pipefail
RUN=${RUN:-/data/teutonic/ckpt/main_upper_ema_m1}
KEEP=${KEEP:-"step-150"}
while true; do
  # numeric order of the number after the LAST "step-" (sort -t- -k2 on the path broke on run names with a dash, e.g.
  # math_step-60_...: step-100/120 sorted before step-60/80 and the two NEWEST checkpoints were deleted, Oct 13 08:48/09:53)
  mapfile -t SAVED < <(ls -d $RUN/step-* 2>/dev/null | while read d; do [[ -f $d/SAVED && ! -L $d ]] && echo "${d##*step-} $d"; done | sort -n | cut -d' ' -f2-)
  n=${#SAVED[@]}
  for ((i = 0; i < n - 2; i++)); do
    d=${SAVED[$i]}; name=$(basename $d)
    [[ " $KEEP " == *" $name "* ]] && continue
    if grep -q "\"$d\"" $RUN/uploads.json 2>/dev/null; then
      rm -rf "$d" && echo "$(date -u +%H:%M:%S) deleted $name (on HF: $(grep -o "\"repo\": \"[^\"]*\", \"ckpt\": \"$d\"" $RUN/uploads.json | cut -d'"' -f4))"
    fi
  done
  free=$(df -BG --output=avail /data/teutonic | tail -1 | tr -dc 0-9)
  (( free < 500 )) && echo "$(date -u +%H:%M:%S) WARNING: only ${free} GB free on /data/teutonic"
  sleep 300
done
