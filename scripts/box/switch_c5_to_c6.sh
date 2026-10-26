#!/usr/bin/env bash
# User 23:20 JST Oct 11: "after step 25, halve the batch and reduce the lr". Wait for c5's step-25 checkpoint and its screen,
# stop c5 and the v1 sequence, then run train_sequence2.sh (c6 continues from c5/step-25).
set -uo pipefail
D=/data/teutonic; L=$D/logs
N0=$(wc -l < $L/pm2_main_watch.log)
until [[ -f $D/ckpt/text_upper_ema_c5/step-25/SAVED ]] && tail -n +$((N0 + 1)) $L/pm2_main_watch.log | grep -q "step 25: training resumed"; do sleep 20; done
echo "$(date -u +%H:%M:%S) c5 step 25 saved and screened: $(tail -n +$((N0 + 1)) $L/pm2_main_watch.log | grep -E '  (TEXT|MAIN) ' | sed 's/  */ /g' | cut -c1-90 | tr '\n' ' ')"
for name in train_sequence train_guard main_watch ckpt_retention launch_text_upper_ema_c5 text_upper_ema_c5; do pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 36); do P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c5/PAUSE
echo "$(date -u +%H:%M:%S) c5 stopped; starting train_sequence2 (1 024 seqs/step, lr 1.0e-6)"
exec bash $L/train_sequence2.sh
