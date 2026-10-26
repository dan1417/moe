#!/usr/bin/env bash
# MATH training from the TEXT milestone checkpoint (text_milestone.sh): stop the TEXT run, open-ended MATH run with the c7 recipe
# (4 096 seqs/step, lr 3e-6, EMA 0.96, val every 10, save + MATH / MAIN screens every 20, stochastic-rounding A/B).
set -uo pipefail
D=/data/teutonic; L=$D/logs; START=${1:?start}
source <(sed -n '/^free_gb()/,/^}$/p;/^log()/p;/^LR=/p;/^stage()/,/^}$/p' $L/train_sequence5.sh)
for name in train_guard main_watch ckpt_retention $(pm2 jlist 2>/dev/null | python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(sys.stdin) if 'text_upper_ema_c7' in p['name']))"); do pm2 delete $name >/dev/null 2>&1; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
cd /root/workspace/MoE; P=$D/pools
POOLS=$(ls -d $P/math_cpt_v* | sort -V | paste -sd,)
stage math_upper_ema_a1 "$START" none 1000 1000 0 $POOLS $P/math_val_v1 math,main
