#!/usr/bin/env bash
# Wave 2 of the #43 merge campaign on the ping-pong replica pool (src/parity/mppool.py: two persistent replica sets on
# the 4 x 2 GPU groups, one loads the next recipe while the other scores). Lets wave 1 finish k43_dc1_100, stops it
# (and its replica processes), then screens wave 2 + wave 1's remaining recipes on MAIN/MATH/CODE/TEXT.
set -uo pipefail
cd /root/workspace/MoE && source /data/teutonic/env.sh
export PARITY_MERGE_SLOTS=8  # B300: more chunks in flight per GPU = faster loads (src/parity/weights.py)
W1=/data/teutonic/logs/pm2_campaign43_w1.log
until true || ! pgrep -f "[c]ampaign43.py --recipe configs/merge/k43/k43_raw.json" >/dev/null; do
  echo "$(date -u +%H:%M:%S) waiting for wave 1's k43_dc1_100"; sleep 15; done
sleep 5
if P=$(pgrep -f "[c]ampaign43.py --recipe configs/merge/k43/k43_raw.json"); then
  KIDS=$(pgrep -P "$P" | tr '\n' ' ')
  echo "$(date -u +%H:%M:%S) stopping wave 1 (pid $P, children $KIDS)"
  pm2 stop campaign43_w1 >/dev/null
  sleep 10
  for k in $KIDS; do kill -9 "$k" 2>/dev/null; done
  sleep 10
fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
python scripts/merge/recipes43_w2.py
# 2026-10-10 02:00 user: CODE and TEXT kings are about to be replaced by 5DAL -> screen MAIN/MATH only (vs #43) until
# the new kings are crowned; 4 800 sequences per competition (the GPUs would otherwise wait on loads)
R=(); for f in k43_o42_125 orth_k43_o42_100 ties_k43_o42 sign_k43_o42_100 k43_o42_050 k43_o42_ck1_025 k43_ck1_075 \
  k43_dc1_100_mom010e k43_mom010e k43_mom020e k43_o42e_100 k43_o42d_100 k43_o42_100_mn tiestrim_k43_o42_05 slerp_k43_c1_050 \
  k43_ours025; do
  R+=(--recipe configs/merge/k43/$f.json); done
exec python scripts/merge/campaign43.py "${R[@]}" --limit 4800 --comps main,math --sample tag:A --replicas 4 --batch-size 32 --pingpong 1
