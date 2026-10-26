#!/usr/bin/env bash
# Launch (or resume) a MAIN training run with preflight checks. Env (defaults = the m3 plan):
#   M_RUN (run dir name), M_START (init checkpoint dir), M_STEPS, M_SKIP (sequences of the seed-0 data order already used),
#   M_WARMUP, M_DECAY_START (warmup-stable-decay: hold the LR until this step of THIS run, then cosine), M_LR, M_MIN_LR_FRAC,
#   M_LOSS_WEIGHTS. Plan m3 (user 14:35): from m1/step-175, LR 1.5e-6 / dense 3e-6, code-reasoning + sft-reasoning x2,
#   425 steps with the LR decaying over the last 85 (user 15:05: "make the training go well and stably").
set -euo pipefail
D=/data/teutonic; source $D/env.sh; cd /root/workspace/MoE
K43=/data/teutonic/models/9b202f1afae9c3575cdf6a57a37326aeb0a8ab8704658e3b64224bee72c537da
# M_* names: env.sh (sourced above) exports START etc. for the c1 setup -- plain names here were silently overridden
# (15:12: m3 started from start_a7745d17833c instead of m1/step-175; stopped within seconds)
RUN=${M_RUN:-main_upper_ema_m3}; START=${M_START:-$D/ckpt/main_upper_ema_m1/step-175}
STEPS=${M_STEPS:-425}; SKIP=${M_SKIP:-358400}; WARMUP=${M_WARMUP:-10}; DECAY_START=${M_DECAY_START:-340}
LR=${M_LR:-1.5e-6}; MIN_LR_FRAC=${M_MIN_LR_FRAC:-0.1}; LOSS_WEIGHTS=${M_LOSS_WEIGHTS:-code-reasoning=2,sft-reasoning=2}
# ---- preflight: refuse to start into a known failure
fail() { echo "PREFLIGHT FAILED: $*"; exit 3; }
[[ -f $START/model.safetensors.index.json ]] || fail "start checkpoint $START incomplete"
[[ $START == "$D/ckpt/main_upper_ema_m1/step-175" || -n ${M_START:-} ]] || fail "unexpected start $START (set M_START explicitly)"
free=$(df -BG --output=avail $D | tail -1 | tr -dc 0-9); (( free >= 1500 )) || fail "only ${free} GB free (need 1500: 220 GB per checkpoint, two kept)"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); (( n == 0 )) || fail "$n processes on the GPUs"
[[ ! -e $D/ckpt/$RUN ]] || fail "run dir $D/ckpt/$RUN already exists"
POOLS=$(ls -d $D/pools/main_cpt_v* | sort -V | paste -sd,)
for p in ${POOLS//,/ } $D/pools/main_val_v1; do [[ -f $p/meta.json ]] || fail "pool $p missing"; done
echo "$(date -u +%H:%M:%S) preflight OK (${free} GB free, GPUs idle, start $START)"
EXTRA="--freeze \\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr $LR --lr-dense-frac 2.0 --warmup $WARMUP --min-lr-frac $MIN_LR_FRAC --decay-start $DECAY_START --ema 0.98 --val-raw 0 --contract-dir $T1 --save-every 25 --eval-every 25 --steps $STEPS --pool $POOLS --val $D/pools/main_val_v1 --seed 0 --skip-seqs $SKIP --loss-weights $LOSS_WEIGHTS"
echo "$(date -u +%H:%M:%S) starting $RUN from $START: $STEPS steps, skip $SKIP, warmup $WARMUP, decay from $DECAY_START, lr $LR, weights $LOSS_WEIGHTS"
exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool 10 --batches 16x16 \
  --min-tok-s 0 --extra-args "$EXTRA" --king $K43 --max-runs 1
