#!/usr/bin/env bash
# TEXT launcher (user 08:00 JST: train TEXT, then MAIN): main_run.sh with the TEXT pools (fresh v1 + v12-v14) and the TEXT
# val set, no source weighting; same M_* env, so train_guard resumes it like a MAIN run.
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
RUN=${M_RUN:?run name}; START=${M_START:?start checkpoint}
STEPS=${M_STEPS:-240}; SKIP=${M_SKIP:-0}; WARMUP=${M_WARMUP:-10}; DECAY_START=${M_DECAY_START:-180}
LR=${M_LR:-1.5e-6}; MIN_LR_FRAC=${M_MIN_LR_FRAC:-0.1}
DENSE_FRAC=${M_DENSE_FRAC:-2.0}; VAL_RAW=${M_VAL_RAW:-0}  # lr multiplier for attention / shared / lm_head; 1 = also validate the raw weights
FREEZE=${M_FREEZE:-'\.mlp\.gate\.|^model\.embed_tokens\.|^model\.layers\.([0-9]|1[0-5])\.'}  # frozen params (regex)
# ---- preflight: refuse to start into a known failure
fail() { echo "PREFLIGHT FAILED: $*"; exit 3; }
[[ -f $START/model.safetensors.index.json ]] || fail "start checkpoint $START incomplete"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); (( n == 0 )) || fail "$n processes on the GPUs"
[[ ! -e $D/ckpt/$RUN ]] || fail "run dir $D/ckpt/$RUN already exists"
POOLS=${M_POOLS:-$D/pools/text_cpt_v1,$D/pools/text_cpt_v12,$D/pools/text_cpt_v13,$D/pools/text_cpt_v14}  # TEXT default (c1 used v10 + v11)
VAL=${M_VAL:-$D/pools/text_val_v1}
BATCHES=${M_BATCHES:-16x16}; SAVE_EVERY=${M_SAVE_EVERY:-25}; EMA=${M_EMA:-0.98}; SAVE_SR=${M_SAVE_SR:-0}; EVAL_EVERY=${M_EVAL_EVERY:-$SAVE_EVERY}; START_VAL=${M_START_VAL:-}  # known step-0 val of the start: skips the val pass at launch  # micro x accum per GPU; save + val interval; EMA decay
for p in ${POOLS//,/ } $VAL; do [[ -f $p/meta.json ]] || fail "pool $p missing"; done
# disk: retention keeps the two newest step-N plus final (3 checkpoints at once, x2 with sr- twins), 220 GB each, + 200 GB headroom
need=$(( 3 * (SAVE_SR > 0 ? 2 : 1) * 220 + 200 ))
free=$(df -BG --output=avail $D | tail -1 | tr -dc 0-9); (( free >= need )) || fail "only ${free} GB free (need ${need}: 3 checkpoints of 220 GB$( (( SAVE_SR > 0 )) && echo " + sr twins") + headroom)"
echo "$(date -u +%H:%M:%S) preflight OK (${free} GB free, GPUs idle, start $START)"
EXTRA="--freeze $FREEZE --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr $LR --lr-dense-frac $DENSE_FRAC --warmup $WARMUP --min-lr-frac $MIN_LR_FRAC --decay-start $DECAY_START --ema $EMA --val-raw $VAL_RAW --contract-dir $T1 --save-every $SAVE_EVERY --eval-every $EVAL_EVERY --save-sr $SAVE_SR --steps $STEPS --pool $POOLS --val $VAL --seed 0 --skip-seqs $SKIP${START_VAL:+ --start-val $START_VAL}"
echo "$(date -u +%H:%M:%S) starting $RUN from $START: $STEPS steps, skip $SKIP, warmup $WARMUP, decay from $DECAY_START, lr $LR (dense x$DENSE_FRAC), val-raw $VAL_RAW, freeze $FREEZE, batches $BATCHES, save every $SAVE_EVERY, val every $EVAL_EVERY, ema $EMA, save-sr $SAVE_SR"
exec python scripts/train/chain_runs.py --no-scorer --start $RUN --init-ckpt $START --first-pool 10 --batches $BATCHES \
  --min-tok-s 0 --extra-args "$EXTRA" --king $T1 --max-runs 1
