#!/usr/bin/env bash
# train.sh — train a model for one competition.
#
#   bash train.sh <merge label | checkpoint dir | model name> <competition>
#     merge label   a recipe label from the merge campaign, e.g. pub_846a6b3de4 (search/public.json "best" -> recipe file name)
#     checkpoint    a directory with model.safetensors.index.json, e.g. /data/teutonic/ckpt/text_upper_ema_c10/step-50
#     model name    a name from configs/merge/k41/models.json, e.g. TX5 or C10_S50
#     competition   main | math | code | text   (or 1 | 2 | 3 | 4)
#
#   e.g.  bash train.sh pub_846a6b3de4 math
#         STEPS=300 LR=4e-6 bash train.sh TX5 text
#
# What it does: writes the merge (if a label), stops the merge campaign (training needs every GPU), builds the
# competition's data pools if missing, then launches the run under pm2 with the settings that worked best for that
# competition, plus the guard (auto-resume after a crash / hang), checkpoint screens vs the reigning kings every
# EVERY steps (main_watch) and checkpoint retention. Follow it with: pm2 logs launch_<run>, bash dashboard.sh.
# DRY=1 prints the plan and changes nothing. Overrides (env): STEPS LR DENSE WARMUP BATCHES EVERY SCREEN_EVERY SKIP EVAL_EVERY EMA FREEZE DECAY_START NPOOLS FORCE=1 (replace a running
# training run).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
D=/data/teutonic; L=$D/logs; P=$D/pools
source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) [train] $*"; }
die() { echo "train.sh: $*" >&2; exit 1; }
SRC=${1:-}; COMP=$(echo "${2:-}" | tr 'A-Z' 'a-z')
case $COMP in 1) COMP=main;; 2) COMP=math;; 3) COMP=code;; 4) COMP=text;; main|math|code|text) ;; *) die "usage: bash train.sh <merge label | ckpt dir | model name> <main|math|code|text>";; esac
[[ -n $SRC ]] || die "missing the model (merge label, checkpoint dir or model name)"

# ---- 1 resolve the start checkpoint ------------------------------------------------------------------------------
RECIPE=""; START=""
if [[ -f $SRC/model.safetensors.index.json ]]; then START=$(cd "$SRC" && pwd)
else
  START=$(python3 -c "import json,sys,os; m=json.load(open('configs/merge/k41/models.json')); p=m.get(sys.argv[1],''); print(p if os.path.exists(os.path.join(p,'model.safetensors.index.json')) else '')" "$SRC")
  if [[ -z $START ]]; then
    RECIPE=$(ls -t configs/merge/*/"$SRC".json 2>/dev/null | head -1 || true)
    [[ -n $RECIPE ]] || die "'$SRC' is neither a checkpoint dir, a model name in configs/merge/k41/models.json, nor a merge label (configs/merge/*/$SRC.json)"
    START=$D/ckpt/start_$SRC
  fi
fi
LABEL=$(basename "$SRC" | tr -c 'A-Za-z0-9_\n-' '_')
RUN=${COMP}_${LABEL}_$(date -u +%m%d%H%M)
log "competition $COMP, start $START${RECIPE:+ (merge $RECIPE)}, run $RUN"

# ---- 4 per-competition settings (the best runs so far; every value can be overridden from the environment) ----------
if [[ $COMP == text ]]; then   # c11 recipe: long documents + windows (8 192-token rows), layers 0-7 and routers frozen
  : "${STEPS:=400}" "${LR:=6.0e-6}" "${DENSE:=1.0}" "${WARMUP:=20}" "${BATCHES:=4x8}" "${EVERY:=25}" "${EVAL_EVERY:=25}" "${EMA:=0.98}"
  : "${FREEZE:=\\.mlp\\.gate\\.|^model\\.layers\\.[0-7]\\.}"; SEQS=256; SEQ_LEN=8192; SCREENS=text,main
else                           # MATH / MAIN / CODE recipe (train_sequence5): 4 096 x 2 048 tokens per step
  : "${STEPS:=200}" "${LR:=3.0e-6}" "${DENSE:=1.0}" "${WARMUP:=15}" "${BATCHES:=16x32}" "${EVERY:=20}" "${EVAL_EVERY:=10}" "${EMA:=0.96}"
  : "${FREEZE:=\\.mlp\\.gate\\.|^model\\.embed_tokens\\.|^model\\.layers\\.([0-9]|1[0-5])\\.}"; SEQS=4096; SEQ_LEN=2048
  SCREENS=$COMP,main; [[ $COMP == main ]] && SCREENS=main,math
fi
: "${DECAY_START:=$(( STEPS * 3 / 4 ))}"   # warmup-stable-decay: cosine over the last quarter
: "${SKIP:=0}"                             # sequences of the seed-0 data order already used (a run continued from a checkpoint)
: "${SCREEN_EVERY:=$EVERY}"                # king screens every SCREEN_EVERY steps (a multiple of EVERY; each pauses the trainer ~8 min)
(( SCREEN_EVERY % EVERY == 0 )) || die "SCREEN_EVERY=$SCREEN_EVERY must be a multiple of EVERY=$EVERY (screens score saved checkpoints)"

if [[ ${DRY:-0} == 1 ]]; then
  log "DRY RUN: would train $RUN from $START${RECIPE:+ (written from $RECIPE)}: $STEPS steps x $SEQS seqs of $SEQ_LEN tokens"
  log "  lr $LR (dense x$DENSE), warmup $WARMUP, cosine from step $DECAY_START, batches $BATCHES per GPU, EMA $EMA, freeze $FREEZE"
  log "  checkpoints every $EVERY steps, $SCREENS screens every $SCREEN_EVERY; pools: $( [[ $COMP == text ]] && echo text_mix_v1 + text_val_v1 || echo "${COMP}_cpt_v1..v$(( (STEPS * SEQS + SKIP + 199999) / 200000 )) + ${COMP}_val_v1")"
  exit 0
fi

# ---- 2 free the GPUs -----------------------------------------------------------------------------------------------
if pgrep -f "[s]cripts/train/cpt_fsdp.py" >/dev/null && [[ ${FORCE:-0} != 1 ]]; then
  die "a training run is on the GPUs (pm2 ls); FORCE=1 bash train.sh ... replaces it"
fi
for name in public_merge train_guard main_watch ckpt_retention $(pm2 jlist 2>/dev/null | python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(sys.stdin) if p['name'].startswith('launch_') and p['pm2_env']['status'] == 'online'))"); do
  pm2 stop "$name" >/dev/null 2>&1 || true
done
desc() { local p; for p in $(pgrep -P "$1"); do echo "$p"; desc "$p"; done; }
for c in $(pgrep -f "[s]cripts/merge/campaign43.py"); do kill -9 $c $(desc $c) 2>/dev/null || true; done
pkill -f "[s]cripts/train/cpt_fsdp.py" 2>/dev/null || true
for i in $(seq 1 60); do
  [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] && break
  (( i > 6 )) && kill -9 $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) 2>/dev/null || true
  sleep 5
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]] || die "GPUs still busy: $(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | head -3)"

# ---- 3 write the merge ---------------------------------------------------------------------------------------------
if [[ -n $RECIPE && ! -f $START/SAVED ]]; then
  log "writing the merge $SRC -> $START ($(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('description','')[:120])" "$RECIPE"))"
  python scripts/merge/write_merge.py --recipe "$RECIPE" --out "$START" --force 2>&1 | tail -2
  touch "$START/SAVED"
fi
[[ -f $START/model.safetensors.index.json ]] || die "start $START incomplete"

# ---- 5 data pools (built from the validator's data sources if missing; ~3 min per 200 000-sequence pool) ------------
pool() {  # name n block [excluded pools...]
  local name=$1 n=$2 block=$3; shift 3
  [[ -f $P/$name/meta.json ]] && return 0
  local ex=(); for e in "$@"; do ex+=(--exclude-pool "$P/$e"); done
  local comp=$COMP; [[ $name == text_* ]] && comp=text
  log "building pool $name ($n sequences, competition $comp)"
  python scripts/data/build_diverse_pool.py --competition "$comp" --n "$n" --block "$block" --tag "$name" --out "$P/$name" "${ex[@]}"
}
if [[ $COMP == text ]]; then
  pool text_val_v1 2048 1
  if [[ ! -f $P/text_mix_v1/meta.json ]]; then
    for v in 1 2 3 4; do pool text_cpt_v$v 200000 32 text_val_v1 $(for w in $(seq 1 $((v - 1))); do echo text_cpt_v$w; done); done
    [[ -f $P/text_long_v1/meta.json ]] || python scripts/data/build_long_pool.py --pools $P/text_cpt_v1,$P/text_cpt_v2,$P/text_cpt_v3,$P/text_cpt_v4 --out $P/text_long_v1 --join 4
    python scripts/data/build_mix_pool.py --long $P/text_long_v1 --windows $P/text_cpt_v1,$P/text_cpt_v2,$P/text_cpt_v3,$P/text_cpt_v4 --out $P/text_mix_v1 --n 200000
  fi
  POOLS=$P/text_mix_v1; VAL=$P/text_val_v1
else
  pool ${COMP}_val_v1 2048 1
  NEED=$(( (STEPS * SEQS + SKIP + 199999) / 200000 )); NEED=${NPOOLS:-$NEED}
  EX=(${COMP}_val_v1)
  for v in $(seq 1 "$NEED"); do pool ${COMP}_cpt_v$v 200000 32 "${EX[@]}"; EX+=(${COMP}_cpt_v$v); done
  POOLS=$(for v in $(seq 1 "$NEED"); do echo -n "$P/${COMP}_cpt_v$v,"; done | sed 's/,$//'); VAL=$P/${COMP}_val_v1
fi

# ---- 6 launch ------------------------------------------------------------------------------------------------------
python3 - "$RUN" "$START" "$STEPS" "$DECAY_START" "$LR" "$DENSE" "$WARMUP" "$SEQS" "$SEQ_LEN" "$EVAL_EVERY" "$SCREENS" "$POOLS" "$VAL" "$BATCHES" "$EVERY" "$EMA" "$FREEZE" "$SRC" "$SCREEN_EVERY" "$SKIP" <<'PY'
import json, sys
(run, start, steps, decay, lr, dense, warm, seqs, seq_len, eval_every, comps, pools, val, batches, every, ema, freeze, src, screen_every, skip) = sys.argv[1:]
json.dump({"base": run, "start": start, "total": int(steps), "skip0": int(skip), "decay_abs": int(decay), "lr": float(lr), "dense_frac": float(dense),
           "min_frac": 0.1, "warmup": int(warm), "seqs_per_step": int(seqs), "seq_len": int(seq_len), "eval_every": int(eval_every),
           "max_resumes": 6, "launcher": "text_run.sh", "comps": comps, "val_raw": 1, "pools": pools, "val": val, "batches": batches,
           "save_every": int(every), "screen_every": int(screen_every), "ema": float(ema), "save_sr": 0, "freeze": freeze, "why": f"train.sh {src} {comps.split(',')[0]}"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
for name in launch_$RUN main_watch ckpt_retention train_guard; do pm2 delete "$name" >/dev/null 2>&1 || true; done
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log -- -c \
  "M_RUN=$RUN M_START=$START M_STEPS=$STEPS M_SKIP=$SKIP M_DECAY_START=$DECAY_START M_LR=$LR M_DENSE_FRAC=$DENSE M_WARMUP=$WARMUP M_EVAL_EVERY=$EVAL_EVERY M_VAL_RAW=1 M_POOLS=$POOLS M_VAL=$VAL M_BATCHES=$BATCHES M_SAVE_EVERY=$EVERY M_EMA=$EMA M_SAVE_SR=0 M_FREEZE='$FREEZE' exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=$SCREENS EVERY=$SCREEN_EVERY STEPS=$STEPS SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh $SCREEN_EVERY" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
grep -q "PREFLIGHT FAILED" $L/pm2_launch_$RUN.log 2>/dev/null && die "$(grep PREFLIGHT $L/pm2_launch_$RUN.log | tail -1)"
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "$RUN launched: $STEPS steps x $SEQS seqs ($BATCHES per GPU), lr $LR, checkpoints every $EVERY steps, $SCREENS screens every $SCREEN_EVERY"
log "follow: pm2 logs launch_$RUN | screens: tail -f $L/pm2_main_watch.log | checkpoints: $D/ckpt/$RUN | bash dashboard.sh"
