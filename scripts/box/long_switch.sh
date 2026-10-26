#!/usr/bin/env bash
# 04:10 JST Oct 12 (autonomous, user: "if training is not going well, adjust the parameters and start again"): c7 at 2 048 tokens
# is flat (TEXT +0.001245 at step 20 vs +0.001233 start); the TEXT score is 8 % long documents (2,049-8,192 tokens, our lead
# +0.0053/token) + 92 % 2,048-token windows (+0.00087), and training never saw > 2,048 tokens. c8: the same recipe on 8,192-token
# rows of real contiguous text (text_long_v1 = v18-v27 un-shuffled, 4 consecutive rows joined), 1 024 rows x 8 192 = the same
# 8.4M tokens per step (micro 4 x accum 32), from the best c5 / c7 TEXT checkpoint (nearest rounding: stochastic lost the A/B).
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE
log() { echo "$(date -u +%H:%M:%S) $*"; }
until [[ -f $D/pools/text_long_v1/meta.json ]]; do sleep 30; done
log "long pool ready: $(python3 -c "import json; m=json.load(open('$D/pools/text_long_v1/meta.json')); print(m['n_sequences'], 'x', m['seq_len'], m['source_counts'])")"
BEST=$(python3 - <<'PY'
import glob, json
best = (-1.0, "/data/teutonic/ckpt/text_upper_ema_c5/step-25")
for f in glob.glob("/data/teutonic/eval_cache/results/*_duel.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        m = t[0].get("model", "") if len(t) == 1 else ""
        v = r.get("verdict") or {}
        if ("/text_upper_ema_c5/" in m or "/text_upper_ema_c7" in m) and "/sr-" not in m and r.get("sample", "").startswith("0a2840c9") \
           and v.get("n_sequences") == 4800 and v["mu_hat"] > best[0]:
            best = (v["mu_hat"], m)
print(best[1])
PY
)
log "c8 starts from $BEST"
for name in train_sequence5 train_guard main_watch ckpt_retention $(pm2 jlist 2>/dev/null | python3 -c "import json,sys; print(' '.join(p['name'] for p in json.load(sys.stdin) if 'text_upper_ema_c7' in p['name']))"); do pm2 delete $name >/dev/null 2>&1; done
for c in $(pgrep -f "[s]cripts/merge/campaign43"); do kill -9 $c; done
for i in $(seq 1 36); do P_=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | tr '\n' ' '); [[ -z ${P_// } ]] && break; [[ $i -gt 6 ]] && kill -9 $P_; sleep 5; done
rm -f $D/ckpt/text_upper_ema_c7*/PAUSE
RUN=text_upper_ema_c8; POOLS=$D/pools/text_long_v1; VAL=$D/pools/text_val_v1
python3 - "$BEST" <<'PY'
import json, sys
json.dump({"base": "text_upper_ema_c8", "start": sys.argv[1], "total": 1000, "skip0": 0, "decay_abs": 1000, "lr": 3e-6, "dense_frac": 1.0,
           "min_frac": 0.1, "warmup": 15, "seqs_per_step": 1024, "seq_len": 8192, "eval_every": 10, "max_resumes": 6, "launcher": "text_run.sh",
           "comps": "text,main", "val_raw": 1, "pools": "/data/teutonic/pools/text_long_v1", "val": "/data/teutonic/pools/text_val_v1",
           "batches": "4x32", "save_every": 20, "ema": 0.96, "save_sr": 0,
           "why": "8 192-token long-context TEXT training (long documents are 8 % of the TEXT score with a +0.005/token lead)"},
          open("/data/teutonic/ckpt/current_plan.json", "w"), indent=1)
PY
rm -f $D/ckpt/$RUN.lineage.json
pm2 start bash --name launch_$RUN --no-autorestart --time --merge-logs --output $L/pm2_launch_$RUN.log --error $L/pm2_launch_$RUN.log \
  -- -c "M_RUN=$RUN M_START=$BEST M_STEPS=1000 M_SKIP=0 M_DECAY_START=1000 M_LR=3.0e-6 M_DENSE_FRAC=1.0 M_WARMUP=15 M_EVAL_EVERY=10 M_VAL_RAW=1 M_POOLS=$POOLS M_VAL=$VAL M_BATCHES=4x32 M_SAVE_EVERY=20 M_EMA=0.96 M_SAVE_SR=0 exec bash $L/text_run.sh" >/dev/null
pm2 start bash --name main_watch --no-autorestart --output $L/pm2_main_watch.log --error $L/pm2_main_watch.log -- -c "COMPS=text,main EVERY=20 SR=0 RUN=$D/ckpt/$RUN exec bash $L/main_watch.sh 20" >/dev/null
pm2 start bash --name ckpt_retention --no-autorestart --output $L/pm2_ckpt_retention.log --error $L/pm2_ckpt_retention.log -- -c "RUN=$D/ckpt/$RUN KEEP=none exec bash $L/ckpt_retention.sh" >/dev/null
sleep 60
pm2 start $L/train_guard.py --name train_guard --interpreter python3 --no-autorestart --output $L/pm2_train_guard.log --error $L/pm2_train_guard.log >/dev/null
log "c8 LAUNCHED (8 192-token rows, 1 024 x 8 192 per step, lr 3e-6)"
