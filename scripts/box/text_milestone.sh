#!/usr/bin/env bash
# User 02:10 JST Oct 12: "if the model's TEXT score is 0.0026+, save the checkpoint, prepare the upload and continue with MATH
# training from it". Every 2 min: the first c7-lineage checkpoint (nearest or stochastic-rounding twin) whose TEXT screen
# (4 800 seqs vs the TEXT king) has mu >= 0.0026 is (1) protected (hard-linked copy, survives retention), (2) prepared for
# submission like t005 / t007 (random hidden permutation, neutral contract files only, key in submit/private, verified) as the
# next free submit/tNNN, (3) the TEXT run stops and MATH training starts from it (open-ended, MATH + MAIN screens).
# The on-chain submission itself stays with the user.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE; source $D/env.sh
log() { echo "$(date -u +%H:%M:%S) $*"; }
log "watching c7 TEXT screens for mu >= 0.0026"
while true; do
  HIT=$(python3 - <<'PY'
import glob, json, os
for f in sorted(glob.glob("/data/teutonic/eval_cache/results/*_duel.json"), key=os.path.getmtime):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        m = t[0].get("model", "") if len(t) == 1 else ""
        v = r.get("verdict") or {}
        if "/text_upper_ema_c" in m and "/text_upper_ema_c5" not in m and "/text_upper_ema_c6" not in m and r.get("sample", "").startswith("0a2840c9") and v.get("n_sequences") == 4800 and v.get("mu_hat", 0) >= 0.0026:
            print(m, v["mu_hat"], v.get("lcb")); raise SystemExit
PY
)
  [[ -n $HIT ]] && break
  sleep 120
done
CK=$(echo $HIT | cut -d' ' -f1); MU=$(echo $HIT | cut -d' ' -f2); LCB=$(echo $HIT | cut -d' ' -f3)
NAME=text_milestone_$(basename $(dirname $CK))_$(basename $CK)
log "TEXT MILESTONE: $CK TEXT mu $MU (lcb $LCB) >= 0.0026"
cp -al "$CK" "$D/ckpt/$NAME" && log "protected as $D/ckpt/$NAME"
N=9; while [[ -e $D/submit/t$(printf %03d $N) ]]; do N=$((N + 1)); done; T=t$(printf %03d $N)
log "preparing the submission $T from $NAME"
sed -e "s#^SRC=.*#SRC=$D/ckpt/$NAME; OUT=$D/submit/$T; PERM=$D/submit/private/$T.perm.pt#" -e "s#^\# MATH submission.*#\# TEXT milestone submission prep ($NAME, TEXT screen mu $MU)#" \
  $L/permute_t008.sh > $L/permute_$T.sh && chmod +x $L/permute_$T.sh
pm2 start bash --name permute_$T --no-autorestart --time --output $L/pm2_permute_$T.log --error $L/pm2_permute_$T.log -- $L/permute_$T.sh >/dev/null
log "permute_$T running (CPU); switching training to MATH from $NAME"
pm2 delete train_sequence5 >/dev/null 2>&1
exec bash $L/math_after_text.sh "$D/ckpt/$NAME"
