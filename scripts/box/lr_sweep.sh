#!/usr/bin/env bash
# LR sweep (08:00 JST Oct 12, user: "you didn't find the correct parameters?"): same start (c9/step-50), same mixed pool
# (text_mix_v1), same recipe; 50 steps each at 6e-6 (c10, running), 1e-5 (c10b), 3e-6 (c10c); judged on the step-50 screens
# (TEXT long documents vs windows). The best continues from its own step-50 checkpoint.
set -uo pipefail
D=/data/teutonic; L=$D/logs; cd /root/workspace/MoE
log() { echo "$(date -u +%H:%M:%S) $*"; }
wait_screen() {  # run, step -> wait until that checkpoint has been screened on TEXT (4 800 seqs)
  until python3 - "$1" "$2" <<'PY'
import glob, json, sys
run, step = sys.argv[1], sys.argv[2]
for f in glob.glob("/data/teutonic/eval_cache/results/*_duel.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        if len(t) == 1 and t[0].get("model", "").rstrip("/").endswith(f"/{run}/step-{step}") and r.get("sample", "").startswith("0a2840c9"):
            sys.exit(0)
sys.exit(1)
PY
  do sleep 60; done
}
probe() {  # run lr -> launch the probe from c9/step-50 on the mixed pool, 50 steps (constant lr after warmup)
  local RUN=$1 LR=$2
  sed -e "s#text_upper_ema_c10#$RUN#g" -e "s#M_LR=6.0e-6#M_LR=$LR#" -e "s#\"lr\": 6e-6#\"lr\": $LR#" \
      -e "s#M_STEPS=250 M_SKIP=0 M_DECAY_START=200#M_STEPS=50 M_SKIP=0 M_DECAY_START=1000#" -e "s#\"total\": 250, \"skip0\": 0, \"decay_abs\": 200#\"total\": 50, \"skip0\": 0, \"decay_abs\": 1000#" \
      -e "s#launch_text_upper_ema_c9 text_upper_ema_c9#launch_text_upper_ema_c10 text_upper_ema_c10 launch_text_upper_ema_c10b text_upper_ema_c10b#" \
      $L/start_c10.sh > $L/start_$RUN.sh
  bash $L/start_$RUN.sh
  log "$RUN launched (lr $LR, 50 steps from c9/step-50)"
}
wait_screen text_upper_ema_c10 50; log "c10 (6e-6) step 50 screened"
probe text_upper_ema_c10b 1.0e-5
wait_screen text_upper_ema_c10b 50; log "c10b (1e-5) step 50 screened"
probe text_upper_ema_c10c 3.0e-6
wait_screen text_upper_ema_c10c 50; log "c10c (3e-6) step 50 screened"
python3 - <<'PY'
import glob, json
rows = {}
for f in glob.glob("/data/teutonic/eval_cache/results/*_duel.json"):
    d = json.load(open(f))
    for r in d.get("results", []):
        t = (r.get("challenger") or {}).get("terms") or []
        m = t[0].get("model", "") if len(t) == 1 else ""
        for run in ("text_upper_ema_c10", "text_upper_ema_c10b", "text_upper_ema_c10c"):
            if m.rstrip("/").endswith(f"/{run}/step-50"):
                comp = {"0a2840c9": "TEXT", "e8ef9ae1": "MAIN"}.get(r["sample"][:8])
                c = r["verdict"].get("component_scores") or {}
                rows.setdefault(run, {})[comp] = (r["verdict"]["mu_hat"], (c.get("long_documents") or {}).get("mu_hat"), (c.get("windows") or {}).get("mu_hat"))
for run, v in sorted(rows.items()):
    print("SWEEP", run, {k: tuple(round(x, 6) for x in vv) for k, vv in v.items()})
PY
log "SWEEP DONE"
