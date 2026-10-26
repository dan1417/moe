#!/usr/bin/env bash
# Finish a training line and decide on a submission candidate, unattended:
#   1. (cooldown mode) touch COOLDOWN and wait for the linear LR decay to reach its last steps
#   2. touch STOP -> the trainer writes raw + EMA exports and exits (frees the GPUs)
#   3. parity-screen raw and EMA vs king #30 on the first 8000 seqs of sample A (8 single-GPU replicas)
#   4. full 30k confirmation of the better one on samples A and B, then verify_submission.py
#   bash scripts/train/finish_line.sh line_a2            # cooldown first (default)
#   bash scripts/train/finish_line.sh line_a2 now        # stop right now, no cooldown
# Output: /workspace/teutonic/logs/finish_<name>.log and /workspace/teutonic/runs/<name>/FINISH.json
set -euo pipefail
NAME=${1:?run name}; MODE=${2:-cooldown}
source /workspace/teutonic/env.sh
cd "$REPO_ROOT"
RUN=/workspace/teutonic/runs/$NAME
PM2LOG=/workspace/teutonic/logs/pm2_${NAME}.out.log
EXPORTS=$(python -c "import yaml;print(yaml.safe_load(open('$RUN/config.yaml'))['export_dir'])")
last_step() { python - "$RUN/history.jsonl" <<'PY'
import json, sys
steps = [json.loads(l)["step"] for l in open(sys.argv[1]) if '"loss"' in l]
print(steps[-1] if steps else -1)
PY
}
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

if [[ "$MODE" == "cooldown" ]]; then
  touch "$RUN/COOLDOWN"; say "COOLDOWN armed"
  until grep -q "COOLDOWN triggered" "$PM2LOG"; do sleep 10; done
  TOTAL=$(grep "COOLDOWN triggered" "$PM2LOG" | tail -1 | sed -E 's/.*total now ([0-9]+).*/\1/')
  say "cooldown running; schedule ends at step $TOTAL"
  until (( $(last_step) >= TOTAL - 3 )); do sleep 20; done
fi
touch "$RUN/STOP"; say "STOP armed at step $(last_step)"
until [[ -f "$RUN/DONE" ]]; do sleep 10; done
for i in $(seq 1 60); do [[ "$(pm2 jlist 2>/dev/null | python -c "import json,sys;d=sys.stdin.read();print(next((p['pm2_env']['status'] for p in json.loads(d[d.index('['):]) if p['name']=='$NAME'),'absent'))")" != "online" ]] && break; sleep 5; done
until [[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" == "0" ]]; do sleep 5; done
STEP=$(ls -d "$EXPORTS"/step*_ema | sort | tail -1 | sed -E 's/.*step0*([0-9]+)_ema/\1/')
RAW=$(printf "%s/step%06d_raw" "$EXPORTS" "$STEP"); EMA=$(printf "%s/step%06d_ema" "$EXPORTS" "$STEP")
touch "$RAW/KEEP" "$EMA/KEEP" 2>/dev/null || true
say "GPUs free; final exports: $RAW $EMA"

PE=(python scripts/eval/parity_eval.py --attn flash_attention_4 --batch-size 16 --replicas 8)
say "screen: raw and EMA on the first 8000 of sample A"
"${PE[@]}" --limit 8000 duel --king reign:30 --challenger "$RAW" --challenger "$EMA" --sample tag:A 2>&1 | grep -v -E "FutureWarning|warnings.warn"
BEST=$(python - "$RAW" "$EMA" <<'PY'
import json, sys
from pathlib import Path
res = sorted(Path("/workspace/teutonic/eval_cache/results").glob("*_duel.json"))[-1]
rows = json.loads(res.read_text())["results"]
score = {r["challenger"]["label"]: r["verdict"]["mu_hat"] for r in rows}
raw, ema = Path(sys.argv[1]).name, Path(sys.argv[2]).name
print(sys.argv[1] if score.get(raw, -1) > score.get(ema, -1) else sys.argv[2])
PY
)
say "best on the screen: $BEST — full 30k confirmation on samples A and B"
"${PE[@]}" duel --king reign:30 --challenger "$BEST" --sample tag:A --sample tag:B 2>&1 | grep -v -E "FutureWarning|warnings.warn"
say "validator checks"
python scripts/verify/verify_submission.py --model-dir "$BEST" --king-dir /dev/shm/teutonic/king30 | tail -4
python - "$BEST" "$RUN/FINISH.json" <<'PY'
import json, sys
from pathlib import Path
res = sorted(Path("/workspace/teutonic/eval_cache/results").glob("*_duel.json"))[-1]
rows = json.loads(res.read_text())["results"]
out = {"candidate": sys.argv[1], "results": [{"sample": r["sample"], **{k: r["verdict"].get(k) for k in
       ("mu_hat", "lcb", "robust_lcb", "se_shard_jackknife", "accepted", "win_rate")},
       "sources": {s: v["mu_hat"] for s, v in r["verdict"]["source_scores"].items()}} for r in rows]}
ok = all(r["mu_hat"] >= 0.0036 and (r["robust_lcb"] or 0) > 0.003 and min(r["sources"].values()) > -0.001 for r in out["results"])
out["submit_rule_met"] = ok
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
print("SUBMIT RULE MET" if ok else "SUBMIT RULE NOT MET")
PY
