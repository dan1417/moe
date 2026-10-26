#!/usr/bin/env bash
# Run after container/GPU reset (clear D-state from interrupted DCP save).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"
source .venv/bin/activate
export NCCL_IB_DISABLE=1 NCCL_NET=Socket TEUTONIC_DIST_BACKEND=nccl TEUTONIC_ATTN=eager
export GPUS_PER_NODE=8 CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=600
export NCCL_TIMEOUT=600

echo "==> 0) Clean wedged equality temps"
rm -rf checkpoints/4090_equality/.step_*.tmp

echo "==> 1) Equality uninterrupted"
rm -rf checkpoints/4090_equality sequence_pool_equality
python scripts/data/prepare_train_sequence_pool.py --config configs/4090/equality.yaml --world-size 8
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/equality.yaml 2>&1 | tee logs/4090_equality_uninterrupted.log

MID=$(ls -d checkpoints/4090_equality/step_* | sort | head -1)
echo "MID_CKPT=$MID"

echo "==> 2) Equality resume from $MID"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/equality.yaml --resume "$MID" \
  2>&1 | tee logs/4090_equality_resume.log

python - <<'PY'
import re
from pathlib import Path

def losses(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.search(r"step=(\d+) lm_loss=([0-9.]+)", line)
        if m:
            out[int(m.group(1))] = float(m.group(2))
    return out

a = losses("logs/4090_equality_uninterrupted.log")
b = losses("logs/4090_equality_resume.log")
# resume starts after mid ckpt step
common = sorted(set(a) & set(b))
print("common steps", common)
for s in common:
    d = abs(a[s]-b[s])
    print(f"step {s}: uninterrupted={a[s]} resume={b[s]} delta={d}")
if not common:
    raise SystemExit("FAIL: no overlapping steps to compare")
# allow tiny bf16 noise
ok = all(abs(a[s]-b[s]) < 1e-3 for s in common)
print("EQUALITY", "PASS" if ok else "FAIL (check delta; may be bf16 nondeterminism)")
PY

echo "==> 3) Soup via --trigger-decay (uses stable DCP ckpts in output_dir)"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/equality.yaml \
  --resume "$(ls -d checkpoints/4090_equality/step_* | sort | tail -1)" \
  --trigger-decay 2>&1 | tee logs/4090_soup.log
grep -E "Averaging|pre-decay soup|triggered decay|No stable" logs/4090_soup.log | tail -20

echo "==> 4) seq512 OOM probe"
python scripts/data/prepare_train_sequence_pool.py --config configs/4090/seq512_probe.yaml --world-size 8
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/seq512_probe.yaml 2>&1 | tee logs/4090_seq512_probe.log || true
grep -E "lm_loss|out of memory|OOM" logs/4090_seq512_probe.log | tail -10

echo "==> 5) 8-GPU LoRA smoke"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/lora_smoke_8gpu.yaml 2>&1 | tee logs/4090_lora_8gpu.log
grep -E "LoRA|lm_loss|step=" logs/4090_lora_8gpu.log | tail -20

echo "==> 6) Longer burn-in (100 steps @ seq256)"
python scripts/data/prepare_train_sequence_pool.py --config configs/4090/mini_burnin.yaml --world-size 8
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/mini_burnin.yaml 2>&1 | tee logs/4090_burnin.log

echo "==> 7) Offline eval smoke (--eval-n 48) vs full if cache ready"
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/mini_eval.yaml --eval-only \
  --checkpoint checkpoints/4090_mini_real/step_0000049 --eval-n 48 \
  --eval-output results/offline_eval_smoke48.json 2>&1 | tee logs/4090_eval_smoke.log

echo "DONE remaining tests"
