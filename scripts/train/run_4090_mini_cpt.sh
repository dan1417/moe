#!/usr/bin/env bash
# Full CPT on depth-shrunk full-width MiMo-V2 (~8.6B) across 8× GPU (RTX 4090).
# Exercises NCCL + FSDP FULL_SHARD + Muon/MoE with LoRA disabled.
#
# Usage:
#   bash scripts/train/run_4090_mini_cpt.sh
#   BRINGUP=1 bash scripts/train/run_4090_mini_cpt.sh          # 5 steps, random tokens
#   GPUS_PER_NODE=8 STEPS=200 bash scripts/train/run_4090_mini_cpt.sh
#   RESUME=checkpoints/4090_mini_cpt/step_0000004 STEPS=10 bash scripts/train/run_4090_mini_cpt.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_repo_root.sh"
ROOT="$REPO_ROOT"
cd "$ROOT"

PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" && -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/setup/detect_gpus.sh"
detect_gpus

GPUS_PER_NODE="${GPUS_PER_NODE:-$DETECTED_GPUS}"
CONFIG="${CONFIG:-configs/4090/mini_cpt.yaml}"
BRINGUP="${BRINGUP:-0}"
STEPS="${STEPS:-}"
RESUME="${RESUME:-}"

if [[ ! -f "$ROOT/pipeline_model/config.json" ]]; then
  echo "==> Building depth-shrink full-width MiMo-V2 king (~8.6B)"
  "$PYTHON" scripts/data/prepare_pipeline_model.py --output "$ROOT/pipeline_model"
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed — fix NVML/driver before 8-GPU NCCL train" >&2
  bash "$ROOT/scripts/setup/fix_nvml.sh" 2>/dev/null || true
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TEUTONIC_NODES=1
export GPUS_PER_NODE
# Prefer NCCL on multi-GPU; only set gloo explicitly for broken NVML boxes.
if [[ -z "${TEUTONIC_DIST_BACKEND:-}" ]]; then
  if [[ "$GPUS_PER_NODE" -gt 1 ]]; then
    export TEUTONIC_DIST_BACKEND=nccl
  else
    export TEUTONIC_DIST_BACKEND=gloo
  fi
fi
export TEUTONIC_ATTN="${TEUTONIC_ATTN:-eager}"
# expandable_segments can fail under near-OOM MoE all-gathers on 24GB cards.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
# Prefer empty/default allocator unless the caller sets a value.

RUNTIME_CFG="$ROOT/checkpoints/4090_mini_cpt/runtime_config.yaml"
mkdir -p "$(dirname "$RUNTIME_CFG")"
"$PYTHON" - <<PY
import copy
import yaml
from pathlib import Path
root = Path("$ROOT")
cfg = yaml.safe_load((root / "$CONFIG").read_text())
cfg = copy.deepcopy(cfg)
cfg.setdefault("hardware", {})["gpus_per_node"] = int("$GPUS_PER_NODE")
cfg.setdefault("training", {})
if "$BRINGUP" == "1":
    cfg["training"]["smoke_steps"] = 5
    cfg["training"]["total_steps"] = 5
    cfg["training"]["smoke_patterned_data"] = False
    cfg["training"]["smoke_seq_len"] = 32
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["log_every"] = 1
    # Expert-level FSDP2 keeps MoE unshards tiny; CPU offload is unnecessary and
    # thrashing-slow (H2D per expert) on 24GB with ~8GB resident.
    cfg.setdefault("parallelism", {})["fsdp_cpu_offload"] = False
    # AC doubles every expert FSDP unshard; disable for short NCCL/Muon bring-up.
    cfg["training"]["activation_checkpointing"] = False
    cfg["scheduler"]["warmup_steps"] = 1
    cfg["scheduler"]["stable_steps"] = 3
    cfg["scheduler"]["decay_steps"] = 1
    print("BRINGUP: 5 steps, seq=32, accum=1, no-AC, random synthetic, NCCL/FSDP check")
elif "$STEPS":
    steps = int("$STEPS")
    cfg["training"]["total_steps"] = steps
    cfg["training"]["smoke_steps"] = 0
    # 4090 practicality: one microbatch/step; keep patterned from config.
    cfg["training"]["gradient_accumulation_steps"] = 1
    cfg["training"]["log_every"] = max(1, min(10, steps // 20))
    cfg.setdefault("parallelism", {})["fsdp_cpu_offload"] = False
    # Seq 64 is enough for period-32 patterned learning without multi-hour steps.
    cfg["training"]["smoke_seq_len"] = int(cfg["training"].get("smoke_seq_len", 256))
    if steps <= 200:
        cfg["training"]["smoke_seq_len"] = min(int(cfg["training"]["smoke_seq_len"]), 64)
        cfg["training"]["activation_checkpointing"] = False
    # Keep WSD phases proportional for short runs
    warm = max(1, steps // 20)
    decay = max(1, steps // 10)
    stable = max(1, steps - warm - decay)
    cfg["scheduler"]["warmup_steps"] = warm
    cfg["scheduler"]["stable_steps"] = stable
    cfg["scheduler"]["decay_steps"] = decay
    print(
        f"STEPS={steps}: patterned={cfg['training'].get('smoke_patterned_data')} "
        f"seq={cfg['training']['smoke_seq_len']} accum=1 "
        f"ac={cfg['training'].get('activation_checkpointing')}"
    )
cfg["lora"]["enabled"] = False
Path("$RUNTIME_CFG").write_text(yaml.dump(cfg))
print(f"runtime config -> $RUNTIME_CFG")
print(f"gpus={cfg['hardware']['gpus_per_node']} lora={cfg['lora']['enabled']} "
      f"steps={cfg['training']['total_steps']} patterned={cfg['training'].get('smoke_patterned_data')}")
PY

TRAIN_ARGS=(--config "$RUNTIME_CFG")
if [[ -n "$RESUME" ]]; then
  if [[ -f "$ROOT/$RESUME/COMMITTED" ]]; then
    RESUME="$ROOT/$RESUME"
  fi
  if [[ ! -f "$RESUME/COMMITTED" ]]; then
    echo "ERROR: RESUME checkpoint missing COMMITTED marker: $RESUME" >&2
    exit 1
  fi
  TRAIN_ARGS+=(--resume "$RESUME")
  echo "==> Resuming from $RESUME"
fi

echo "==> Mini full CPT: nproc=${GPUS_PER_NODE} backend=${TEUTONIC_DIST_BACKEND} config=${RUNTIME_CFG}"
"$PYTHON" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="$GPUS_PER_NODE" \
  -m src.train "${TRAIN_ARGS[@]}"

echo "==> 4090 mini CPT complete"
