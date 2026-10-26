# Prompt: 8×4090 mini CPT with real Teutonic datasets

Copy everything below the line into a new Cursor agent chat on a fresh **8× RTX 4090** machine that has working internet (DNS + HTTPS to GitHub and `teutonic.ai`).

---

## Mission

Set up [wilfrid51/MoE](https://github.com/wilfrid51/MoE) from `main`, build the **~8.59B depth-shrink full-width** MiMo-V2 pipeline king, download **exact Teutonic R2 training shards** (not synthetic), and run an **8-GPU FSDP2 + Muon** continued-pretrain smoke / short CPT on real data.

Do **not** use `BRINGUP=1` for the real-data run (that forces random synthetic tokens). Do **not** treat mini success as 110B/B300 proof.

## Target hardware / software

- 8× NVIDIA RTX 4090 (24GB), single node
- Working `nvidia-smi`, NCCL, outbound HTTPS
- Ubuntu-like Linux, Python 3.11+ preferred
- Disk: ≥200GB free (venv + ~17GB king weights + shard cache)

## What “exact Teutonic dataset” means

Live manifest: `https://teutonic.ai/datasets/manifest.json`

Three pre-tokenized sources (uint32 `.npy`, packed **2048-token** sequences), proportions from the live manifest (typically finewebedu / automathtext-v2 / dclm-baseline-1.0). Training must sample **all three** in manifest proportions — not finewebedu alone.

## End-to-end steps (execute in order)

### 0) Preconditions

```bash
nvidia-smi
curl -fsSL -o /dev/null -w "%{http_code}\n" https://teutonic.ai/datasets/manifest.json
curl -fsSL -o /dev/null -w "%{http_code}\n" https://github.com
df -h .
```

If DNS/HTTPS fails, stop and fix network before continuing.

### 1) Clone repo (latest main)

```bash
git clone https://github.com/wilfrid51/MoE.git
cd MoE
git checkout main
git pull
git log -1 --oneline
```

Expect recent commit about depth-shrink mini / FSDP2 / Muon (e.g. `5e5ac1f` or newer).

### 2) Create venv + install deps + build mini king

```bash
bash scripts/setup/setup_pipeline_venv.sh
source .venv/bin/activate
```

This should:

- Install PyTorch CUDA wheels + requirements (skip flash-attn / TE failures on SM89 if needed)
- Run `python scripts/data/prepare_pipeline_model.py` → `./pipeline_model` (~8.59B, ~16GB BF16)
- Run design verifier

Confirm:

```bash
python - <<'PY'
import json, torch
from pathlib import Path
info = json.loads(Path("pipeline_model/king_info.json").read_text())
print(info["param_count"], info.get("num_hidden_layers"), info.get("n_routed_experts"))
print("cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
assert info["param_count"] > 8_000_000_000
assert info["num_hidden_layers"] == 4
assert info["n_routed_experts"] == 256
PY
nvidia-smi -L
```

### 3) Optional: NCCL bring-up only (synthetic, 5 steps)

Skip if you only care about real data. Validates FSDP2/Muon before downloads.

```bash
BRINGUP=1 NCCL_IB_DISABLE=1 NCCL_NET=Socket bash scripts/train/run_4090_mini_cpt.sh
```

### 4) Download real Teutonic data

**A — Eval cache (required for offline eval later; good connectivity check):**

```bash
source .venv/bin/activate
python scripts/data/prepare_eval_cache.py --config configs/4090/mini_cpt.yaml --workers 8
```

**B — Small train sequence pool (recommended for short real-data CPT + resume):**

Edit `configs/4090/mini_cpt.yaml` (or write a runtime overlay) so `data.sequence_pool` looks like:

```yaml
data:
  cache_dir: ./shard_cache
  cache_max_gb: 200
  manifest_url: https://teutonic.ai/datasets/manifest.json
  sequence_pool:
    enabled: true
    pool_dir: ./sequence_pool
    # Keep small on first rent — increase later
    shards_per_source_range: [2, 8]
    seqs_per_shard_range: [32, 128]
```

Also set training for real data (see step 5), then:

```bash
python scripts/data/prepare_train_sequence_pool.py --config configs/4090/mini_cpt.yaml --world-size 8
ls -lh sequence_pool/ shard_cache/ | head
```

Inspect a shard:

```bash
python - <<'PY'
import numpy as np
from pathlib import Path
seqs = np.load("sequence_pool/pool_sequences.npy", mmap_mode="r")
print("pool", seqs.shape, seqs.dtype, seqs[0, :8])
assert seqs.shape[1] == 2048
PY
```

### 5) Configure real-data 8×4090 CPT

Default `configs/4090/mini_cpt.yaml` still has `smoke_synthetic_data: true`. For **real Teutonic data**, create/use a runtime config with at least:

```yaml
training:
  king_dir: ./pipeline_model
  output_dir: ./checkpoints/4090_mini_real
  micro_batch_size: 1
  gradient_accumulation_steps: 1   # start conservative on 24GB
  activation_checkpointing: true
  activation_checkpointing_mode: selective
  total_steps: 50                  # short first rent; raise later
  smoke_steps: 0
  log_every: 1
  # CRITICAL: use live shards / sequence pool, not random tokens
  smoke_synthetic_data: false
  smoke_patterned_data: false
  # Truncate 2048 → shorter for 4090 memory on full-width MoE; null/0 = full 2048 (may OOM)
  smoke_seq_len: 256
  attn_implementation: eager
  profile: mini

data:
  sequence_pool:
    enabled: true
    pool_dir: ./sequence_pool

lora:
  enabled: false

parallelism:
  fsdp: true
  fsdp_cpu_offload: false
```

**Memory note:** full 2048 seq on this ~8.6B width-MoE + eager + per-expert FSDP is heavy on 24GB. Start at `smoke_seq_len: 256` (or 128). Only raise toward 2048 after a clean few steps.

Write the overlay, e.g. `configs/4090/mini_real.yaml`, by copying `4090_mini_cpt.yaml` and applying the overrides above (keep MoE/optimizer/scheduler blocks).

### 6) Launch 8-GPU training on real data

```bash
source .venv/bin/activate
export NCCL_IB_DISABLE=1
export NCCL_NET=Socket
export TEUTONIC_DIST_BACKEND=nccl
export TEUTONIC_ATTN=eager
export GPUS_PER_NODE=8

# Prefer extending run_4090_mini_cpt.sh via CONFIG=..., or torchrun directly:
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m src.train --config configs/4090/mini_real.yaml
```

If you extend `scripts/train/run_4090_mini_cpt.sh`, ensure it does **not** set `BRINGUP=1` and does **not** force `smoke_synthetic_data: true`.

Watch:

```bash
watch -n1 nvidia-smi
# logs should say it is NOT using "random synthetic smoke batches"
# should prefetch/read sequence_pool or shard_cache
```

### 7) Success criteria (first rent)

- [ ] `pipeline_model` ~8.59B, 4 layers, 256 experts
- [ ] Manifest fetch + shard download succeed (files under `shard_cache/` / `sequence_pool/`)
- [ ] Train log shows **real data path** (no “random synthetic smoke batches”)
- [ ] 8 ranks, NCCL, FSDP2 expert wrap, no hang/OOM for ≥5–10 steps
- [ ] `lm_loss` is finite; checkpoint under `checkpoints/4090_mini_real/` with `COMMITTED` if save runs
- [ ] Append results to `docs/TEST.md`

### 8) If OOM or extremely slow

1. Lower `smoke_seq_len` (256 → 128 → 64)
2. Keep `micro_batch_size: 1`, `gradient_accumulation_steps: 1`
3. Keep `activation_checkpointing: true` for real-data (unlike BRINGUP)
4. Confirm `fsdp_cpu_offload: false` (offload was thrashing-slow)
5. Expect low tok/s: eager MoE + per-expert FSDP is a dress rehearsal, not production throughput

### 9) Do not do on this rent (unless asked)

- Full 110B genesis download / B300 MXFP8
- Submission upload / validator contract
- Claiming Teutonic mining readiness from mini loss alone

## Agent operating rules

- Prefer existing scripts (`setup_pipeline_venv.sh`, `scripts/data/prepare_pipeline_model.py`, `prepare_eval_cache.py`, `prepare_train_sequence_pool.py`, `run_4090_mini_cpt.sh`) over reinventing.
- Never commit secrets; don’t force-push.
- Update `docs/TEST.md` with commands, pass/fail, GPU memory, step time, and whether data was real vs synthetic.
- If anything fails, capture the exact error + the last 50 log lines and fix forward (network → build → download → short real train).

## Quick command cheat sheet

```bash
git clone https://github.com/wilfrid51/MoE.git && cd MoE
bash scripts/setup/setup_pipeline_venv.sh && source .venv/bin/activate
python scripts/data/prepare_eval_cache.py --config configs/4090/mini_cpt.yaml --workers 8
# after editing yaml for real data + small sequence_pool:
python scripts/data/prepare_train_sequence_pool.py --config configs/4090/mini_real.yaml --world-size 8
NCCL_IB_DISABLE=1 NCCL_NET=Socket GPUS_PER_NODE=8 \
  python -m torch.distributed.run --standalone --nproc_per_node=8 \
  -m src.train --config configs/4090/mini_real.yaml
```
