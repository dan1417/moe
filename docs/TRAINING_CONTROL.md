# Training Control Guide

How to control the Teutonic SN3 pipeline: what to run in each situation, which flags/config to use, and what to expect.

---

## Quick decision table

| Situation | What to do |
|-----------|------------|
| Starting from scratch on a fresh machine | `bash setup.sh` → download king → launch training |
| New king crowned while you sleep | `monitor.py --auto-download` (or manual `download_king.py`) → restart training from new king |
| Dataset manifest changed | Review proportions in `manifest.json` → clear shard cache if needed → continue/restart training |
| Normal training run | `bash scripts/train/launch.sh` |
| Multi-node (2×8 GPUs) | Set `hardware.num_nodes: 2`, launch both nodes via `launch.sh` |
| GPU OOM during training | Reduce `micro_batch_size` first; enable `activation_checkpointing` or `fsdp_cpu_offload` as last resort |
| Training finished stable phase, ready to submit | `--trigger-decay` (or let schedule finish) → `--eval-only` → `--export` → upload |
| Check if checkpoint beats king | `bash scripts/train/launch.sh -- --eval-only [--checkpoint PATH]` |
| Package for validator upload | `bash scripts/train/launch.sh -- --export [--checkpoint PATH]` |
| Verify king download integrity | `python scripts/data/download_king.py --verify-only` |
| Watch dashboard without downloading | `python scripts/monitor.py` |
| Watch + auto-download new king | `python scripts/monitor.py --auto-download` |

---

## End-to-end workflow (happy path)

```bash
# 1. One-time setup
bash setup.sh

# 2. (Optional) Watch for new kings 24/7
python scripts/monitor.py --auto-download --output ./king

# 3. Download current king (required before first train)
python scripts/data/download_king.py --output ./king

# 4. Train (8 GPUs, distributed)
bash scripts/train/launch.sh

# 5. Offline eval vs king — only submit if LCB > 0.15
bash scripts/train/launch.sh -- --eval-only

# 6. Export submission package
bash scripts/train/launch.sh -- --export

# 7. Upload to subnet
teutonic-miner auth --hotkey h0
teutonic-miner upload --hotkey h0 --name teutonic-II-110B-<suffix> ./submit_model
teutonic-miner ready --hotkey h0
```

---

## Entry points

### `scripts/monitor.py` — competition watcher

Polls `dashboard.json` and `datasets/manifest.json`. Persists state in `.teutonic_monitor_state.json`.

| Flag | Purpose |
|------|---------|
| `--interval SEC` | Poll period (default 60s) |
| `--auto-download` | Run `download_king.py` when king digest changes |
| `--output DIR` | King download dir (default `./king`) |
| `--once` | Single poll; exit `1` if king changed (good for cron) |
| `--beep` | Terminal beep on king/manifest change |
| `--state-file PATH` | Custom state file |

**When to use**

- **24/7 mining:** `--auto-download` so you react within one poll interval of a new king.
- **Cron hook:** `--once` every minute; if exit code is `1`, trigger download + training restart.
- **Manifest alert only:** run without `--auto-download`; watch logs for `config_version changed`.

---

### `scripts/data/download_king.py` — king checkpoint

| Flag | Purpose |
|------|---------|
| `--output DIR` | Where to write king (default `./king`) |
| `--verify-only` | SHA256-check existing files, no download |
| `--verbose` | Debug logging |

**When to use**

- **Before every training run** — challenger must start from king (`king_info.json` must exist in `training.king_dir`).
- **After new king** — re-download; old king weights are invalid for submission.
- **Integrity check** — `--verify-only` before a long training job.

---

### `scripts/train/launch.sh` — distributed launcher

Wraps `torchrun` with NCCL tuning for B300 + InfiniBand. Forwards extra args to `train.py`.

| Env var | Default | Purpose |
|---------|---------|---------|
| `NUM_NODES` | `1` | Number of nodes |
| `NODE_RANK` | `0` | This node's rank (0 or 1) |
| `MASTER_ADDR` | `127.0.0.1` | Coordinator IP (node 0) |
| `MASTER_PORT` | `29500` | Coordinator port |
| `GPUS_PER_NODE` | `8` | GPUs per node |
| `CONFIG` | `configs/pro6000/train.yaml` | Config path (`configs/b300/train.yaml` on B300) |

**Single node (8 GPUs)**

```bash
bash scripts/train/launch.sh
bash scripts/train/launch.sh -- --eval-only    # pass train.py flags after --
```

**Multi-node:** currently fail-closed. SparseLoCo and cross-node gradient
synchronization are not production-validated.

---

### `-m src.train` — train / eval / export

All modes require `torchrun` (via `launch.sh`) except that you can inspect `--help` locally.

| Mode | Flag | Exit code |
|------|------|-----------|
| Train (default) | *(none)* | `0` |
| Offline eval | `--eval-only` | `0` if safe to submit, `1` if not |
| Export submission | `--export` | `0` on success, `2` if no checkpoint |
| Pre-decay soup + decay | `--trigger-decay` | *(combined with train)* |

**Shared flags**

| Flag | Purpose |
|------|---------|
| `--config PATH` | YAML config (default `configs/b300/train.yaml`) |
| `--checkpoint DIR` | Specific checkpoint dir; default = latest in `training.output_dir` |
| `--submission-dir DIR` | Export output (default `submission.output_dir`) |
| `--genesis-dir DIR` | Contract reference for export (default `submission.genesis_dir`) |

---

## Training modes in detail

### 1. Default training

```bash
bash scripts/train/launch.sh
```

**What happens**

- Loads model from `training.king_dir` (must contain `king_info.json`).
- WSD schedule: warmup (200) → stable (4000) → decay (800) = 5000 steps total.
- Loss = `lm_loss + z_loss + aux_loss`; expert bias update every step.
- Checkpoints saved during **stable** phase every `checkpoint_averaging.save_every_stable` steps (default 100).
- Final checkpoint at end of `total_steps`.

**Config knobs** (`configs/b300/train.yaml`)

| Key | When to change |
|-----|----------------|
| `training.total_steps` | Longer/shorter run |
| `training.micro_batch_size` | OOM → lower; 2-node → raise to 8 |
| `training.gradient_accumulation_steps` | Adjust global tokens/step |
| `training.activation_checkpointing` | `true` only if OOM on B300 |
| `training.activation_checkpointing_mode` | `selective` (MLP/MoE) or `full` |
| `training.low_precision` | `mxfp8` / `nvfp4` / `none` (TE on sm_10x B300) |
| `training.low_precision_strict` | fail rather than silently fall back when TE is unavailable |
| `training.fp8_training` | legacy alias for `mxfp8` when `low_precision` unset |
| `scheduler.*` | LR schedule shape |
| `optimizer.*` | Muon/AdamW LRs, grad clip |
| `moe.*` | Router stability (z-loss, aux loss, bias speed) |
| `moe.dispatch_overlap` | unsafe custom expert dispatch; must remain `false` |
| `packing.enabled` | must remain `false` with FA4; only eager packed attention is supported |
| `lora.enabled` | freeze 110B base and train attention LoRA + routers (required for Pro 6000 smoke) |
| `compilation.enabled` | `false` to disable `torch.compile` (debugging) |
| `training.smoke_steps` | Cap run length for smoke validation (used by `scripts/verify/smoke_train.py`) |
| `qk_norm.enabled` | unsupported; must remain `false` |

**Resume training:**

```bash
bash scripts/train/launch.sh -- --resume ./checkpoints/step_0002500
```

Loads FSDP model/optimizer state and the scheduler, validates topology, and
continues from the next step. Deterministic resume requires
`data.sequence_pool.enabled: true`; streaming resume is rejected.

**Smoke validation:**

```bash
python scripts/verify/smoke_train.py --skip-torchrun   # unit checks only
NPROC=8 python scripts/verify/smoke_train.py --steps 2 --config configs/pro6000/train.yaml
```

GPU smoke trains **LoRA adapters** (frozen 110B base, trainable attention LoRA + routers)
with FA4 + selective activation checkpointing. Full-parameter pre-training is not
viable on 8× RTX PRO 6000; do not force eager attention for smoke (it is pathologically slow).

Verifies: LoRA wrap (not expert `gate_proj`), gate training patch, router hooks,
z-loss/aux-loss, bias updates, packing fail-closed.

---

### 2. Pre-decay soup + decay phase (`--trigger-decay`)

```bash
bash scripts/train/launch.sh -- --trigger-decay
```

**When to use**

- You have stable-phase checkpoints and want to enter decay before submission.
- Typically run when `lm_loss` has plateaued in stable phase.

**What happens (first stable step)**

1. Collects last `checkpoint_averaging.n_checkpoints` stable checkpoints (min gap `min_step_gap`).
2. Averages weights on rank 0 → `checkpoints/pre_decay_soup.pt`.
3. Loads soup into model on all ranks.
4. Calls `scheduler.trigger_decay()` → LR cosine decay begins.

If no stable checkpoints exist, decay is triggered anyway (with a warning).

**Alternative:** Let training run through full 5000 steps (decay phase is built into WSD schedule at step 4200).

---

### 3. Offline evaluation (`--eval-only`)

```bash
bash scripts/train/launch.sh -- --eval-only
bash scripts/train/launch.sh -- --eval-only --checkpoint ./checkpoints/step_0002500
```

**When to use**

- Before every submission attempt.
- After decay phase or any checkpoint you consider a candidate.
- To compare multiple checkpoints — run eval on each with `--checkpoint`.

**What happens**

- Loads king from `training.king_dir`.
- Loads challenger from `--checkpoint` (or latest; if none, king vs king → `mu_hat ≈ 0`).
- Paired bootstrap on 2000 eval sequences (all 3 datasets, seed 99999).
- Writes `checkpoints/offline_eval_result.json` on rank 0.

**Decision thresholds**

| Metric | Condition | Meaning |
|--------|-----------|---------|
| `accepted` | `LCB > delta` (0.1) | Would pass validator accept bar |
| `safe_to_submit` | `LCB > delta + safety_margin` (0.15) | Recommended before upload |

```bash
# Exit code 0 = safe to submit; 1 = not safe
echo $?
```

**If FAIL:** keep training, try `--trigger-decay`, eval a different stable checkpoint, or tune hyperparams — do **not** export yet.

---

### 4. Export submission (`--export`)

```bash
bash scripts/train/launch.sh -- --export
bash scripts/train/launch.sh -- --export --checkpoint ./checkpoints/step_0004200 --submission-dir ./submit_model
```

**When to use**

- Only after `--eval-only` returns safe (`LCB > 0.15`).

**What happens (rank 0)**

1. Loads checkpoint weights into model.
2. Strips MTP tensors (asserts none remain).
3. Saves safetensors via `save_pretrained`.
4. Copies contract files byte-identical from king; SHA256 vs `submission.genesis_dir`.
5. Removes `manifest.json` if present.
6. Writes `submission_info.json`.

**Then upload** with `teutonic-miner upload` / `ready`.

---

## Training schedule phases

Controlled by `scheduler` in config (WSD):

| Phase | Steps (default) | LR | Checkpointing |
|-------|-----------------|-----|---------------|
| Warmup | 0–199 | 0 → `lr_max` | No scheduled saves |
| Stable | 200–4199 | `lr_max` constant | Every 100 steps (`save_every_stable`) |
| Decay | 4200–4999 | `lr_max` → `lr_min` cosine | No extra scheduled saves |

**Practical guidance**

| Goal | Action |
|------|--------|
| Fast iteration / mid-stable snapshot | Use stable checkpoint from `step_XXXXXXX/` |
| Best quality before submit | `--trigger-decay` or train through decay, then eval |
| Multi-node sync point | Use `step_XXXXXXX_outer` after SparseLoCo sync |

---

## Situation → action reference

### New king crowned

1. Stop current training (if running on old king).
2. `python scripts/data/download_king.py --output ./king` (or rely on monitor `--auto-download`).
3. **Discard** old checkpoints trained from previous king — they cannot be submitted.
4. Restart: `bash scripts/train/launch.sh`.

### Dataset manifest `config_version` changed

1. Monitor logs: `config_version changed`.
2. Fetch manifest: `https://teutonic.ai/datasets/manifest.json`.
3. Check new sources/proportions — training sampler auto-refreshes every 100 yields, but review before long runs.
4. Optional: clear `data.cache_dir` if shard URLs changed.

### Out of memory (OOM)

Apply in order (edit `configs/b300/train.yaml`):

1. `training.micro_batch_size: 2` (or `1`)
2. `training.gradient_accumulation_steps: 8` (keep tokens/step similar)
3. `training.activation_checkpointing: true`
4. `parallelism.fsdp_cpu_offload: true`
5. `compilation.enabled: false` (reduces compile memory spike)

### Router instability (dead experts, exploding logits)

| Symptom | Config change |
|---------|---------------|
| Logit explosion | Raise `moe.z_loss_coeff` (e.g. `2e-4`) |
| Expert collapse | Raise `moe.seq_aux_alpha` or `moe.bias_update_speed` |
| Router grad spikes | Lower `optimizer.gradient_clip.router` (e.g. `0.3`) |

Watch `dead=` and `max_load=` in training logs.

### Per-dataset divergence

Watch `datasets={...}` in logs (EMA per source). If one dataset loss spikes:

- Manifest proportion issue — verify all 3 datasets are represented.
- Not a separate CLI mode; adjust training length or LR.

### Eval passes (`LCB > 0.15`) but export fails

- **Contract mismatch:** run `bash setup.sh` to refresh `./genesis`; ensure king contract files intact.
- **MTP keys:** should be stripped automatically; if not, check checkpoint for `mtp` in state dict.

### Eval fails (`LCB ≤ 0.15`)

| Next step | Command / action |
|-----------|------------------|
| Continue stable training | `bash scripts/train/launch.sh -- --resume ./checkpoints/step_XXXXXXX` |
| Try soup + decay | `bash scripts/train/launch.sh -- --trigger-decay` |
| Eval another checkpoint | `--eval-only --checkpoint ./checkpoints/step_XXXXXXX` |
| Extend stable (code) | `scheduler.extend_stable(n)` — requires wiring or longer `stable_steps` in config |

---

## Checkpoint layout

```
checkpoints/
├── step_0000200/
│   ├── model.pt
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── sparse_loco.pt      # if multi-node
│   └── meta.json
├── step_0000200_outer/     # SparseLoCo sync checkpoint
├── pre_decay_soup.pt       # after --trigger-decay
└── offline_eval_result.json
```

**Latest checkpoint** = highest `step_XXXXXXX` (ignores `_outer` suffix for `find_latest_checkpoint`).

---

## Config file map

Single source of truth: `configs/b300/train.yaml`

| Section | Controls |
|---------|----------|
| `hardware` | Nodes, GPUs per node |
| `training` | Batch size, steps, paths, FP8, checkpointing |
| `scheduler` | WSD warmup/stable/decay, LR |
| `optimizer` | Muon + AdamW, grad clip, router LR scale |
| `moe` | Router losses, EP dispatch overlap |
| `sparse_loco` | Multi-node gradient sync (auto-noop on 1 node) |
| `parallelism` | FSDP, CPU offload |
| `compilation` | `torch.compile` layers + optimizer |
| `packing` | Document-boundary sequence packing |
| `checkpoint_averaging` | Stable saves + soup size |
| `qk_norm` | Optional QK norm injection |
| `submission` | Export dir, genesis dir, safety margin |
| `data` | Shard cache, manifest URL, DataLoader workers |

**Before each run**, at minimum verify:

- `training.king_dir` points to current king
- `hardware.num_nodes` matches your launch command
- `training.micro_batch_size` matches node count

---

## Submission safety checklist

- [ ] Trained from **current** king (`king_info.json` digest matches dashboard)
- [ ] `--eval-only` → `safe_to_submit: true` (`LCB > 0.15`)
- [ ] `--export` succeeded; contract SHA256 verified vs `./genesis`
- [ ] No `manifest.json` in `./submit_model`
- [ ] No MTP keys in weights
- [ ] `teutonic-miner upload` + `ready` on correct hotkey

---

## Command cheat sheet

```bash
# Setup
bash setup.sh

# King
python scripts/data/download_king.py --output ./king
python scripts/data/download_king.py --output ./king --verify-only

# Monitor
python scripts/monitor.py --auto-download
python scripts/monitor.py --once

# Train
bash scripts/train/launch.sh
bash scripts/train/launch.sh -- --trigger-decay

# Validate & ship
bash scripts/train/launch.sh -- --eval-only
bash scripts/train/launch.sh -- --eval-only --checkpoint ./checkpoints/step_0003000
bash scripts/train/launch.sh -- --export --checkpoint ./checkpoints/step_0004200
```
