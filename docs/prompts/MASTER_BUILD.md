# Teutonic SN3 MoE Training Pipeline — Master Build Spec

**Authoritative implementation document.** Merges `prompts/first.md`, `prompts/second.md`, and `prompts/third.md`.

Always read `.cursor/rules/CONTEXT.md` first for competition protocol, datasets, submission rules, and workflow.

---

## Precedence Rules

When specs conflict, apply in this order:

1. **third.md** (audit fixes, Muon, packing, compile, router hooks) — wins
2. **second.md** (B300 hardware, process groups, SparseLoCo, FP8) — wins over first
3. **first.md** (base pipeline) — fills gaps not covered above
4. **CONTEXT.md** — protocol/submission rules always apply; training hardware details defer to second/third

---

## Conflict Resolution Table

| Topic | first.md | second.md | third.md | **Use** |
|-------|----------|-----------|----------|---------|
| GPU | 8× H200 (96 GB) | 8× B300 (288 GB), optional 2 nodes | B300 (inherits second) | **B300** |
| `micro_batch_size` | 1 (H200 OOM) | 4 (1 node), 8 (2 nodes) | 4 | **4** (1 node); **8** (2 nodes) |
| `gradient_accumulation_steps` | 8 | 4 | 4 | **4** |
| Tokens/step | ~64–128 seqs | 8×4×4 = 131K tokens | same | **131K tokens/step** (1 node) |
| `activation_checkpointing` | on (OOM defense) | off | off | **false** on B300 |
| `fp8_training` | — | true (MXFP8) | — | **true** (falls back to BF16) |
| Optimizer | AdamW all params | AdamW (inherits first) | Muon + AdamW hybrid | **Muon + AdamW** via `optimizer.py` |
| AdamW LR | 1.5e-5 | — | `adamw_lr: 1.5e-5` | **1.5e-5** peak (AdamW groups) |
| Muon LR | — | — | `muon_lr: 0.005` | **0.005** (continued pretraining) |
| Router LR | lr × 0.1 | — | `router_lr_scale: 0.1` | **adamw_lr × 0.1** |
| Optimizer states | FP32 | — | BF16 for AdamW | **BF16** (`adamw_bf16_states: true`) |
| Muon momentum buffer | — | — | FP32 | **FP32** |
| Grad clipping | global 1.0 | — | per-module | **per-module** (router 0.5, others 1.0) |
| FSDP process group | global | `ep_group` (intra-node) | inherits second | **`pg.ep_group`** |
| FSDP wrap policy | one DecoderLayer unit | same | expert vs attention split | **Separate wrap: expert modules + attention modules** |
| Multi-node sync | — | SparseLoCo | inherits second | **SparseLoCo** when `dp_size > 1` |
| Router monitoring | `get_router_stats(model)` post-forward | no change | forward hooks | **`RouterHookManager` hooks** |
| Z-loss input | `outputs.router_logits` | — | `hook_manager` | **`compute_z_loss(hook_manager)`** |
| Bias update frequency | every 10 steps | — | every step | **every step** |
| Seq-level aux loss | — | — | α=1e-4 | **`lm_loss + z_loss + aux_loss`** |
| `torch.compile` | — | — | enabled | **true** (per-layer + optimizer step) |
| Sequence packing | flat 2048 chunks | — | EOS + `cu_seqlens` | **`packing.py`** when `packing.enabled` |
| EP dispatch overlap | — | implicit in topology | `ep_dispatch.py` | **`ExpertDispatcher`** async all-to-all |
| Checkpoint strategy | save on schedule | sync ckpt on outer step | pre-decay soup (K=5) | **both SparseLoCo sync ckpt + stable-phase soup** |
| QK norm | — | — | check first, inject only if safe | **`check_and_inject_qk_norm`** — do NOT inject if king lacks it |
| Launch | `torchrun --standalone` | `scripts/train/launch.sh` | — | **`scripts/train/launch.sh`** |
| OOM fallback | grad checkpoint + cpu offload | cpu offload false | — | **Reduce micro_batch first**; cpu offload only if forced on smaller GPUs |

---

## Hardware Target

**Primary:** 8× NVIDIA B300 per node (288 GB HBM3e each, 2.3 TB per node)

- NVLink 5 intra-node: 14.4 TB/s
- InfiniBand inter-node (optional 2nd node): 800 Gb/s per GPU (ConnectX-8)
- **2 nodes:** 16× B300 = 4.6 TB total; `world_size=16`, `dp_size=2`, `ep_size=8`

**Model memory:** ~220 GB BF16 for 110B MiMo-V2 MoE

| Config | Params/GPU | Free memory/GPU | micro_batch |
|--------|------------|-----------------|-------------|
| 1 node (8 GPU) | 27.5 GB | 260+ GB | 4 |
| 2 nodes (16 GPU) | 13.75 GB | abundant | 8 |

**Distributed topology (critical):**

```
EP_GROUP = 8 GPUs within one node  (expert all-to-all stays on NVLink)
DP_GROUP = paired GPUs across nodes (inter-node; SparseLoCo sync)

1-node  (world_size=8):  EP=[0..7],  DP=trivial (size 1 each)
2-node  (world_size=16): EP=[0..7] + [8..15],  DP=[i, i+8] for i in 0..7
```

---

## Complete File Tree

```
teutonic-miner/
├── configs/
│   └── train_config.yaml          # single source of truth for all hyperparams
├── src/
│   ├── process_groups.py          # NEW (second) — build first
│   ├── sparse_loco.py             # NEW (second) — multi-node sync
│   ├── optimizer.py               # NEW (third) — Muon + AdamW
│   ├── ep_dispatch.py             # NEW (third) — async EP all-to-all
│   ├── packing.py                 # NEW (third) — document masking
│   ├── checkpoint_averaging.py    # NEW (third) — pre-decay model soup
│   ├── qk_norm.py                 # NEW (third) — QK norm check/inject
│   ├── model.py                   # load, FP8, FSDP, compile helpers
│   ├── data.py                    # unchanged per second/third
│   ├── scheduler.py               # unchanged per second/third
│   ├── moe_utils.py               # REWRITE (third) — router hooks
│   ├── offline_eval.py            # unchanged per second/third
│   └── train.py                   # full assembly
├── scripts/
│   ├── download_king.py           # enhance (first)
│   ├── monitor.py                 # referenced in CONTEXT
│   └── launch.sh                  # NEW (second)
├── requirements.txt
└── setup.sh
```

---

## Global Code Standards

Apply to every file (from first.md; still binding):

1. Type hints on every function signature; explicit return types
2. Every class has `__repr__`
3. Every blocking I/O call has a `timeout` parameter
4. `log.info` / `log.warning` rank-0 gated **except** per-rank failure warnings (download errors, shard skips)
5. No bare `except` — always `except (SpecificError, ...) as e`
6. FSDP file writes: only when `dist.get_rank() == 0` (or `pg.is_rank0`)
7. `optimizer.zero_grad(set_to_none=True)`
8. Constants in `SCREAMING_SNAKE_CASE` at top of file
9. Do **not** use `model.half()` or `model.float()` — dtype via FSDP MixedPrecision / FP8
10. Random seeds: `(base_seed + rank)` for per-rank diversity; multi-node data seed: `(base_seed + pg.dp_rank)`

---

## File Specifications

### `src/process_groups.py` (second — foundation)

Central registry of distributed topology. **All other distributed code imports from here.**

```python
class ProcessGroups:
    ep_group: ProcessGroup      # intra-node, 8 ranks
    dp_group: ProcessGroup      # inter-node
    ep_rank: int                # 0-7
    dp_rank: int                # 0 or 1
    ep_size: int                # always 8
    dp_size: int                # 1 or 2
    local_rank: int             # GPU on node
    global_rank: int
    world_size: int             # 8 or 16
    node_rank: int              # 0 or 1
    is_rank0: bool              # global rank == 0
```

- `init_process_groups() -> ProcessGroups` — call once after `dist.init_process_group("nccl")`
- `get_ep_group()`, `get_dp_group()` — convenience accessors
- `log_topology(pg)` — print topology on rank 0 at startup

---

### `src/sparse_loco.py` (second)

SparseLoCo inter-node gradient sync (arxiv 2508.15706). **Auto-disabled when `dp_size == 1`.**

```python
class SparseLoCo:
    inner_steps: int = 200
    sparsity_k: float = 0.01      # top 1% by magnitude
    quant_bits: int = 2
    outer_lr: float = 0.7
    outer_momentum: float = 0.9   # Nesterov
```

Key methods:
- `should_sync(step)` — `enabled and step > 0 and step % inner_steps == 0`
- `snapshot_params()` — save `theta_init` at start and after each outer sync
- `outer_step(step)` — pseudo-grad → error feedback → top-k → quantize → all-reduce on `dp_group` → dequantize → Nesterov SGD → snapshot

Training loop calls `outer_step` **after** `optimizer.step()`, no branching on `dp_size` in train.py.

Log: `compression_ratio`, `sync_time_ms`, `pseudo_grad_norm`.

---

### `src/optimizer.py` (third — critical)

**Replaces** first.md's single AdamW optimizer.

#### `NewtonSchulzOrthogonalizer`
- 5-step NS iteration; coefficients `a=3.4445, b=-4.7750, c=2.0315`
- Cast to float32 internally for orthogonalization

#### `DistributedMuon`
- All-gather full grad on `ep_group` → orthogonalize → reduce-scatter update
- Nesterov momentum (default 0.95), `ns_steps=5`
- Scale: `update *= max(1, m/n)^0.5`
- Only 2D matrices; not router/embedding/norm

#### `build_optimizer(model, cfg, pg) -> CombinedOptimizer`

| Group | Optimizer | Params |
|-------|-----------|--------|
| A | Muon | 2D attention + expert FFN weights |
| B | AdamW | biases, norms, embeddings, LM head, other 1D |
| C | AdamW | router/gate/expert_bias — `lr × router_lr_scale`, `weight_decay=0` |

AdamW states in **BF16** when `adamw_bf16_states: true`. Muon momentum in **FP32**.

#### `CombinedOptimizer`
Unified interface: `.step()`, `.zero_grad()`, `.state_dict()`, `.load_state_dict()`, `.param_groups` (union for scheduler).

---

### `src/ep_dispatch.py` (third)

Async expert parallelism dispatcher for 256 experts, top-8 routing, EP=8.

```python
class ExpertDispatcher:
    def dispatch_async(tokens, expert_indices, expert_scores) -> AsyncDispatchHandle
    def combine_async(expert_outputs, dispatch_handle) -> AsyncCombineHandle
    def compute_experts(dispatched_tokens, expert_weights, expert_indices) -> Tensor
```

- Node-limited routing (`node_limit=1`): tokens never cross node boundaries
- Uses `dist.all_to_all_single` async; overlap with compute (`delay_wgrad`)
- Grouped GEMM when available; loop fallback otherwise

---

### `src/packing.py` (third)

Document-boundary-aware sequence packing for `flash_attn_varlen_func`.

```python
def find_document_boundaries(token_ids, eos_id) -> Tensor  # cu_seqlens
def pack_batch_with_masking(batch, eos_id) -> dict
def flash_attn_varlen_forward(q, k, v, cu_seqlens, max_seqlen, ...) -> Tensor
```

Output batch keys: `input_ids`, `labels` (-100 at EOS), `cu_seqlens`, `max_seqlen`, `position_ids`, `datasets`.

If no EOS in sequence: `cu_seqlens = [0, seq_len]`.

---

### `src/checkpoint_averaging.py` (third)

Pre-decay model soup for WSD stable phase.

```python
def average_checkpoints(paths, output_path, device="cpu", dtype=bf16) -> dict
def collect_stable_phase_checkpoints(output_dir, scheduler, max_n=5, min_step_gap=50) -> list[str]
def run_pre_decay_soup(model, output_dir, scheduler, rank, n_checkpoints=5) -> None
```

Workflow: collect last N stable checkpoints → average on rank 0 → load into model (all ranks) → save → `scheduler.trigger_decay()`.

---

### `src/qk_norm.py` (third)

```python
def check_and_inject_qk_norm(model, cfg) -> bool
```

**Safety rule:** Only inject if `model.config` has `qk_norm=False` **AND** `cfg` explicitly sets `qk_norm: true`. If king/genesis trained without QK norm, **do not inject** — eval would be invalid.

---

### `src/moe_utils.py` (third — full rewrite)

Replaces first.md implementation entirely.

| Function | Purpose |
|----------|---------|
| `register_router_hooks(model) -> RouterHookManager` | Capture router logits + routing weights via forward hooks |
| `RouterHookManager.get_stats()` | Per-layer `RouterLayerStats` |
| `RouterHookManager.clear()` / `.remove()` | Reset / cleanup hooks |
| `update_expert_biases(model, hook_manager, ...)` | Aux-loss-free bias update (DeepSeek-V3 §2.3.2); create `expert_bias` buffer if missing |
| `compute_seq_auxiliary_loss(hook_manager, alpha=1e-4)` | Sequence-wise aux loss (differentiable) |
| `compute_z_loss(hook_manager, z_loss_coeff=1e-4)` | `mean(logsumexp(logits)^2)` with LSE trick |
| `get_grad_norms(model, pg)` | FSDP-aware norms: router, expert_ffn, attention, embed_lm_head, total |
| `clip_grad_norms_per_module(model, pg, ...)` | Per-module clip; router max 0.5 |
| `log_expert_utilization(...)` | Compact one-line summary (from first.md, use hook stats) |

**Loss composition:**
```
total_loss = lm_loss + z_loss + aux_loss_seq
```

Bias update: **every training step** (not every 10).

---

### `src/model.py` (first + second + third)

#### Load
- `AutoModelForCausalLM` + `trust_remote_code=True`
- At load time:
  - `model.config.output_router_logits = True`
  - `model.config.use_cache = False`
- Assert `king_info.json` exists in `king_dir` (never train from genesis alone)

#### `get_parameter_groups(model, weight_decay)`
Still used for logging/reference; actual routing via `build_optimizer`.

#### `print_model_summary(model)`
Total params, active params/token, expert count, layers, attention pattern, **FP8 status**, per-node memory.

#### `enable_fp8_training(model, pg) -> bool` (second)
- Transformer Engine `MXFP8BlockScaling()` on Blackwell (cap >= 9)
- **Exclude:** router, gate, embed, lm_head
- **Include:** expert FFN + attention Q/K/V/O linears
- Fallback: log warning, return False

#### `fp8_context(enabled)` (second)
Returns `te.fp8_autocast()` or `nullcontext()`.

#### `apply_fsdp(model, pg, cfg) -> FSDP` (second + third)
- `process_group=pg.ep_group` (NOT global)
- `ShardingStrategy.FULL_SHARD`
- `MixedPrecision(param=bf16, reduce=bf16, buffer=bf16)`
- `use_orig_params=True`, `limit_all_gathers=True`
- **Wrap policy (third):** separate units for expert modules vs attention modules; expert unit = one expert
- Detect decoder layer class via `model.model.layers[0]`
- `fsdp_cpu_offload: false` on B300

#### `enable_gradient_checkpointing(model, decoder_layer_cls)` (first)
Only when `activation_checkpointing: true` (default **false** on B300).

#### `compile_decoder_layers(model, pg)` (third)
- Per decoder layer: attention `mode="max-autotune"`, MoE `mode="reduce-overhead"`
- `torch._dynamo.config.cache_size_limit = 64`
- Skip embeddings + LM head

---

### `src/data.py` (first — unchanged)

Multi-dataset weighted streaming pipeline.

- **Manifest** (`https://teutonic.ai/datasets/manifest.json`): cached 60s; warn on `config_version` change
- **ShardCache**: LRU, thread-safe downloads, mmap `np.load(..., mmap_mode='r')`
- **WeightedShardSampler**: infinite iterator; `shards[rank::world_size]`; manifest refresh every 100 yields
- **TeutonicDataset**: `input_ids`, `labels` (2048), `dataset` name; worker-aware rank/world_size
- **build_train_loader(cfg, rank, world_size)**: `num_workers=4`, `prefetch_factor=8`, `pin_memory=True`, `persistent_workers=True`
- **EvalDataset**: pre-download N seqs per source, seed=99999; CPU storage

**Multi-node (second):** pass `rank=pg.global_rank`, `world_size=pg.world_size`; data seed uses `dp_rank` for node diversity.

---

### `src/scheduler.py` (first — unchanged)

- **WSDScheduler**: warmup (linear) → stable (constant) → decay (cosine)
  - `get_lr(step)`, `step()`, `trigger_decay()`, `extend_stable(n)`
  - `phase`: `"warmup" | "stable" | "decay"`
  - Full `state_dict` / `load_state_dict`
- **CosineWarmupScheduler**: backup with `extend_total_steps(n)`
- **build_scheduler(optimizer, cfg)**

Default schedule (CONTEXT):
- warmup: 200 steps
- stable: 4000 steps
- decay: 800 steps
- peak LR: 1.5e-5 (AdamW groups); min: 1.5e-6

---

### `src/offline_eval.py` (first — unchanged)

Validator-matching paired bootstrap eval.

- `paired_bootstrap_lcb(diffs, n_bootstrap=10000, alpha=0.001, seed=0)`:
  ```python
  rng = np.random.default_rng(seed=0)
  LCB = 2*mu_hat - np.quantile(boot_means, 1-alpha)
  ```
- `compute_per_sequence_ce`: manual CE per sequence (NOT `model(..., labels=).loss`)
- **OfflineEvaluator.evaluate(challenger)** → **EvalResult**
- `EvalResult.is_safe_to_submit(safety_margin=0.05)`: `lcb > delta + 0.05`
- Submit threshold: **offline LCB > 0.15** (delta 0.1 + margin 0.05)

Always runs rank 0, single-node.

---

### `-m src.train` (first + second + third — full assembly)

#### CLI
```
--config          configs/b300/train.yaml
--export          export submission from latest checkpoint, exit
--eval-only       offline eval vs king, exit
--trigger-decay   run pre-decay soup + decay phase
```

#### Setup
```python
def setup_distributed(cfg) -> ProcessGroups:
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pg = init_process_groups()
    log_topology(pg)  # rank 0
    return pg
```

Read `TEUTONIC_NODES` env var for logging only; topology from `world_size`.

#### Init sequence
1. Load challenger from **king** checkpoint
2. `check_and_inject_qk_norm(model, cfg)` (if configured)
3. `fp8_enabled = enable_fp8_training(model, pg)`
4. `model = apply_fsdp(model, pg, cfg)`
5. `compile_decoder_layers(model, pg)` if `compilation.enabled`
6. `optimizer = build_optimizer(model, cfg, pg)`
7. Optionally `torch.compile(optimizer.muon.step)` and `torch.compile(optimizer.adamw.step)`
8. `scheduler = build_scheduler(optimizer, cfg)`
9. `hook_manager = register_router_hooks(model)`
10. `sparse_loco = SparseLoCo(model, pg.dp_group, ..., enabled=(pg.dp_size > 1))`
11. `sparse_loco.snapshot_params()`
12. `ExpertDispatcher` if `moe.dispatch_overlap`

#### Training loop (per step)
```python
for step in range(start_step, total_steps):
    for acc in range(grad_accum_steps):
        batch = next(train_iter)
        if cfg.packing.enabled:
            batch = pack_batch_with_masking(batch, eos_id=tokenizer.eos_token_id)

        with fp8_context(fp8_enabled):
            outputs = model(input_ids=..., labels=..., use_cache=False)

        lm_loss = outputs.loss
        z_loss = compute_z_loss(hook_manager, cfg.moe.z_loss_coeff)
        aux_loss = compute_seq_auxiliary_loss(hook_manager, cfg.moe.seq_aux_alpha)
        total_loss = lm_loss + z_loss + aux_loss
        (total_loss / grad_accum_steps).backward()
        hook_manager.clear()

    grad_norms = clip_grad_norms_per_module(model, pg, ...)
    optimizer.step()
    lr = scheduler.step()
    optimizer.zero_grad(set_to_none=True)

    update_expert_biases(model, hook_manager, cfg.moe.bias_update_speed)

    if sparse_loco.should_sync(step):
        sparse_loco.outer_step(step)
        save_checkpoint(..., suffix="_outer")  # sync checkpoint

    if scheduler.phase == "stable" and step % cfg.checkpoint_averaging.save_every_stable == 0:
        save_checkpoint(...)

    if args.trigger_decay and not decay_triggered:
        run_pre_decay_soup(model, output_dir, scheduler, rank)
        decay_triggered = True

    # rank 0 logging: lm_loss, z_loss, aux_loss, lr, phase, grad_norms,
    #   tokens/sec, fp8_enabled, dp_size, ep_size, compression_ratio,
    #   dataset EMA losses, router stats (dead experts, max load)
```

#### `DatasetLossTracker` (first)
EMA `alpha=0.98` per dataset name — early warning for dataset divergence.

#### Checkpointing
- Save: `model.pt`, `optimizer.pt`, `scheduler.pt`, `meta.json`
- FSDP full state dict: `offload_to_cpu=True`, `rank0_only=True`, consolidate from `ep_group`
- `meta.json`: step, lm_loss, lr, phase, timestamp, dataset_losses
- SparseLoCo outer sync: additional `_outer` checkpoint
- `dist.barrier()` at sync points

#### `export_submission(model, king_dir, output_dir, rank)`
- Strip MTP keys (assert none remain)
- `save_pretrained` safetensors
- Copy contract files **byte-identical** from king_dir; SHA256 verify vs genesis
- Remove `manifest.json` if present
- Write `submission_info.json`

---

### `scripts/data/download_king.py` (first — enhance)

Add:
- Verify total shard count matches manifest before returning
- Print download speed (GB/s) per shard
- `--verify-only`: checksum check without re-download

---

### `scripts/train/launch.sh` (second)

```bash
# Single node:
bash scripts/train/launch.sh

# 2-node:
NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh  # node 0
NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh  # node 1
```

NCCL tuning for B300 + IB: `NCCL_IB_DISABLE=0`, `NCCL_NET_GDR_LEVEL=5`, `NCCL_IB_GID_INDEX=3`, `NCCL_CROSS_NIC=1`, `NCCL_SOCKET_IFNAME=bond0`, `CUDA_DEVICE_MAX_CONNECTIONS=1`, `TEUTONIC_NODES=$NUM_NODES`.

---

### `configs/b300/train.yaml`

Single source of truth. **Every value must have a comment explaining WHY.**

```yaml
# ── Hardware ──────────────────────────────────────────
hardware:
  gpus_per_node: 8
  num_nodes: 1              # 1 or 2 — update before each run

# ── Training ──────────────────────────────────────────
training:
  micro_batch_size: 4       # B300: 3× H200 memory; 1-node default
  gradient_accumulation_steps: 4   # 8×4×4×2048 ≈ 131K tokens/step
  activation_checkpointing: false  # B300 has abundant memory
  fp8_training: true        # MXFP8 on Blackwell; falls back to BF16
  total_steps: 5000         # warmup + stable + decay
  log_every: 10
  base_seed: 42
  king_dir: ./king
  output_dir: ./checkpoints
  eval_batch_size: 4

# ── WSD Schedule ──────────────────────────────────────
scheduler:
  type: wsd
  warmup_steps: 200
  stable_steps: 4000
  decay_steps: 800
  # AdamW peak/min — Muon uses optimizer.muon_lr
  lr_max: 1.5e-5
  lr_min: 1.5e-6

# ── Optimizer (third) ─────────────────────────────────
optimizer:
  muon_lr: 0.005            # continued pretraining (0.02 for scratch)
  muon_momentum: 0.95
  muon_nesterov: true
  muon_ns_steps: 5
  muon_weight_decay: 0.0
  adamw_lr: 1.5e-5
  adamw_lr_min: 1.5e-6
  adamw_betas: [0.9, 0.95]
  adamw_eps: 1.0e-8
  adamw_weight_decay: 0.1
  adamw_bf16_states: true
  router_lr_scale: 0.1
  gradient_clip:
    expert_ffn: 1.0
    attention: 1.0
    router: 0.5

# ── MoE (third) ───────────────────────────────────────
moe:
  bias_update_speed: 0.001   # set 0.0 for final 500B tokens if needed
  z_loss_coeff: 1.0e-4
  seq_aux_alpha: 1.0e-4
  node_limit: 1
  dispatch_overlap: true
  delay_wgrad: true

# ── SparseLoCo (second) ─────────────────────────────
sparse_loco:
  enabled: true              # auto-noop when dp_size==1
  inner_steps: 200
  sparsity_k: 0.01
  quant_bits: 2
  outer_lr: 0.7
  outer_momentum: 0.9

# ── Parallelism (second) ──────────────────────────────
parallelism:
  fsdp: true
  fsdp_shard_params: true
  fsdp_cpu_offload: false

# ── Compilation (third) ───────────────────────────────
compilation:
  enabled: true
  decoder_mode: reduce-overhead
  attention_mode: max-autotune
  compile_optimizer: true
  cache_size_limit: 64

# ── Packing (third) ───────────────────────────────────
packing:
  enabled: true

# ── Checkpoint averaging (third) ──────────────────────
checkpoint_averaging:
  enabled: true
  n_checkpoints: 5
  save_every_stable: 100
  min_step_gap: 50

# ── QK norm (third) ───────────────────────────────────
qk_norm:
  enabled: false             # ONLY true if genesis has qk_norm=False AND you accept eval risk

# ── Data ──────────────────────────────────────────────
data:
  cache_dir: ./shard_cache
  cache_max_gb: 500
  manifest_url: https://teutonic.ai/datasets/manifest.json
  manifest_refresh_interval: 100
```

---

### `requirements.txt` + `setup.sh`

**requirements.txt:**
```
torch>=2.4.0
transformers>=4.47.0
accelerate>=0.35.0
safetensors>=0.4.0
numpy>=1.26.0
requests>=2.31.0
tqdm
pyyaml
flash-attn>=2.6.0
transformer-engine>=2.0.0
triton>=3.0.0
wandb
```

**Install order:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformer-engine --extra-index-url https://pypi.nvidia.com
pip install -r requirements.txt
pip install flash-attn --no-build-isolation || echo "flash-attn optional, skipping"
```

**setup.sh:**
```bash
pip install -r requirements.txt
pip install flash-attn --no-build-isolation || echo "flash-attn optional, skipping"
huggingface-cli download dendriteholdings/teutonic-II-110B-genesis \
  --revision 8f971f030f5d801f812540007b396058f69cc08a \
  --local-dir ./genesis --include "*.py" "*.json" "*.jinja*"
```

---

## Phased Build Order

Implement in phases. Each phase should leave the repo in a runnable (or testable) state.

### Phase 1 — Foundation (no distributed extras)
1. `src/data.py`
2. `src/scheduler.py`
3. `src/offline_eval.py`
4. `scripts/data/download_king.py`
5. `configs/b300/train.yaml` (skeleton)
6. `requirements.txt`, `setup.sh`

### Phase 2 — Distributed topology + core model
7. `src/process_groups.py`
8. `src/model.py` (load, FSDP on ep_group — no FP8/compile yet)
9. `src/moe_utils.py` (router hooks rewrite)
10. `src/optimizer.py`
11. `-m src.train` (minimal loop: forward → loss → step)
12. `scripts/train/launch.sh`

### Phase 3 — B300 + multi-node
13. `src/sparse_loco.py`
14. FP8 in `model.py`
15. SparseLoCo + FP8 wiring in `train.py`
16. Full `train_config.yaml` hardware/sparse_loco sections

### Phase 4 — Throughput + stability optimizations
17. `src/packing.py`
18. `src/ep_dispatch.py`
19. `src/checkpoint_averaging.py`
20. `src/qk_norm.py`
21. `compile_decoder_layers` in `model.py`
22. Full `train.py` rewrite (all losses, soup, logging)

### Phase 5 — Validate
23. `--eval-only` against king
24. `--export` submission
25. Fix until `offline LCB > 0.15`

**Within-phase file order (third.md build order, adjusted for dependencies):**

```
process_groups → sparse_loco → optimizer → ep_dispatch → packing
→ moe_utils → checkpoint_averaging → qk_norm → model → train → config
```

`data.py`, `scheduler.py`, `offline_eval.py` can be built anytime in Phase 1.

---

## What Winning Looks Like

1. **Stable training** — `lm_loss` decreases; no NaN/inf
2. **All 3 datasets** in manifest proportions (not finewebedu-only)
3. **Balanced routing** — no dead experts, no collapsed routing
4. **Offline eval** — `LCB > 0.15` before submit (`delta 0.1 + margin 0.05`)
5. **Speed** — training starts within ~10 minutes of new king

## Failure Modes & Defenses

| Failure | Defense |
|---------|---------|
| Exploding router logits | Z-loss (`1e-4`) |
| Expert collapse (batch-level) | Bias update every step |
| Expert collapse (seq-level) | Seq aux loss (`1e-4`) |
| Catastrophic forgetting | Per-dataset loss EMA |
| Training from genesis | Assert `king_info.json` in `king_dir` |
| MTP in submission | Strip + assert no `mtp` keys |
| Wrong contract files | SHA256 vs genesis |
| Cross-doc attention noise | `packing.py` + `cu_seqlens` |
| Inter-node EP traffic | EP stays intra-node; SparseLoCo for DP only |
| QK norm eval mismatch | Only inject when explicitly safe |

---

## End-to-End Workflow

```bash
python scripts/monitor.py --auto-download
python scripts/data/download_king.py --output ./king
bash scripts/train/launch.sh                                    # or 2-node variant
python -m src.train --config configs/b300/train.yaml --eval-only
python -m src.train --config configs/b300/train.yaml --export
teutonic-miner upload --hotkey h0 --name teutonic-II-110B-<suffix> ./submit_model
teutonic-miner ready --hotkey h0
```

---

## Cursor Usage

When implementing, tell Cursor:

```
Read .cursor/rules/CONTEXT.md and prompts/MASTER_BUILD.md.
Implement Phase N only: [file list].
Do not implement later phases.
```

After each phase, verify imports and that `train.py --help` runs.

---

## Source Documents

| File | Role |
|------|------|
| `.cursor/rules/CONTEXT.md` | Protocol, competition, datasets, submission rules |
| `prompts/first.md` | Base pipeline (9 files) |
| `prompts/second.md` | B300 hardware, process groups, SparseLoCo, FP8 |
| `prompts/third.md` | Muon, packing, compile, router hooks, gap fixes |
| **`prompts/MASTER_BUILD.md`** | **This file — single implementation authority** |
