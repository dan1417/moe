Read CONTEXT.md first. Now build the complete Teutonic SN3 MoE training pipeline.

## Hardware Target
8× H200 SXM5 (96 GB HBM3e each). NVLink for intra-node GPU comms.
Total GPU memory: 768 GB. Model in BF16 is ~220 GB. Plan for it.

## What to build (one file per section)

─────────────────────────────────────────
FILE 1: src/model.py
─────────────────────────────────────────
Load and configure the MiMo-V2 model for training.

Requirements:
- Load with AutoModelForCausalLM + trust_remote_code=True
- Tie all MoE-specific training knobs at load time:
    model.config.output_router_logits = True   (needed for aux loss monitoring)
    model.config.use_cache = False              (never True during training)
- Implement get_parameter_groups(model, weight_decay) that returns:
    Group 1 — weight decay:    all 2D+ params except router/embedding/norm
    Group 2 — no weight decay: biases, norms, embeddings
    Group 3 — router only:     all params whose name contains "router" or "gate"
               weight_decay=0, lr_scale=0.1  (routers need lower LR for stability)
- Implement print_model_summary(model) that shows:
    total params, active params/token, expert count, layers, attn pattern
- Implement apply_fsdp(model, local_rank, cpu_offload=False) using FSDP2:
    ShardingStrategy.FULL_SHARD
    MixedPrecision(param=bf16, reduce=bf16, buffer=bf16)
    auto_wrap_policy: detect decoder layer class by introspecting
      model.model.layers[0] — wrap each DecoderLayer individually
    use_orig_params=True  (required for param group LR scaling)
    limit_all_gathers=True
    Return (fsdp_model, decoder_layer_cls)
- Implement enable_gradient_checkpointing(model, decoder_layer_cls) using
    torch.distributed.fsdp.wrap.checkpoint_wrapper on each DecoderLayer

─────────────────────────────────────────
FILE 2: src/data.py
─────────────────────────────────────────
Multi-dataset weighted streaming data pipeline.

Constraints:
- Datasets are pre-tokenized .npy uint32 shards on Cloudflare R2
- Each shard = flat array of uint32 token IDs, packed in 2048-token sequences
- Manifest at https://teutonic.ai/datasets/manifest.json is the source of truth
- Proportions in manifest EXACTLY match eval weights — do not deviate

Requirements:
- class Manifest:
    fetch(force=False) → List[DatasetSource]  (cached 60s)
    properties: delta_threshold, eval_n, config_version
    Log a WARNING if config_version changes between refreshes

- class ShardCache(cache_dir, max_gb):
    LRU eviction by disk usage when max_gb exceeded
    Thread-safe concurrent downloads (one thread per shard, dedup in-flight)
    load_shard(dataset_name, key, url) → np.ndarray[uint32]
    Memory-map shards with np.load(mmap_mode='r') for zero-copy reads

- class WeightedShardSampler:
    Infinite iterator: yields (dataset_name, np.ndarray[2048]) tuples
    Each rank gets disjoint shard subsets: shards[rank::world_size]
    Shuffles shard order each epoch with deterministic per-source seed
    Shuffles sequences within each shard before yielding
    Refreshes manifest every 100 yields, updates weights on proportion change
    Logs proportion changes at WARNING level

- class TeutonicoDataset(IterableDataset):
    Wraps WeightedShardSampler
    Returns dict(input_ids=LongTensor[2048], labels=LongTensor[2048], dataset=str)
    Worker-aware: accounts for num_workers in rank/world_size calculation

- build_train_loader(cfg, rank, world_size) → DataLoader
    num_workers=4, prefetch_factor=8, pin_memory=True, persistent_workers=True
    collate_fn keeps dataset names as list[str] alongside tensors

- class EvalDataset:
    build(): pre-download N sequences per source (deterministic seed=99999)
    get_batch(source_name) → LongTensor[N, 2048]
    Stores on CPU, moves to GPU only during eval

─────────────────────────────────────────
FILE 3: src/scheduler.py
─────────────────────────────────────────
WSD scheduler (primary) + cosine fallback.

- class WSDScheduler:
    Phases: warmup (linear) → stable (constant) → decay (cosine)
    get_lr(step) → float: pure function, no side effects
    step() → float: advances internal counter, sets all param groups
    trigger_decay(): enter decay phase now (call before final submission run)
    extend_stable(n): add N more stable steps (for checkpoint reuse)
    phase property: "warmup" | "stable" | "decay"
    Full state_dict / load_state_dict

- class CosineWarmupScheduler (backup):
    Standard cosine with warmup
    extend_total_steps(n) for flexible continued training

- build_scheduler(optimizer, cfg) → scheduler

─────────────────────────────────────────
FILE 4: src/moe_utils.py
─────────────────────────────────────────
MoE-specific monitoring and stability tools.

This is where you win or lose. Implement ALL of these:

A) Router load monitoring:
   def get_router_stats(model) → dict:
     For each MoE layer, compute:
       - expert_load: fraction of tokens routed to each expert (tensor[n_experts])
       - load_balance_loss: ideal load = 1/n_experts, measure deviation
       - max_expert_load: peak fraction (>0.1 is concerning, >0.2 is collapsing)
       - dead_expert_count: experts receiving <0.001 of tokens
     Return dict keyed by layer index

B) Auxiliary-loss-free load balancing (DeepSeek-V3 / Teutonic approach):
   def update_expert_biases(model, router_stats, bias_update_speed=0.001):
     For each MoE layer:
       For each expert e:
         if expert_load[e] > target_load: bias[e] -= bias_update_speed
         if expert_load[e] < target_load: bias[e] += bias_update_speed
     This nudges routing without penalizing the main loss.
     Check if the model already has expert_bias params before adding.
     target_load = 1.0 / n_experts

C) Z-loss for router logit scale stability:
   def compute_z_loss(router_logits, z_loss_coeff=1e-4) → Tensor:
     z_loss = mean(log(sum(exp(logits)))^2)  per DeepSeekV3 §3.3
     This prevents logit explosion without affecting routing decisions.
     Return z_loss_coeff * z_loss

D) Gradient norm monitoring per component:
   def get_grad_norms(model) → dict:
     Return separate grad norms for:
       "router": all router/gate params
       "experts": all expert FFN params  
       "attention": all attention params
       "embedding": embedding params
       "total": all params combined
     Use torch.nn.utils.clip_grad_norm_ logic but don't clip here

E) Expert utilization logger:
   def log_expert_utilization(router_stats, step, writer=None):
     Print a compact one-line summary per step:
       "step X | dead_experts=Y | max_load=Z | imbalance=W"
     Optionally write to TensorBoard SummaryWriter if provided

─────────────────────────────────────────
FILE 5: src/offline_eval.py
─────────────────────────────────────────
Exact replica of the validator's paired bootstrap evaluation.

- paired_bootstrap_lcb(diffs, n_bootstrap=10000, alpha=0.001, seed=0):
    MUST match the validator byte-for-byte:
      rng = np.random.default_rng(seed=0)
      resample n_bootstrap times with replacement
      LCB = 2*mu_hat - np.quantile(boot_means, 1-alpha)
    Returns (mu_hat, lcb)

- compute_per_sequence_ce(model, token_seqs, batch_size, device):
    IMPORTANT: do NOT use model(..., labels=...).loss
    That returns mean-reduced loss, losing per-sequence info.
    Instead:
      logits = model(input_ids=ids).logits          # (B, T, V)
      shift_logits = logits[:, :-1, :].float()      # (B, T-1, V)
      shift_labels = ids[:, 1:]                     # (B, T-1)
      token_ce = F.cross_entropy(shift_logits.view(-1,V),
                                 shift_labels.view(-1),
                                 reduction='none').view(B, -1)
      seq_ce = token_ce.mean(dim=1)                 # (B,) nats/token
    Use torch.inference_mode() + autocast(bfloat16)
    Use tqdm for progress on large batches

- class OfflineEvaluator:
    __init__(king_model, eval_dataset, manifest, device, eval_batch_size)
    
    evaluate(challenger_model) → EvalResult:
      For each source in manifest:
        n_source = round(manifest.eval_n * source.proportion)
        Score both models on n_source sequences from eval_dataset
        diffs[source] = king_ce - challenger_ce  (positive = challenger wins)
      Concatenate all diffs
      mu_hat, lcb = paired_bootstrap_lcb(all_diffs)
      Return EvalResult with full breakdown

- @dataclass EvalResult:
    mu_hat: float
    lcb: float
    accepted: bool          # lcb > delta_threshold
    delta: float
    n_sequences: int
    per_source: dict        # {source_name: {king, challenger, mu_hat, n}}
    
    __str__: pretty-print with ✅/❌ and per-source table
    is_safe_to_submit(safety_margin=0.05): lcb > delta + safety_margin

─────────────────────────────────────────
FILE 6: -m src.train
─────────────────────────────────────────
Main training entry point. All pieces assembled here.

Requirements:

A) Distributed setup:
   - torchrun compatible: read LOCAL_RANK, RANK, WORLD_SIZE from env
   - init_process_group("nccl")
   - torch.cuda.set_device(local_rank)
   - Set: allow_tf32=True for both matmul and cudnn

B) Training loop structure:
   for step in range(start_step, total_steps):
     
     # Gradient accumulation
     for acc in range(grad_accum_steps):
       batch = next(train_iter)
       with torch.amp.autocast("cuda", dtype=torch.bfloat16):
         outputs = model(input_ids=..., labels=..., use_cache=False)
         lm_loss = outputs.loss
         
         # Z-loss on router logits (if available in outputs)
         z_loss = compute_z_loss(outputs.router_logits) if hasattr(outputs, 'router_logits') else 0
         loss = lm_loss + z_loss
       
       (loss / grad_accum_steps).backward()
     
     # Clip + step
     grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
     optimizer.step()
     lr = scheduler.step()
     optimizer.zero_grad(set_to_none=True)
     
     # MoE load balancing (every 10 steps, not every step — too slow)
     if step % 10 == 0:
       router_stats = get_router_stats(model)
       update_expert_biases(model, router_stats)
     
     # Logging (rank 0 only)
     if rank == 0 and step % log_every == 0:
       log: step, lm_loss, z_loss, lr, phase, grad_norm,
            tokens/sec, dead_experts, max_expert_load
       per_dataset losses (track separate EMA per dataset name from batch["datasets"])

C) Per-dataset loss EMA tracking:
   class DatasetLossTracker:
     Maintains exponential moving average (alpha=0.98) per dataset name
     update(dataset_name, loss)
     summary() → dict[str, float]
   Log this every step — it's your early warning that one dataset is diverging

D) Checkpointing:
   save_checkpoint(model, optimizer, scheduler, step, output_dir, rank):
     FSDP full state dict with offload_to_cpu=True, rank0_only=True
     Save: model.pt, optimizer.pt, scheduler.pt, meta.json
     meta.json: step, lm_loss, lr, phase, timestamp, dataset_losses

   load_checkpoint(model, optimizer, scheduler, ckpt_dir):
     Load and return start_step

E) Submission model export (call at end or via --export flag):
   export_submission(model, king_dir, output_dir, rank):
     Full FSDP state dict → strip MTP keys → save_pretrained safetensors
     Copy contract files byte-identical from king_dir
     Remove manifest.json if present
     Write submission_info.json: step, lm_loss, timestamp

F) Main function:
   parse_args: --config, --export, --eval-only, --trigger-decay
   --export:        export submission model from latest checkpoint, then exit
   --eval-only:     run offline eval against king, print result, then exit
   --trigger-decay: call scheduler.trigger_decay(), continue training to end
   
   Call dist.barrier() at safe synchronization points
   Only rank 0 writes logs, checkpoints, eval results

─────────────────────────────────────────
FILE 7: scripts/data/download_king.py (already written — enhance it)
─────────────────────────────────────────
Add:
- Verify total shard count matches manifest before returning
- Print download speed (GB/s) per shard
- --verify-only flag: check checksums without re-downloading

─────────────────────────────────────────
FILE 8: configs/b300/train.yaml
─────────────────────────────────────────
Single source of truth for all hyperparameters.
All values must be documented with a comment explaining WHY this value.

─────────────────────────────────────────
FILE 9: requirements.txt + setup.sh
─────────────────────────────────────────
requirements.txt:
  torch>=2.4.0
  transformers>=4.47.0
  accelerate>=0.35.0
  safetensors>=0.4.0
  numpy>=1.26.0
  requests>=2.31.0
  tqdm
  pyyaml
  flash-attn  # optional

setup.sh:
  pip install -r requirements.txt
  pip install flash-attn --no-build-isolation || echo "flash-attn optional, skipping"
  huggingface-cli download dendriteholdings/teutonic-II-110B-genesis \
    --revision 8f971f030f5d801f812540007b396058f69cc08a \
    --local-dir ./genesis --include "*.py" "*.json" "*.jinja*"

─────────────────────────────────────────
GLOBAL CODE STANDARDS (apply to every file)
─────────────────────────────────────────
1. Type hints on every function signature. Return type always explicit.
2. Every class has a __repr__.
3. Every blocking I/O call has a timeout parameter.
4. All log.info/warning calls are rank-0 gated EXCEPT warnings about
   per-rank failures (download errors, shard skips).
5. No bare except clauses. Always except (SpecificError, OtherError) as e.
6. For FSDP: always check dist.get_rank() == 0 before file writes.
7. Use set_to_none=True in optimizer.zero_grad() (faster than zeroing).
8. Constants in SCREAMING_SNAKE_CASE at top of each file.
9. Do NOT use model.half() or model.float() — dtype is controlled by FSDP MixedPrecision.
10. All random seeds are (base_seed + rank) to ensure per-rank diversity.

─────────────────────────────────────────
WHAT WINNING LOOKS LIKE
─────────────────────────────────────────
The pipeline wins when:
1. It trains stably — lm_loss decreases monotonically, no NaN/inf.
2. It trains across ALL THREE datasets in correct proportions.
3. Router load stays balanced — no dead experts, no collapsed routing.
4. offline_eval LCB > 0.15 (delta=0.1 + safety=0.05) before we submit.
5. It reacts fast — within 10 minutes of a new king, we start training.

Failure modes to defend against:
- Exploding router logits → Z-loss
- Expert collapse (all tokens → few experts) → bias update + monitoring
- Catastrophic forgetting (loss spikes on one dataset) → per-dataset EMA tracking
- OOM on H200 → gradient checkpointing + set_to_none + limit_all_gathers
- Training from genesis instead of king → assert king_info.json exists in king_dir
- Submitting without stripping MTP → assert no "mtp" keys in state dict
- Submitting with wrong contract files → sha256 compare contract files vs genesis

Build all 9 files now. Start with src/moe_utils.py (most critical),
then -m src.train, then the rest.