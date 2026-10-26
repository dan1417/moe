Read CONTEXT.md. The following REPLACES or ADDS to the original build prompt.
Do not re-read the original prompt; use this as the authoritative delta.

## Hardware Reality Check
Old assumption: 8×H200 (96 GB each, 768 GB total)
New reality:    8×B300 per node (288 GB HBM3e each, 2.3 TB total per node)
                Optionally 2 nodes = 16×B300 = 4.6 TB total
                NVLink 5 intra-node:  14.4 TB/s switch bandwidth
                InfiniBand inter-node: 800 Gb/s per GPU (ConnectX-8)

Implication for 110B model (220 GB BF16):
  1 node:  220 GB / 8 GPUs = 27.5 GB/GPU just for params.
           Each GPU has 260+ GB free → micro_batch_size=4, no grad checkpointing needed.
  2 nodes: 220 GB / 16 GPUs = 13.75 GB/GPU → micro_batch_size=8, zero memory pressure.

Update train_config.yaml accordingly:
  micro_batch_size: 4           (was 1 for H200)
  gradient_accumulation_steps: 4  (was 8; same ~131K tokens/step, fewer accum steps)
  activation_checkpointing: false  (disabled; memory is abundant on B300)
  fp8_training: true            (new; enabled for B300 Blackwell)


## New File: src/process_groups.py
─────────────────────────────────────────────────────────────
The entire distributed topology lives here. Build it first.
All other files import from here.

CRITICAL DESIGN DECISION — read carefully:
  Expert all-to-all (token routing) is the hottest communication path in MoE.
  It must stay INTRA-NODE to use NVLink 5 (14.4 TB/s), NOT InfiniBand (800 Gb/s).
  Therefore:
    EP_GROUP  = 8 GPUs within one node (intra-node only)
    DP_GROUP  = 1 GPU per node, replicated across nodes (inter-node)
  
  For 1 node  (world_size=8):  EP=8, DP=1. No inter-node traffic.
  For 2 nodes (world_size=16): EP=8, DP=2. Inter-node only for DP sync.

Implement this file:

class ProcessGroups:
  """Central registry of all distributed process groups."""
  
  ep_group:    ProcessGroup  # expert parallelism — intra-node, 8 ranks
  dp_group:    ProcessGroup  # data parallelism  — inter-node, 1 per node
  ep_rank:     int           # rank within EP group (0-7)
  dp_rank:     int           # rank within DP group (0 or 1)
  ep_size:     int           # always 8 (GPUs per node)
  dp_size:     int           # 1 (single node) or 2 (two nodes)
  local_rank:  int           # GPU index on this node (0-7)
  global_rank: int           # absolute rank across all processes
  world_size:  int           # total GPU count (8 or 16)
  node_rank:   int           # which node am I on (0 or 1)
  is_rank0:    bool          # global rank == 0 only

def init_process_groups() -> ProcessGroups:
  """
  Call once after dist.init_process_group("nccl").
  Derives node_rank from global_rank and local_rank.
  Creates EP and DP groups for each configuration:

  1-node  (world_size=8):
    ep_group = [0,1,2,3,4,5,6,7]   (one group)
    dp_group = [rank] for each rank  (trivial, size-1)

  2-node  (world_size=16):
    ep_group = [0..7] and [8..15]   (one per node)
    dp_group = [i, i+8] for i in 0..7  (pairs across nodes)
    e.g. dp_group for GPU0_node0 = [0, 8]
         dp_group for GPU3_node0 = [3, 11]
  """

def get_ep_group() -> ProcessGroup:  # convenience accessor
def get_dp_group() -> ProcessGroup:  # convenience accessor
def log_topology(pg: ProcessGroups): # print the full topology at startup


## New File: src/sparse_loco.py
─────────────────────────────────────────────────────────────
SparseLoCo is a DiLoCo-variant that compresses inter-node gradient sync
to 1-3% of the data. Paper: arxiv 2508.15706 (NeurIPS 2025 Workshop).

How it works:
  Inner loop: H steps of normal local training on each node independently.
              No cross-node communication during inner loop.
  Outer step: Compute pseudo-gradient = theta_init - theta_current
              Apply Top-k sparsification (keep top 1% by magnitude)
              Quantize kept values to 2-bit
              All-reduce across DP_GROUP (inter-node)
              Apply outer optimizer step (SGD + Nesterov momentum)
              Reset theta_init = theta_current (start next inner loop)
  Error feedback: Residual (discarded 99%) is accumulated locally.
                  Added back to pseudo-gradient at next outer step.
                  This is what makes SparseLoCo outperform dense DiLoCo.

Key hyperparameters:
  H (inner_steps):        200  (sync every 200 gradient steps)
  k (sparsity):           0.01 (keep top 1% of pseudo-gradient values)
  quantization_bits:      2
  outer_lr:               0.7  (from paper; much higher than inner lr)
  outer_momentum:         0.9  (Nesterov)

Implement:

class SparseLoCo:
  def __init__(
    self,
    model: nn.Module,
    dp_group: ProcessGroup,
    inner_steps: int = 200,
    sparsity_k: float = 0.01,
    quant_bits: int = 2,
    outer_lr: float = 0.7,
    outer_momentum: float = 0.9,
    enabled: bool = True,        # False when dp_size==1 (single node)
  )
  
  def should_sync(self, step: int) -> bool:
    """True if this step is a SparseLoCo outer sync step."""
    return self.enabled and (step % self.inner_steps == 0) and step > 0
  
  def snapshot_params(self):
    """Save theta_init = current params. Call at training start and after each sync."""
    # Deep copy of all param data into self._theta_init dict
  
  def outer_step(self, step: int):
    """
    Full SparseLoCo outer sync. Call when should_sync() is True,
    AFTER inner optimizer step, BEFORE zeroing inner grad.
    
    Steps:
    1. Compute pseudo_grad[p] = self._theta_init[p] - p.data  for each param p
    2. Add error_feedback[p] to pseudo_grad[p]  (residual from last round)
    3. Top-k sparsify: mask = topk(|pseudo_grad|, k=sparsity_k * numel)
    4. Residual = pseudo_grad * (1 - mask)  → store in self._error_feedback
    5. Sparse_grad = pseudo_grad * mask
    6. Quantize sparse_grad to quant_bits (simple uniform quantization is fine)
    7. All-reduce sparse_grad across dp_group (average across DP workers)
    8. Dequantize
    9. Outer optimizer: apply Nesterov SGD step using all-reduced pseudo_grad
       p.data = p.data - outer_lr * (outer_momentum * velocity + pseudo_grad)
       velocity = outer_momentum * velocity + pseudo_grad
    10. self.snapshot_params()  (reset theta_init for next inner loop)
    
    Log: compression_ratio, sync_time_ms, pseudo_grad_norm
    """
  
  def _top_k_sparsify(self, tensor: Tensor, k: float) -> Tuple[Tensor, Tensor]:
    """Returns (sparse_values, mask). k is fraction to keep."""
  
  def _quantize_2bit(self, tensor: Tensor) -> Tuple[Tensor, float, float]:
    """Uniform 2-bit quantization. Returns (quantized, scale, offset)."""
  
  def _dequantize_2bit(self, quantized: Tensor, scale: float, offset: float) -> Tensor:

IMPORTANT: When dp_size == 1 (single node), SparseLoCo is automatically
disabled (enabled=False) and outer_step() is a no-op. The training loop
must NOT branch on this; SparseLoCo handles it internally.


## Updated File: src/model.py
─────────────────────────────────────────────────────────────
Add FP8 support and integrate process_groups.

A) FP8 with Transformer Engine on B300:
  B300 supports MXFP8 (Microscaling FP8) — more accurate than Hopper's FP8.
  ~30-40% throughput improvement over BF16 on Blackwell.
  
  Add:
  def enable_fp8_training(model, pg: ProcessGroups) -> bool:
    """
    Wraps eligible linear layers with transformer_engine FP8 modules.
    Returns True if FP8 was successfully enabled, False otherwise.
    
    Rules for B300/MXFP8:
      - Use te.recipe.MXFP8BlockScaling() recipe (NOT DelayedScaling)
        BlockScaling is more accurate and recommended for Blackwell.
      - EXCLUDE from FP8: router/gate layers (logit scale is sensitive)
      - EXCLUDE from FP8: embedding layers
      - EXCLUDE from FP8: final LM head (logits must be accurate)
      - INCLUDE in FP8:   all expert FFN linear layers
      - INCLUDE in FP8:   all attention projection layers (Q,K,V,O)
      
    Implementation:
      try:
        import transformer_engine.pytorch as te
        # Detect GPU capability
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:  # Hopper=9, Blackwell=10
          return False
        
        recipe = te.recipe.MXFP8BlockScaling()
        # Wrap eligible layers
        for name, module in model.named_modules():
          if isinstance(module, nn.Linear):
            if any(skip in name for skip in ['router','gate','embed','lm_head']):
              continue
            # Replace with te.Linear — preserve weights
            parent = get_parent_module(model, name)
            te_linear = te.Linear(module.in_features, module.out_features, ...)
            te_linear.weight.data = module.weight.data
            setattr(parent, leaf_name, te_linear)
        return True
      except ImportError:
        log.warning("transformer_engine not installed; falling back to BF16")
        return False
    """
  
  Add fp8_autocast context manager for the training loop:
  def fp8_context(enabled: bool):
    """Returns te.fp8_autocast() if enabled, else contextlib.nullcontext()"""

B) Update apply_fsdp() signature:
  def apply_fsdp(model, pg: ProcessGroups, cfg: dict) -> FSDP:
  
  The FSDP group must be the EP_GROUP (intra-node), NOT the global group.
  Each node independently shards its model across its 8 local GPUs.
  SparseLoCo handles cross-node synchronization separately.
  
  Pass process_group=pg.ep_group to FSDP constructor.
  This is the critical change: FSDP is now EP-local, SparseLoCo is DP-global.

C) Update get_parameter_groups() — no change needed.

D) Update print_model_summary() — add FP8 status and per-node memory.


## Updated File: -m src.train
─────────────────────────────────────────────────────────────
Wire everything together. Changes from original:

A) Distributed setup:
  def setup_distributed(cfg) -> ProcessGroups:
    dist.init_process_group("nccl")
    pg = init_process_groups()  # from process_groups.py
    log_topology(pg)            # print full topology on rank 0
    return pg

B) Model init:
  model = load_challenger_model(...)
  fp8_enabled = enable_fp8_training(model, pg)   # NEW
  model = apply_fsdp(model, pg, cfg)             # uses ep_group now
  sparse_loco = SparseLoCo(                       # NEW
    model, pg.dp_group,
    inner_steps=cfg['sparse_loco']['inner_steps'],
    sparsity_k=cfg['sparse_loco']['sparsity_k'],
    enabled=(pg.dp_size > 1),
  )
  sparse_loco.snapshot_params()                  # NEW

C) Training loop changes:
  with fp8_context(fp8_enabled):                  # NEW: wrap forward
    outputs = model(input_ids=..., use_cache=False)
    ...
  
  After optimizer.step():
    if sparse_loco.should_sync(step):              # NEW
      sparse_loco.outer_step(step)
  
  Logging: add fp8_enabled=True/False, dp_size, ep_size, compression_ratio

D) DataLoader per-node:
  Each node has its own independent data stream.
  Seed must be (base_seed + pg.dp_rank) to ensure nodes train on different data.
  This is already handled by WeightedShardSampler if rank uses global_rank.
  VERIFY: pass rank=pg.global_rank, world_size=pg.world_size to build_train_loader.

E) Checkpoint:
  On outer sync steps, also save a "sync checkpoint" with _outer suffix.
  Only pg.is_rank0 writes. All others dist.barrier() wait.
  
  save_checkpoint() must consolidate FSDP state from ep_group only,
  then pg.is_rank0 writes the file. DP workers (if dp_size=2) have identical
  weights after outer sync, so only one node needs to write.

F) Launch detection:
  Read TEUTONIC_NODES env var to set expectation:
    os.environ.get('TEUTONIC_NODES', '1')  → '1' or '2'
  Use it only for logging/assertions; actual topology comes from world_size.


## Updated File: configs/b300/train.yaml
─────────────────────────────────────────────────────────────
Add these sections to the existing config:

hardware:
  gpus_per_node: 8
  # Set automatically from WORLD_SIZE; used for assertions only
  num_nodes: 1          # 1 or 2 — update before each run

training:
  micro_batch_size: 4          # was 1 (H200). B300 has 3x memory.
  gradient_accumulation_steps: 4  # 8 GPUs × 4 seqs × 4 accum = 131K tok/step
  activation_checkpointing: false  # disabled; B300 has plenty of memory
  fp8_training: true           # use MXFP8 on Blackwell; falls back to BF16

sparse_loco:
  # SparseLoCo outer sync (only active when num_nodes > 1)
  enabled: true
  inner_steps: 200        # sync every 200 gradient steps (~26M tokens between syncs)
                          # Higher = less communication, more divergence risk.
                          # 200 is the sweet spot from the paper for 110B-scale.
  sparsity_k: 0.01        # keep top 1% of pseudo-gradient values
                          # 1% = 97-99% compression ratio. Validated in paper.
  quant_bits: 2           # 2-bit quantization on sparse values
  outer_lr: 0.7           # outer optimizer LR (much higher than inner LR)
                          # Paper: outer_lr = 0.7 works across all tested scales.
  outer_momentum: 0.9     # Nesterov momentum for outer optimizer

parallelism:
  # FSDP is now EP-local (intra-node only). SparseLoCo handles inter-node.
  fsdp: true
  fsdp_shard_params: true
  fsdp_cpu_offload: false   # not needed; B300 has abundant memory
  # Expert parallelism is EP=8 (always full node)
  # Data parallelism is DP=1 (1 node) or DP=2 (2 nodes)


## New File: scripts/train/launch.sh
─────────────────────────────────────────────────────────────
#!/bin/bash
# Auto-detects single vs multi-node from NODE_RANK and MASTER_ADDR env vars.
# Usage:
#   Single node:  bash scripts/train/launch.sh
#   Node 0 of 2: NODE_RANK=0 MASTER_ADDR=192.168.1.10 bash scripts/train/launch.sh
#   Node 1 of 2: NODE_RANK=1 MASTER_ADDR=192.168.1.10 bash scripts/train/launch.sh

NUM_NODES=${NUM_NODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=8
CONFIG=${CONFIG:-"configs/b300/train.yaml"}

# NCCL tuning for B300 + InfiniBand
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5       # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3        # RoCE v2
export NCCL_CROSS_NIC=1           # use all NICs
export NCCL_SOCKET_IFNAME=bond0   # adjust to your IB interface name
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1   # needed for NVLink efficiency
export TEUTONIC_NODES=$NUM_NODES

torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m src.train --config $CONFIG $@

# Example 2-node run:
# Node 0: NUM_NODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh
# Node 1: NUM_NODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 bash scripts/train/launch.sh


## Updated requirements.txt
─────────────────────────────────────────────────────────────
Add:
  transformer-engine>=2.0.0    # FP8/MXFP8 for Blackwell

Install order matters:
  pip install torch --index-url https://download.pytorch.org/whl/cu128
  pip install transformer-engine --extra-index-url https://pypi.nvidia.com
  pip install -r requirements.txt


## What does NOT change
─────────────────────────────────────────────────────────────
- src/data.py          — no change. Runs identically on 1 or 2 nodes.
- src/scheduler.py     — no change. WSD schedule is per-node, inner loop.
- src/offline_eval.py  — no change. Always runs on rank 0, single-node eval.
- src/moe_utils.py     — no change. router_stats are intra-node already.
- scripts/data/download_king.py  — no change.
- scripts/monitor.py        — no change.
- Protocol constraints      — no change. Submission is still a single model.


## Build order for Cursor
─────────────────────────────────────────────────────────────
1. src/process_groups.py   (topology foundation — everything depends on this)
2. src/sparse_loco.py      (new inter-node sync algorithm)
3. src/model.py            (update apply_fsdp + add enable_fp8_training)
4. -m src.train            (wire SparseLoCo + FP8 + process groups into loop)
5. configs/b300/train.yaml  (add sparse_loco + hardware sections)
6. scripts/train/launch.sh       (replace torchrun commands)
7. requirements.txt        (add transformer-engine)

Build all 7 now. Start with src/process_groups.py.