Read CONTEXT.md. Below is a complete audit of every suboptimal design decision
in the current pipeline, followed by the full implementation of fixes.
This is authoritative — implement ALL sections in order.

══════════════════════════════════════════════════════════
PART 1: GAP ANALYSIS — WHAT IS WRONG AND WHY
══════════════════════════════════════════════════════════

GAP 1 — WRONG OPTIMIZER (biggest loss-reduction gap)
─────────────────────────────────────────────────────
Current: AdamW for all parameters.
Problem: AdamW treats every weight as an independent scalar. For 2D weight
         matrices (attention, expert FFNs), this ignores the matrix geometry
         and leads to suboptimal update directions.
Fix:     Muon (Momentum Orthogonalized by Newton-Schulz) for all 2D matrix
         weights. Validated on MoE models up to 30B+. Delivers ~2× better
         compute efficiency than AdamW. Moonlight 16B MoE was trained with
         Muon and matches Llama 3.1 8B with half the FLOPs.
Rules:   - Muon ONLY for 2D weight matrices (attention Q/K/V/O, expert FFN)
         - AdamW for 1D params: biases, norms, embeddings, LM head
         - Router/gate params: AdamW with lr_scale=0.1 (routers are fragile)
         - For MoE expert layers: Muon shines most because experts see
           only a fraction of tokens → large effective batch → Muon is stable

GAP 2 — NO EP ALL-TO-ALL COMMUNICATION OVERLAP
───────────────────────────────────────────────
Current: FSDP full-shard with no explicit EP dispatch.
Problem: Expert routing requires an all-to-all: dispatch tokens to expert GPUs,
         compute, combine results. This all-to-all is on the critical path of
         every forward pass. Without overlap, GPU sits idle during token routing.
         For 256 experts across 8 GPUs, each with top-8 routing, this is the
         #1 throughput bottleneck.
Fix:     Async EP dispatcher: issue all-to-all BEFORE starting expert FFN
         computation. While experts on GPU_i compute their tokens, GPU_i's
         tokens are already in-flight to their destination experts.
         Additionally: delay_wgrad_compute — defer expert weight gradient
         computation to overlap with the next layer's forward pass.
         Reference: Megatron-LM --overlap-moe-expert-parallel-comm
                    DeepSeek-V3 "nearly achieving full computation-communication
                    overlap" via DualPipe algorithm.

GAP 3 — NO SEQUENCE PACKING WITH DOCUMENT MASKING
──────────────────────────────────────────────────
Current: Fixed 2048-token chunks fed directly from .npy shards. Documents are
         concatenated without boundaries. Cross-document attention is computed
         (attending to tokens from a DIFFERENT document = noise).
Problem: (a) Cross-document attention pollutes gradients.
         (b) If the shard contains many short documents, context is wasted
             attending across document boundaries. Wastes compute.
Fix:     Flash Attention varlen API (flash_attn_varlen_func) with cu_seqlens.
         Track EOS token positions in each packed sequence.
         cu_seqlens marks document boundaries so attention is blocked:
           each token only attends to tokens in the SAME document.
         15-20% throughput improvement + cleaner gradient signal.
         This is what DeepSeek-V3 uses ("sequence packing improvement").

GAP 4 — torch.compile IS DISABLED
──────────────────────────────────
Current: compile: false in config.
Problem: torch.compile with inductor backend + max-autotune gives 10-23%
         throughput improvement on FSDP models (measured by Meta/PyTorch team).
         On B300 Blackwell with MXFP8, the gain is even larger because
         Triton can auto-tune Blackwell-specific tensor core tiling.
Fix:     Compile EACH DECODER LAYER individually (not the full model).
         Compiling each layer avoids graph breaks at FSDP shard boundaries.
         Use mode="reduce-overhead" for MoE layers (dynamic shapes from
         variable expert assignment). mode="max-autotune" for attention layers.
         Also compile the optimizer step: torch.compile(optimizer.step) for
         AdamW/Muon — 5-15% additional speedup on elementwise ops.

GAP 5 — INCORRECT LOAD BALANCING (sequence-level missing)
──────────────────────────────────────────────────────────
Current: Only auxiliary-loss-free bias update (step-level statistics).
Problem: DeepSeek-V3 uses TWO mechanisms together:
         (a) Bias-based load balancing (implemented ✓) — batch-level
         (b) Complementary SEQUENCE-WISE auxiliary loss — coefficient α=0.0001
             This is NOT the traditional heavy aux loss (α=0.01 which hurts).
             This tiny loss prevents extreme within-sequence imbalance where
             a single sequence sends all its tokens to 2 experts.
             bias update handles batch imbalance; seq-level loss handles
             single-sequence collapse.
Fix:     Add sequence-level auxiliary loss at α=0.0001:
           For each sequence: compute per-sequence expert fractions
           aux_loss_seq = sum_i (f_i * P_i) where f_i = token fraction to expert i
                                               P_i = mean routing prob to expert i
           Total aux_loss = α * mean(aux_loss_seq over sequences in batch)
         Add to main loss: total_loss = lm_loss + z_loss + aux_loss_seq

GAP 6 — BF16 OPTIMIZER STATES (50% memory waste)
────────────────────────────────────────────────
Current: AdamW stores exp_avg + exp_avg_sq in FP32 → 2 × model_size = 440 GB.
Problem: On B300 we have 2.3 TB, so not OOM. But: FP32 optimizer states slow
         down optimizer step (larger tensors, more memory bandwidth).
Fix:     Store exp_avg and exp_avg_sq in BF16. For Muon (no second moment),
         only momentum buffer needed → naturally smaller footprint.
         Precision-aware optimizer: Muon momentum in FP32, AdamW states in BF16.
         Implementation: pass dtype=torch.bfloat16 to optimizer state tensors.

GAP 7 — NO CHECKPOINT AVERAGING
────────────────────────────────
Current: Submit single checkpoint from end of training.
Problem: The WSD stable phase produces many valid checkpoints. Each has slightly
         different noise. Averaging them (model soup) is FREE improvement:
         no extra training, typically 0.02-0.05 nats better loss.
Fix:     Save checkpoints every 100 steps in the stable phase.
         Before decay: average last K=5 checkpoints (linear average of weights).
         Then run decay phase from the AVERAGED checkpoint.
         Call this: "pre-decay soup" — standard in WSD training recipes.

GAP 8 — NO QK NORM (training instability risk)
───────────────────────────────────────────────
Current: MiMo-V2 may have QK norm in its architecture already. MUST CHECK.
         If not present: attention logits can explode with high LR on B300
         especially with larger micro_batch (4 vs 1).
Fix:     Check model config for qk_norm or qk_layernorm field.
         If absent, add RMSNorm to Q and K inside attention:
           q = rms_norm_q(q)  # before rotary embedding
           k = rms_norm_k(k)
         This is a FORWARD-PASS-ONLY change; does not affect saved weights
         compatibility UNLESS the config says it's enabled (then it's fine).
         Check first: print(model.config) and look for qk_norm.

GAP 9 — FSDP WRAPS MoE AND ATTENTION LAYERS THE SAME WAY
──────────────────────────────────────────────────────────
Current: Single FSDP wrap policy for all decoder layers.
Problem: Attention and MoE expert layers have very different optimal parallelism.
         Expert layers: need EP (each GPU holds subset of expert weights = full
                        expert GEMMs = efficient). NO TP on experts.
         Attention layers: benefit from TP (splits Q/K/V heads across GPUs
                           = reduces activation memory per GPU).
         MoE Parallel Folding (Megatron 2025): decouple these configs.
         For our scale (8 GPUs, 2 nodes): with B300's 288 GB, memory is not
         the issue. But proper EP dispatch (all-to-all) IS important.
Fix:     Separate FSDP wrapping for expert modules vs attention modules.
         Expert FSDP unit = one expert (not the whole MoE layer).
         This enables FSDP to shard individual experts across GPUs correctly,
         giving us true EP where each GPU's expert receives dispatched tokens.

GAP 10 — DATA PIPELINE MISSING EOS INJECTION AND BOUNDARY TRACKING
───────────────────────────────────────────────────────────────────
Current: .npy shards are loaded as flat token arrays. No document boundary info.
Problem: Without document boundaries, we can't do intra-doc attention masking
         (Gap 3). Also: the WSD stable phase benefits from fresh data ordering;
         the current shuffling is shard-level only, not document-level within
         shard which reduces gradient diversity.
Fix:     Add _find_eos_positions(token_array, eos_id) that returns cu_seqlens
         (cumulative lengths array for flash_attn_varlen_func).
         The MiMo-V2 tokenizer's EOS token ID must be extracted from the config.
         If documents within a shard are already separated by EOS: use that.
         If not: treat each 2048-chunk as one document (cu_seqlens = [0, 2048]).

══════════════════════════════════════════════════════════
PART 2: NEW FILE — src/optimizer.py (MOST CRITICAL)
══════════════════════════════════════════════════════════

Implement the Distributed Muon optimizer, exactly as validated in the
Moonlight paper (arxiv 2502.16982) for MoE training.

"""
Muon — Momentum Orthogonalized by Newton-Schulz

For 2D weight matrices only. Falls back to AdamW for 1D params.
Validated: 2× compute efficiency over AdamW at MoE scale.
Reference: Jordan et al. 2024, Liu et al. (Moonlight) 2025.
"""

class NewtonSchulzOrthogonalizer:
    """
    5-step Newton-Schulz iteration to approximate matrix orthogonalization.
    Much cheaper than full SVD. Converges to UV^T of the momentum matrix.
    
    Coefficients from Moonlight paper:
      a, b, c = 3.4445, -4.7750, 2.0315  (degree-5 polynomial)
    
    def orthogonalize(G: Tensor) -> Tensor:
        # G shape: (m, n), m >= n assumed
        # If m < n: transpose, orthogonalize, transpose back
        # Normalize: X = G / G.norm() * sqrt(n)
        # 5 iterations of: X = a*X + b*(X @ X.T @ X) + c*(X @ X.T @ X @ X.T @ X)
        # These coefficients minimize distance from orthogonal group in Frobenius norm
        Require: G.dtype = float32 (cast internally if bf16)
    """

class DistributedMuon(torch.optim.Optimizer):
    """
    Distributed Muon for FSDP/ZeRO training.
    
    Key design (from Moonlight §3 "Distributed Muon"):
      Standard ZeRO-1 partitions optimizer states by parameter.
      Muon needs FULL gradient matrix to orthogonalize → can't use ZeRO-1 directly.
      Solution:
        1. Each GPU holds its FSDP shard of the gradient (partial matrix).
        2. All-gather the full gradient across EP group.
        3. Orthogonalize the FULL gradient on each GPU (identical result).
        4. Reduce-scatter the orthogonalized update (each GPU only applies its shard).
        5. This gives correct Muon update with same memory footprint as ZeRO-1.
    
    Args:
        params:          parameter groups (2D matrices only)
        lr:              learning rate (default 0.02 for pretraining, 0.005 for continued)
        momentum:        Nesterov momentum (default 0.95)
        nesterov:        use Nesterov momentum (default True)
        ns_steps:        Newton-Schulz steps (default 5)
        weight_decay:    weight decay (default 0.0 — Muon already has implicit regularization)
        ep_group:        process group for all-gather/reduce-scatter
    
    def step():
        for each param group:
            if grad is None: continue
            
            # 1. All-gather full gradient (Muon needs the full matrix)
            full_grad = all_gather(param.grad, group=ep_group)
            
            # 2. Nesterov momentum: buf = momentum * buf + full_grad
            #    grad_for_update = full_grad + momentum * buf  (Nesterov)
            
            # 3. Orthogonalize: update = NS_orthogonalize(grad_for_update)
            #    Scale: update *= max(1, m/n)^0.5  (per Moonlight eq.4)
            
            # 4. Reduce-scatter back to local shard
            local_update = reduce_scatter(update, group=ep_group)
            
            # 5. Apply update + weight decay
            param.data.mul_(1 - lr * weight_decay).add_(local_update, alpha=-lr)
    
    NOTE: Muon is applied ONLY to parameters where param.ndim == 2
          AND param is NOT a router, embedding, or norm weight.
          The caller is responsible for routing 1D and special params to AdamW.
"""

def build_optimizer(model, cfg: dict, pg: ProcessGroups):
    """
    Build the hybrid Muon + AdamW optimizer.
    
    Parameter routing:
      Group A — Muon:    2D matrix weights in attention + expert FFNs
      Group B — AdamW:   everything else (biases, norms, embeddings, LM head)
      Group C — AdamW:   router/gate params (lr * 0.1, weight_decay=0)
    
    For each parameter, assign by checking:
      is_2d = param.ndim == 2
      is_router = any(k in name for k in ['router', 'gate', 'expert_bias'])
      is_special = any(k in name for k in ['embed', 'norm', 'bias', 'lm_head'])
      
      if is_router:       → Group C (AdamW, lr_scale=0.1)
      elif is_special:    → Group B (AdamW, lr_scale=1.0)
      elif is_2d:         → Group A (Muon, lr_scale=1.0)
      else:               → Group B (AdamW, lr_scale=1.0)
    
    Log the parameter count in each group at rank 0.
    Return (muon_optimizer, adamw_optimizer) — both must be stepped together.
    
    Wrap both in a CombinedOptimizer that exposes .step(), .zero_grad(),
    .state_dict(), .load_state_dict() delegating to both.
    """

class CombinedOptimizer:
    """Wraps Muon + AdamW as one optimizer interface."""
    def __init__(self, muon: DistributedMuon, adamw: torch.optim.AdamW): ...
    def step(self): muon.step(); adamw.step()
    def zero_grad(self, set_to_none=True): ...
    def state_dict(self) -> dict: ...
    def load_state_dict(self, state: dict): ...
    def param_groups(self) -> list: ...  # union of both groups, for scheduler


══════════════════════════════════════════════════════════
PART 3: NEW FILE — src/ep_dispatch.py
══════════════════════════════════════════════════════════

"""
Expert parallelism dispatcher with async communication overlap.

The #1 throughput bottleneck in MoE training is the expert routing all-to-all.
This module implements async dispatch to hide it behind computation.

Architecture for 256 experts, 8 GPUs per node (EP=8):
  Each GPU holds 32 experts (256 / 8).
  For a batch of B*T tokens with top-8 routing:
    Each token dispatched to 8 experts across the 8 GPUs.
    ALL dispatches are intra-node (NVLink, 14.4 TB/s).

Dispatch flow:
  tokens (B, T, D) → router → scores (B, T, 256) → top-8 indices
  
  [ASYNC STEP 1] Begin all-to-all: scatter input tokens to correct GPU
  [COMPUTE]      While tokens are in-flight: continue with next layer's
                 forward pass or finish previous layer's backward.
  [ASYNC STEP 2] Wait for all-to-all completion
  [COMPUTE]      Run expert FFNs on received tokens (grouped GEMM)
  [ASYNC STEP 3] Begin all-to-all: scatter results back to original GPUs
  [COMPUTE]      Overlap with weight gradient computation (delay_wgrad)
  [ASYNC STEP 4] Wait for combine all-to-all

Implement:

class ExpertDispatcher:
    def __init__(
        self,
        n_experts: int,         # 256
        top_k: int,             # 8
        ep_group: ProcessGroup, # intra-node, 8 GPUs
        hidden_dim: int,
        node_limit: int = 1,    # DeepSeek-V3 node-limited routing
                                # each token goes to experts on at most 1 node
    )
    
    def dispatch_async(
        self,
        tokens: Tensor,          # (B*T, D)
        expert_indices: Tensor,  # (B*T, top_k)
        expert_scores: Tensor,   # (B*T, top_k)
    ) -> AsyncDispatchHandle:
        """
        Begin async token dispatch to expert GPUs.
        Returns a handle; call handle.wait() to get dispatched tokens.
        
        Implementation:
        1. Compute send_counts: how many tokens each GPU sends to each GPU
           send_counts[src][dst] = number of tokens from src destined for dst's experts
        2. dist.all_to_all_single (async) to dispatch tokens
        3. Return handle with: work_handle, metadata for combine step
        
        Node-limited routing (DeepSeek-V3 §3.3):
          Before top-k selection, group experts by node.
          For each token, select the top-M nodes (M=1 for single-node EP),
          then select top-k experts from those nodes only.
          This ensures tokens never cross node boundaries during all-to-all.
          Achieves near-full computation-communication overlap.
        """
    
    def combine_async(
        self,
        expert_outputs: Tensor,  # local expert computation results
        dispatch_handle: AsyncDispatchHandle,
    ) -> AsyncCombineHandle:
        """
        Begin async combine: scatter expert outputs back to their source GPUs.
        Call handle.wait() to get the final combined output.
        """
    
    def compute_experts(
        self,
        dispatched_tokens: Tensor,  # tokens received from dispatch
        expert_weights: list,        # local expert FFN weight matrices
        expert_indices: Tensor,      # which expert each token goes to
    ) -> Tensor:
        """
        Run expert FFNs on dispatched tokens using grouped GEMM.
        Each GPU's 32 experts process their received tokens in one batched op.
        
        Uses torch.ops.grouped_gemm (if available) or a loop over experts.
        Grouped GEMM handles variable-size expert inputs without padding.
        """

class AsyncDispatchHandle:
    work: dist.Work
    metadata: dict  # for combine step

class AsyncCombineHandle:
    work: dist.Work
    def wait(self) -> Tensor: ...


══════════════════════════════════════════════════════════
PART 4: NEW FILE — src/packing.py
══════════════════════════════════════════════════════════

"""
Sequence packing with intra-document attention masking.
Uses flash_attn_varlen_func for variable-length efficient attention.

Protocol note: The shards are 2048-token chunks. We treat the entire 2048
tokens as potentially multi-document. The MiMo-V2 tokenizer uses EOS token
(id = tokenizer.eos_token_id) as document separator.

Why this matters:
  Without doc masking: gradient signal leaks across document boundaries.
  Token at position 1500 attends to tokens at 0-1499 regardless of docs.
  With doc masking: attention is blocked at EOS boundaries.
  Result: cleaner gradients, 15-20% throughput improvement (no wasted attention).
"""

EOS_PLACEHOLDER = -1  # sentinel if EOS ID not known

def find_document_boundaries(
    token_ids: Tensor,  # (seq_len,) int64
    eos_id: int,
) -> Tensor:
    """
    Returns cu_seqlens: LongTensor of cumulative document lengths.
    Example: 3 documents of lengths [512, 1024, 512] → [0, 512, 1536, 2048]
    
    If no EOS found in the sequence: return [0, seq_len] (treat as 1 document).
    """

def pack_batch_with_masking(
    batch: dict,  # from DataLoader: input_ids (B, T), datasets list
    eos_id: int,
) -> dict:
    """
    Transform a standard batch into a varlen-packed batch for flash attention.
    
    Input:  batch["input_ids"] (B, T)
    Output: {
        "input_ids":    (B*T,)      # flattened
        "labels":       (B*T,)      # flattened, -100 at EOS positions
        "cu_seqlens":   (total_docs+1,)  # cumulative lengths across all docs in batch
        "max_seqlen":   int          # max single document length in batch
        "position_ids": (B*T,)       # reset to 0 at each document start
        "datasets":     list[str]    # unchanged
    }
    
    The cu_seqlens spanning the ENTIRE batch enables flash_attn_varlen_func
    to compute attention correctly across all B sequences at once.
    """

def flash_attn_varlen_forward(
    q: Tensor,            # (total_tokens, n_heads, head_dim)
    k: Tensor,
    v: Tensor,
    cu_seqlens: Tensor,   # (n_docs + 1,) cumulative doc lengths
    max_seqlen: int,
    causal: bool = True,
    softmax_scale: float = None,
) -> Tensor:
    """
    Wrapper around flash_attn_varlen_func.
    Falls back to standard attention if flash_attn not installed.
    """
    try:
        from flash_attn import flash_attn_varlen_func
        return flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens,
                                      max_seqlen, max_seqlen,
                                      causal=causal, softmax_scale=softmax_scale)
    except ImportError:
        # Fallback: standard attention with block-diagonal mask
        return _standard_varlen_attention_fallback(q, k, v, cu_seqlens, causal)


══════════════════════════════════════════════════════════
PART 5: NEW FILE — src/checkpoint_averaging.py
══════════════════════════════════════════════════════════

"""
Checkpoint averaging (model soup) for WSD stable-phase checkpoints.

Strategy: In the stable phase, save a checkpoint every K=100 steps.
          Before triggering decay: average the last N=5 checkpoints.
          Run decay from the averaged checkpoint.
          
          Why this works: each stable checkpoint has slightly different
          noise from the mini-batch sampling. Averaging them reduces
          variance without introducing bias (all have the same E[loss]).
          
          Expected gain: 0.02–0.05 nats better final loss.
          Cost: O(N × model_size) temporary memory to load and average.
"""

def average_checkpoints(
    checkpoint_paths: list[str],
    output_path: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Linearly average N checkpoints into one.
    All averaging done on CPU to avoid GPU memory pressure.
    Returns the averaged state dict.
    
    Algorithm:
        avg = load(ckpt[0])
        for ckpt in ckpt[1:]:
            state = load(ckpt)
            for key in avg:
                avg[key].add_(state[key])
        for key in avg:
            avg[key].div_(N)
    
    Handles safetensors shards: load each shard, average, save.
    """

def collect_stable_phase_checkpoints(
    output_dir: str,
    scheduler: WSDScheduler,
    max_n: int = 5,
    min_step_gap: int = 50,
) -> list[str]:
    """
    Find the last N checkpoints from the stable phase (not warmup, not decay).
    Returns paths sorted by step, most recent N only.
    """

def run_pre_decay_soup(
    model: FSDP,
    output_dir: str,
    scheduler: WSDScheduler,
    rank: int,
    n_checkpoints: int = 5,
) -> None:
    """
    Full pre-decay averaging workflow:
    1. collect_stable_phase_checkpoints()
    2. average_checkpoints() on rank 0
    3. Load averaged weights back into model (all ranks)
    4. Save averaged checkpoint
    5. scheduler.trigger_decay()
    Call this when offline eval shows LCB is close but not quite at threshold.
    """


══════════════════════════════════════════════════════════
PART 6: UPDATED — src/moe_utils.py
══════════════════════════════════════════════════════════

Replace the current moe_utils.py entirely. All of these must work:

A) Router load monitoring — FIXED implementation:
   The current implementation assumes we can access router stats after the
   forward pass. But router logits are intermediate tensors — they're gone
   unless we hook them.
   
   def register_router_hooks(model) -> RouterHookManager:
       """
       Register forward hooks on every router/gate layer.
       Captures router_logits and routing_weights tensors.
       
       For MiMo-V2, the routers are named like:
         model.model.layers[i].mlp.router (or .gate)
       
       Use model.named_modules() to find all modules where
         type(module).__name__ in ('MoEGate', 'TopKRouter', 'Expert...Router')
       
       Hook: record input (router logits before softmax) and output
             (selected expert indices and weights).
       
       Returns: RouterHookManager with .get_stats() and .clear()
       """
   
   class RouterHookManager:
       def get_stats(self) -> dict[int, RouterLayerStats]:
           """Return per-layer routing statistics since last .clear()."""
       def clear(self): ...
       def remove(self): ...

B) Auxiliary-loss-free bias update — FIXED:
   def update_expert_biases(
       model,
       hook_manager: RouterHookManager,
       bias_update_speed: float = 0.001,
       target_load: float = None,  # default: 1/n_experts
   ):
       """
       Update expert bias terms based on observed load imbalance.
       Implemented per DeepSeek-V3 §2.3.2:
         For each expert e in layer l:
           if load[e] > target: bias[e] -= bias_update_speed
           if load[e] < target: bias[e] += bias_update_speed
       
       IMPORTANT: expert_bias must be a registered buffer/parameter
       in the router module. If it doesn't exist, create it:
         router.register_buffer('expert_bias', torch.zeros(n_experts))
       The bias is added to routing scores BEFORE top-k selection
       but NOT included in the final routing weights (only affects selection).
       """

C) Sequence-level auxiliary loss — NEW:
   def compute_seq_auxiliary_loss(
       hook_manager: RouterHookManager,
       alpha: float = 0.0001,
   ) -> Tensor:
       """
       DeepSeek-V3 complementary sequence-wise auxiliary loss.
       Prevents extreme imbalance WITHIN a single sequence.
       
       For each layer l and each sequence s in batch:
         f_i^s = fraction of tokens in sequence s routed to expert i
         P_i^s = mean routing probability for expert i in sequence s
         seq_loss^s = sum_i(f_i^s * P_i^s)
       
       Loss = alpha * mean(seq_loss over all sequences and layers)
       
       Note: This is SEPARATE from the bias update.
       Bias update handles batch-level imbalance (non-differentiable).
       Seq aux loss handles within-sequence collapse (differentiable).
       Both are needed.
       
       Returns scalar tensor (requires_grad=True, flows through router).
       """

D) Z-loss — FIXED signature:
   def compute_z_loss(
       hook_manager: RouterHookManager,
       z_loss_coeff: float = 1e-4,
   ) -> Tensor:
       """
       Z-loss on router logits to prevent scale explosion.
       z_loss = mean(log(sum(exp(logits)))^2) per DeepSeek-V3 eq.3
       Use log-sum-exp trick for numerical stability:
         lse = torch.logsumexp(logits, dim=-1)  # (B*T,)
         z_loss = mean(lse ** 2)
       """

E) Gradient norms per component — FIXED:
   def get_grad_norms(model, pg: ProcessGroups) -> dict[str, float]:
       """
       Compute gradient norms for each component.
       IMPORTANT with FSDP: grads are sharded. Must use FSDP.clip_grad_norm_
       or manually all-reduce norms across EP group.
       
       Returns: {
           "router": float,
           "expert_ffn": float,
           "attention": float,
           "embed_lm_head": float,
           "total": float,
       }
       """

F) Per-module gradient clipping — NEW:
   def clip_grad_norms_per_module(
       model,
       pg: ProcessGroups,
       expert_max_norm: float = 1.0,
       attention_max_norm: float = 1.0,
       router_max_norm: float = 0.5,  # routers need more conservative clipping
   ) -> dict[str, float]:
       """
       Clip gradients separately per module type.
       Returns dict of actual norms before clipping (for logging).
       """


══════════════════════════════════════════════════════════
PART 7: UPDATED — -m src.train
══════════════════════════════════════════════════════════

Rewrite the training loop to incorporate all the above. Key changes:

A) Imports and setup:
   from optimizer import build_optimizer, CombinedOptimizer
   from ep_dispatch import ExpertDispatcher
   from packing import pack_batch_with_masking, find_document_boundaries
   from checkpoint_averaging import run_pre_decay_soup
   from moe_utils import (register_router_hooks, update_expert_biases,
                          compute_seq_auxiliary_loss, compute_z_loss,
                          clip_grad_norms_per_module, get_grad_norms)

B) Model compilation (after FSDP wrap):
   # Compile each decoder layer individually to avoid graph breaks
   def compile_decoder_layers(model, pg: ProcessGroups):
       """
       Apply torch.compile to each decoder layer module.
       Attention submodule: mode="max-autotune" (static shapes)
       MoE submodule:       mode="reduce-overhead" (dynamic shapes)
       Embeddings + LM head: no compile (rarely the bottleneck)
       
       Use use_orig_params=True in FSDP before compiling.
       Set torch._dynamo.config.cache_size_limit = 64 to avoid recompilation.
       """
   
   Also compile the optimizer step:
       optimizer.muon = torch.compile(optimizer.muon.step, fullgraph=False)
       optimizer.adamw = torch.compile(optimizer.adamw.step, fullgraph=False)

C) Hook registration:
   hook_manager = register_router_hooks(model)

D) Training loop:
   optimizer.zero_grad(set_to_none=True)
   
   for step in range(start_step, total_steps):
       accum_lm_loss = 0.0
       accum_z_loss = 0.0
       accum_aux_loss = 0.0
       
       for acc in range(grad_accum):
           batch = next(train_iter)
           
           # Document-boundary-aware packing
           batch = pack_batch_with_masking(batch, eos_id=tokenizer.eos_token_id)
           
           input_ids = batch["input_ids"].to(device)
           labels    = batch["labels"].to(device)
           cu_seqlens = batch["cu_seqlens"].to(device)
           
           with fp8_context(fp8_enabled):
               outputs = model(
                   input_ids=input_ids,
                   labels=labels,
                   use_cache=False,
                   # Pass cu_seqlens if model's attention accepts it
                   # MiMo-V2 may need custom attention_mask construction
               )
           
           lm_loss = outputs.loss
           
           # Z-loss (router logit scale control)
           z_loss = compute_z_loss(hook_manager, z_loss_coeff=cfg['moe']['z_loss_coeff'])
           
           # Sequence-level auxiliary loss (within-sequence imbalance)
           aux_loss = compute_seq_auxiliary_loss(
               hook_manager, alpha=cfg['moe']['seq_aux_alpha']
           )
           
           total_loss = lm_loss + z_loss + aux_loss
           (total_loss / grad_accum).backward()
           
           accum_lm_loss  += lm_loss.item()
           accum_z_loss   += z_loss.item() if isinstance(z_loss, Tensor) else z_loss
           accum_aux_loss += aux_loss.item() if isinstance(aux_loss, Tensor) else aux_loss
           
           hook_manager.clear()  # clear captured router stats after backward
       
       # Per-module gradient clipping
       grad_norms = clip_grad_norms_per_module(
           model, pg,
           expert_max_norm=1.0,
           attention_max_norm=1.0,
           router_max_norm=0.5,
       )
       
       # Optimizer step (Muon + AdamW combined)
       optimizer.step()
       lr = scheduler.step()
       optimizer.zero_grad(set_to_none=True)
       
       # MoE load balancing (every step, not every 10 — bias update is cheap)
       update_expert_biases(model, hook_manager,
                            bias_update_speed=cfg['moe']['bias_update_speed'])
       
       # SparseLoCo outer sync (if multi-node)
       if sparse_loco.should_sync(step):
           sparse_loco.outer_step(step)
       
       # Pre-decay checkpoint averaging
       if scheduler.phase == "stable" and step % 100 == 0:
           save_checkpoint(model, optimizer, scheduler, step, ...)
       
       # Trigger pre-decay soup before decay
       if args.trigger_decay and not decay_triggered:
           run_pre_decay_soup(model, output_dir, scheduler, rank)
           decay_triggered = True
       
       # Logging (rank 0 only)
       if rank == 0 and step % log_every == 0:
           log_dict = {
               "step": step,
               "lm_loss": accum_lm_loss / grad_accum,
               "z_loss": accum_z_loss / grad_accum,
               "aux_loss": accum_aux_loss / grad_accum,
               "total_loss": (accum_lm_loss + accum_z_loss + accum_aux_loss) / grad_accum,
               "lr": lr,
               "phase": scheduler.phase,
               "grad_norm_total": grad_norms["total"],
               "grad_norm_router": grad_norms["router"],
               "grad_norm_expert": grad_norms["expert_ffn"],
               **dataset_tracker.summary(),   # per-dataset EMA losses
               **router_stats_summary(hook_manager),  # dead experts, max load
           }
           # Write to: stdout (always), wandb/tensorboard (if enabled)
           log_training_step(log_dict)


══════════════════════════════════════════════════════════
PART 8: UPDATED — configs/b300/train.yaml
══════════════════════════════════════════════════════════

Add these sections (others unchanged):

optimizer:
  # PRIMARY: Muon for 2D matrix weights
  muon_lr: 0.005           # continued pretraining; was 0.02 for scratch
  muon_momentum: 0.95
  muon_nesterov: true
  muon_ns_steps: 5         # Newton-Schulz iterations (5 is standard)
  muon_weight_decay: 0.0   # Muon has implicit regularization via orthogonalization
  
  # SECONDARY: AdamW for 1D params + special params
  adamw_lr: 1.5e-5         # matches previous config
  adamw_lr_min: 1.5e-6
  adamw_betas: [0.9, 0.95]
  adamw_eps: 1.0e-8
  adamw_weight_decay: 0.1
  adamw_bf16_states: true  # store exp_avg/exp_avg_sq in BF16 (50% memory reduction)
  
  # ROUTER: separate AdamW with very low LR
  router_lr_scale: 0.1     # router LR = adamw_lr * 0.1
  
  gradient_clip:
    expert_ffn: 1.0
    attention: 1.0
    router: 0.5            # more conservative for router stability

moe:
  # Auxiliary-loss-free load balancing
  bias_update_speed: 0.001  # from DeepSeek-V3 §3.3 (first 14.3T tokens)
                             # set to 0.0 for last 500B tokens of training
  
  # Z-loss (router logit scale control)
  z_loss_coeff: 1.0e-4     # from DeepSeek-V3 §3.3
  
  # Sequence-level auxiliary loss (within-sequence imbalance)
  seq_aux_alpha: 1.0e-4    # TINY — prevents collapse without hurting performance
                            # traditional aux loss uses 0.01 which hurts; this is 100× smaller
  
  # Expert dispatch
  node_limit: 1            # tokens sent to at most 1 node's experts (keep intra-node)
  dispatch_overlap: true   # async all-to-all overlap with computation
  delay_wgrad: true        # defer expert wgrad to overlap with next layer forward

compilation:
  enabled: true            # was false — enable on B300
  decoder_mode: "reduce-overhead"   # for MoE layers (dynamic shapes)
  attention_mode: "max-autotune"    # for attention (static shapes)
  compile_optimizer: true           # compile optimizer .step() for elementwise speedup
  cache_size_limit: 64              # torch._dynamo cache to avoid recompilation

packing:
  enabled: true            # document-boundary-aware attention masking
  # EOS token ID extracted from tokenizer config at runtime

checkpoint_averaging:
  enabled: true
  n_checkpoints: 5         # average last 5 stable-phase checkpoints
  save_every_stable: 100   # save every 100 steps during stable phase
  min_step_gap: 50         # minimum steps between averaged checkpoints

training:
  # Updated for B300 (288 GB per GPU)
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  activation_checkpointing: false


══════════════════════════════════════════════════════════
PART 9: UPDATED — requirements.txt
══════════════════════════════════════════════════════════

# Add to existing requirements:
flash-attn>=2.6.0            # varlen attention for document masking
                             # install: pip install flash-attn --no-build-isolation
transformer-engine>=2.0.0    # MXFP8 for Blackwell B300
triton>=3.0.0               # Muon Newton-Schulz uses Triton kernels
wandb                        # optional: training metrics logging


══════════════════════════════════════════════════════════
PART 10: NEW FILE — src/qk_norm.py
══════════════════════════════════════════════════════════

"""
QK normalization check and injection for MiMo-V2.

Check first; inject only if absent.
"""

def check_and_inject_qk_norm(model, cfg) -> bool:
    """
    1. Check model.config for qk_norm / qk_layernorm / use_qk_norm field.
       If True: QK norm already in architecture. Return True (no action needed).
    
    2. If absent: for each attention module, register a forward hook that
       applies RMSNorm to Q and K before the attention computation.
       
       Caveat: this changes the forward pass behavior, which affects
       the eval loss compared to the king (the king was trained without it
       if genesis didn't have it). If the king doesn't have it, DON'T add it —
       you'd be comparing apples to oranges in the eval.
       
       Therefore: ONLY inject if model.config explicitly has qk_norm=False
       AND we explicitly set qk_norm=True in our training config.
       If genesis has it off, leave it off.
    
    Return: bool indicating whether QK norm is active.
    """


══════════════════════════════════════════════════════════
BUILD ORDER (implement in this exact sequence)
══════════════════════════════════════════════════════════

1. src/optimizer.py          — Muon + CombinedOptimizer (everything depends on this)
2. src/ep_dispatch.py        — async EP all-to-all dispatcher
3. src/packing.py            — sequence packing + document masking
4. src/moe_utils.py          — rewrite with router hooks + fixed stats
5. src/checkpoint_averaging.py — model soup
6. src/qk_norm.py            — QK norm check/inject
7. src/model.py              — update apply_fsdp + compile_decoder_layers
8. -m src.train              — full rewrite wiring all of the above
9. configs/b300/train.yaml — add all new config sections

══════════════════════════════════════════════════════════
SUMMARY: WHAT THIS GIVES YOU vs CURRENT DESIGN
══════════════════════════════════════════════════════════

| Optimization          | Gain                          | Evidence               |
|-----------------------|-------------------------------|------------------------|
| Muon optimizer        | ~2× compute efficiency        | Moonlight, Kimi 2025   |
| EP all-to-all overlap | 30–60% step time reduction    | DeepSeek-V3, Megatron  |
| Document masking      | 15–20% throughput + cleaner ∇ | DeepSeek-V3, SkyLadder |
| torch.compile         | 10–23% throughput             | Meta/PyTorch blog      |
| Checkpoint averaging  | 0.02–0.05 nats loss           | WSD/MiniCPM recipe     |
| Seq-level aux loss    | Prevents expert collapse      | DeepSeek-V3 §3.3       |
| BF16 optimizer states | 50% optimizer memory          | Megatron --precision   |
| Per-module grad clip  | Router stability              | Common practice        |
| Router hooks (real)   | Accurate load monitoring      | Required for bias upd. |

Combined expected effect: Faster loss reduction per step AND more steps per
second. The Muon gain alone should cut the number of training steps needed
to beat the king's LCB by ~40-50%.