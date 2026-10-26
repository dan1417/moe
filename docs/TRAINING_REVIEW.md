# Teutonic SN3 CPT stack: read-only review (2026-09-24)

Repo reviewed: `/root/workspace/MoE` @ `0b3d274` (clean tree, nothing modified).
Reference used: live validator clone (`teutonic/evaluator/{engine,sources}.py`, `teutonic/evaluation/{policy,categories,configuration,early_stopping}.py`, `chain.toml`), live king code (`web/king_small`), live datasets manifest (config `e4a9f831…`, eval_n 30000, delta 0.003), live dashboard, the validator's transformers fork (`tf_fork`, v5.15.1 + commit 9e8a181 "Fix FA4 asymmetric value head dimensions"), and the live `sft-reasoning/manifest.json` (fetched with curl; no shards downloaded).

---

## 0. Bottom line

The production config (`configs/b300/train.yaml`) cannot produce a competitive challenger. There are four blocking problems:

1. **The Muon learning rate would wreck the king** (C1). `muon_lr: 0.005` combined with Moonlight RMS scaling (0.2·√max(m,n)) behaves like an AdamW learning rate of 0.005. That is roughly 10–25× a typical *pretraining* peak for this model size, and it applies to 99.1% of the parameters.
2. **There are no fp32 master weights** (C2). Both optimizers update bf16 parameters in place. At the configured AdamW LR, about 84% of updates round to zero (100% for norm weights). Muon weight decay is a no-op. Fixing C1 alone would turn training into a near no-op.
3. **The MoE training path is too slow to be useful** (C3). It uses 11,308 per-expert FSDP units, a Python loop over 256 experts, and 8 redundant `index_add_` calls per expert. The repo's own `docs/TEST.md` says 1B tokens would take "~7–8 months" on 8×. Kings currently change about once a day.
4. **The offline submission gate is out of date for the live protocol** (H1). It applies an absolute 0.05 margin against a 0.003 delta. Building its 30k-sequence eval set means downloading about 30,000 shards (~1.5 TB). Its numerics also differ from the validator's.

Several parts are right and verified:
- Manifest parsing and URL construction for the 5 new datasets.
- Sequence semantics: aligned 2048-token windows, no masking, positions 0..2047, HF label shift, per-token mean.
- The `noaux_tc` gate training patch.
- FA4 attention with native sinks and unpadded asymmetric V on B300.
- SWA window semantics.
- Export naming and dtype.

---

## 1. Live-protocol facts this review relies on (verified today)

- **Accept rule** (`policy.py:28-64`). A challenger is accepted when the 0.1% percentile of 10k bootstrap means of `king − challenger` is above `delta_threshold`, which is 0.003 (`chain.toml:64`, live manifest).
  - The 8 duels under delta 0.003 (from the dashboard):
    - 09-21 17:57 had mu +0.00331 and lcb 0.00297, and was **rejected**.
    - The last two winners had mu 0.00357 / lcb 0.00349 (per-sequence SD ≈ 0.0046) and mu 0.00323 / lcb 0.00310 (SD ≈ 0.0075).
    - Changes that were larger or uneven had SD ≈ 0.02–0.075.
  - A robust target is therefore mu ≥ 0.0035–0.004 with no per-source regression.
  - The rejected duel on 09-21 15:42 gained on math/code (+0.0035 each) and lost on txt360 (−0.0134), ending at mu −0.0019.
- **Scoring** (`engine.py:1305-1369`). `model.model(input_ids)` runs with no mask. Then `lm_head` is applied in 1024-position chunks and `F.cross_entropy` is computed on **bf16** logits with `reduction='none'`, summed over 2047 positions and divided by 2047. The model is loaded as bf16 with `strict=True` (`engine.py:1132-1140`) and uses a grouped MoE with an **fp32** combine (`engine.py:846-890`, specifically `:887-888`). TF32 is not enabled anywhere in the evaluator, so router GEMMs run in true fp32.
- **Sampling** (`configuration.py:166-234`, `categories.py:143-219`). Samples are category-stratified by shard count, with `shards_per_dataset=4` as a floor and at least one shard per category. Each source's quota is drawn from that handful of shards.
- **Dataset manifests** (sft-reasoning fetched). Top-level keys are `version: shard-pipeline-v1`, `tokenizer`, `dtype: uint32`, `seq_len: 2048` and `shards[]`. Each shard entry has `{key: "shards/sft-reasoning__<category>--<file>…npy", n_tokens, size_bytes, sha256}`. `size_bytes = 4·n_tokens + 128`, and `n_tokens % 2048 == 0` for all 10,242 shards. The two categories are `3efforts-pretrain` (6,558 shards) and `instruction-following` (3,684 shards). A full shard is 12,582,912 tokens (≈50.3 MB). Manifest sizes (HEAD): pretrain-behaviors 98.8 MB, math-reasoning 93.5 MB, txt360-v2 68.5 MB, code-reasoning 61.6 MB (modified 09-22 08:21, which matches config e4a9f831), sft-reasoning 2.9 MB. Total ≈ 325 MB.
- **King**. The index `total_size` is 220,561,730,944 B, which is exactly 2 B × 110,280,865,472 parameters. Every tensor is bf16, including `mlp.gate.e_score_correction_bias` and `self_attn.attention_sink_bias`. The checkpoint has no `mtp` keys. Full-attention layers are 0, 5, 11, 17, 23, 29, 34, 39 and 44 (9 layers); the other 36 are SWA. Upload cap is `MAX_MINER_UPLOAD_BYTES = 250e9` (`access/storage.py:25`). Extra files such as README are allowed (the king ships one). Contract files are hash-checked (`storage.py:177-190`).

---

## 2. Findings summary (ranked)

| ID | Sev | Area | One-line |
|----|-----|------|----------|
| C1 | CRITICAL | Optimizer | Muon peak LR 0.005 with RMS scaling gives per-element updates of 9.5e-4 per step (~5% of weight RMS) on 109.3B params. This destroys the king. |
| C2 | CRITICAL | Optimizer/precision | No fp32 master weights. bf16 params are updated in place, so AdamW at 1.5e-5 is ~84% rounded away (norms 100%), the router is ~98% frozen, and Muon WD is a no-op. With sane LRs, training becomes a near no-op. |
| C3 | CRITICAL (practical) | MoE/dist throughput | 11,308 per-expert FSDP units, a Python loop over 256 experts and 8 index_adds per expert. The repo's own numbers give tens of tok/s, i.e. months per 1B tokens. |
| H1 | HIGH | Protocol drift / submit gate | Submit threshold is LCB > delta + 0.05 = 0.053. The eval set needs ~30k shard downloads (~1.5 TB, 500 GB cache). Sampling is not stratified/clustered, and numerics differ (autocast bf16 router, TF32, eager, bf16 MoE combine). |
| H2 | HIGH | Numerics | MXFP8 (TE) forward/backward is used to optimize a bf16-evaluated model toward a 0.3% target. `gate_proj` stays bf16 while `up`/`down` are MXFP8. The repo measured no speedup. |
| H3 | HIGH | MoE | `e_score_correction_bias` gets ±0.001 each step. This drifts routing for no CE benefit, and bf16 storage distorts the step (0, 1× or 2× depending on magnitude). |
| H4 | HIGH | MoE objective | z-loss 1e-4 (≈1e-3 in magnitude, comparable to the target gain) and seq-aux 1e-4 add non-CE router gradients. They are currently masked by C2 rounding and will bite once C2 is fixed. |
| H5 | HIGH | Dist/memory | Every rank loads a full 220 GB CPU copy (1.76 TB of host RAM). Export needs another ~440 GB on rank 0. Impossible on this 503 GB box and risky on 2 TB B300 hosts. |
| H6 | HIGH | Data pipeline | One 50 MB shard download per 8 KB sequence (~32 TB for the planned run). 32 workers each re-fetch ~325 MB of manifests every 100 yields. Per-worker LRU caches let disk grow to 32×500 GB and crash. |
| H7 | HIGH | Data/resume | Resume requires the sequence pool, but the pool builder raises `ValueError` for sft-reasoning (1%) under the live mix. So a crash in production cannot be resumed. |
| H8 | HIGH | Validation gap | The "mini" rehearsal model is non-causal: SWA attends to the next 127 tokens and full attention is unmasked. It lacks v-scale, partial RoPE and real sinks, and never calls FA4. The genesis code path has never been trained, and TESTPLAN gate G (110B) was never run. |
| H9 | HIGH | Monitoring | No held-out per-source eval during training. `DatasetLossTracker` gives every dataset the same step-mean loss, so per-source regressions are invisible. |
| H10 | HIGH | Checkpointing | About 662 GB per DCP checkpoint every 100 stable steps (~26 TB over the run) with no retention policy. |
| M1–M11 | MEDIUM | Various | See §3.3: train/eval MoE and router numerics, broken latent packing path, loose transformers pin, launcher default config, 8× redundant eval compute, bf16 grad reduce and accumulate, 1-D shard assumption, in-place king overwrite, gaps in export verification, dynamic class cache, schedule. |
| L1–L8 | LOW | Various | See §3.4. |

---

## 3. Detailed findings

### 3.1 CRITICAL

**C1. The Muon learning rate would destroy the king.** *Area: optimizer/schedule (item 5)*
- Code:
  - `configs/b300/train.yaml:38-43` sets `muon_lr: 0.005`, `muon_rms_scale: 0.2` and `muon_weight_decay: 0.05`.
  - `src/optim/optimizer.py:573-577` stores `lr_ratio = 0.005 / 1.5e-5 = 333.3`.
  - `src/optim/scheduler.py:87-91` sets each group's LR to `lr × lr_ratio`, so Muon runs at 0.005 during the stable phase.
  - `optimizer.py:272-278` scales the Newton–Schulz output by `0.2·√max(m,n)`.
  - `optimizer.py:416-418` applies `local_param.add_(update, alpha=-lr)`.
- Which parameters: every 2D matrix except embeddings, lm_head and routers (`optimizer.py:602-617`). That covers attention qkv/o, the dense MLP, all 11,264 routed experts and the shared experts: 109.31B of 110.28B params.
- Numbers, reproduced on CPU with the repo's NS coefficients: the NS-plus-scale output RMS is 0.19, so at lr 0.005 the per-element update RMS is **9.5e-4 per step**. That is ≈4.7% of a weight with RMS 0.02, every step, for 4,000 stable steps. The Moonlight "matched RMS" design exists precisely so that the Muon LR can be read on the AdamW-LR scale. For comparison, pretraining peaks at this scale are about 2e-4–4e-4, and the AdamW CPT LR in this same config is 1.5e-5.
- Impact: loss diverges during warmup; any submission from it would lose on every source.
- Fix: for CPT, use AdamW; or, if keeping Muon, set `muon_lr` to about 1e-5–3e-5 with `rms_scale 0.2` and `muon_weight_decay 0`. This must be done together with C2.

**C2. No fp32 master weights: bf16 parameters are updated directly.** *Area: optimizer/numerics (items 2, 5)*
- Code:
  - The model is loaded in bf16 (`src/model/core.py:674-679`).
  - FSDP2 `MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=bf16)` (`core.py:1039-1043`) keeps the sharded params in bf16, with no master copy.
  - Muon writes bf16 (`optimizer.py:417-418, 205, 219, 269`).
  - `BF16AdamW` is a stock `torch.optim.AdamW` on bf16 params, so its states are already bf16 (`optimizer.py:528-551`).
- Measured on CPU:
  - An update of 1.5e-5 changes only **15.6%** of N(0, 0.02) bf16 weights, and each changed element jumps a full ULP.
  - It changes **0%** of norm weights (~1.0).
  - At the router LR of 1.5e-6, only 2% change.
  - Muon decoupled WD `(1 − 2.5e-4)` changes **0.00%** of elements.
- Impact: once C1 is fixed (Muon LR ~2e-5, giving per-element updates of about 4e-6), nearly every update is lost. The effective update becomes a biased "only tiny-magnitude weights move" rule. The AdamW-managed tensors (embeddings, lm_head, norms, router) are effectively frozen even now.
- Fix, with the memory cost on 8×B300 (full tables in §4):
  - (a) fp32 master params with FSDP2 `param_dtype=bf16, reduce_dtype=fp32` and bf16 Adam m/v: about 165 GB/GPU plus activations, so selective AC is needed.
  - (b) Kahan-compensated or stochastically-rounded bf16 updates (about 10 B/param).
  - (c) Freeze the routed experts and train the other ~3.98B params with full fp32 Adam: about 34.5 GB/GPU in total.

**C3. MoE training throughput is 1–2 orders of magnitude too low.** *Area: MoE/distributed (items 3, 7)*
- Code:
  - `apply_fsdp` shards each expert separately: 44 × 257 = 11,308 FSDP units, plus 45 layer units and the root (`core.py:1076-1092`).
  - The patched MoE forward (`core.py:447-474`) runs a Python loop over 256 experts per layer. Every expert is entered (forcing an all-gather) even with zero tokens, per-expert batches are padded to multiples of 32, and each expert does 8 `slot`-masked `index_add_` calls.
- Evidence from the repo itself (`docs/TEST.md:22-30, 105-114`): the depth ladder extrapolates to ≈5–8 tok/s on 1×B300; "ideal 8× → ~50 tok/s … ~7–8 months" per 1B tokens. The planned run is 1.31B tokens (5000 steps × 262K tokens).
- Competitive context: kings changed on 09-21 (×3), 09-22 and 09-23. A challenger trained against a king that has since been replaced must beat the newer king.
- Fix: stack expert weights per layer and use grouped GEMM (`torch._grouped_mm`, or the validator's `torch._foreach_mm` pattern for forward). Shard FSDP per decoder layer (on B300 NVLink a 4.85 GB per-layer all-gather is milliseconds), or implement proper EP. Accumulate the routed combine in fp32 as the validator does.

### 3.2 HIGH

**H1. The offline gate is out of date and does not reproduce the validator.** *Area: protocol drift, eval parity (items 1, 2, 8)*
- **The margin cannot be met.** `configs/b300/train.yaml:122` sets `safety_margin: 0.05`, `src/eval/offline.py:24` defaults it to 0.05, and `is_safe_to_submit` (`offline.py:257-258`) and `cli.py:1091, 1179-1186` require LCB > 0.003 + 0.05 = **0.053**. Every accepted challenger under delta 0.003 had lcb between 0.0031 and 0.0109. The docs still say "LCB > 0.15" and "2000 sequences" (`docs/TRAINING_CONTROL.md:41, 246-254`).
- **The eval set cannot be built.** `EvalDataset._collect_sequences` (`src/data/dataset.py:1027-1066`) picks a random shard with replacement for *each* sequence. At eval_n 30,000 (live) that is about 30,000 distinct ~50 MB shards, roughly 1.5 TB, against `cache_max_gb: 500` and this box's 1.5 TB disk. Meanwhile ranks ≥1 wait at a barrier with a 30–60 min NCCL timeout (`cli.py:1100-1110`). The validator, by contrast, uses a handful of shards per dataset.
- **Sampling structure differs.** The offline sample is not category-stratified and not clustered per shard, whereas each validator draw is clustered (a few shards per category). Uniform-over-shards matches the validator's category proportions in expectation but misses between-draw cluster variance. It also uses one fixed seed (99999), so repeated checkpoint selection on it produces a winner's-curse bias.
- **Numerics differ from the validator:**
  - `torch.autocast(bf16)` (`offline.py:121-122`) casts the gate's `F.linear(hidden.float(), W.float())` to **bf16**. I verified that autocast turns an fp32 linear into bf16 output. Offline routing therefore differs from the validator's fp32 routing.
  - `torch.backends.cuda.matmul.allow_tf32 = True` (`src/train/cli.py:187`) affects every mode.
  - The eval model is always eager (`cli.py:893`), while production uses FA4.
  - The MoE combine is the bf16 patched path (M1).
  - CE is computed on fp32 logits, whereas the validator uses bf16 logits.
  - The LCB is the basic bootstrap `2μ − q_{1−α}` (`offline.py:43`) rather than the validator's percentile `q_α`.
  - Paired differences cancel much of this, but these are the three different routings (fp32 validator, TF32 training, bf16 offline).
- Fix: import and use the validator's own functions:
  - `pretokenized_dataset_request` with the bundled category rules and random `block_hash`/`hotkey` to build draws.
  - `sample_pretokenized_sequences`.
  - `compute_per_sequence_loss`, `enable_grouped_mimo_moe` and `paired_bootstrap_verdict`.
  - No autocast and no TF32.
  - Decision rule: lcb > delta + ~0.0005 on at least 2–3 independent draws, and no source regressing beyond noise.

**H2. MXFP8 training for a model evaluated in bf16.** *Area: numerics (item 2)*
- `train.yaml:15-16` sets `low_precision: mxfp8` (strict). `core.py:884-981` swaps `nn.Linear` for `te.Linear` using `MXFP8BlockScaling`.
- `LP_SKIP_KEYWORDS` contains `"gate"` (`core.py:36, 848-850`), so every expert `gate_proj` (and the layer-0 `gate_proj`) stays bf16 while `up_proj`, `down_proj`, `qkv_proj` and `o_proj` run in MXFP8.
- E4M3 blocks carry about 2–6% relative per-element quantization noise on every GEMM input, far above the 0.3% loss improvement being sought. The optimizer partly fits the quantization noise, which does not carry over to bf16 scoring.
- The repo's own measurement: "MXFP8 is not faster than BF16" (`docs/TEST.md:90`).
- Fix: `low_precision: none`.

**H3. The e_score_correction_bias is updated every step.** *Area: MoE (item 3)*
- `train.yaml:62` sets `bias_update_speed: 0.001`. `src/moe/utils.py:339-443` applies ±γ per expert per step toward a uniform load (`add_` at `:431/:436`, clamp ±10).
- The validator has no capacity limit, so balance does not affect CE. These updates only move routing away from what the experts were trained with.
- The bias is stored in bf16. Measured step sizes for +0.001: |b| < 0.25 → about 0.00098; 0.25 ≤ |b| < 0.5 → 0.00195 (doubled); |b| ≥ 0.5 → 0 (lost). The rule is therefore magnitude-dependent.
- Fix: `bias_update_speed: 0`. This also keeps the 44 bias tensors byte-identical to the king.

**H4. Non-CE router objectives.** *Area: MoE (item 3)*
- `train.yaml:63-64` sets `z_loss_coeff: 1e-4` and `seq_aux_alpha: 1e-4`, implemented in `moe/utils.py:473-529`.
- For a sigmoid router over 256 experts, logsumexp is about 3–5, so the z-loss term is about 1e-3. That is the same order as the whole target improvement, and it pushes top logits down, changing the normalized top-k weights. Seq-aux is about α × Σ f·P ≈ 1e-4.
- Neither is part of the eval. Both are currently neutralized by C2 (router updates round away) and will act once C2 is fixed.
- Fix: set both to 0 for CPT.

**H5. Host-RAM blow-up at load and export.** *Area: distributed/memory (item 7)*
- `load_challenger_model_distributed` makes every rank call `from_pretrained` for the full model on CPU (`core.py:689-707`, docstring says so). That is 8 × 220.6 GB ≈ **1.76 TB** of host RAM.
- `run_export` repeats this on all ranks (`cli.py:1221`), then rank 0 holds the gathered full state (220 GB) plus a second full `reference` model (`src/submit/export.py:201-207`), about 440 GB.
- This box has 503 GB. DGX-class B300 hosts typically have about 2 TB, which leaves no margin.
- The meta-device approach was abandoned after producing "random logits". The likely cause is that non-persistent rotary `inv_freq` buffers are not recomputed after materialization.
- Fix: load once on rank 0 and use `set_model_state_dict(..., broadcast_from_rank0=True)`, or use meta init plus per-rank sharded safetensors reads, then recompute the rotary buffers. In export, stream the gathered bf16 state straight to safetensors with the king's key order; do not build a second model.

**H6. Data pipeline is download-amplified, refetches manifests and grows disk without bound.** *Area: data (items 1, 8)*
- **One shard per sequence.** `WeightedShardSampler.__next__` (`dataset.py:596-610`) picks a new shard for every yielded sequence, `load_shard`s it (a ≈50 MB download), and takes one 2048-token window. That is about 6,144× amplification; the 640k sequences planned need about 32 TB of downloads.
- **Manifest refetch per worker.** Each DataLoader worker builds its own sampler (`dataset.py:686-693`). Its constructor and every 100th yield call `manifest.fetch(force=True)` (`:597-598, 612-613`), each re-downloading and parsing about 325 MB of JSON (≈1.15M `ShardInfo` objects). With 8 ranks × 4 workers that is roughly one full refetch per training step node-wide, plus about 1–2 GB of RAM per worker.
- **Unbounded disk.** The `ShardCache` is forked into each worker with its own `_lru`, and `_current_size_bytes` only counts that worker's own entries (`dataset.py:541-560`). Global disk use can reach 32 × 500 GB before any eviction. `_download_shard` then raises `OSError` on low disk (`:515-521`), which kills the DataLoader.
- Fix: build the training set up front, taking many sequences per shard (e.g., at least 256). Fetch manifests once on rank 0 and share shard lists. Use one global cache with eviction, or none.

**H7. Resume is impossible in production, and the pool builder fails on the live mix.** *Area: data/checkpoint (items 6, 8)*
- `train()` rejects `--resume` unless `data.sequence_pool.enabled` (`cli.py:1283-1287`). The production config disables the pool (`train.yaml:139-140`), so any crash means starting over.
- Enabling the pool fails. `prepare_train_sequence_pool.py:184-192` computes, for sft-reasoning, `n = 0.01 × 640,000 = 6,400`, which gives `shards_high = min(2000, 6400 // 100) = 64 < shards_low = 500`, and raises `ValueError("Impossible selection ranges")`. I replicated this; the other four sources pass.
- This is protocol drift: the old 3-source mix had no 1% source.
- Fix: scale the shard ranges by source size, e.g. `shards = clamp(ceil(n / seqs_per_shard), 1, …)`.

**H8. No test has validated training on the real model code.** *Area: validation (items 3, 4, 8)*
- The rehearsals in `docs/TEST.md` ran on `pipeline_models/mimo_v2_mini/modeling_mimo_v2.py`, which is a rewrite, not genesis code (`prepare_pipeline_model.py:146-147` copies it).
- That rewrite:
  - Builds an SWA mask that lets query i see keys `[i, i+127]`, i.e. **the future** (`:204-210`). I verified this: query 10 with window 4 sees keys `[10, 11, 12, 13]`.
  - Has **no causal mask** in full-attention layers unless a 2-D `attention_mask` is passed (`:268`), and training never passes one.
  - Applies RoPE to the full 192 dims with theta 1e4 for every layer (`:190`), whereas genesis uses 64 dims, theta 1e7 for full attention and 1e4 for SWA.
  - Has no `attention_value_scale`.
  - Implements the "sink" by adding to key 0's logit (`:213`) instead of adding a softmax-denominator term.
  - Never dispatches through `ALL_ATTENTION_FUNCTIONS`, so the "FA4" gates (C2, C3) never ran FA4.
- The real 110B gate G was "NOT RUN" (`docs/TEST.md:47`).
- Plumbing results (FSDP, DCP, export) are still informative. Loss, learning and FA4 correctness are not validated.

**H9. Per-source regressions cannot be seen during training.** *Area: item 8*
- `DatasetLossTracker.update(batch_datasets, mean_lm_loss)` (`cli.py:93-96, 1528`) assigns the *step-mean* loss to every dataset present in the step, so all per-dataset EMAs are the same number.
- There is no held-out evaluation during training.
- Duels are lost to single-source regressions (e.g., txt360 −0.0134 on 09-21 15:42).
- Fix: compute per-sequence loss (`reduction='none'`) and aggregate per source. Every N steps, run a fixed held-out set (~256 sequences per source) through the validator's scorer.

**H10. Checkpoint volume.** *Area: checkpointing (item 6)*
- `save_checkpoint` runs every 100 stable steps (`cli.py:1560-1586`) and saves bf16 params (220.6 GB), fp32 Muon momentum (437.2 GB) and bf16 AdamW states (3.9 GB), about **662 GB** each. The stable phase alone has 40 saves, about 26 TB.
- No pruning code exists (checked by grep).
- Fix: keep the last N full checkpoints, and save model-only bf16 snapshots for the soup.

### 3.3 MEDIUM

- **M1. MoE and router numerics differ between training and eval.**
  - The patched MoE casts top-k weights to bf16 and accumulates the 8-expert combine in a bf16 `out` (`core.py:452-453, 466-470`). The validator and the reference code use fp32 weights and an fp32 accumulator (`engine.py:882-890`; `modeling_mimo_v2.py:201, 212`).
  - Router GEMMs run in TF32 during training (`cli.py:187`) and bf16 during offline eval (autocast); the validator uses fp32. This flips near-tie top-8 selections.
  - The gradient is therefore taken for a slightly different function.
  - Fix: fp32 combine and TF32 off (at least for the gate).
- **M2. The latent packing path is wrong (currently disabled).**
  - `pack_batch_with_masking` (`src/data/packing.py:91-142`) resets `position_ids` per document and masks EOS labels.
  - `_forward_model` (`cli.py:370-378`) forwards `position_ids` but never a document mask, so attention still crosses documents while RoPE positions restart. `cu_seqlens` is moved to the device but never used (`cli.py:342-343`).
  - It also uses `tokenizer.eos_token_id` = `<|im_end|>`, which is an end-of-*turn* token in chat and reasoning data (`cli.py:1295-1296`).
  - The validator scores with absolute positions 0..2047 across document boundaries and includes EOS targets.
  - Fix: keep `packing.enabled: false` or delete the path.
- **M3. transformers is pinned too loosely.** `requirements.txt` allows `transformers>=4.47.0`. Older 4.x releases passed `window_size=(w, w)` to FlashAttention, which gives a 129-token SWA window. The fork used by the validator passes `(w−1, w−1)` (`tf_fork/.../modeling_flash_attention_utils.py:650-655`), matching eager `kv > q − 128` (`masking_utils.py:98-99`). Pin transformers 5.15.1 plus the FA4 asymmetric-V fix, as the validator does. The eval venv present on this box already has 5.15.1.
- **M4. Wrong default config in the launcher.** `scripts/train/launch.sh:26` defaults `CONFIG` to `configs/pro6000/train.yaml`, so a bare `launch.sh` on B300 runs the wrong profile.
- **M5. Offline eval wastes 7/8 of its compute.** `compute_per_sequence_ce_distributed` broadcasts the *same* batch to all 8 ranks and all of them run the full forward (`offline.py:114-131`). Fix: shard sequences across ranks and gather the losses.
- **M6. Gradient precision.** `reduce_dtype=bf16` (`core.py:1041`) and gradient accumulation into bf16 sharded grads over 4 micro-batches (no `set_requires_gradient_sync(False)`) cost accumulation precision and 4× the reduce-scatter traffic.
- **M7. Shard layout is assumed to be 1-D.** `EvalDataset` slices `array[start:start+2048]` (`dataset.py:1053-1056`) and the pool builder does `seq_mem[i] = array[start:end]` (`prepare_train_sequence_pool.py:263-265`). For a 2-D `(N, 2048)` `.npy` both slice *rows*. The training sampler (`dataset.py:646-658`) and the validator handle both layouts. The live shard layout was not checked (I did not download shards). Add an explicit `ndim`/shape assertion.
- **M8. The king directory is overwritten in place.** `scripts/data/monitor.py --auto-download` rewrites `./king` in place (`monitor.py:198-205`), so an eval or export running at the same time can read a mix of old and new files. Download to a fresh directory and swap it atomically.
- **M9. Export verification has gaps.**
  - `verify_contract_files` compares only the contract files *present in `king_dir`* against `./genesis` (`export.py:82-115`). If `tokenizer.json` is missing from the king directory, the check passes vacuously and the submission is then rejected. The expected hashes are in `chain.toml`.
  - Export does not check the key set, shapes or dtypes against the king index.
  - It does not check that the weights actually differ from the king. A re-sharded no-op would get a different digest, get evaluated, and burn the hotkey's only submission.
  - It does not reload and re-score through the validator's loader.
- **M10. Dynamic class caching can skip patches.** `_MIMO_MOE_CLASS` and `_MIMO_GATE_CLASS` are cached module-globally (`core.py:346-362`). A second model loaded from a *different* directory gets a distinct remote-code class and is left unpatched: it keeps the original MoE with fp32 combine. In the non-swap eval path (`cli.py:748, 809`) the king and challenger would then be scored with different numerics. The weight-swap path is fine.
- **M11. Schedule and averaging.** WSD warmup is 200 steps and decay goes to 10% (`train.yaml:28-34`). `adamw_weight_decay` and `adamw_lr_min` are read but ignored (`optimizer.py:686`). Soup averaging exists but only via `--trigger-decay` (`cli.py:1588-1598`), and there is no EMA. For small-margin CPT, averaging late checkpoints (LAWA/EMA) is among the cheapest reliable gains. Keep it, but save model-only snapshots.

### 3.4 LOW

- **L1.** The offline bootstrap uses the basic interval instead of the validator's percentile interval (`offline.py:43`).
- **L2.** Offline CE uses fp32 logits while the validator uses bf16 CE. The rounding is roughly unbiased, so this is minor.
- **L3.** FA4 prewarm compiles 64 q-heads and a `(128, 0)` window (`core.py:592-596`), while the model uses 48 heads and runs at `(127, 127)`. This only wastes compile time.
- **L4.** On SM120 (RTX 5090 / Pro 6000), FA4 does not accept sinks, so the repo strips the sink and rescales the output by `sigmoid(lse − sink)` (`core.py:143-232`). Correct gradients then require FA4's backward to propagate `dlse`. Upstream `main` does (`FlashAttnFunc.backward(ctx, dout, dlse)`); the pinned 4.0.0b28 was not verified. Check the gradient against eager before training SWA layers on SM120.
- **L5. Documentation drift.**
  - `.cursor/rules/CONTEXT.md` still describes 3 datasets, 2000 sequences, delta 0.1, 48 layers and a "~128K" vocab, all outdated.
  - `docs/TRAINING_CONTROL.md` still says "LCB > 0.15" and "all 3 datasets".
  - The D real-data test used dclm/finewebedu/automathtext (`docs/TEST.md:98`).
- **L6.** `merge.py:120-133` classifies the layer-0 dense `mlp.gate_proj` as `ROUTER` because of the "gate" keyword, and `post_attention_layernorm` as `ATTENTION`.
- **L7.** The king CE cache key uses the `TEUTONIC_ATTN` tag (`cli.py:494-500`) while the eval model is always eager (`cli.py:893`).
- **L8.** Step 0 runs at LR 0 (`get_lr(0) = 0`), which wastes one step.

### 3.5 Verified correct (no action needed)

- **Manifest parsing and URLs.**
  - `Manifest.fetch` reads `sources[].manifest_url/proportion/total_shards` and each shard's `key/sha256/size_bytes/n_tokens` (`dataset.py:196-252`).
  - The URL is `base_url + "/" + key`, e.g. `…/sft-reasoning/shards/sft-reasoning__…npy`. That equals the validator's `_shard_url`/`urljoin` result.
  - `eval_n` and `delta_threshold` are read live (30000 and 0.003); the 2000/0.1 defaults are used only if the fields are absent.
- **Mixture.** Sources are sampled with the manifest proportions, and shards uniformly within a source. Category shares are therefore proportional to shard count, which is the validator's apportionment rule in expectation.
- **Sequence semantics.**
  - Aligned 2048-token windows (`index × 2048`) from uint32 shards (`dataset.py:643-658`), the same unit the validator uses (`engine.py:1220-1267`).
  - No BOS or padding.
  - With packing off: no attention mask, `position_ids = 0..2047`.
  - HF `ForCausalLMLoss` shift gives 2047 targets per sequence and a token mean, which equals the validator's per-sequence sum/2047 averaged over sequences (all sequences have equal length). Loss is computed on fp32 logits.
- **Gate patch** (`core.py:365-399`).
  - Toggling `training = False` around the locked forward reproduces the validator's `noaux_tc` routing exactly: fp32 logits, the bias used only for selection, and `norm_topk_prob`.
  - Gradients reach `gate.weight` through the gathered sigmoid scores.
  - `e_score_correction_bias` is removed from autograd (`moe/utils.py:324-336`) and from the optimizer (`optimizer.py:629-637`).
  - The shared expert is always added (`core.py:471-473`).
- **Attention.**
  - Training runs the genesis attention code: v-scale 0.707, 64-dim partial RoPE, dual RoPE thetas, 4/8 KV heads.
  - On B300 (SM103), FA4 gets `learnable_sink` natively. `_patch_fa4_asymmetric_value_heads` (`core.py:235-343`) mirrors the validator fork's commit 9e8a181 (no V padding).
  - SWA semantics match: FA4 `(127, 127)` causal equals eager `kv > q − 128`, i.e. 128 keys.
  - The sink parameter is frozen (`requires_grad=False` in the locked code) and round-trips unchanged.
- **Activation checkpointing** (selective, `core.py:1123-1144`) uses non-reentrant `checkpoint_wrapper`. Router hooks fire again during recompute, but the duplicate records are cleared and the doubled counts do not change the sign-based bias update. TE autocast is kept through backward (`cli.py:1450-1481`).
- **DCP save/resume** (`src/checkpoint/dcp.py`) is sharded, committed atomically with a marker, checks topology, and restores RNG, scheduler and optimizer state.
- **Export key names and dtypes.**
  - Export loads the checkpoint into a *plain* HF model, so the extra `_extra_state` keys from `te.Linear` are ignored by the DCP load.
  - It saves bf16 with the original FQNs. No conversion mapping is registered for `model_type: mimo_v2` or `MiMoV2*` in the validator's transformers fork (`conversion_mapping.py`; only `mimo_v2_flash` has one).
  - `strict_except_mtp` load, MTP strip, `manifest.json` removal, and contract files copied *after* `save_pretrained`, so `config.json` is byte-identical to the king.
  - Size is ≈220.6 GB, under the 250 GB cap.
  - Untouched tensors (sinks, and e_score biases if γ = 0) stay bit-identical: the path is bf16 end to end with no casts, and averaging identical values is exact.

---

## 4. Memory and feasibility

Parameter split (exact, from the config):

| Group | Params |
|---|---|
| Muon set: attention 2.442B + dense 0.151B + routed experts 106.300B + shared 0.415B | 109.309B |
| AdamW set: embed + lm_head 0.937B, routers 0.0346B, norms 0.28M | 0.972B |
| Frozen: sinks 1,728 and e_score biases 11,264 | 12,992 |
| **Total** | **110.281B** (= 220.56 GB in bf16) |

**8×B300 (≈268 GiB usable per GPU):**

| Setup | Bytes/param | Total | Per GPU (before activations) | Verdict |
|---|---|---|---|---|
| **As configured**: bf16 params + bf16 grads + fp32 Muon momentum + bf16 AdamW m/v | ≈8.0 | 882 GB | **110 GB** | Fits even without AC (activations at mbs 4 × 2048 are about 80–130 GB; logits and loss about 12 GB). Numerically broken (C1, C2). |
| fp32 master AdamW, fp32 m/v | 16 | 1.76 TB | 220 GB | Does not fit with activations. |
| fp32 master, bf16 m/v | 12 | 1.32 TB | 165 GB | Fits with selective AC. |
| Kahan bf16 params + bf16 grads + bf16 m/v | ≈10 | 1.10 TB | 138 GB | Fits. |
| **Freeze routed experts**, train the other 3.98B in fp32 Adam | 16 on trainable | 63.7 GB + 212.6 GB frozen bf16 | **34.5 GB** | Trivially fits; large batches; far faster optimizer. Recommended first experiment. |

Host RAM with the current loader is 1.76 TB (H5).

**8×RTX 5090 (this machine: 8×32 GB = 256 GB VRAM, 503 GB RAM, PCIe only with no P2P on GeForce, SM120):**

- **The current stack cannot run here.** Per-rank CPU loading needs 1.76 TB. MXFP8 is gated off on SM120 (`core.py:818-827`). The frozen bf16 weights alone take 27.6 GB of each GPU when sharded 8 ways.
- **Full fine-tuning** needs at least about 1.1 TB of state. Impossible.
- **The "freeze routed experts" variant** needs 276 GB total, which exceeds 256 GB. It would need CPU offload.
- **Feasible options**, in order of practicality:
  1. **Offline eval.** Load one bf16 copy with a layer-wise `device_map` across all 8 GPUs (~5.6 layers and ~27.6 GB per GPU) and score at batch 1–2 with the validator's `compute_per_sequence_loss`, `enable_grouped_mimo_moe` and eager attention (the validator supports eager). Throughput is a few sequences per second, so 2 × 30k sequences takes hours. This is the most useful thing this box can do.
  2. **Head-only or top-K-layer training on cached activations.**
     - Run the frozen model once in pipeline mode and cache final hidden states, or the inputs to layer 45−K (6 KB per token; 50M tokens is about 300 GB on the 1.5 TB disk).
     - Train `norm` + `lm_head` (0.47B params, about 7.5 GB with fp32 Adam), or the last K = 1–2 MoE layers + norm + lm_head (K = 2 is about 4.9B params, about 78 GB of fp32 Adam state, about 10 GB per GPU when sharded).
     - The diff stays confined to those tensors; everything else is byte-identical, which is low risk for per-source regressions.
     - Gains are probably modest, but the loop is fast enough to iterate.
  3. **LoRA or attention-only training with the frozen base.** Either pipeline the frozen base across GPUs (≈27.6 GB/GPU plus activations, very tight, needing CPU offload of a few experts' weights) or use FSDP2 `CPUOffloadPolicy` for the base (about 220 GB pinned host memory with a single-copy loader). The CPU-offload route moves roughly 440 GB of weight traffic per step over PCIe, about 20–40 s per 16K-token step (~0.5–1K tok/s).
     - Caveats: `lora.py` puts adapters on Muon at LR 0.005 (C1). With `save_full_checkpoint: false` nothing is saved (`cli.py:1632-1655`). There is no LoRA-merge export path.
     - Verify SM120 sink gradients first (L4).

---

## 5. Recommended changes (minimal set to be competitive under the live protocol)

1. **Optimizer (C1, C2, H4).**
   - Use AdamW for all trained tensors, with fp32 master weights (load fp32; FSDP2 `param_dtype=bf16`, `reduce_dtype=fp32`) and bf16 or fp32 m/v.
   - Peak LR about 1e-5 (sweep 5e-6, 1e-5, 2e-5 with short probes), warmup 20–50 steps, decay to about 0, weight decay 0, global clip 1.0.
   - If Muon is kept: `muon_lr` about 1e-5–3e-5 (RMS-matched), `muon_weight_decay: 0`.
   - Set `moe.bias_update_speed: 0`, `z_loss_coeff: 0`, `seq_aux_alpha: 0`.
   - The first experiment to run is freezing routed experts (§4).
2. **Precision (H2, M1).** Set `low_precision: none`. Remove `allow_tf32 = True` (`cli.py:187-188`), or at least keep the gate GEMM in fp32. Accumulate the MoE combine in fp32 with fp32 routing weights, as the validator does.
3. **MoE throughput (C3).** Use grouped-GEMM experts with stacked weights (`torch._grouped_mm`), FSDP per decoder layer instead of per expert, and remove the 8× `slot` `index_add_` loop.
4. **Data (H6, H7).**
   - Pre-build a training pool: take many sequences per downloaded shard, and scale shard-count ranges by source size so sft-reasoning (6,400 sequences) is valid.
   - Fetch manifests once on rank 0.
   - Use a single global cache budget.
   - Enable the pool so resume works.
   - Keep the manifest proportions (optionally upweight txt360 and code if held-out gains are there, but guard every source).
5. **Offline gate (H1).** Replace `EvalDataset`/`offline.py` scoring with the validator's own sampling and scoring functions (§3.2 H1). Use the live delta, percentile LCB, margin about 0.0005, at least 2–3 independent category-stratified draws, and a per-source non-regression check. Delete `safety_margin: 0.05`.
6. **Loading and export (H5, M9).**
   - Load on a single rank and broadcast, or use meta init with sharded loads (recompute rotary `inv_freq`).
   - Export by streaming the gathered bf16 state to safetensors.
   - Assert: key set, shapes and dtypes equal the king index; frozen tensors are byte-identical to the king; contract-file SHA-256 matches `chain.toml`; no `mtp`, no `manifest.json`, no symlinks; total < 250e9 B; weights differ from the king.
   - Finally, reload through the validator's `load_eval_model` and re-score about 256 sequences against the offline numbers.
7. **Ops and monitoring (H9, H10, M4, M8).**
   - Held-out per-source eval every N steps with automatic stop or rollback on regression.
   - Keep the last N checkpoints plus model-only soup snapshots.
   - Default `CONFIG` to the B300 config in `launch.sh`.
   - King download into a new directory with atomic swap.
   - Before any 110B run, validate the genesis code path: a short real-model forward/backward on B300, and an FA4-vs-eager loss parity check on real data at 2048 tokens. The mini model cannot provide this (H8).
8. **Pins (M3).** transformers 5.15.1 plus FA4 asymmetric-V fix, matching the validator. Record the FA4 build.

---

## 6. Verification log

- **Read in full:**
  - Training stack: `src/train/cli.py`, `src/model/{core,lora,qk_norm}.py`, `src/moe/*.py`, `src/data/{dataset,packing}.py`, `src/optim/{optimizer,scheduler}.py` (plus `sparse_loco` snapshot and outer-step logic), `src/checkpoint/*.py`, `src/submit/export.py` (plus the relevant parts of `merge.py`), `src/eval/offline.py`, `src/dist/process_groups.py`.
  - Configs and scripts: `configs/b300/train.yaml`, `configs/pro6000/train.yaml`, the 1-GPU B300 configs (geometry only), `scripts/train/launch.sh`, `scripts/data/{download_king,prepare_eval_cache,prepare_train_sequence_pool}.py`.
  - Mini model: `pipeline_models/mimo_v2_mini/modeling_mimo_v2.py`.
  - Docs: `docs/TEST.md`, `docs/TRAINING_CONTROL.md`, `docs/TESTPLAN.md` (key sections), `README.md`, `.cursor/rules/CONTEXT.md`.
- **Computed on CPU** (script in scratchpad `numerics_check.py`):
  - Muon update RMS at lr 0.005.
  - bf16 rounding of AdamW, weight-decay and bias updates.
  - autocast downcasting an fp32 linear.
  - The mini SWA mask direction.
  - Pool-builder range arithmetic for the live mix.
  - Exact parameter counts matching the king's `total_size`.
- **Fetched:** `sft-reasoning/manifest.json` (full), and HEAD requests on the other four manifests. FA4 `flash_attn/cute/interface.py` from upstream `main` (to check `dlse`). No shards were downloaded.
- **Repo state note:** at the end of this review, `git status` showed untracked items I did not create: `.venv-eval/`, `UPDATE.md`, `eval_cache/`, `scripts/eval/parity_eval.py`, `scripts/verify/verify_parity_scorer.py`, `src/parity/`. They appeared during the session, presumably from a parallel agent, and were not reviewed. This review only wrote to the scratchpad.
- **Not verified:**
  - Live shard array layout (1-D vs 2-D; affects M7 only).
  - Behaviour of the pinned FA4 4.0.0b28 on SM120 backward (L4).
  - Actual e_score bias magnitudes in the king (affects how H3's bf16 distortion plays out).
  - Real 8×B300 step time for the current code; the extrapolation is the repo's own.
