# Teutonic SN3 Mining Pipeline — Full Context

## What is this project?
I am mining Bittensor Subnet 3 (Teutonic). The subnet trains a 110B MoE language
model collaboratively. Miners compete by submitting challenger model checkpoints.
The validator runs a paired bootstrap duel: if my challenger's loss is lower than
the king's by a statistically significant margin (LCB > delta_threshold), my model
becomes the new king and I earn TAO rewards.

## Competition Mechanics (from teutonic.ai/llms.txt + dashboard.json)
- King model:        current best checkpoint on-chain
- Challenger:        my submission — must start from the king checkpoint
- Eval metric:       per-sequence cross-entropy (nats/token), averaged over 2048 tokens
- Eval sequences:    2000 sequences drawn from ALL 3 datasets (weighted)
- Bootstrap:         10,000 resamples, alpha=0.001, one-sided LCB
- Accept condition:  LCB( mean(king_loss - challenger_loss) ) > delta_threshold
- delta_threshold:   currently 0.1 nats (was 0.5, tightened as competition matured)
- Current king loss: ~2.02 nats (reign #12 as of late Aug 2026)
- Speed matters:     first to submit a valid challenger after a new king is crowned wins
- Contract rule:     config.json, tokenizer*, modeling_mimo_v2.py, configuration_mimo_v2.py,
                     chat_template.jinja.txt must be BYTE-IDENTICAL to genesis.
                     Only .safetensors weight files may differ.
                     MTP tensors (keys containing "mtp") must be STRIPPED before submission.

## Model Architecture (MiMo-V2, trust_remote_code=True)
- Total params:      ~110B  (active per token: ~7B)
- Experts:           256 routed experts, top-8 routing per token (DeepSeek-style)
- Layers:            48 decoder layers
- Attention:         Hybrid — sliding window (SWA, window=128) every 5 layers,
                     full attention every 6th layer; attention-sink bias on SWA
- Genesis repo:      dendriteholdings/teutonic-II-110B-genesis
- Genesis revision:  8f971f030f5d801f812540007b396058f69cc08a
- Tokenizer:         XiaomiMiMo/MiMo-V2.5-Pro  (vocab size ~128K)
- Sequence length:   2048 tokens (fixed by protocol)
- Dtype:             bfloat16

## Datasets (from teutonic.ai/datasets/manifest.json)
Three pre-tokenized datasets hosted on Cloudflare R2 as .npy uint32 shards.
Each shard is packed 2048-token sequences.

| Dataset             | Proportion | Total Tokens | Shards  |
|---------------------|------------|--------------|---------|
| finewebedu          | 0.22       | 1.57T        | 125,441 |
| automathtext-v2     | 0.26       | 1.90T        | 151,131 |
| dclm-baseline-1.0   | 0.52       | 3.73T        | 297,617 |

- New datasets WILL be added over time. Proportions are normalized automatically.
- The eval samples from ALL three datasets in these exact proportions.
- Many competing miners only train on finewebedu — they lose on automathtext + dclm.
  Training across all three datasets in correct proportions is critical.
- Manifest is live at: https://teutonic.ai/datasets/manifest.json
- config_version field tells you if the manifest changed.

## Key API Endpoints
- https://teutonic.ai/dashboard.json     — live competition state, king, queue, history
- https://teutonic.ai/datasets/manifest.json — dataset sources, proportions, eval config
- https://teutonic.ai/llms.txt           — full human-readable protocol documentation
- Model shards:  https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/{digest}/

## Training Strategy (best practices applied)
- Start from king checkpoint (not genesis — king is better)
- WSD (Warmup-Stable-Decay) LR schedule — best for continued pretraining:
    warmup: 200 steps (linear 0→lr_max)
    stable: 4000 steps (constant lr_max) — checkpoints valid any time
    decay:  800 steps (cosine lr_max→lr_min) — trigger before submission
- LR: peak 1.5e-5, min 1.5e-6 (10x lower than scratch pretraining)
- Optimizer: AdamW, β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1, grad_clip=1.0
- No weight decay on 1D params (norms, biases, embeddings)
- BF16 mixed precision
- Gradient checkpointing (saves memory, +15% compute)
- Micro batch: 1-2 seqs/GPU, grad accumulation 8 steps → 64-128 global seqs/step
- FSDP2 (full shard) across 8×H200 GPUs
- Offline eval runs the SAME paired bootstrap before submission
  (different seed from validator, but same distribution — reliable LCB proxy)
- Safety margin: only submit if offline_LCB > delta + 0.05 (extra buffer)

## Project File Structure
```
MoE/
├── README.md
├── configs/
│   ├── b300/train.yaml          # 8×B300 production (canonical)
│   ├── b300/1gpu_*.yaml         # 1×B300 TESTPLAN gates
│   ├── 4090/                    # mini CPT dress rehearsal
│   ├── pro6000/train.yaml
│   └── smoke/pipeline.yaml
├── scripts/
│   ├── setup/                   # venv, FA4, NVML, bootstrap_ml, check_env
│   ├── train/                   # launch + gate runners
│   ├── data/                    # king, pools, mini build
│   ├── eval/ merge/ verify/
│   └── _repo_root.sh
├── src/
│   ├── train/                   # python -m src.train
│   ├── model/ moe/ data/ optim/
│   ├── checkpoint/ eval/ dist/ submit/
│   └── ...
├── pipeline_models/             # depth-shrink modeling sources
└── docs/                        # TEST.md, TESTPLAN.md, prompts/
```

## Workflow (end to end)
1.  bash scripts/setup/setup_env.sh                 # hardware-aware venv (FA4/TE on B300)
2.  python scripts/data/monitor.py --auto-download  # watch for new king 24/7
3.  python scripts/data/download_king.py --output ./king
4.  CONFIG=configs/b300/train.yaml bash scripts/train/launch.sh
    # or: torchrun --standalone --nproc_per_node=8 -m src.train --config configs/b300/train.yaml
5.  teutonic-miner auth / upload / ready

## Critical Rules (never violate)
- NEVER modify: config.json, tokenizer*, modeling_mimo_v2.py,
  configuration_mimo_v2.py, chat_template.jinja.txt
- ALWAYS strip MTP tensors (keys containing "mtp") before submission
- NEVER include manifest.json in submission directory
- Eval MUST use all 3 datasets in manifest proportions — not just finewebedu
- Checkpoint must be safetensors format (not pytorch .bin)
- challenger_model MUST initialize from king, not genesis

## Current Pain Points / Open TODOs
- [ ] Verify the exact field names in dashboard.json for king digest
      (key may be "king_digest" or "model_sha256" — check live dashboard)
- [ ] Confirm shard URL format: base_url + shard["key"] or shard["path"]
      (fetch one manifest.json from a dataset source and check structure)
- [ ] Test FSDP wrap policy: _find_decoder_layer_class() uses heuristics —
      print(type(layer).__name__) for layers in model to confirm class name
- [ ] For 8×H200 OOM: enable fsdp_cpu_offload: true in config, or reduce
      micro_batch_size to 1 and gradient_accumulation to 4
- [ ] DeepSpeed ZeRO-3 alternative to FSDP (add ds_config.json if preferred)
- [ ] The offline eval uses a different seed than the validator —
      results are directionally correct but not numerically identical
- [ ] Monitor manifest for new dataset additions (config_version change)

## Dependencies
torch>=2.4.0, transformers>=4.47.0, accelerate>=0.35.0,
safetensors, requests, numpy, tqdm, pyyaml, flash-attn (optional)

Install:
  pip install torch transformers accelerate safetensors requests numpy tqdm pyyaml
  pip install flash-attn --no-build-isolation   # optional, for throughput
