# Start the TEXT continued-pretraining run on a fresh 8× B300 box

Goal: beat the TEXT king T1 by ≥ +0.0026 on the 70/15/15 text mix (bar 0.002 + margin), starting from our best merge.
Trainer: `scripts/train/cpt_fsdp.py` (FSDP, grouped-GEMM experts, routers frozen, validator-ready checkpoints).
Plan and numbers: `docs/TRAINING_PLAN_TEXT.md`. Everything below is copy-paste; times are for a node with
8× B300 (≥ 180 GB each), ≥ 1 TB host RAM, ≥ 2 TB NVMe and a fast uplink.

## 0. Timeline (ETA)
| step | wall time |
|---|---|
| code + env (§1) | 20 min |
| start checkpoint from Hugging Face (§2, 220 GB) | 10–30 min (parallel with §1) |
| pools (§3, 1.7 GB, already built on the old box) | 2 min |
| training, 410 M tokens (§4), incl. 8 checkpoint saves | 2–4 h (first validation number after ~15 min, first checkpoint ~25 min) |
| score a checkpoint on our screen vs T1 (§5) | 10 min each (T1 download 1 h, in parallel with training) |
| confirm on the exact text sample + submission (§6) | 1.5 h |
| **total** | **5–7 h** |

## 1. Code and environment
```bash
git clone https://github.com/wilfrid51/MoE.git && cd MoE
# if GitHub is behind: apply the bundle copied from the old box
#   git fetch /path/moe_handoff_*.bundle main && git merge FETCH_HEAD
python3 -m venv .venv-eval && source .venv-eval/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu130      # torch ≥ 2.8 with CUDA 13 (B300 = sm_103)
pip install transformers==5.15.1 accelerate safetensors huggingface_hub hf_transfer numpy bitsandbytes
pip install -e .   2>/dev/null || true                                       # repo deps if any
bash scripts/setup/setup_eval_venv.sh                                        # validator f5bfc3f + fa4sim for the screen (§5)
mkdir -p /data/teutonic/{models,ckpt,pools,logs,eval_cache,submit}
cat > /data/teutonic/env.sh <<'EOF'
cd /path/to/MoE && source .venv-eval/bin/activate
export TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache TEUTONIC_MODELS_DIRS=/data/teutonic/models PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=1
EOF
# no LD_PRELOAD shim here unless this test prints False:
python -c "import torch;a=torch.arange(10,device='cuda:0');print(torch.equal(a.to('cuda:1').cpu(),a.cpu()))"
```

## 2. Start checkpoint
`yamalies/rt_smoke` on Hugging Face is our best TEXT recipe (#42 + 0.25·(#41−#40) + experts 0.15·(T1−#40), +0.0020 vs T1)
with 6 tiny router steps; the routers are frozen in training anyway, so it is the start point.
```bash
source /data/teutonic/env.sh
HF_TOKEN=hf_... python - <<'EOF'
import os
from huggingface_hub import snapshot_download
print(snapshot_download("yamalies/rt_smoke", token=os.environ["HF_TOKEN"], local_dir="/data/teutonic/ckpt/start", max_workers=16))
EOF
python - <<'EOF'   # sanity: 42 shards + index + modeling files
import json, os; d="/data/teutonic/ckpt/start"; print(len(json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]), "tensors;", sorted(f for f in os.listdir(d) if not f.endswith(".safetensors")))
EOF
```

## 3. Data
Copy from the old box (or rebuild with `scripts/data/build_diverse_pool.py --competition text --n 200000 --block 32 --tag text_cpt_v1`,
≈ 3 min; the val set with `--n 2048 --block 1 --tag text_val_v1 --exclude-pool <train>`):
```bash
rsync -a oldbox:/data/teutonic/pools/text_cpt_v1 oldbox:/data/teutonic/pools/text_val_v1 /data/teutonic/pools/
# text_cpt_v1: 200 000 seqs × 2048 tokens (0.41 B), sources 70/15/15 text/math/code, shards of our eval samples excluded
```

## 4. Train
```bash
source /data/teutonic/env.sh
nohup python -m torch.distributed.run --nproc_per_node 8 scripts/train/cpt_fsdp.py \
  --init /data/teutonic/ckpt/start --pool /data/teutonic/pools/text_cpt_v1 --val /data/teutonic/pools/text_val_v1 \
  --out /data/teutonic/ckpt/text_a --steps 800 --batch 2 --accum 16 --lr 1e-5 --warmup 50 --min-lr-frac 0.1 \
  --dtype fp32 --optim adamw8bit --attn eager --eval-every 50 --val-n 512 --save-every 100 \
  > /data/teutonic/logs/cpt_text_a.log 2>&1 &
tail -f /data/teutonic/logs/cpt_text_a.log
```
* global batch = 2 × 16 × 8 = 256 sequences = 0.52 M tokens; 800 steps = 1 epoch of the pool.
* `--dtype fp32` keeps fp32 master weights (sharded: 55 GB/GPU) — needed at lr 1e-5 (bf16 weights would swallow the updates);
  8-bit AdamW states 27 GB/GPU; activations with per-layer checkpointing ≈ 20–40 GB. Host RAM: rank 0 holds one fp32 copy
  (440 GB) while loading. If the node has 180 GB GPUs and < 600 GB RAM, use `--dtype bf16 --lr 3e-5` (second choice).
* The log prints `train`, `|g|`, `tok/s`, `eta` every 10 steps and `val` (held-out CE vs the start value) every 50;
  the run is doing its job when `val` falls steadily. Stop early (Ctrl-C / kill) if `val` rises for 100 steps.
* Checkpoints: `/data/teutonic/ckpt/text_a/step-100 … step-700, final/` — each a complete bf16 HF/validator layout.
* If OOM: `--batch 1 --accum 32` (same global batch). If NCCL complains about P2P on a PCIe box: `NCCL_P2P_DISABLE=1`.
* Resume after a crash: `--init /data/teutonic/ckpt/text_a/step-N --steps <remaining> --warmup 0`.

## 4b. 8× H100 80 GB: LoRA instead of full-parameter CPT
fp32 master weights of the whole model (440 GB, 55 GB/GPU sharded) do not fit next to the bf16 compute copy and the
activations on 80 GB GPUs, and bf16 master weights at lr 1e-5 swallow the updates. The H100 route is therefore LoRA:
the base stays frozen in bf16 (27.5 GB/GPU), rank-16 adapters with fp32 master weights (their own FSDP units, gradients
reduced in fp32) sit on the attention projections, the shared expert, the layer-0 dense MLP and — per expert, through
the same grouped GEMMs — the routed experts (`--lora-targets attn,shared,dense,experts`, 2.3 B adapter params). B is
zero at start, so step 0 is exactly the start model; at every save the adapters are folded into the weights
(W + α/r · B A) and the checkpoint has the same plain bf16 validator layout as in §4 (42 shards + index + modeling files).
```bash
source /data/teutonic/env.sh
export OMP_NUM_THREADS=16
nohup python -m torch.distributed.run --nproc_per_node 8 scripts/train/cpt_fsdp.py \
  --init /data/teutonic/ckpt/start --pool /data/teutonic/pools/text_cpt_v1 --val /data/teutonic/pools/text_val_v1 \
  --out /data/teutonic/ckpt/text_lora_d --steps 200 --batch 4 --accum 32 --lr 5e-5 --warmup 10 --min-lr-frac 0.1 \
  --lora-r 16 --lora-alpha 32 --optim adamw --attn flex_attention --eval-every 10 --val-n 512 --save-every 50 \
  > /data/teutonic/logs/cpt_text_lora_d.log 2>&1 &
# or under pm2: bash scripts/train/pm2_cpt_lora.sh <name> <same args>; score checkpoints alongside with
# scripts/train/score_ckpts.py --run /data/teutonic/ckpt/<name> --king <T1 dir>
```
* Global batch 1 024 sequences = 2.1 M tokens per step (4 × 32 × 8 GPUs), 200 steps = 1 epoch of the pool; adapter lr 5e-5,
  AdamW fp32 states (tiny), routers frozen as always. `--attn flex_attention` matches eager numerically and is ~1.7× faster;
  never use `sdpa` with this model (wrong losses).
* Smoke test first: `--steps 2 --save-every 2 --eval-every 1 --val-n 16` on a small pool; the saved `final/` must have
  42 shards, `model.safetensors.index.json` and the `*.py` / tokenizer files, and its weights must equal the start
  checkpoint up to the two tiny steps.
* Measured on 8× H100: 16.7 k tok/s (31 s per 256 sequences, 125 s per 1 024), 43.6 GB allocated / 62 GB reserved per GPU
  at batch 4; model load + FSDP 2.5 min, val-512 30 s, checkpoint save (adapters folded) 4 min, screen duel 5.5 min.
  Step-0 val on `text_val_v1` (512 seqs) = 1.2058; a value near 11.9 means the model is broken (uniform logits).
* lr findings (docs/LORA_PLAN.md): 1e-4 at 256 seq/step is flat on the mix, 2e-4 is destructive (val +0.0008…+0.0012 by
  step 50, with or without warmup), 5e-5 at 1 024 seq/step is the first setting with a clear val trend. Stop if val rises
  for 100 steps.
* Resume after a crash: `--init /data/teutonic/ckpt/text_lora_a/step-N` (folded weights) with fresh adapters,
  `--steps <remaining> --warmup 0` (the optimizer state is not carried over).

## 5. Score every checkpoint on our screen (while training continues)
Needs T1 locally (the TEXT king) and the samples. In parallel with §4:
```bash
bash scripts/merge/ingest_king41.sh 95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0 T1   # ~1 h
rsync -a oldbox:/data/teutonic/eval_cache/samples oldbox:/data/teutonic/eval_cache/losses /data/teutonic/eval_cache/   # or rebuild: parity_eval.py --competition mix sample --sample tag:A
# kings.json / models.json: set "text" -> the T1 dir; register the checkpoint and score it (10 min each on 8 GPUs):
python scripts/eval/parity_eval.py --competition text duel --king /data/teutonic/models/<T1 dir> \
    --challenger /data/teutonic/ckpt/text_a/step-200 --sample tag:A --limit 2400
```
(Or start `scripts/merge/sweep41.py --base /data/teutonic/ckpt/start …` and queue `queue41.py --base none --vec CKPT=1` items —
same tooling as on the old box; see docs/HANDOFF_K41.md §6.) Target on the screen: **TEXT mu ≥ +0.0026**; also check
that MATH/CODE parts did not fall more than the text part gained (`catdiag41.py`).

## 6. Confirm and submit (only if §5 passes)
```bash
# exact-sample confirmation (validator bootstrap) via confirm41 plugin or:
python scripts/eval/parity_eval.py --competition text duel --king <T1 dir> --challenger <ckpt> --sample b5318eaf-e83a2099ebe9d06f --limit 3000
# then the usual: verify_submission.py → fresh hotkey → ONE upload → prefix_check.py → ready --competition text
```
Rules that still apply: one upload per prefix, ≥ +0.0006 above the bar on the exact sample before spending a hotkey,
coldkey never on the box.
