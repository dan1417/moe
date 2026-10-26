# Move the TEXT campaign to a fresh 8-GPU box (B300 / B200 / H200)

## Current: 8× B300, `MODE=full_ema` (handover from the B200 box, 2026-10-09)

Recipe (42.md "text_full_ema_a", lessons from UPDATE.md): full-parameter (all but the routers), fp32 masters, 8-bit
AdamW with states offloaded to host RAM, **experts lr 1.5e-6 / dense 3e-6**, constant after 20 warmup steps, **2 048
sequences = 4.2M tokens per step**, **fp32 weight EMA 0.98/step** (val, checkpoints and screen scores are the EMA),
checkpoint + screen vs T1 every 25 steps, 200 steps over two fresh pools. Target on the screen: ≥ +0.0024 (validator
scores came in ~0.0004 below our screen), ideally +0.0026.

**Start checkpoint:** `START_REPO_PLACEHOLDER`. Fallbacks: `yamalies/85e70b14d610` (run d s150, +0.002117).

### 0. On the B200 box, before it is shut down
```bash
git -C /root/workspace/MoE push origin main      # EMA trainer, multi-pool, dashboard_server.py live only in local commits
```

### 1. On the B300 box (~1.5 h to the first training step)
```bash
git clone https://github.com/wilfrid51/MoE.git /root/MoE && cd /root/MoE     # private repo: token or SSH key
mkdir -p /data/teutonic && (umask 077; echo -n hf_xxx > /data/teutonic/hf_token)   # read token of yamalies (no gate approval needed for the owner)
MODE=full_ema START_REPO=yamalies/<repo> bash scripts/setup/bootstrap_h200.sh
```
What it does (idempotent, re-run after any failure): torch with `_grouped_mm` + validator venv + bitsandbytes + pm2 →
start checkpoint from HF (206 GB) → eval samples (mix A, text A/B) → pools `text_cpt_v1`, `text_val_v1` and the two
training pools `text_cpt_v10`, `v11` (~4 min each) → T1 from R2 (~26 min) + its screen losses → dashboard
(`dashboard_server.py` on 127.0.0.1:8787, render loop every 10 s) → `chain_runs.py` launches `text_full_ema_b1`, trying
per-GPU batch 16 × 16 first, then 8 × 32, then 4 × 64 (same 2 048-sequence step; a bigger batch must reach 30k tok/s at
step 1). The scorer duels step 0 first (paired baseline), then every EMA checkpoint.

### 2. Checks
* B300 is sm_103: if step 1 crashes in flex attention or bitsandbytes, run the 1-GPU smoke first:
  `CUDA_VISIBLE_DEVICES=0 .venv-eval/bin/python -m torch.distributed.run --nproc_per_node 1 scripts/train/cpt_fsdp.py --init $START --pool /data/teutonic/pools/text_cpt_v10 --val /data/teutonic/pools/text_val_v1 --out /tmp/smoke --steps 2 --lr 1e-3 --warmup 1 --min-lr-frac 1.0 --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --attn flex_attention --batch 2 --accum 2 --eval-every 2 --val-n 16 --save-every 2 --debug-layers 2 --ema 0.98`
  (bitsandbytes missing → the trainer falls back to torch AdamW with fp32 states; fine with offload).
* Host RAM ≥ 1.2 TB (EMA 440 GB + Adam 210 GB pinned + loading).
* Expected: step 1 within ~10 min of launch, 26k tok/s at batch 4 (B200 number), more at 8/16; EMA update ~2 s/step.
* Logs: `pm2 logs text_full_ema_b1`, `pm2 logs chain_text`, dashboard `ssh -L 8787:localhost:8787 <box>`.
* After run 1 the chain would start run 2 with the same `--pool v10,v11` (EXTRA overrides its pool) → before step 200,
  build `v12`/`v13` and relaunch run 2 with them (or stop the chain).

---

## Older: LoRA on 8× H200 / 8× B200

Same LoRA recipe as the H100 box (docs/LORA_PLAN.md: rank 16, lr 5e-5, 1 024 sequences per step, 200-step runs on fresh
200k-sequence pools, every 50 steps a checkpoint scored against T1), continued from the best checkpoint so far.

## What the H100 box leaves behind
* After every finished run the chain uploads that run's best checkpoint to Hugging Face (`yamalies/<random name>`,
  public but gated = manual approval; the free plan has no room for 220 GB private). The repo id is in
  `/data/teutonic/ckpt/<run>/uploads.json` and in `pm2 logs chain_text` ("UPLOADED yamalies/…").
* Pools `text_cpt_v1`, `v2` are deterministic (seeded by their tag), so the new box rebuilds them only to exclude their
  shards and trains on `v3` onwards.

Current best: `yamalies/85e70b14d610` = H100 run d step 150, screen +0.002111 vs T1 (uploaded 2026-10-09 08:11 JST).

## Steps (≈ 45 min to the first training step, mostly downloads)
```bash
git clone https://<wilfrid51 token>@github.com/wilfrid51/MoE.git /root/MoE && cd /root/MoE && git remote set-url origin https://github.com/wilfrid51/MoE.git
mkdir -p /data/teutonic && (umask 077; echo -n hf_xxx > /data/teutonic/hf_token)
START_REPO=yamalies/<repo from uploads.json> bash scripts/setup/bootstrap_h200.sh
```
`bootstrap_h200.sh`: torch with `_grouped_mm` + validator venv + pm2 → start checkpoint from HF → eval samples (mix A, text
A and B) → pools v1/v2/val → T1 download + screen losses → dashboard (localhost:8787) → `chain_runs.py --init-ckpt <start>
--batches 16x8,8x16,4x32`. The driver tries per-GPU batch 16 first (141 GB per GPU) and falls back to 8, then 4, if a
batch runs out of memory, crashes or is not faster than 17k tok/s at step 1.

## Modes
* `MODE=full` (default): full-parameter training of the routed experts of layers 23–44 with fp32 master weights, lr 1e-5,
  8-bit AdamW; estimated ~121 GB/GPU (fp32 weights 56 + gradients of the trained layers 27 + optimizer 13 + activations ~25).
  Per-GPU batch 4, falling back to 2 and 1 on OOM. This is the lever the H100 box cannot pull: LoRA there saturated near +0.0021.
* `MODE=full_all` (use on B200/B300, ≥ 180 GB per GPU): every parameter except the routers, fp32 master weights, lr 1e-5,
  8-bit AdamW; ~140 GB/GPU plus activations; needs ≥ 600 GB host RAM for loading. Per-GPU batch 4, fallback 2, 1.
* `MODE=lora`: the H100 LoRA recipe with per-GPU batch 16 (fallback 8, 4).

## Expectations
* H100 measured: batch 4 = 16.7k tok/s (7 h per 200-step run incl. scoring). Batch 8 is measured at the start of run e.
* H200: same compute per GPU as H100 but 1.8× the memory and 1.4× the bandwidth; batch 16 amortizes the base all-gathers
  4× better than batch 4 → expected 1.6–2× H100 throughput (≈ 3.5–4.5 h per run). Measure, do not assume: the step-1
  line of `pm2 logs text_lora_h1` prints tok/s and GPU memory.
* Stop rules are the same: screen ≥ +0.0026 → full-sample confirmation (tag:A and tag:B, 30 000 seqs each) → stop and
  upload; two runs in a row gaining < +0.0001 → stop for a decision. Nothing is ever submitted automatically.
