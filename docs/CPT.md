# CPT recipe for 8×B300 (`src/cpt`)

Replaces the old `src/train` stack for challenger training (see `docs/TRAINING_REVIEW.md`
for why the old stack cannot win: Muon LR ~1000× too high, no fp32 master weights,
per-expert Python MoE loop, outdated submit gate).

## What wins right now (from the dashboard, the lineage analysis and king #28's own metadata)
- Kings win by **0.003–0.012 nats** over 30k sequences of the 5-dataset mix.
- King #28's `SOURCE.json`/`merge_spec.json`: *"0.5·trunk v33 EMA-100 + 0.5·lineC v34 EMA-100, both
  warm-resumed from the v60 king"* → continued training from the king on **two data mixes in
  parallel, EMA of weights, then a 50/50 average**. Internal gain +0.0034.
- King #29 = #28 + an update to upper-layer attention/experts only; routers, router bias, norms and
  layers ≤12 untouched.

## Recipe implemented here
| Choice | Why |
|---|---|
| start from the current king, full-parameter AdamW, lr ~1e-5, wd 0, clip 1.0 | tiny, stable steps; the gap to beat is 0.3% |
| **fp32 master weights** (FSDP2 shards fp32, computes bf16) | bf16 in-place updates round ~84% of 1e-5-scale steps to zero |
| routers + `e_score_correction_bias` frozen, **pure CE** (no aux/z-loss/bias updates) | routing drift is the fastest way to regress a source; eval is CE only |
| objective = eval objective: unpadded 2048-token windows, no doc masking, mean over 2047 | exactly how the validator scores |
| grouped-GEMM experts (`torch._grouped_mm`, stacked [256, out, in]) | the old per-expert loop made 1B tokens take months; bit-identical per expert |
| data = validator's own category-stratified shard planner, quotas scaled | train on precisely the scored mixture; local-eval shards excluded |
| EMA (decay 0.99, host RAM) + warmup 50 / linear cooldown | EMA/soups are what the top miner ships |
| bf16 exports every N steps → parity eval → `write_merge.py` soups of lines | decide with the validator's math, ship exactly what was scored |

Memory per GPU (110.3B params, 8 ranks): fp32 params 55 GB + fp32 grads 55 GB + AdamW m,v 110 GB
= 220 GB, + checkpointed activations (micro-batch 4) → fits 288 GB B300. EMA: 55 GB host RAM per rank.
B200 (192 GB) does **not** fit full-parameter; freeze e.g. experts of layers 1–12 (`freeze:` regexes).

## Runbook (8×B300 node, local NVMe at /data)
```bash
# env: torch>=2.9 cu13, validator transformers fork, FA4 as used by the validator
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install "transformers @ git+https://github.com/tony-dendrite/transformers.git@9e8a1817d44502602b5b50e09b2a5ab0271bdb39" \
            "flash-attn-4==4.0.0b28" accelerate safetensors pyyaml requests httpx fastapi pydantic
pip install --no-deps "git+https://github.com/unarbos/teutonic.git@5479810a8c8b8635ede24e5be93515ffa9db4c6f"

python scripts/data/download_king.py --output /data/king --workers 32          # current king
python scripts/data/build_train_pool.py --out /data/pools/line_a --rounds 10 --scale 2 --tag line_a
python scripts/data/build_train_pool.py --out /data/pools/line_b --rounds 10 --scale 2 --tag line_b
python scripts/data/build_train_pool.py --out /data/pools/val --rounds 1 --scale 0.05 --tag val

# line A, then line B (different pool / seed / lr), each ~0.6B tokens
python -m torch.distributed.run --standalone --nproc_per_node=8 -m src.cpt.train --config configs/cpt/b300.yaml
```
Then on the eval box (or the same node): `parity_eval.py duel --king <king> --challenger <export_ema>`
on two fresh samples; soup lines with a recipe JSON (`0.5·A + 0.5·B`), re-score, `write_merge.py`,
`verify_submission.py`.

## Node runbook: the 2026-09-25 8×B300 pod (RAM-disk storage)

This pod has ~200 GiB of free disk and 2 TB of RAM, so checkpoints live on tmpfs. Everything below assumes
`source /workspace/teutonic/env.sh` (venv `.venv-cpt`, `TEUTONIC_EVAL_CACHE=/workspace/teutonic/eval_cache`,
`TEUTONIC_MODELS_DIRS=/dev/shm/teutonic/models`, FA4 kernel cache).

| What | Where |
|---|---|
| king #30 (`reign:30`) | `/dev/shm/teutonic/king30` (linked under `/dev/shm/teutonic/models/<digest>`) |
| start body (merged, from HF) | `/dev/shm/teutonic/start` ← `scripts/data/fetch_hf_checkpoint.sh <repo_id>` |
| exports (bf16 EMA, 220 GB each) | `/mnt/ram/runs/<line>/exports/stepNNNNNN_ema` (`export_keep: 2`; `touch <dir>/KEEP` to pin) |
| pools, val, king val losses | `/workspace/teutonic/pools/{line_a,line_b,val}`, `pools/val/king30_losses.npy` |
| run dirs (history.jsonl, triggers) | `/workspace/teutonic/runs/<line>` |
| logs | `/workspace/teutonic/logs/train_<config>_latest.log` |

```bash
# 0. gate (once, from the king): memory, tok/s, export round-trip, king val losses
bash scripts/train/node_train.sh configs/cpt/node/gate_full.yaml --max-steps 6
cp /workspace/teutonic/runs/gate_full/start_val_losses.npy /workspace/teutonic/pools/val/king30_losses.npy
# 1. start body arrives on HF
bash scripts/data/fetch_hf_checkpoint.sh <repo_id>                 # -> /dev/shm/teutonic/start, verified vs king
bash scripts/eval/screen_export.sh /dev/shm/teutonic/start 8000    # expect ≈ +0.0018 vs reign:30 on A and B
# 2. train line A (foreground; use tmux/nohup)
bash scripts/train/node_train.sh configs/cpt/node/line_a_upper.yaml
#    history: /workspace/teutonic/runs/line_a/history.jsonl  (val every 100 steps: vs_king / ema_vs_king per source)
#    controls: touch /workspace/teutonic/runs/line_a/{EXPORT,COOLDOWN,STOP}
# 3. judge an export (GPUs must be free: STOP the line or wait for it to finish)
bash scripts/eval/screen_export.sh /mnt/ram/runs/line_a/exports/step000750_ema 8000      # screen
bash scripts/eval/screen_export.sh /mnt/ram/runs/line_a/exports/step000750_ema 30000     # full, both samples
# 4. submit per docs/SUBMIT.md (verify_submission.py, then the user's wallet steps)
```

RAM budget (keep < ~1.8 TB): king 220 + start 220 (both `/dev/shm`) + ≤2 exports 440 (`/mnt/ram`) + host EMA
(55 GB/rank full, 18 GB/rank upper) + ~120 GB processes. Evict the king (`rm -rf /dev/shm/teutonic/king30`,
re-download ≈ 9 min) if a third export must be kept. No DCP checkpoints here; resume = train from an export.

### How much to train (decision rule, not a fixed number)
- Bar: validator LCB > 0.003 vs the king *at eval time*. Our own samples: **mu_hat ≥ +0.0036 on both A and B,
  robust_lcb > 0.003, no source negative**. The start body banks +0.0018, so training must add ≈ +0.002.
- Reference: the last three king steps gained +0.0032…+0.0036 each and took the top miners ~1–1.5 days; their
  weight deltas (2–4 % relative in layers 16–44, ~9 % lm_head) match ~1000–2000 AdamW steps at lr 1e-5.
- Procedure: watch `ema_vs_king.mu_hat` on the 2048-sequence val (SE ≈ 1e-4) every 100 steps
  (52 M tokens). When it exceeds ≈ +0.0040 (start body + training) `touch COOLDOWN` (200 steps of linear
  decay), let the final EMA export finish, confirm with the parity eval on A and B (full 30k). Hard cap
  2000 steps ≈ 1 B tokens per line. If a new king is crowned mid-run, the bar moves up by its gain (~0.003):
  keep training rather than restarting, and re-screen against `reign:<new>` once it is public.

## Handoff: 8×B300 run from the +0.0018 merged body (decided 2026-09-25)

**Start body** = `configs/merge/b300_start_body.json` = `#30u + 0.17·(#26−#22) + 0.13·(#25−#23)` with the deltas'
routed-expert part ×1.15 and attention / lm_head / shared-expert parts ×0.7 → **+0.00175 (A) / +0.00182 (B)
vs king #30** (see MERGE.md). `#30u` is king #30 with its hidden-dimension permutation undone
(`configs/merge/k30_hidden_perm.json`, exact). Two ways to get it onto the node:

*A. Rebuild from public downloads (no large transfer; ~1.5 h with 32 streams):*
```bash
git clone <this repo> && cd MoE && bash scripts/setup/setup_cpt_b300.sh && source .venv-cpt/bin/activate
export DATA=/data          # ≥ 2.5 TB NVMe: 5 kings (1.1 TB) + start body + exports + DCP checkpoints
START_RECIPE=configs/merge/b300_start_body.json \
START_UNPERMUTE=1a9d7c9de018864987af3c4163be49d5790019b5064087219c3be967bf5b2b9a:configs/merge/k30_hidden_perm.json \
BASE_CONFIG=configs/cpt/b300_upper.yaml LINES="a b" TOKENS_PER_LINE_ROUNDS=10 SCALE=2 \
bash scripts/train/launch_cpt.sh
```
(downloads #30/#26/#22/#25/#23, un-permutes #30, writes `$DATA/start_b300_start_body` with `write_merge.py`,
builds pools + val, trains line a then line b; exports every 300 steps under `$DATA/runs/line_*/exports`.)

*B. Download the written body from HuggingFace (public, verified: 15/15 validator gates, SHA-256 matched):*
```bash
pip install -U huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download touristAI/teutonic-II-110B-merge-k30u --local-dir /data/start_body
DATA=/data BASE_CONFIG=configs/cpt/b300_upper.yaml LINES="a b" bash scripts/train/launch_cpt.sh   # with king_dir: /data/start_body in the config
```
(`merge_recipe.json` inside the repo is the exact recipe; `README.md` has the scores.)

**Which config:** `b300_upper.yaml` trains layers 16–44 + lm_head (what every recent king's own step did;
≈195 GB/GPU) — first line. `b300.yaml` (all parameters, ≈250 GB/GPU) — second line if memory allows. Soup the
two EMA exports 50/50 (`configs/merge` recipe with the two export dirs as terms), score with the parity eval.

**Target:** the validator's LCB must exceed 0.003 **against #30** (or whatever king is current at submission).
The start body already banks +0.0018, so training has to add ≈ +0.0018 without regressing txt360. Judge every
export with `parity_eval.py duel --king reign:30 --challenger <export> --sample tag:A --sample tag:B` (on the
node with `--attn flash_attention_4`, or here) — never with the training loss alone.

**Before spending a hotkey** follow `docs/SUBMIT.md` (30k-sequence confirmation on two samples, write, verify,
re-score the written weights). Note #30's owner permutes their kings; our submission can be in either basis.

## Recommended starting point: the best merged body, not the bare king
Merge screening (see `MERGE.md`) found that stacking the two public sibling steps #29 lacks,
`#29 + 0.2·(#26 − #22) + 0.2·(#25 − #23)`, is ≈ +0.0014 better than king #29 on its own. Starting training
from it banks that gain:
```bash
START_RECIPE=configs/merge/stack_26_25_b0.2.json BASE_CONFIG=configs/cpt/b300_upper.yaml \
DATA=/data LINES="a b" bash scripts/train/launch_cpt.sh
```
(`launch_cpt.sh` downloads kings #29/#26/#22/#25/#23 and writes the body with `write_merge.py`.) Note the
trainer's `vs_king` numbers are then relative to the merged start; judge exports against the real king with
`parity_eval.py duel --king reign:<current>`.

## Gates before the first real run (do these on the B300 node, ~30 min)
1. Step-0 validation loss per source equals the parity scorer's king loss on the same sequences (±1e-3).
2. One optimizer step at lr 0 leaves exported weights byte-identical to the king (export round-trip).
3. Memory/throughput at micro-batch 4 with FA4; target ≥ 300M tokens/hour.

## Status
- Smoke-tested end-to-end on 8×RTX 5090 with a 4-layer slice of the real king (`configs/cpt/smoke_5090.yaml`):
  see UPDATE.md for the result. Not yet run on B300 hardware.
