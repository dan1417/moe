# Training plan: win TEXT (and attack MAIN) with a continued-pretraining run

Written 2026-10-08 07:30 UTC. Context: every merge family tops out at +0.0019…+0.0020 on the TEXT mix vs T1
(target +0.0026 = bar 0.002 + 0.0006 margin); MAIN/MATH are flat vs #42; router-only tuning gives nothing
(42.md). The kings that beat us were produced by training on the validator's data distribution. This is the plan
to do the same, sized so the decision can be made from one page.

## 1. Objective and success test

| competition | opponent | our best merge | target (screen) | confirm |
|---|---|---|---|---|
| TEXT | T1 (`95ef7a26…`) | +0.00195 | mu ≥ +0.0026 on the equal-thirds screen re-weighted 70/15/15 | `confirm41` on the exact text sample `b5318eaf-e83a2099…`, lcb30k > 0.002 |
| MAIN | #42 (`67643e86…`) | ≈ 0 | mu ≥ +0.0036 on the MAIN mix (32/28/22/17/1) | `confirm41` on the exact MAIN sample |

A trained checkpoint is scored exactly like a merge candidate (queue item with the checkpoint as a single term;
`sweep41` → 41.md/42.md), then merged *back* with the proven recipes (0.25·D41, experts momentum) if that still helps.

## 2. Start point and data

* Start point: `ck3_d41_025_pmom015` written with `write_merge.py` (= #42 + 0.25·(#41−#40) + experts 0.15·(T1−#40)),
  or plain #42 (`CK3_b41`) for a MAIN-first run. Both are in our basis (un-permuted).
* Data: `scripts/data/build_diverse_pool.py --competition text --n 200000 --block 32 --tag text_cpt_v1`
  (≈ 410 M tokens, one random 32-sequence block from each of ~6 000 shards, sources in the live 70/15/15
  proportions, the validator's vocab filter, shards of our local eval samples excluded). Build time ≈ 1 h at the
  r2.dev rate limit (the 49k-sequence math pool took 15 min). A second pool with `--competition main` for MAIN.
* Held-out: `--n 2048 --block 1 --tag text_val_v1 --exclude-pool <train>` for early stopping; the exact competition
  samples stay untouched for the final confirm.

## 3. Training recipe (what the winning kings' trajectories look like)

5DALwjE4's successive kings differ by small steps (|#41−#40| ≈ 230 in our int8 norm units, cos ≈ 0.3–0.4 between
consecutive steps): short, low-LR continued pretraining on the eval distribution, not a new model.

* Full-parameter CPT, bf16 weights, fp32 master + 8-bit AdamW (or AdamW with ZeRO-3 on 16 GPUs), seq 2048,
  global batch 256 sequences (0.5 M tokens), lr 1e-5 peak (50-step warmup, cosine to 1e-6), weight decay 0,
  1 epoch over the 410 M-token pool ≈ 800 steps. Routers (`mlp.gate.*`) and `e_score_correction_bias` frozen
  (keeps routing stable; see the router result in 42.md). No load-balancing loss (noaux routing).
* Checkpoint every 100 steps; score each one on our screen (10 min on this box) — the first useful signal is at
  step 200, before most of the money is spent.
* Variant if TEXT alone is the goal: sample the pool 80/10/10 instead of 70/15/15 (over-weight text slightly;
  the duel also measures the 15/15 parts, so do not go further).

## 4. Hardware, time, cost

| option | memory | tokens/s (est.) | 410 M tokens | cost (est.) |
|---|---|---|---|---|
| 8× B300 192 GB (1.5 TB) | full FT with 8-bit Adam ≈ 660 GB + activations: fits | 25–35 k | 3.5–4.5 h | ≈ $60–70/h → **$300–400 for the run, ~$1 000 for a day incl. setup/retries** |
| 16× H200 141 GB (2.2 TB) | fits with ZeRO-3 + fp32 Adam | 30–40 k | 3–4 h | ≈ $80–100/h → similar |
| this box (8× RTX 6000 Ada, 48 GB) | dense path / LoRA only, 0.3 seq/s | 0.6 k | 190 h | free, but a week |

The CODE crown pays ≈ 18 alpha/h ≈ $135/h at today's price, so a full rented day is covered by ~8 h of emission.

## 5. Pipeline on the rented box (one day)

1. `rsync` the start checkpoint (220 GB bf16) and the pools; install the eval venv (`src/parity`, `fa4sim`) so the
   same screen runs there in parallel with training.
2. Train with the recipe above (HF `MiMoV2ForCausalLM` + DeepSpeed ZeRO-3 / FSDP; `trust_remote_code`; the
   validator's loader needs the same shard layout, so save with `safetensors` + index, bf16).
3. Every 100 steps: score on the screen vs T1 and vs #42 (re-weighted per competition); stop when the TEXT margin
   stops improving or the MATH/CODE parts start losing more than the TEXT part gains.
4. Pick the checkpoint; optionally re-apply the merge recipe on top (`queue41 --base <ckpt> --vec D41=0.25 …`).
5. Confirm on the exact samples (`confirm41`), check ≥ +0.0006 margin above the bar, then the usual submission:
   `write_merge.py` → `prefix_check.py` → one upload → `ready`.

## 6. Risks

* Validator rejects models that are byte-copies of promoted weights only; a trained model is new weights — fine.
* Over-fitting the 15/15 parts: watch the per-category table (`catdiag41`) on each checkpoint.
* Permutation: we train in our basis; the validator does not care. Keep the hidden-dimension layout as is.
* Rate limit on r2.dev while building the pool: 48 workers, blocks of 32, retries are built in.
