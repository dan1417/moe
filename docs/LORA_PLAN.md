# LoRA plan for TEXT: what we measured, what the instrument can resolve, and the fastest path to the right parameters

Written 2026-10-08 15:10 UTC on the 8× H100 box. Goal unchanged: TEXT mu ≥ +0.0026 vs T1 on the text sample
(start model `rt_smoke` = +0.00182 on the first 2 400 sequences of `b5318eaf-e83a2099…` → we need **+0.0008 on the
70/15/15 mix** without losing the 15/15 parts).

## 1. Facts from today's runs (all 8× H100, FSDP, bf16 base + fp32 rank-16 adapters, routers frozen)

| run | recipe | result |
|---|---|---|
| a | lr 1e-4, warmup 50, 256 seq/step, all targets | 200 steps (105 M tok): val −0.0001 ± 0.0001; screen +0.00182 → +0.00192 (s100) → +0.00188 (s200). **Text split +0.0002, math −0.0003, code −0.0002: the mix nets zero.** |
| b | lr 2e-4, no warmup, from a/s200 | val +0.0012 at step 50: destructive |
| c | lr 2e-4, warmup 50, from a/s200 | val +0.0008 at step 50: still destructive → 2e-4 is above the stable range for these adapters |
| d (running) | lr 5e-5, warmup 10, **1 024 seq/step**, 200 steps = 1 epoch | val vs step 0 (1.20579): step 10 −0.00025, 20 −0.00011, 30 −0.00029, 40 −0.00032, 50 −0.00022; **screen step 50 +0.00201, step 100 +0.00209 vs start +0.00182** (step 100: text +0.00038, math −0.00004, code +0.00003 vs start): step 150 +0.00211: increments +0.000188 / +0.000077 / +0.000025 per 50 steps → saturates near +0.00212 |

Costs: 16.7 k tok/s → 31 s per 256-seq step, 125 s per 1 024-seq step; model load + FSDP 2.5 min; val-512 ≈ 30 s;
checkpoint save 4 min; screen duel (pause + 2 400 seq text + 2 400 seq mix) 5.5 min.

**The instrument problem.** The effects we are hunting are 1–2 × 10⁻⁴ per 50 M tokens. The noise of the tools:
val-512 ± 1 × 10⁻⁴ (paired), screen vs T1 SE 1.3 × 10⁻⁴ unpaired but **≈ 3 × 10⁻⁵ for the paired difference of two
checkpoints of the same run**, and the per-split val on the full 2 048-seq pool ± 6 × 10⁻⁵ (text) / ± 1.3 × 10⁻⁴
(math, code). Any probe shorter than ~100 steps (50 M tokens, 1 h) is below the resolution of val-512 — a 25-minute
probe tells us nothing. Rule: **decide on paired screen deltas and per-split val, never on val-512 alone.**

## 2. Why the mix does not move (and what can change it)

Every token passes the attention, shared-expert and layer-0 adapters; only 8 of 256 experts per token see the
routed-expert adapters. Text gradients dominate the shared paths (70 % of tokens) and pull them toward text, which is
exactly where math and code lose. The routed-expert adapters are the only domain-separable lever. Three cheap ways to
tilt the trade-off, in the order they will be tried:

1. **Group learning rates**: experts at lr, shared paths (attn/shared/dense) at lr/5 (`--lr-dense-frac 0.2`).
2. **Experts-only adapters** (`--lora-targets experts`): the limiting case of 1; also 3× fewer FSDP units (faster).
3. **Loss weights per source** (`--loss-weights text=1,math=2,code=2`): protects the 15/15 parts with the same data.

Not worth GPU time now: rank 32 (capacity is not the limit — the adapters barely move the loss at r = 16), lr ≥ 2e-4
(destructive twice), bigger batch *without* a lr change (same drift, fewer steps).

## 3. Search procedure (fastest that can actually resolve the effects)

Trainer additions (no GPU needed, done while run d runs): per-split val (text/math/code CE on the full val pool every
eval, so the trade-off is visible live), `--lr-dense-frac`, `--loss-weights`, `--val-n 2048`.

Then **100-step probes at 256 seq/step** (52 M tokens, ≈ 55 min + 4 min save + 5.5 min score ≈ 65 min each), each
scored on the screen paired against the start on the same 2 400 sequences, all from the same start checkpoint:

| # | probe | what it answers |
|---|---|---|
| P1 | lr 1e-4, group lr dense ×0.2 | does protecting the shared paths keep math/code while text gains? |
| P2 | experts-only, lr 1e-4 | is the gain in the experts at all? |
| P3 | lr 1e-4, loss weights math/code ×2 | can data weighting buy the 15/15 back? |
| (d) | lr 5e-5, batch 1 024 (running) | does a cleaner gradient at lower lr net positive? (judge at step 50 = 100 M tokens, 16:50 UTC, screen on `step-50`) |

Decision metric per probe: paired screen delta vs start at 100 steps (text / math / code split) → mix gain per 50 M
tokens. A winner needs ≥ +0.0002 per 100 steps on the mix with no split below −0.0001; then it runs the epoch
(800 steps ≈ 4.5 h incl. scoring) with checkpoints every 100 steps and the usual early-stop rules. If no probe clears
+0.0002/100 steps, LoRA CPT cannot reach +0.0026 in a day on this box and the honest answer is to stop renting and go
back to the merge route (or a larger, full-parameter run elsewhere).

## 4. Zero-training search in parallel: the trained delta as a merge ingredient

`a/step-200 − start` is a task vector (text +0.0002 / math −0.0003 / code −0.0002 at dose 1). The merge tooling can
score **doses and module placements of that vector without any training** (5 min per recipe during a training pause):
experts-only part at dose 1, 2, 3; attention-only; dose 2 overall. If an experts-only ×2–3 extrapolation gives
text +0.0006 at math/code ≥ −0.0002, that is a +0.0003 mix gain for free and stacks with whatever probe wins.
Recipe: register `A200`, define `TA = A200 − START` in `configs/merge/k41/vectors.json`, queue
`--base START --ov experts:TA=2.0` etc. (`queue41.py`), score with `parity_eval duel` on the text sample.

**Result (15:21 UTC):** experts ×2 +0.00152, experts ×3 +0.00049, all ×2 +0.00084, attention ×2 +0.00170 — all below the start
(+0.00182). The trained direction does not extrapolate; §4 is closed. This also says the lr-1e-4 direction is mostly noise,
which is the case for the larger-batch run d.

## 5. Timeline from 15:10 UTC

| when | what |
|---|---|
| 15:10–15:40 | trainer additions (per-split val, group lr, loss weights); run d continues |
| 15:40–16:50 | run d to step 50 (100 M tokens) → save → screen. In parallel nothing else fits on the GPUs. |
| 17:13 | **decision 1 (taken)**: run d +0.00019 at step 50, below the +0.0003 bar but with no split loss → continue. **17:20, user: run d trains the full epoch (200 steps, ≈ 22:30 UTC); no early stop on the screen.** Only stop: val rising for 50 steps (5 readings) or a split falling below −0.0003 on the screen. Probes come after run d. Original rule: run d mix gain ≥ +0.0003 with no split < −0.0001 → keep d to the epoch (done ≈ 22:00). Otherwise stop d and start the probes. |
| 17:00–20:20 | P1, P2, P3 (65 min each) + the 4 merge-dose recipes during the pauses |
| 20:30 | **decision 2**: best probe → epoch run from start (≈ 4.5 h, ready ≈ 01:00 UTC) or stop. |
| each checkpoint | screen; at ≥ +0.0026: confirm on the full text sample (30 000 seqs, lcb30k) → report → hotkey |

## 6. After run d: chain until the king is beaten (user decision, 17:30 UTC)

`scripts/train/chain_runs.py` (pm2 `chain_text`) watches the scores. Each finished run is followed by the next one
(`text_lora_e`, `_f`, …) from the previous run's `final` weights, on a **fresh** 200k-sequence pool (`text_cpt_v2`, `v3`, …;
eval-sample, val and earlier-pool shards excluded), same recipe as run d (1 024 seq/step, lr 5e-5, 200 steps). A checkpoint
with screen mu ≥ +0.0026 is confirmed on the **full** text samples tag:A and tag:B (30 000 sequences each, trainer paused):
mu ≥ +0.0026 and validator LCB > 0.002 on both → training stops, `<run>/CONFIRMED.json`, nothing submitted. The chain stops for
a human decision after two consecutive runs gaining < +0.0001 each, or after 4 new runs.

