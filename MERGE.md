# MERGE.md — merge campaign against the current king (#30)

## Current king: #30 (`teutonic-II-110B-A7B-281b2710`, crowned 2026-09-25 00:29 UTC by +0.00325 over #29)
- Owner 5Hmim6 (also owns #29 and #26 / lineage B).
- **#30 is a hidden-dimension-permuted model**: every tensor differs from #29 by |Δ|/|W| ≈ 1.41 (= √2, i.e.
  element-wise uncorrelated) yet embeddings columns match #29's one-to-one with cosine 1.0000. The permutation
  is function-preserving (an obfuscation against exactly this kind of weight-space merging). We work on
  **#30u** = #30 re-indexed into #29's basis (`scripts/merge/unpermute_checkpoint.py`, exact, no arithmetic;
  installed as `reign:30` on the fast disk).
- After un-permuting, #30's real step vs #29: lower layers/embeddings/routers **bit-identical**, upper layers
  +2–4%, shared experts +11%, lm_head +9% (same recipe as #29's own step). Its direction is nearly
  orthogonal to our two sibling deltas (cos 0.06–0.2), so the deltas are *not* inside #30.

## Ingredients (fixed — the only public training steps outside #30's ancestry)
| delta | definition | on disk |
|---|---|---|
| **D26** | #26 − #22 (lineage B's last step, owner 5Hmim6) | `/opt/teutonic/deltas/d26_22` (bf16, verified exact) |
| **D25** | #25 − #23 (5DALwjE4's step, sibling of #24) | `/opt/teutonic/deltas/d25_23` |

Everything else public (#0–#24, #27, #28, #29) is an ancestor of #30 and therefore already inside it; adding
more of an ancestor's step = extrapolating the king along its own path, which measured negative every time.

## Results vs #30u (live table; `python scripts/merge/merge_report.py --king reign:30 --prefix u30_ --sample tag:A --sample tag:B`)
<!-- K30_RESULTS -->
_Regenerated 2026-09-25 06:36 UTC from the loss cache; mu_hat = mean(king CE − candidate CE), positive = beats the king; need ≈ +0.0036._

| candidate | method | what it does | sample | seqs | **mu_hat** | LCB | win % | code | math | pretrain | sft | txt360 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `u30_dstack_0.17_0.13` | linear | #30u + 0.17*D26 + 0.13*D25 | tag:A | 4000 | **+0.00142** | +0.00122 | 70 | +0.0029 | +0.0015 | +0.0029 | +0.0025 | -0.0022 |
| `u30_dortho_0.17_0.13` | delta_orthogonal | #30u + 0.17*(D26 orth. to step) + 0.13*(D25 orth. to step) | tag:A | 4000 | **+0.00135** | +0.00116 | 70 | +0.0028 | +0.0014 | +0.0027 | +0.0021 | -0.0019 |
| `u30_dstack_0.1_0.1` | linear | #30u + 0.1*D26 + 0.1*D25 | tag:A | 4000 | **+0.00131** | +0.00116 | 72 | +0.0023 | +0.0011 | +0.0023 | +0.0021 | -0.0007 |
| `u30_dagree_0.25_0.2` | delta_agree | #30u: sign-agreement weighting (0.25,0.2) vs (0.1,0.08) | tag:A | 4000 | **+0.00102** | +0.00078 | 67 | +0.0033 | +0.0015 | +0.0028 | +0.0024 | -0.0041 |
| `u30_dstack_0.25_0.0` | linear | #30u + 0.25*D26 + 0.0*D25 | tag:A | 4000 | **+0.00081** | +0.00060 | 65 | +0.0024 | +0.0013 | +0.0022 | +0.0021 | -0.0032 |
| `u30_dstack_0.0_0.15` | linear | #30u + 0.0*D26 + 0.15*D25 | tag:A | 4000 | **+0.00068** | +0.00055 | 62 | +0.0013 | +0.0005 | +0.0013 | +0.0013 | -0.0005 |
| `u30_dstack_0.0_0.25` | linear | #30u + 0.0*D26 + 0.25*D25 | tag:A | 4000 | **+0.00040** | +0.00022 | 59 | +0.0015 | +0.0006 | +0.0015 | +0.0010 | -0.0023 |
| `u30_extrap_-0.2` | linear | #30u - 0.2*(#30u-#29) | tag:A | 4000 | **+0.00007** | -0.00003 | 51 | -0.0001 | -0.0001 | -0.0000 | +0.0004 | +0.0004 |
| `u30_extrap_+0.2` | linear | #30u + 0.2*(#30u-#29) | tag:A | 4000 | **-0.00048** | -0.00058 | 40 | -0.0004 | -0.0002 | -0.0004 | -0.0008 | -0.0010 |
| `u30_undo23_-0.1` | linear | #30u - 0.1*(#23-#21): extend the new-mix un-learning of an old-mix step | tag:A | 4000 | **-0.00149** | -0.00161 | 26 | -0.0022 | -0.0011 | -0.0023 | -0.0021 | -0.0001 |
<!-- /K30_RESULTS -->

Planned waves (each 4k-sequence screen ≈ 7 min now that the deltas load from the fast disk):
1. **re-screen**: `#30u + a·D26 + b·D25` at (0.17,0.13), (0,0.25), (0.25,0), (0.1,0.1), (0,0.15)
2. **wave 1**: extrapolate / partially undo #30's own step (±0.2); *orthogonal-to-step* (deltas minus their
   projection on #30−#29); sibling sign-agreement weighting
3. **diagnostics**: per-token loss attribution (which target tokens the stack hurts on txt360 → per-row
   `lm_head` repair from the king), routing stats of #30u (for routing-aware expert merges)
4. **wave 2** (`scripts/merge/wave2.py`): TIES / DARE / breadcrumbs on the deltas, per-module ×0.7/×1.3,
   per-depth scaling, then a coordinate search over grouped coefficients
5. best candidate → 12k-sequence confirmation on two samples → `write_merge.py` → `verify_submission.py`

## Conclusion vs #30 (2026-09-25 05:50 UTC)
**Best merge: `#30u + 0.17·D26 + 0.13·D25` with the deltas' routed-expert part ×1.15 and their attention and
lm_head/shared-expert parts ×0.7 → +0.00175 on sample A, +0.00182 on independent sample B (LCB ≈ +0.0016,
win rate 72–73%; code +0.0034, pretrain +0.0034, sft +0.0026, math +0.0017, txt360 −0.0015…−0.0019).**
The bar is +0.0036 (LCB > 0.003 on the validator's 30k draw): **merging public kings reaches half of it.**

Everything tried vs #30u (all 4k-sequence screens, same sequences): plain stacks over (a,b); single deltas;
extrapolating / undoing #30's own step (±0.2: −0.00048 / +0.00007); orthogonal-to-step; sibling sign-agreement;
delta-TIES, breadcrumbs, delta-DARE; per-module ×0.7/×1.3 (experts↑, attention↓, head↓ — the only structure
found); per-depth splits; the combinations and a coordinate probe (flat plateau +0.00170…0.00175); routing-aware
expert scaling, old-mix ancestor-step ±, token-level diagnostic → pending in the queue, none expected to move
the needle by more than ~1e-4.

Why the ceiling: the two sibling steps carry ≈ +0.003 of reasoning/code gains but ≈ −0.002 (weighted) of web
damage, and every way of separating the two (modules, depth, experts by routing, sparsification, projections)
loses more gain than it removes damage. There are no other public steps outside #30's ancestry.

**What the +0.0018 is worth:** as the *starting point* for training it halves the distance a B300 run has to
cover (needs ≈ +0.0018 more instead of +0.0036). `write_merge.py` + `verify_submission.py` produce the exact
scored weights (the finalizer does this automatically when the queue drains).

Reference point: the same two deltas gave a flat optimum of **+0.0016** vs #29 (history below); the bar is
**+0.0036** (LCB > 0.003 on the validator's own 30k draw).

---

## History: results vs king #29 (2026-09-24; superseded)

Last updated 2026-09-24 21:20 UTC. Newest numbers are regenerated from the loss cache
(`python scripts/eval/parity_eval.py leaderboard --king reign:29 --sample tag:A`).

## How to read the score
- **mu_hat** = mean over sequences of (king #29 loss − merged-model loss), in nats/token, i.e. exactly the
  validator's statistic. **Positive = the merge beats the king.**
- Sequences: a fresh validator-distribution sample (`tag:A`, 30k built with the validator's own
  category-stratified sampler; first 4,000 or 8,000 used for screening). Same sequences for every row.
- **To take the crown** the validator needs LCB > **0.003** on its own 30k draw; because a different draw
  moves mu_hat by ~±2e-4, we want **mu_hat ≥ ~0.0036** here before spending a hotkey.
- Per-source columns are mu_hat per dataset (eval weights: pretrain 0.32, math 0.28, txt360 0.22,
  code 0.17, sft 0.01).

## The scorer is trustworthy
Replaying the latest real duel (#29 vs #28) on the validator's exact 30k sequences (seed recovered
on-chain) gave mu_hat **0.003240** / LCB **0.003104** vs the published **0.003234** / **0.003101**.

## Models used as ingredients (all public kings)
| reign | repo | lineage | role |
|---|---|---|---|
| **#29** | teutonic-II-110B-A7B-d2fb75a4 | A (tip) | **current king** (= #28 + update to layers 16–44 + lm_head) |
| #28 | teutonic-II-110B-A7B-5ek5koe5-v61 | A | previous king (itself a 50/50 merge of two training lines) |
| #26 | teutonic-II-110B-A7B-e6f3595e | B (tip) | the other lineage's last king |
| #22 | 5Hmi/teutonic-II-110B-A7B-e6466d4c | B | #26's parent |
| #20 | teu/teutonic-II-110B-rbefd0a96-o19 | B | #22's parent |
| #17 | teutonic-II-110B-A7B-5Dz7-6 | common ancestor of A and B | |
| #25 | teutonic-II-110B-A7B-5DALwjE4-77 | A2 (sibling of #24, from #23) | its step (#25 − #23) is not in #29's ancestry |
| #23 | teutonic-II-110B-A7B-5Fjj-25 | A | #25's parent (and an ancestor of #29) |

## Results (sorted by score)
| # | candidate | method | what it does | seqs | **mu_hat** | LCB | win % | code | math | pretrain | sft | txt360 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0a | **`dstack_0.17_0.13`** | linear, stacked (delta checkpoints) | `#29 + 0.17·(#26−#22) + 0.13·(#25−#23)` — the fitted optimum | 4000 | **+0.00160** | +0.00141 | – | – | – | – | – | -0.0018 |
| 0a' | `dstack_0.2_0.15` | linear, stacked (delta checkpoints) | `#29 + 0.2·(#26−#22) + 0.15·(#25−#23)` | 4000 | **+0.00159** | +0.00137 | – | – | – | – | – | -0.0027 |
| 0b' | `dstack_0.18_0.1` | linear, stacked (delta checkpoints) | `#29 + 0.18·D26 + 0.10·D25` | 4000 | **+0.00154** | +0.00136 | – | – | – | – | – | -0.0017 |
| 0c | `dstack_0.15_0.2` | linear, stacked (delta checkpoints) | `#29 + 0.15·D26 + 0.20·D25` | 4000 | **+0.00145** | +0.00123 | – | – | – | – | – | -0.0025 |
| 0 | **`stack_26_25_b0.2`** | linear, stacked sibling steps | `#29 + 0.2·(#26 − #22) + 0.2·(#25 − #23)` — lineage B's last step **plus** the 5DALwjE4 branch step (sibling of #24), both absent from #29's ancestry | 4000 | **+0.00140** | +0.00116 | 69 | +0.0036 | +0.0018 | +0.0032 | +0.0024 | -0.0035 |
| 0b | `stack_26x0.2_25x0.4` | linear, stacked | `#29 + 0.2·(#26−#22) + 0.4·(#25−#23)` | 4000 | **-0.00056** | -0.00093 | 56 | +0.0033 | +0.0013 | +0.0016 | +0.0006 | -0.0092 |
| 1 | `transplant26_b0.25` | linear (task arithmetic) | `#29 + 0.25·(#26 − #22)` — add lineage B's last training step | 8000 | **+0.00100** | +0.00084 | 67 | +0.0026 | +0.0015 | +0.0023 | +0.0022 | -0.0027 |
| 2 | `expertB_a1.0_e0.3_r0.15` | routing-aware expert merge (MoE) | B's step ×0.3 only into experts with txt360 affinity ≤1.0 (4925/11264), ×0 web experts, ×0.15 attention/shared/head | 4000 | **+0.00084** | +0.00070 | 64 | +0.0026 | +0.0013 | +0.0012 | +0.0016 | -0.0017 |
| 3 | `expertB_a0.5_e0.5_r0.2` | routing-aware expert merge (MoE) | B's step ×0.5 into experts with affinity ≤0.5 (2002), ×0.1 other experts, ×0.2 non-experts | 4000 | **+0.00063** | +0.00047 | 59 | +0.0024 | +0.0011 | +0.0010 | +0.0017 | -0.0019 |
| 4 | `ties26_d0.3_b0.35` | TIES | `#29 + 0.35·trim30%(#26 − #22)` — keep only the top-30% largest-magnitude Δ entries per tensor | 4000 | **+0.00021** | -0.00000 | 59 | +0.0019 | +0.0012 | +0.0016 | +0.0015 | -0.0045 |
| 5b | `t26_upper` | linear, per-depth | B's step ×0.25 on layers 29–44 + lm_head only | 4000 | **+0.00054** | +0.00039 | 62 | +0.0016 | +0.0009 | +0.0015 | +0.0014 | -0.0022 |
| 5c | `transplant25_b0.25` | linear (task arithmetic) | `#29 + 0.25·(#25 − #23)` — the 5DALwjE4 branch step alone (sibling of #24) | 4000 | **+0.00045** | +0.00026 | 58 | +0.0017 | +0.0007 | +0.0014 | +0.0007 | -0.0022 |
| 5a | `stack_26x0.25_25x0.3` | linear, stacked | `#29 + 0.25·(#26−#22) + 0.3·(#25−#23)` | 4000 | **+0.00050** | +0.00016 | 63 | +0.0039 | +0.0018 | +0.0027 | +0.0021 | -0.0071 |
| 5 | `expertB_a0.8_e0.35_r0.25` | routing-aware expert merge (MoE) | B's step ×0.35 into experts with affinity ≤0.8 (3645), ×0 web experts, ×0.25 non-experts | 4000 | **-0.00011** | -0.00028 | 52 | +0.0022 | +0.0010 | -0.0000 | +0.0008 | -0.0035 |
| 6 | `transplant26_b0.25_nohead` | linear, per-module | B's step ×0.25 everywhere except lm_head | 8000 | **-0.00034** | -0.00050 | 51 | +0.0008 | +0.0006 | +0.0008 | +0.0004 | -0.0041 |
| 7 | `slerp29_26_t0.1` | SLERP | per-tensor spherical interpolation SLERP(#29, #26, t=0.1) | 4000 | **-0.00041** | -0.00052 | 42 | -0.0005 | -0.0003 | -0.0006 | -0.0008 | -0.0003 |
| 8 | `dare26_p0.5_b0.25` | DARE | `#29 + 0.25·DARE(#26 − #22)` — drop 50% of Δ entries at random, rescale ×2 | 4000 | **-0.00074** | -0.00096 | 49 | +0.0013 | +0.0008 | +0.0001 | +0.0007 | -0.0055 |
| 9 | `transplant22_b0.25` | linear (task arithmetic) | `#29 + 0.25·(#22 − #20)` — lineage B's earlier step | 4000 | **-0.00157** | -0.00178 | 37 | -0.0005 | -0.0003 | -0.0004 | -0.0005 | -0.0059 |
| 10 | `orthoB_b0.25` | orthogonal projection | `#29 + 0.25·⊥(#26 − #17)` — only the part of B's path (since common ancestor #17) orthogonal to #29's path | 8000 | **-0.00173** | -0.00188 | 31 | -0.0010 | -0.0006 | -0.0018 | -0.0017 | -0.0036 |
| 11 | `t26_lower` | linear, per-depth | B's step ×0.25 on layers 0–28 + lm_head only | 4000 | **-0.00176** | -0.00192 | 28 | -0.0017 | -0.0007 | -0.0016 | -0.0008 | -0.0035 |
| 12 | `transplant26_b0.25_headonly` | linear, per-module | B's step ×0.25 on lm_head only | 8000 | **-0.00192** | -0.00200 | 20 | -0.0026 | -0.0010 | -0.0018 | -0.0024 | -0.0026 |
| 13 | `extrap29_a0.5` | linear (extrapolation) | `#29 + 0.5·(#29 − #28)` — continue #29's own update | 8000 | **-0.00198** | -0.00208 | 21 | -0.0017 | -0.0009 | -0.0024 | -0.0030 | -0.0029 |
| 14 | `transplant26_b0.5` | linear (task arithmetic) | `#29 + 0.5·(#26 − #22)` | 8000 | **-0.00236** | -0.00269 | 46 | +0.0014 | +0.0010 | -0.0001 | -0.0002 | -0.0127 |
| 15 | `dareties17_w0.3` | DARE-TIES (mergekit) | base #17 + TIES(sign election, normalized) over DARE'd deltas of #29 (w1, density 1.0) and #26 (w0.3, density 0.5) | 4000 | **-0.00827** | -0.00868 | 9 | -0.0049 | -0.0030 | -0.0080 | -0.0062 | -0.0181 |

Methods implemented (all in `src/parity/weights.py::_fill`, the same code `scripts/merge/write_merge.py`
uses to write a submission, so a scored merge is shipped bit-for-bit): linear/task-arithmetic with
per-tensor & per-regex coefficients, sign-agree, orthogonal projection, TIES, DARE, DARE-TIES (mergekit
semantics), SLERP, routing-aware expert merge (`scripts/merge/make_expert_recipe.py` + `parity_eval.py
routing`). Recipes: `configs/merge/*.json`.

## Other levers tried
| lever | result |
|---|---|
| logit temperature (scale lm_head by c*) | c* = 1.0012, predicted gain **+0.000001** — king is perfectly calibrated |
| head tuning (fit lm_head only on 16k seqs, frozen transplant body, validated on held-out sample) | **−0.00004 ± 0.00004** after 250 steps — head already optimal for its body; stopped |

## What we learned
1. **#29 sits at the optimum of its own update**: extrapolating it (row 13) loses 0.002; a parabola
   through #28/#29/extrapolation puts the optimum at 0.84× of #29's update (worth only ~1e-4).
2. **Lineage B's last step (#26 − #22) is the only ingredient with new information**: +0.001 at β=0.25,
   best β≈0.18–0.25 (β=0.5 overshoots, row 14). It helps every reasoning/code/math/planning category
   (+0.002…+0.003) but hurts all three txt360 web categories (−0.002…−0.004).
3. **Its parts are coupled**: head-only, body-only, per-depth, per-expert, trimmed (TIES) or sparsified
   (DARE) versions are all worse than the plain uniform step — removing any piece breaks the rest.
4. **Both lineages learned mostly the same thing since #17** (cosine of their updates +0.57…+0.95), so
   interpolating toward #26 (SLERP), B's whole path (orthogonal), or TIES/DARE-TIES averaging all lose.
5. Routing-aware expert merging does reduce the txt360 damage (−0.0017 instead of −0.0028) but gives up
   as much on the other sources — B's useful changes also live in web-heavy experts.

**Update: stacking two sibling steps reached +0.0016 at the fitted optimum (rows 0a/0a') — the surface is flat there, so ≈ +0.0016 is the ceiling of this family.**

**Bottom line (before stacking): the best merge was +0.0010 — about a third of what's needed. Merging public kings cannot
take the crown from #29; the remaining ~+0.0026 has to come from training** (8×B300 with `src/cpt`,
ideally starting from the +0.001 transplant body — see `docs/CPT.md`).

## Independent validation of the best stack
`#29 + 0.2·D26 + 0.2·D25` on a second, independent 4k-sequence sample (**tag:B**): **+0.00162** (LCB +0.00134,
win 71%; code +0.0039, math +0.0019, pretrain +0.0032, sft +0.0027, txt360 −0.0028) vs +0.00144 on sample A.
The gain is real and its per-source pattern is stable. (Delta-checkpoint build vs direct 5-model merge on
the same sequences: +0.001435 vs +0.001403 — consistent; both deltas verified bit-exact = bf16(fp32 diff).)

## Stack-weight model
A quadratic fit through the measured (a, b) points of `#29 + a·(#26−#22) + b·(#25−#23)`
[(0.25,0)=+0.00099, (0.5,0)=−0.00236, (0,0.25)=+0.00045, (0.2,0.2)=+0.00140, (0.2,0.4)=−0.00056] puts the
optimum at **a≈0.165, b≈0.13, worth ≈ +0.0016**. The grid being run now brackets it. The two deltas are
now precomputed as bf16 checkpoints (`/opt/teutonic/deltas/`), so each variant loads in minutes.
Anatomy of the ingredients (`compare_checkpoints.py`): #26's step touches layers 8–44, ~10–13% relative
on upper experts and **29% on lm_head** (hence head/body inseparable); #29's own step touched layers
16–44 at ~2.5%.

## Why this is the ceiling for public-king merges
The eval switched to the current 5-dataset mix on 2026-09-21. Kings crowned since then (#25–#29) are the
only ones trained for it; #27 and #28 are ancestors of #29 (already inside it), so the **only two public
steps #29 does not contain are #25's (from #23) and #26's (from #22)** — exactly the two in the stack.
Older kings' steps were trained for the old mix (finewebedu/automathtext/dclm) and test negative (e.g.
row 9, #22−#20). Expected ceiling for merging public kings: ≈ +0.0014–0.0016.

## Still running / pending
- (done) `t26_upper` = +0.00054: upper half alone works, lower half alone (−0.00176) does not, but together (+0.00100) they beat either half.
- (done) #25 step alone +0.00045, #26 step alone +0.00099, both stacked at 0.2 +0.00140 → the two sibling steps are
  complementary (slightly better than additive). Now probing stack weights (0.2, 0.4), (0.25, 0.3), (0.15, 0.3).
