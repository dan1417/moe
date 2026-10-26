# UPDATE log — Teutonic SN3 mining work

Newest entries on top. Each entry: what was done, why, and what's next.

---

## 2026-09-25 14:20 UTC — Plain CPT plateaus at +0.0024; distillation from MiMo-V2.5-Base (line D1)

- **C1 result:** raw vs start +0.00056 → +0.00050 → +0.00043 at steps 50/100/150; EMA@150 = **+0.00244 vs #30**
  (code +0.0039, math +0.0020, pretrain +0.0039, sft +0.0041, txt360 −0.0002). Same plateau as A2 despite 115× more
  shard diversity → not pool overfitting: after the quick txt360 repair the CE gradient is ~95% noise
  (B_noise ≈ 10–20M tokens). EMA@150 backed up (16.8 GB delta, 7.8% of bf16 values changed).
- **Teacher:** the eval data's top token id (151668) is Qwen3's last token → MiMo's tokenizer == Qwen3's (same vocab,
  merges, specials). Among compatible models, **XiaomiMiMo/MiMo-V2.5-Base** (~310B-A15B, FP8, ~48T pretraining tokens)
  is a scaled-up MiMo-V2 (same classes). Loading it with our code needed: FP8 128×128 block dequant, rope parameters
  rebuilt (assigning `rope_scaling` silently dropped `partial_rotary_factor` → full-width RoPE), and **de-interleaving
  the fused qkv_proj** (Xiaomi stores it as num_kv_heads TP chunks [q_i|k_i|v_i]; SGLang de-interleaves on load;
  without it the model scored 11.9 nats = random).
- **Teacher vs us (2,048 val seqs):** teacher 1.238 vs ours 0.924 — worse on every source. But the per-token
  **probability mixture** (1−w)·p_ours + w·p_teacher: txt360 **+0.109** at w=0.5, sft +0.011, pretrain +0.002, code
  +0.001, math 0 → **+0.025 weighted** (7× the +0.0036 bar). The teacher knows what our synthetic-data specialist
  doesn't (web/QA knowledge), 95% of the value on txt360.
- **Line D1** (`configs/cpt/node/line_d1_kd.yaml`, pm2 `line_d1`, 14:12): from C1 EMA@150; 24k txt360 training
  sequences labelled with the teacher's top-64 log-probs (`train.py --teacher-topk-out`, 74–108k tok/s, 95% of the
  teacher's mass, true token inside top-64 95%); txt360 rows train on CE + 1.0·soft-CE(teacher), other sources on CE.
  KD gradient noise scale ≈ 170k tokens (vs ~10M for labels) → a strong, consistent signal. Target: txt360 from
  −0.0001 to ≈ +0.0051 vs #30 (5% of the mixture's headroom) with the other sources held.

---

## 2026-09-25 12:35 UTC — Pod #2: diverse data, 72k tok/s trainer, line C1 under pm2

Pod #1 (and A2/A3b's exports) is gone; the new 8×B300 node started from scratch at 11:24. King is still #30.
- **Data diversity (new):** the old pools took ~2,200 sequences from each of 270 shards; every shard is a contiguous
  6,144-sequence chunk of one sub-dataset, so a pool was 270 topics (the shard-jackknife SE of a duel is ~4× its iid
  SE for the same reason) — a plausible cause of A2's plateau. `scripts/data/build_diverse_pool.py` reads random
  blocks with HTTP ranges: **train = 1M sequences from 32,189 shards** (blocks of 32), **val = 8,192 sequences from
  8,440 shards** (one each), live source proportions, parity-sample shards excluded. On this val, king #30 = 0.92705
  and the start body = **+0.001764 ± 0.000076** vs #30 (A/B screens said +0.00175/+0.00182 → calibrated).
- **Trainer speed 40k → 72k tok/s** (profiled: FSDP all-gathers were ~27% of CUDA time → bigger micro-batches):
  lm_head+CE in checkpointed 512-position chunks, fused AdamW (foreach allocated a 36 GB sqrt(v) temporary),
  GPU-resident EMA (host EMA cost ~23 s per update), no per-micro-batch GPU sync → micro-batch 16 fits (243 GB peak).
- **Line C1** (`configs/cpt/node/line_c1.yaml`, pm2 `line_c1`, 12:17): 1M-token steps, dense lr 4e-6, routed experts
  ×0.5, EMA 0.99/step, val every 50 steps, EMA export every 150. Step 50: raw **+0.00232 vs #30** (+0.00056 vs start;
  txt360 +0.0014, code +0.0005, pretrain +0.0003), EMA still lagging.
- **Backups:** `backup` (pm2) pushes each new-best export to the private HF repo `touristAI/teutonic-cpt-backup` as a
  bit-exact delta vs the public start body (`scripts/data/ckpt_delta.py restore` rebuilds byte-identical shards);
  `kingwatch` (pm2) logs king changes / queue entries.

---

## 2026-09-25 09:45 UTC — Parity eval confirms training gain; extrapolation dead; line A3b launched

A2 plateaued (EMA vs king +0.00212 / +0.00251 / +0.00245 at steps 100/200/300; raw vs start +0.00056 → +0.00031 →
+0.00009) → stopped at step ~310. Parity eval (validator math, FA4, first 8k of fresh sample A) vs king #30:

| candidate | mu_hat | code | math | pretrain | sft | txt360 |
|---|---|---|---|---|---|---|
| start body (HF merge) | +0.00168 | +0.0032 | +0.0017 | +0.0032 | +0.0027 | −0.0018 |
| **A2 EMA@250** | **+0.00237** | +0.0038 | +0.0020 | +0.0038 | +0.0034 | −0.0002 |
| start + 2·(EMA250 − start) | +0.00165 | +0.0031 | +0.0015 | +0.0028 | +0.0027 | −0.0009 |
| start + 3·(EMA250 − start) | −0.00058 | +0.0010 | +0.0001 | +0.0001 | +0.0004 | −0.0037 |

- Training gain confirmed (+0.0007, mostly txt360 recovery); the trainer's val metric (+0.0025) agrees within 0.00015.
- Extrapolating along the training direction hurts — the EMA sits at the bottom of its own direction (same as kings).
- **Line A3b** (09:41, pm2 `line_a3b`, `configs/cpt/node/line_a3b_from_a2.yaml`): start = A2 EMA@250 (+0.00247 on val),
  fresh `line_b` pool, 1M-token steps, β1 0.95, routed experts (70.1B params) at lr 1e-6, dense (2.3B) at 3e-6.
  First noise-scale reading: **B_noise ≈ 9.2M tokens** (|g_true| 0.048 vs |g_1M| 0.152) → at 0.5–1M tokens/step the
  update direction is 3–4× more noise than signal, which explains A1's damage and A2's plateau.
- Parity eval on this node: `--replicas 8` (whole model per GPU) now clones replicas over NVLink after one disk load
  (46 s instead of 173 s), preloads the FA kernels (transformers' lazy import raced across replica threads), frees
  replicas between models (reference cycles kept 266 GB/GPU → OOM on the next model), batch 16 → **69 seq/s**
  (layer pipeline: 20). A 30k sample now takes ~7 min.

---

## 2026-09-25 09:15 UTC — Line A2 (lr 3e-6) is climbing; finish pipeline + loss-cache fix

| step | raw vs king | EMA vs king | raw vs start | EMA vs start |
|---|---|---|---|---|
| 0 | +0.00184 | – | 0 | – |
| 100 | +0.00240 | +0.00212 | +0.00056 | +0.00028 |
| 200 | +0.00215 | **+0.00251** | +0.00031 | **+0.00066** |

lr 3e-6 fixed the degradation: every source positive at step 200 (EMA: code +0.0040, pretrain +0.0038, sft +0.0035,
math +0.0021, txt360 ±0). Raw jitters ±0.0003 between evals; the EMA extracts the drift at ≈ +0.0004 / 100 steps →
the +0.0036 bar needs roughly step 450–550 plus cooldown. Step-250 EMA export written (`/mnt/ram/runs/line_a2/exports`).
- **Bug fixed — loss-cache identity collision:** `model_identity()` keyed a local checkpoint by index + shard names +
  sizes; every trainer export has the same layout, so raw/EMA/later exports would have reused each other's cached
  losses. Now: index + 3 × 1 MiB content samples per shard (0.07 s; kings keep their R2-digest identity).
- `scripts/train/finish_line.sh <run> [now]`: COOLDOWN → STOP near the end (raw + EMA export) → parity screen both
  (first 8k of A, `--replicas 8`) → full 30k on A and B for the better → `verify_submission.py` → `runs/<run>/FINISH.json`
  with the submit rule (mu ≥ 0.0036, robust LCB > 0.003, no source < −0.001 on both samples).
- `scripts/train/watch_run.py <run>`: history-file watcher (evals, exports, pm2 state) — the log-grep watchers dropped events.
- Trainer (next launches): per-module LR groups (`optimizer.groups: [{pattern, lr_mult}]`), gradient-noise-scale
  logging (`noise_every`), final export always includes raw, tok/s fix after evals. Prepared
  `configs/cpt/node/line_a3_1m_split.yaml` (1M-token steps, experts at ⅓ lr) in case A2 stalls.

---

## 2026-09-25 08:05 UTC — Line A at lr 1e-5 degraded the body; relaunched as line A2 at lr 3e-6

Held-out (2048 seqs, paired vs king #30 / vs the start body):

| step | raw vs start | raw vs king | EMA vs start | EMA vs king |
|---|---|---|---|---|
| 0 | 0 | +0.00186 | – | – |
| 100 | −0.00258 | −0.00072 | +0.00041 | +0.00227 |
| 200 | **−0.00414** | −0.00229 | **−0.00048** | +0.00138 |

Every source got worse (pretrain −0.0061, code −0.0040, sft −0.0036, math −0.0033, txt360 −0.0026 at step 200);
train loss flat (0.90–1.0), grad norm 0.22 throughout — no instability, just a converged model being pushed off its
optimum by Adam's ≈lr-sized per-element steps from 524k-token batches whose gradient is mostly noise (per-element
grad RMS ≈ 8e-7 ≫ eps, so the normalized step is ~lr regardless of signal). Consistent with the merge finding that
0.2× a king step in weight space already costs −0.0005. **Stopped at step ~230 (no export; the EMA was below the
start).** History kept in `runs/line_a_lr1e-5_failed`.
- **Line A2** (`configs/cpt/node/line_a2_lr3e-6.yaml`, pm2 `line_a2`, 08:03 UTC): identical except lr **3e-6**, warmup
  100 → noise cost should drop ~10× (∝ lr²); if the drift is real the EMA must beat the start by step 200–300.
  Rule: raw vs start must stay ≥ −0.001 and EMA vs start must be > 0 and rising; otherwise next levers are a larger
  batch (accum 8 = 1M tokens/step), β2 0.99, or training from #30u instead of the merged body.

---

## 2026-09-25 07:05 UTC — Gate passed; line A training from the merged body under pm2 (40k tok/s)

- **Gate (`configs/cpt/node/gate_full.yaml`, king #30, all params, mb 4×8):** model sharded-loaded from tmpfs in 5 s;
  step-0 export round-trip **bit-identical on all 34,234 tensors** (220.6 GB written in 2 min); **19.7k tok/s**
  (26.5 s / 524k-token step) with peak 275.7 GB/GPU — full-parameter fits, but with no headroom. EMA pinned OK.
  The king's per-sequence val losses (2048 seqs, mean 0.926929) are saved as `pools/val/king30_losses.npy`.
- **Start body** `touristAI/teutonic-II-110B-merge-k30u` (= `u30w3_e1.15_a0.7_h0.7`) fetched to
  `/dev/shm/teutonic/start` (~1 GB/s), `verify_submission.py` **READY** (15/15 checks; digest `2fbf93c8…`).
  Trainer step-0 check on the val pool: **start body vs king #30 = +0.001858 ± 0.00010** (win 74 %; code +0.0034,
  pretrain +0.0032, sft +0.0031, math +0.0018, txt360 −0.0013) — reproduces the merge campaign's +0.0018.
- **Line A launched 06:58 UTC** (`pm2` process `line_a`, `configs/cpt/node/line_a_upper_mb8.yaml`): layers 16–44 +
  lm_head + final norm trainable (**72.4B** params; layers 0–15, embeddings, routers frozen = 37.9B), AdamW lr 1e-5,
  warmup 50, EMA-100, micro-batch 8 × accum 4 = 524k tokens/step, eval every 100 steps, EMA export every 250 steps
  (`/mnt/ram/runs/line_a/exports`, keep 2), cap 2000 steps ≈ 1.05B tokens. **40.1k tok/s (13.1 s/step), peak 200 GB/GPU.**
  First EMA export ≈ 08:00 UTC (step 250); `python scripts/train/progress.py` summarizes; `pm2 logs line_a`.
- Parity eval on this node: the 8-stage layer pipeline reaches only ~20 seq/s (49 min per 30k sample) — too slow
  for the race. Added `--replicas N` (`parity_eval.py`): N full model copies (one 110B model fits one B300) score
  disjoint slices concurrently; `screen_export.sh` uses 8. Untested on GPU yet (GPUs are training) — first use at
  the first export pause. King #30 baseline: 6k of sample A cached before the GPUs were handed to training.
- Ops lessons: `pkill -f <pattern>` from the tool shell kills the tool shell itself (pattern in its own cmdline) — use
  PIDs; killing a nohup wrapper leaves the python child alive (it kept 30 GB/GPU) — kill the child PID.

---

## 2026-09-25 06:50 UTC — Code pushed, merged model on HuggingFace

- Code: `github.com/wilfrid51/MoE` main @ `8ed5eb0` (+ this docs update).
- Model: **https://huggingface.co/touristAI/teutonic-II-110B-merge-k30u** (public; 42 shards, 220.6 GB, all
  SHA-256 verified against the local write; `verify_submission.py` READY 15/15; `merge_recipe.json` + README inside).
  First attempt under `scoutminer/…` stalled on HF's 128-commits/hour limit (upload_large_folder commits in many
  batches); the second account used a single-commit `upload_folder` and finished in 4.6 min since the blobs were
  already server-side. The partial `scoutminer` repo can be deleted.
- B300 runbook: `docs/CPT.md` → Handoff option B now downloads from HF.
## 2026-09-25 06:30 UTC — 8×B300 pod live: king #30 in RAM, env verified, node trainer + configs ready

Node: 8×B300 SXM6 (275 GB, NVLink NV18), 240 CPUs, **2 TB RAM but only ~200 GiB free disk** on `/`
(shared host disk; `/root` is a slow 79 GB gocryptfs). The 220 GB king does not fit the disk, so large
checkpoints live on RAM-backed tmpfs: `/dev/shm/teutonic` (602 GB: king #30 + start body) and `/mnt/ram`
(760 GB tmpfs mounted by us: exports). Pools/venv/eval cache/logs on `/workspace/teutonic`. `source
/workspace/teutonic/env.sh` sets everything (venv, `TEUTONIC_EVAL_CACHE`, `TEUTONIC_MODELS_DIRS`, FA4 cache).
- King #30 (`1a9d7c9d…`) downloaded to `/dev/shm/teutonic/king30` at ~430 MB/s (32 streams, ~9 min), sha-verified;
  linked as `reign:30` for the parity eval. Re-downloading is cheap, so the king can be evicted if RAM gets tight.
- `.venv-cpt` on the system torch 2.12.0+cu130: validator transformers fork 5.15.1, `flash-attn-4==4.0.0b28`
  (pip *does* install it although `pip index` says otherwise), validator package @5479810. Verified on a B300:
  FA4 fwd+bwd with sinks + asymmetric V (SWA 128 and full), `torch._grouped_mm` bit-exact vs per-expert GEMMs.
- Fresh samples A/B rebuilt (same ids as on the eval box: `e4a9f831-16160cb5…`, `e4a9f831-1792f11a…`), val
  pool (2999 seqs, excludes A/B shards), train pools line_a/line_b (10 rounds × scale 2 ≈ 600k seqs ≈ 1.2B tokens
  each, excluding A/B/val shards) building.
- Trainer (`src/cpt/train.py`) upgraded for this node: `data.king_val_losses` (per-seq losses of the *real* king
  on the val pool → `vs_king` is vs #30 even when starting from the merged body; `vs_start` reported too);
  `--score-model/--score-out` score-only mode; `export_dir` + `export_keep` (RAM-disk retention; `KEEP` file pins);
  run-time triggers `COOLDOWN` / `STOP` / `EXPORT` files in the run dir (pull the LR cooldown forward when the
  parity eval says we are close); `--max-steps`; pinned-memory fallback for the host EMA; GPU memory in logs.
  `build_train_pool.py --exclude-pool` keeps val shards out of the train pools.
- Configs `configs/cpt/node/`: `gate_full.yaml` (6-step full-parameter gate from the king: memory, tok/s,
  export round-trip, king val losses), `line_a_upper.yaml` (layers 16–44 + head, the kings' own recipe),
  `line_b_full.yaml`. Scripts: `scripts/train/node_train.sh`, `scripts/data/fetch_hf_checkpoint.sh <repo>`
  (HF → `/dev/shm/teutonic/start` + `verify_submission.py` vs king), `scripts/eval/screen_export.sh <dir> [limit]`.
- No DCP checkpoints here (1.3 TB); resume = restart from the last EMA export.

---

## 2026-09-25 06:10 UTC — Decision: rent 8×B300, train from the merged body → handoff prepared

- `configs/merge/b300_start_body.json`: the +0.0018 leader expressed in public kings (#30u, #26, #22, #25, #23)
  with the module multipliers; verified equal to the delta-based recipe up to bf16 delta rounding (≤2e-5).
- `configs/merge/k30_hidden_perm.json` + `scripts/merge/unpermute_checkpoint.py` (accepts JSON): rebuild #30u
  anywhere. `scripts/train/launch_cpt.sh` gained `START_UNPERMUTE=<digest>:<perm.json>`.
- Runbook: `docs/CPT.md` → "Handoff: 8×B300 run from the +0.0018 merged body" (rebuild-from-downloads or rsync
  the body the finalizer writes to `/root/submit/u30w3_e1.15_a0.7_h0.7`).
- Note: the repo has extensive uncommitted work (src/parity, src/cpt, scripts, configs, docs) — commit or copy
  the working tree before cloning onto the node.

## 2026-09-25 05:50 UTC — Merge campaign vs #30 converged: best ≈ +0.0018 (bar +0.0036)

- Plateau mapped: (1.15/0.7/0.7) +0.00175, (1.0/0.7/0.7) +0.00172, (1.15/0.7/0.85) +0.00172, (1.15/0.8/0.8)
  +0.00171, (1.15/0.85/0.7) +0.00170, whole-stack ×1.1 ≈ +0.00173 — all within noise. Leader on sample B: +0.00182.
- Every method family has now been tried against #30 (see MERGE.md "Conclusion vs #30"). Remaining queued tail
  (routing-aware ×2, ancestor-step ±, token diagnostic) is unlikely to add > 1e-4; the finalizer then writes the
  best merge to `/root/submit/<name>` and runs the validator checks.
- Bottom line: **merging reaches +0.0018 of the +0.0036 needed**. Submitting it would be rejected. It is, however,
  a starting point that halves what training must add.

## 2026-09-25 01:50 UTC — First results vs the un-permuted #30 (#30u)

- `#30u + 0.17·D26 + 0.13·D25` = **+0.00142** (LCB +0.00122, win 70%; code +0.0029, pretrain +0.0029,
  sft +0.0025, math +0.0015, txt360 −0.0022). The stack transfers onto #30 as predicted; gap to the bar
  ≈ 0.0022. Single-delta variants, wave 1 (extrapolation / orthogonal-to-step / agreement), the token
  diagnostic and routing stats are running; wave 2 follows. MERGE.md's #30 table auto-refreshes every 10 min.
- Equivalence proof: #30u vs the permuted #30 on the same 4,000 sequences → mean loss diff −7e-6, sd 1.7e-3
  (= run-to-run fp noise). The un-permuted copy is the same function.
- D25-only (`#30u + 0.25·D25`) = +0.00040 (same as vs #29); D26-only (0.25) = +0.00081; (0.1, 0.1) = +0.00131;
  (0, 0.15) = +0.00068 → the two deltas are ~additive on #30u; optimum still ≈ (0.17, 0.13).
- Wave 1: extrapolating #30's own step +0.2 → −0.00048 (all sources worse); partial undo −0.2 → +0.00007
  (nothing). Same as with #29: each king sits at the optimum of its own update.
- Orthogonal-to-step (deltas minus their projection on #30−#29) = +0.00135 (plain stack +0.00142): no gain.
- #30's step profile = #29's: layers 0–15 untouched, 16–44 ≈3% (attention/experts), shared experts 5–13%.
- Sibling sign-agreement weighting = +0.00102 (worse; txt360 −0.0041). Wave 1 done: nothing beats the plain stack.
- Wave 2 structure: experts ×1.3 = +0.00156 / ×0.7 = +0.00093; attention ×0.7 = **+0.00157** / ×1.3 = +0.00118;
  head+shared ×0.7 = +0.00155 / ×1.3 = +0.00113 → the deltas' expert part wants more weight, attention/head
  parts less. Experts ×1.6 = +0.00133 (worse than ×1.3) → the expert axis peaks near ×1.3; ×2.0 jobs dropped.
  **Combination (experts 1.3, attention 0.7, head 0.7) = +0.00165 — new best** (sub-additive: singles sum to +0.0018).
  **(1.15 / 0.7 / 0.7) = +0.00175 — new best; validated on sample B: +0.00182** (LCB +0.00159); (1.3/0.6/0.6) = +0.00155. Now: sample-B validation of the
  leader + coordinate probes (1.0/0.7/0.7), (1.15/0.8/0.8), (1.15/0.7/0.85), (1.15/0.85/0.7), whole-stack ×1.1.
- **Wave 2: routed experts ×1.3 = +0.00156 — new best** (experts ×0.7 = +0.00093): the expert part of the
  sibling steps wants more weight. Queued: experts ×1.6, ×2.0, ×1.6 with attention ×0.7.
- Wave 2 running (delta-TIES top-50% = +0.00127, below the plain stack; breadcrumbs, DARE, per-module ×0.7/×1.3,
  per-depth to come), then wave 3a (routing-aware expert scaling from #30u's routing stats), the old-mix
  ancestor-step tests (±0.1·(#23−#21)), the token diagnostic, wave 3b (combined winning directions), finalizer.

## 2026-09-25 01:30 UTC — 🔍 King #30 is a *permuted* model (anti-merge obfuscation) — undoing it

`|#30 − #29| / |#29| ≈ 1.41 = √2` for **every** tensor, including embeddings and router weights that no
king in this lineage ever changed → element-wise uncorrelated, yet the same loss. Test: each of the 3072
embedding columns of #30 matches exactly one column of #29 with cosine **1.0000** (no sign flips) →
**#30 = (#29-lineage model) with the hidden dimension permuted**: function-preserving, but it breaks
weight diffs and merges against #29-basis checkpoints (our deltas). The owner (5Hmim6, also owner of #29/#26)
presumably did this to block exactly this kind of merging (and it also defeats the validator's
same-weights digest checks).
Diagnosis after undoing the hidden permutation: embeddings, routers, router biases, layer-0 MLP and
lower-layer attention are **bit-identical** to #29; upper layers changed 2–4%, shared experts 11%, lm_head 9%
(= #29's own recipe again); experts / heads / expert-intermediate dims are NOT permuted (identity matches).
So #30 = perm(#29 + normal step). `scripts/merge/unpermute_checkpoint.py` writes #30u (exact re-indexing);
`swap_k30u.sh` installs it as `reign:30` on the fast disk once the permuted baseline finishes, then queues:
equivalence check (#30u must score like #30), re-screen `#30u + a·D26 + b·D25`, wave-1 methods, token
diagnostic, routing stats.
#30's true step (D30 = #30u − #29) vs our deltas: cos(D30, D26) = 0.07–0.16, cos(D30, D25) = 0.06–0.2 on
typical tensors (0.48/0.49 only on the last attention layer) → **the sibling steps are not inside #30**; the
stack should transfer (≈ +0.0016 expected vs #30u), leaving ≈ +0.002 still to find.
Plan: (1) check whether experts / attention heads / expert-intermediate dims were permuted too (per layer);
(2) write `#30u` = #30 mapped back into #29's basis (a permutation is exact, no rounding); (3) verify the
true step `#30u − #29` is a normal ~1–3% update; (4) redo the D26/D25 re-screen + method sweep on #30u.
The permuted-basis jobs were dropped from the queue (their results would be meaningless).

## 2026-09-25 01:00 UTC — Campaign: beat #30 with merges only ("try every possible merge method")

Ingredients are fixed (two public steps outside #30's ancestry: D25 = #25−#23 and D26 = #26−#22, the latter
possibly already inside #30 since #30's owner also owns #26). Method sweep vs #30, in waves:
1. handler re-screen: `#30 + a·D26 + b·D25` at (0.17,0.13), (0,0.25), (0.25,0), (0.1,0.1), (0,0.15) — tells which delta still adds.
2. wave 1 (queued): extrapolate / partially undo #30's own step (±0.2); **orthogonal-to-step** (new mode:
   each delta minus its projection on #30−#29, i.e. only what the siblings add beyond what #30 just learned);
   sibling sign-agreement weighting.
3. wave 2 (after results): TIES / DARE / breadcrumbs (new: drop largest+smallest entries) on the winning
   delta(s), per-module and per-depth scaling, routing-aware expert version, then a coordinate search over
   grouped coefficients; finally full-30k confirmation on two samples of the best.
Pending #29-only jobs pruned from the queue. Odds: the same ingredients gave +0.0016 vs #29 with a flat
optimum; beating #30 by +0.0036 with them alone would need a method that extracts >2× more — I'll report
each wave's numbers honestly.

## 2026-09-25 00:40 UTC — ⚠️ NEW KING #30 crowned (same miner as #29 + #26); re-targeting

- Duel finished 00:29: challenger accepted, mu 0.003249 / LCB 0.003113 over #29, gains on **every** source
  (txt360 +0.0049, pretrain +0.0036, sft +0.0032, code +0.0029, math +0.0018). Owner: coldkey 5Hmim6 —
  who also owns #29 **and** #26 (lineage B). They may have folded their own B step into #30.
- Consequence: every merge score so far is relative to #29. Our best (+0.0016 vs #29) is ≈ −0.0016 vs #30.
  **Do not submit `/root/submit/*` written for #29** (finalizer cancelled before writing).
- `new_king_handler.sh` (armed; fires when the dashboard flips, polled every 5 min): #29 → slow disk, #30 →
  fast disk (retries until the public bucket has it), then **first in the queue**: #30 baseline on A/B and
  re-screens `#30 + a·D26 + b·D25` at (0.17,0.13), (0,0.25), (0.25,0), (0.1,0.1), (0,0.15) — single-delta
  variants tell whether D26 is already inside #30.
- Merge ceiling vs #29 (for the record): flat at ≈ +0.0016 [(0.17,0.13) +0.00160, (0.2,0.15) +0.00159,
  (0.18,0.10) +0.00154, (0.15,0.20) +0.00145].

## 2026-09-25 00:15 UTC — Stack optimum found: +0.0016 (confirmed on 2 samples); finalization running

- Delta checkpoints built (D26, D25 verified bit-exact); stacked variants now load in ~3 min (was 15–20).
- `#29 + 0.2·D26 + 0.2·D25` validated on independent sample B: **+0.00162** (A: +0.00144).
- Grid: (0.17, 0.13) = **+0.00160**, (0.2, 0.15) = +0.00159 → flat optimum ≈ **+0.0016** (txt360 −0.002).
- Rest of grid + sibling-agreement variants running; then `finalize_best.sh` confirms the best on 12k
  sequences of A and B, writes `/root/submit/<best>`, runs `verify_submission.py`, re-scores the written dir.
- Your call (00:20): **submit the current best from this box, and keep merging.** Preparing the submission:
  `finalize_best.sh` writes the best merge to `/root/submit/<name>` (CPU, ~12 min), runs every validator
  check (`verify_submission.py`), then re-scores the written weights and confirms on 12k sequences of A+B.
  ⚠️ Expected verdict if submitted now: **rejected** (mu ≈ 0.0016 < delta 0.003); a rejection consumes the
  hotkey + its ~1 TAO registration. Wallet/registration/`ready` are yours (see `docs/SUBMIT.md`).
- 00:20: a challenger is being evaluated live (started 23:47, 55% done, provisional mu 0.00324 / LCB 0.00306
  — on the line). If it is crowned, `new_king_handler.sh` moves #29 to the slow disk, downloads #30 to the
  fast disk, scores it on samples A/B and re-screens the stack recipes against it automatically.
- Merge search continues: 8 per-module / per-depth scalings of the optimum queued; king #28 re-downloading
  for the "partial undo of #29's own step" family (predicted +1e-4, but free to combine); a king-change
  monitor writes `/opt/teutonic/logs/KING_CHANGED.alert` if a new king appears.

## 2026-09-24 23:15 UTC — "Keep merging": faster loop + closing in on the stack optimum

- (0.25, 0.3) stack = +0.0005, matching the quadratic fit → the family's optimum is ≈ +0.0016 at
  a≈0.17, b≈0.13 (both deltas, not either alone). Grid around it queued: (0.2,0.2) [also on
  independent sample B], (0.17,0.13), (0.2,0.15), (0.15,0.2), (0.18,0.1), (0.15,0.15), (0.22,0.12), (0.12,0.12).
- Speed: the two sibling steps are being precomputed as bf16 delta checkpoints on the fast disk
  (`/opt/teutonic/deltas/d26_22`, `d25_23`, built on CPU with the same combine code) → each stacked
  variant then reads 3 fast-disk checkpoints instead of 5 (2 on the slow encrypted disk): ~4 min loads
  instead of 15–20. Dead-end kings #17/#20/#27 deleted (re-downloadable); #26 moved to `/root`.
- New mode `delta_agree`: per element, where the two independent miners' steps agree in sign use one
  pair of coefficients, elsewhere another (agreement = evidence; plus a reversed control). Queued after the grid.
- King still #29; no challenger in the validator queue.
- **Honest ceiling: ~+0.0016 from public-king merges vs +0.0036 needed.** The rest must come from
  training; the B300 stack is ready and can start from the merged body.

## 2026-09-24 21:25 UTC — All merge results consolidated in `MERGE.md`

`MERGE.md` (repo root) lists every merge tried against king #29: what it does, its score (mu_hat vs
the king on the same fresh validator-distribution sequences), LCB, win rate and per-dataset breakdown,
plus the ingredients, the non-merge levers and the lessons. Best: `29 + 0.25·(26−22)` = **+0.0010**
(need ≈ +0.0036). Newest: B's step on layers 29–44 + head only = +0.00054 (lower half alone −0.00176).
**Update 22:15 — stacking sibling steps compounds:** #26 step alone +0.00099, #25 step (5DALwjE4
branch, sibling of #24) alone +0.00045, **both stacked at 0.2 each = +0.00140** (new best; code +0.0036,
pretrain +0.0032, math +0.0018, txt360 −0.0035). Probing stack weights now: (0.2, 0.4), (0.25, 0.3),
(0.15, 0.3). Loader now reads safetensors with large `preadv` calls (bit-exact vs safetensors) to
speed up merges that touch the slow `/root` disk.

---

## 2026-09-24 19:00 UTC — More merge methods: TIES, DARE, DARE-TIES, SLERP + MoE routing-aware expert merge

You asked whether only linear merges were tried — yes, until now (task arithmetic/extrapolation,
soups, per-module overrides, orthogonal projection). Added as combine modes in `_fill` (so eval ==
shipped weights) and unit-tested (each reduces to linear in its degenerate case; DARE masks are
deterministic per tensor/chunk):
- `ties` — trim delta to top-density |x| per tensor, scale β
- `dare` — Bernoulli drop p + 1/(1−p) rescale
- `dare_ties` — mergekit semantics: base + λ·TIES(sign election, disjoint normalized mean) over DARE'd deltas
- `slerp` — per-tensor spherical interpolation
- **routing-aware expert merge (MoE-specific)**: `parity_eval.py routing` counts which experts king #29
  routes each dataset's tokens to (held-out sample H, 1500 seqs). Experts are strongly specialized:
  web(txt360) affinity quantiles 0.31/0.66/1.11/1.7/2.5 — ~2000 of 11,264 experts are web
  specialists (>2×), ~2000 barely see web (<0.5×). Since lineage B helps reasoning/code/math but hurts
  web, `make_expert_recipe.py` transplants B's step only into web-light experts (routers are frozen in
  both lineages, so expert identities match). Also added: per-category results (all 3 txt360 categories
  lose, every reasoning/code/math/planning category gains under the plain transplant).

Other results since last entry: orthogonal transplant `29 + 0.25·⊥(26−17)` = −0.0017 (all sources worse).

Expert-merge results (same first 4k sequences of sample A, vs king #29):

| candidate | mu_hat | txt360 |
|---|---|---|
| plain transplant β=0.25 (reference) | **+0.00099** | −0.0028 |
| web-light experts (aff ≤1.0) β0.3, web experts 0, rest 0.15 | +0.00084 | −0.0017 |
| aff ≤0.5 β0.5, web experts 0.1, rest 0.2 | +0.00063 | −0.0019 |
| aff ≤0.8 β0.35, web experts 0, rest 0.25 | −0.00011 | −0.0035 |

Routing-awareness cuts the txt360 damage but loses as much elsewhere — lineage B's gains are spread
over web-heavy experts too. Ceiling along lineage B stays ≈ +0.001.

Classic methods (same first 4k sequences, vs king #29):

| method | mu_hat | txt360 | note |
|---|---|---|---|
| plain transplant `29+0.25·(26−22)` (reference) | **+0.00099** | −0.0028 | still best |
| TIES (top-30% of 26−22, β 0.35) | +0.00021 | −0.0045 | the big entries carry the web conflict |
| DARE (drop 50%, ×2, β 0.25) | −0.00074 | −0.0055 | drop+rescale noise hurts on large deltas |
| SLERP(29, 26, t=0.1) | −0.00041 | −0.0003 | moving toward #26 costs everywhere |
| DARE-TIES (mergekit, base #17; #29 w1 d1.0 + #26 w0.3 d0.5, normalized) | **−0.0083** | −0.0181 | normalized average destroys #29's precision |

**Conclusion of the merge sweep:** no merge of public kings gets near +0.0036; the ceiling is ≈ +0.001
(β≈0.2–0.25 lineage-B step transplant). Why: #29 sits at the optimum of its own update; both lineages
learned mostly the same thing since #17; B's unique knowledge helps reasoning/code/math but conflicts
with #29 on web text, and all variants that avoid the conflict also lose the gain.

**Recommendation:** rent the 8×B300 now rather than after a crown — merges won't produce the crown.
Start from the +0.001 transplant body (`START_RECIPE=configs/merge/transplant26_b0.25.json bash
scripts/train/launch_cpt.sh`), 2 lines × ~0.6B tokens on the exact eval mix, EMA, soup the lines,
score here with the parity eval, submit per `docs/SUBMIT.md`. Head tuning (lm_head only) on the
transplant body is running here meanwhile (~75 min) as the last cheap lever.

Head tuning (lm_head only, on the transplant body, lr 3e-6, validated on held-out sample H):
segment 1 (8k seqs, 125 steps) → gain over the body's own head −0.00003 ± 0.00004, head moved 0.35%;
segment 2 (16k, 250 steps) → −0.00004 ± 0.00004, head moved 0.5%. Online train CE dips but held-out
doesn't → the head is already at its optimum for the body (as calibration suggested). **Stopped** to
free the GPUs; head tuning is not a lever here.

Sibling-step results so far: `29 + 0.25·(22−20)` (lineage B's earlier step) = **−0.0016** (every source
worse, txt360 −0.0059) → only B's *last* step (26−22) carries something #29 lacks.

Also queued after head tuning (models downloading to `/root`): **stacked sibling-step transplants** —
`29 + 0.25·(25−23)` (the 5DALwjE4 branch, sibling of #24 in #29's ancestry), `29 + 0.25·(22−20)`
(lineage B's earlier step), and `29 + 0.2·(26−22) + 0.2·(25−23)`. Each sibling step carries knowledge
#29 does not contain; if they stack, this is the last merge family with real upside.

B300 prep: `configs/cpt/b300_upper.yaml` mirrors king #29's own recipe (train layers 16–44 + lm_head,
freeze 0–15/embeddings/routers; ≈195 GB/GPU → safe on B300, not B200). `launch_cpt.sh` accepts
`START_RECIPE=` (downloads the recipe's kings and writes the body with `write_merge.py`) and
`BASE_CONFIG=`; its embedded steps were dry-run here.

---

## 2026-09-24 18:00 UTC — Merge scoreboard (vs king #29, 8k fresh seqs, need ≈ +0.0036)

| candidate | mu_hat | notes |
|---|---|---|
| `29 + 0.25·(26−22)` (lineage-B transplant) | **+0.0010** | best; helps 4 sources, txt360 −0.0027 |
| same without lm_head | −0.0003 | txt360 −0.0041: B's head change *compensates* B's body change |
| same, lm_head only | ≈ −0.0019 (partial) | B's head only helps together with B's body |
| `29 + 0.5·(29−28)` | −0.0020 | #29 is at the optimum of its own update |
| `29 + 0.5·(26−22)` | −0.0024 | overshoot (txt360 −0.0127) |
| logit temperature | +0.000001 | already calibrated |

No merge beats #29 yet. Head/body changes of lineage B are coupled, so per-module splitting was
dropped.

Lineage geometry since their common ancestor #17 (sampled tensors): A (29−17) and B (26−17) moved
**mostly the same way** — cosine +0.57…+0.95, sign agreement 67–84%, deltas 16–49% of weight norm
(shared experts ~80%). So adding B's path mostly extrapolates #29 along its own path (known to hurt);
the sign-agree idea would keep exactly that part, so it was dropped. Replaced by an **orthogonal
transplant** (`mode: orthogonal`): add only the component of (26−17) orthogonal to (29−17), per tensor
(fp32 math, fp64-accumulated projection) = what B learned that A didn't; plus plain soups (1−β)·#29+β·#26.
(First attempt OOM'd: fp64 temporaries × 32 loader threads on one GPU → merges now run in 128 MB fp32
chunks with ≤3 in flight per GPU.) Head-only transplant final: −0.0019 (worse on every source).

Also added: `docs/SUBMIT.md` (one-shot submission runbook: decide on 2 full samples → write_merge →
verify_submission → re-score the written dir → register Ed25519 hotkey → upload → ready), and the B300
trainer now logs **paired mu_hat vs the king** on the held-out set at every eval (validator's statistic). Next (queued): sign-agree / common-ancestor transplants with king #17 (downloaded), then
head tuning on the best body. If these stay < +0.003, merges alone won't take the crown and the
8×B300 run (`src/cpt`, ready) is the way.

---

## 2026-09-24 17:20 UTC — First positive merge: lineage-B transplant (+0.0010), refining it

`29 + 0.25·(26 − 22)` on 8k fresh sequences vs king #29: **mu_hat +0.000996**, LCB +0.00084,
win rate 67%. β=0.5: **−0.00236** (txt360 −0.0127, code +0.0014, math +0.0010). Quadratic fit in β
→ best global β≈0.18 worth only ≈+0.0012. The four non-txt360 sources want more of lineage B;
txt360 wants none → next: per-module variants (running), and a **sign-agree transplant**
(new combine mode in `_fill`): `29 + β·(26−17)` only where lineage A (29−17) and lineage B (26−17)
moved in the same direction since their common ancestor #17 (downloading) — conflicts are skipped.

| code | math | pretrain | sft | txt360 |
|---|---|---|---|---|
| +0.0026 | +0.0015 | +0.0023 | +0.0022 | **−0.0027** |

Lineage B's last update carries knowledge #29 lacks on 4 of 5 sources, but fights #29 on txt360
(weighted: +0.0016 from the four, −0.0006 from txt360). Not enough yet (need ≥ ~0.0036), so now:
β=0.5 (running), then per-module variants (no-head / head-only / experts / attention / shared experts)
to keep the helpful parts and drop the txt360-damaging part; then head tuning on the best body.

Ops: 3-model merge loads took 722 s because king #22 sits on the slow encrypted `/root` disk → king #28
(not needed anymore, re-downloadable) was deleted from the fast disk and #22 moved there (now ~530 s:
660 GB of reads exceed the page cache). A merge load then crashed scoring with
`CUBLAS_STATUS_ALLOC_FAILED` (merge temporaries left cached in PyTorch's allocator) → `load_model` now
`empty_cache()`s every GPU after loading. Queue is controlled by `scripts/eval/queue_ctl.sh
start|stop|restart|pause|resume|status` (PID file; pattern-based kills had hit my own shell).
New `parity_eval.py leaderboard` ranks all cached candidates against the king from the loss cache.

---

## 2026-09-24 16:50 UTC — First merge results: extrapolating #29 loses; calibration is a no-op

| candidate (8k fresh seqs, sample A, vs king #29) | mu_hat | win rate | note |
|---|---|---|---|
| `29 + 0.5·(29−28)` (extrapolate #29's update) | **−0.00198** | 21% | worse on all 5 sources |
| logit temperature (calibrate) | +0.0000012 predicted | – | king is perfectly calibrated (c* = 1.0012) |

A parabola through #28 / #29 / this point puts the optimum at ≈0.84 of #29's update (predicted gain
only ~1e-4): **#29 already sits at the bottom of its own update direction**. So I cut the rest of
that family (α 0.25/1.0/1.5, soup, per-module variants) — ~2 h of GPU saved. Kept: lineage-B
transplant `29 + β(26−22)` (new information; running), then **head tuning on #29** (lm_head only,
frozen body, 48k seqs ≈ 98M tokens, lr 3e-6, validated on held-out sample H), then the long shots
(re-apply #28's line-merge step, 0.9·#29+0.1·#26).

Honest assessment: public-king merges alone are unlikely to clear +0.0036 over #29 (every public
king is either an ancestor already folded into #29 or the losing lineage). Head tuning is the best
cheap lever here; the reliable path is a short 8×B300 run with `src/cpt` (ready, smoke-tested).

Fixes today: queue runner treated `--attn …` jobs as never-done (`ugrep` parses a leading `--` as an
option → fixed with `grep -- pattern`); pool builder retries shard download timeouts.

---

## 2026-09-24 16:25 UTC — Merge screening running; two extra levers added (calibration, head tuning)

All five merge ingredients are local: kings #29, #28, #26 (`/opt/teutonic/models`), #22, #27
(`/root/teutonic_models`). Queue order (8k-sequence screen on fresh sample A, vs king #29):
1. `calibrate` — one pass estimating the optimal logit temperature for #29 (lm_head scale c*:
   dCE/dc = E_p[z] − z_y, d²CE/dc² = Var_p[z]); a weights-only change the locked architecture allows.
2. extrapolation α=0.5/1.0 + soup sanity → lm_head split variants → lineage-B transplant →
   α=0.25/1.5, experts-only, attention+shared-only → re-apply #28's update (28−27) → 0.9·#29+0.1·#26.

New: **head tuning** (`parity_eval.py headtune`, `src/parity/headtune.py`). With the body frozen,
fitting only `lm_head` is convex, so it can't damage the model; king #29's own update moved lm_head
8.6%, so it's a real lever. Runs on this box through the eval pipeline (~16 seq/s ≈ 120M tokens/h);
the last GPU keeps an fp32 master head + AdamW. Online loss is scored before each update (honest),
validation uses the validator's exact CE on a separate held-out sample (tag:H, never used for
screening). Output: best head + a recipe (`replacements: {lm_head.weight: …}`) that composes with any
merged body; `write_merge.py` and scoring honour replacements.

---

## 2026-09-24 16:10 UTC — ✅ Local eval reproduces the validator's verdict; ✅ B300 trainer smoke-tested

### Replay of the latest duel (#29 vs #28) on the validator's exact 30,000 sequences
| | ours (`fa4sim`) | validator published |
|---|---|---|
| avg king (#28) loss | 0.935177 | 0.935162 |
| avg challenger (#29) loss | 0.931937 | 0.931928 |
| **mu_hat** | **0.003240** | **0.003234** |
| **LCB** | **0.003104** | **0.003101** |
| accepted | yes | yes |

Per-source mu_hat matches within ~3e-5 (code .002625/.002597, math .001822/.001803, pretrain
.003105/.003079, txt360 .005847/.005893; sft has only 300 seqs). **The local eval is the validator.**

Risk insight: the shard-level jackknife SE is 2.0e-4 vs the bootstrap's 4.5e-5 → a different validator
draw can move mu_hat by ~±2e-4. **Target mu_hat ≥ ~0.0036 on our own samples** before spending a hotkey
(#29 itself only had 0.00324 — it was lucky-safe, robust LCB 0.0026).

### Trainer smoke test (8×5090, 2-layer slice of the real king, `configs/cpt/smoke_5090.yaml`)
- FSDP2 sharded load from safetensors in ~1 s; **export round-trip bit-exact** (788/788 tensors identical
  to the king before training).
- Loss 11.47 → 10.19 in 12 steps, val 11.61 → 10.16, EMA lags as expected, LR warmup/decay correct.
- Frozen routers stay bit-identical after training; DCP checkpoint → **resume** at step 12 works.
- Fixes found by the test: transformers-5 mask-API compat patch, val pool shuffling.
- 32 GB cards can't hold full-size fp32 AdamW states (expected); the real run is for 8×B300.

### Now running (unattended queue)
Fresh samples A/B → king #29 baseline → merge screening on 8k sequences: extrapolations of #29's update
(α 0.25/0.5/1.0/1.5, experts-only, attention+shared-only, with/without lm_head), soup sanity check,
lineage-B transplant (26−22) once #22 finishes downloading.

---

## 2026-09-24 15:20 UTC — Parity confirmed on the validator's exact sample; trainer + merge tooling written

### Parity: our score for king #29 on duel 81bcc8fa's exact 30k sequences
| | ours (fa4sim) | validator (published) |
|---|---|---|
| king #29 mean per-seq CE | **0.931937** | **0.931928** (avg_challenger_loss) |

Difference **9e-6** — inside our run-to-run noise (~1e-5). This also indicates production runs FA4
numerics (what `fa4sim` emulates). King #28 is being scored now to compare mu_hat / LCB / per-source.

### What king #29 actually trained (`scripts/verify/compare_checkpoints.py` #29 vs #28, all 34,234 tensors)
- Changed: **layers 16–44 only** — all 256 experts (~2.5% relative), shared experts (7–11%), attention
  qkv/o (~2.5–3%), plus **lm_head (8.6%)**. Norms/sink biases only at bf16-rounding scale.
- Untouched: layers 0–15, embeddings, all routers and router biases, final norm.
- I.e. "freeze bottom 16 layers + embeddings + routers" (likely to save optimizer memory). Extra merge
  candidates split lm_head from the rest (`extrap29_nohead_a0.5`, `extrap29_headonly_a0.5`).

### Your plan (from your answers)
Become king with merges now; I only *prepare* the 8×B300 trainer (you rent after winning); you
register hotkeys when a model is ready. I will never submit anything myself.

### Competitive intel: king #28 was a merge
King #28's upload contains `SOURCE.json` + `merge_spec.json`: *"0.5·trunk v33 EMA-100 + 0.5·lineC v34
EMA-100, both warm-resumed from the v60 king"* (task arithmetic, 50/50). So the top miner trains
several lines from the king on different data mixes, keeps an EMA, and averages them.

### New tools
- `scripts/merge/write_merge.py` — writes a recipe as a submission dir using the *same* tensor-combine
  code the evaluator scored (what you score is what you ship); contract files copied byte-for-byte.
- `scripts/verify/verify_submission.py` — every validator gate: contract SHA-256 vs live chain.toml,
  no manifest.json/symlinks, ≤250 GB, key/shape/dtype == king, no MTP, weights differ from king
  (validator's digest), config lock keys.
- `configs/merge/*.json` — first merge candidates (extrapolate #29's update, transplant lineage-B's
  txt360 update, soup sanity check).
- `scripts/eval/run_queue.sh` + `eval_cache/queue.txt` — unattended GPU job queue (touch
  `eval_cache/queue.pause` to hold GPUs between jobs).
- `scripts/data/download_king.py` — now resumable (range log + legacy prefix), 16 streams; king #28
  (8×30 GB files) finished in 4.8 min after resume.
- `docs/PARITY_EVAL.md` (how to use/read the eval), `scripts/setup/setup_eval_venv.sh`.

### New trainer for 8×B300: `src/cpt/` (+ `docs/CPT.md`, `configs/cpt/b300.yaml`)
Written from scratch around the review's fixes: FSDP2 per layer with **fp32 master weights**, AdamW
lr 1e-5 / wd 0, routers frozen, pure CE on the exact eval objective, **grouped-GEMM experts**
(`torch._grouped_mm` autograd verified bit-identical to per-expert matmuls incl. dX/dW), per-rank
sharded loading straight from safetensors (no 220 GB CPU copy per rank), host-RAM EMA, per-source
validation, DCP resume, exports in the king's exact tensor layout. Training data =
`scripts/data/build_train_pool.py`: the validator's own category-stratified planner with scaled
quotas (smoke pool mix: code 255 / math 420 / pretrain 479 / sft 14 of 1497 ≈ 0.17/0.28/0.32/0.01).
Next: end-to-end smoke test on this box with a 4-layer slice of the real king.

---

## 2026-09-24 14:55 UTC — Eval is 6× faster; king lineage mapped; training review done

### Eval throughput: 2.9 → 17.8 seq/s (30k-sequence sample ≈ 28 min per model)
- Profiling showed CPU/GIL-bound pipeline (tiny kernels, per-head attention loop).
- Fixes: 4 sequences per pipeline step (`--batch-size 4`), vectorized fa4sim attention, sliding-window
  layers computed blockwise (256 keys instead of 2048; **bit-identical** to the reference implementation,
  5× faster), chunked fp32 MoE combine to fit 32 GB cards.
- Batch 4 vs batch 1: mean per-seq diff −1e-6 over 400 seqs → no bias.
- Determinism check: with `torch.use_deterministic_algorithms(True)` two runs are **exactly equal**, so the
  1.6e-3 per-sequence run-to-run noise comes only from atomic `index_add_` ordering flipping near-tie
  expert routing (the validator has the same noise). It moves a 30k mean by ~1e-5 → irrelevant for delta 0.003.

### King lineage (`scripts/merge/king_lineage.py`, reads 13 tensors per king via HTTP ranges)
- Two lineages: **A** …21→23→24→27→28→**29** and **B** 17→20→22→26.
- King #29 = #28 + update to *upper layers only* (attention/experts/shared experts; routers, router bias,
  norms, layers ≤12 untouched). #28 = #27 + tiny update.
- #26 (lineage B, the txt360 specialist that won by +0.073 on txt360) is on the *other* branch → its update
  (26 − 22) is information king #29 does not contain.
- Successive updates are nearly orthogonal (cos 0…+0.25).
- Merge plan: (A) extrapolate `29 + α(29 − 28)`, (B) transplant `29 + β(26 − 22)`, (C) both; score each on
  independent local samples vs king #29. Downloading #26 (fast disk), #22 and #27 (`/root/teutonic_models`).

### Training-code review → `docs/TRAINING_REVIEW.md` (full report with file:line evidence)
As configured the training stack **cannot** produce a winner. Critical issues:
- **C1** Muon LR 0.005 (+ Moonlight RMS scaling) ≈ 5% weight change per step on 99% of params → wrecks the king.
- **C2** no fp32 master weights: bf16 params updated in place → at LR 1.5e-5 ~84% of updates round to zero
  (norms 100%) → training is a near no-op once C1 is fixed.
- **C3** MoE training path loops over 256 experts in Python with 11k FSDP units → months per 1B tokens.
- **H1** offline gate demands LCB > 0.053 (delta + 0.05) — impossible under delta 0.003; replaced by `src/parity`.
- Also: router bias updates / z-loss / aux loss should be 0 for pure-CE gains; MXFP8 noise ≫ 0.3% target;
  data pipeline downloads a 50 MB shard per 8 KB sequence; resume broken for the 1% sft source; 110B path
  never actually trained end-to-end. Training rework is next after merges.

### Other changes
- `scripts/data/download_king.py`: ranged multi-stream downloads (256 MB ranges over 16 streams, per-file
  sha256), because some kings ship 28 GB shards that single-stream at ~50 MB/s.
- `parity_eval.py` searches `TEUTONIC_MODELS_DIRS` (colon-separated) for `reign:N` / `digest:…`.

---

## 2026-09-24 14:30 UTC — Status: validator-parity eval built, first correctness checks passed

### Big finding: the repo was built for an outdated protocol
The live validator (https://github.com/unarbos/teutonic, commit `5479810`, updated today) and
`https://teutonic.ai/datasets/manifest.json` differ a lot from what `.cursor/rules/CONTEXT.md`,
`src/eval/offline.py` and the training configs assume:

| | repo assumption | live now |
|---|---|---|
| eval datasets | finewebedu / automathtext-v2 / dclm | **pretrain-behaviors 0.32, math-reasoning 0.28, txt360-v2 0.22, code-reasoning 0.17, sft-reasoning 0.01** |
| eval size | 2,000 seqs | **30,000 seqs × 2048 tokens** |
| delta (accept margin) | 0.1 nats | **0.003 nats** (LCB of mean paired diff must exceed it) |
| sampling | random shards | **category-stratified**: each dataset split by shard-name category (24 categories), ≥4 shards/dataset |
| attention in eval | eager | production env uses **flash_attention_4** (eager also supported) |

Current king: **reign #29** `teutonic-II-110B-A7B-d2fb75a4` (digest `3325dd0d…`), crowned 2026-09-23
19:41 UTC with mu_hat 0.003234 / LCB 0.003101 over reign #28. Recent kings win by only 0.003–0.012
nats. A king earns ~$214/h (20% weight share, shared by the last 5 kings).

Per-dataset losses in recent duels: math ≈0.45, code ≈0.72, sft ≈0.63, pretrain-behaviors ≈0.93,
**txt360-v2 ≈1.73** (highest loss → most headroom; reign #26 won purely on txt360 +0.073).

### Why the old offline eval can't be trusted (`src/eval/offline.py`)
- wrong LCB formula (basic bootstrap `2μ−q(1−α)` vs validator's percentile `q(α)`), different resampling RNG call
- fp32 logits CE instead of validator's bf16-logit CE; autocast instead of pure bf16 model
- no category-stratified sampling; old datasets; `round()` instead of largest-remainder source targets

### What I built: `src/parity/` + `scripts/eval/parity_eval.py`
Design principle: **import the validator's own code** (pinned `teutonic` package @ `5479810` +
the validator's transformers fork `tony-dendrite/transformers@9e8a1817`, v5.15.1) for everything
that decides a duel: shard planning, sequence sampling, per-sequence CE (`compute_per_sequence_loss`),
paired bootstrap (`paired_bootstrap_verdict`), early stopping. Only the multi-GPU plumbing is mine.

- `src/parity/upstream.py` — pinned imports; `upstream_drift()` diffs scoring files vs validator `main`.
- `src/parity/sample.py` — loads live config, verifies every dataset-manifest SHA-256, and **reproduces the
  live `config_version` hash exactly** (proves shards_per_dataset=4 + category rules match). Builds
  validator-identical 30k samples for any seed `block_hash|hotkey`.
- `src/parity/chain.py` — finds a past duel's seed on-chain (block with the miner's `r2ready:v1`).
  **Verified: latest duel (#29 vs #28) seed = block 9132267; our plan reproduces all 27 shards it used.**
  So we can replay real duels on the *exact* validator sequences and compare to published numbers.
- `src/parity/weights.py` — loads a checkpoint (or a *merge recipe* of several, combined per tensor in
  fp32 → bf16) layer-sharded over the 8×RTX 5090 (24–28 GiB each). Full 220 GB king loads in ~60 s.
- `src/parity/moe.py` — validator's grouped-MoE math via `torch._grouped_mm` (verified bit-identical
  per expert to `F.linear`; this torch build lacks the `torch._foreach_mm` the validator calls).
- `src/parity/attention.py` — `fa4sim`: FA4-faithful attention (fp32 scores, sink in denominator,
  bf16 P·V with fp32 accumulate) because FA4 kernels don't run on sm_120; `eager` also available.
- `src/parity/pipeline.py` — 8-stage pipeline (one sequence in flight per GPU) calling the
  validator's `compute_per_sequence_loss`.
- `src/parity/verdict.py` — validator verdict + per-source/category scores + early-stop simulation +
  **shard-level jackknife SE / robust LCB** (sequences from the same shard are correlated, so the
  validator's iid bootstrap understates how much mu_hat moves between different validator draws).
- `src/parity/cache.py` — per-(sample, model, scorer) loss cache: the king is scored once, then each
  candidate only needs its own pass.

### Correctness checks so far (`scripts/verify/verify_parity_scorer.py`, 16 real sequences)
- ours vs validator's `grouped_mimo_moe`: max per-seq diff 3.8e-3 — **same size as run-to-run noise of
  either implementation** (atomic `index_add_` order flips near-tie expert routing). Averages out to
  ~1e-5 on a 30k mean → negligible vs delta 0.003.
- fa4sim vs eager: per-seq loss shift sd 1.8e-3, mean −1e-4.

### Environment notes
- Machine: 8× RTX 5090 32 GB, 503 GB RAM, 256 CPUs. R2 downloads ~120–300 MB/s (~15–30 min/model).
- `/root` is an encrypted FUSE mount (282 MB/s writes); models go to the fast overlay disk
  `/opt/teutonic/models/<digest>` (812 GB ≈ 3 models). Eval caches in `eval_cache/` (gitignored).
- Eval venv: `.venv-eval` (system torch 2.12+cu130, transformers fork, validator package).

### In progress / next
1. Throughput: 0.86 s/seq sequential, pipelined steady-state being measured → optimize (target ≥8 seq/s,
   i.e. ≤1 h per model per 30k sample).
2. Download king #28 (running) → **replay duel #29 vs #28 on its exact sample** and compare mu_hat/LCB and
   per-source numbers to the published ones (eager vs fa4sim tells us what production runs).
3. Merge search (king #29 with earlier kings / extrapolation) scored on independent local samples.
4. Training-code review (background agent running) → competitive training plan.
