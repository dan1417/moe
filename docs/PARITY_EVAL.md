# Validator-parity evaluation (`src/parity`, `scripts/eval/parity_eval.py`)

Goal: local king-vs-challenger scores that match what the Teutonic validator computes,
so a hotkey (one submission, forever) is only spent on a challenger that wins.

## Since 2026-10-09: document-masked, token-weighted scoring + whole documents (validator `14a9bb2`)

Live for all four competitions from ~13:00 UTC 2026-10-09 (policy `document-masked-token-bootstrap-v1`,
sampling `long-documents-2k-8k-v1`; owners' note: "excludes cross-document token predictions" and "6M tokens
from documents longer than 2048 tokens"). Every number scored before that is on the old policy and not comparable.

| | before | now |
|---|---|---|
| windows | 30 000 × 2 048, causal over the whole row | same windows, split at EOS `151645`: positions restart at 0 after each EOS, attention never crosses it; a fragment's first token is not a target (EOS targets count) |
| extra sample | — | complete documents of 2 049–8 192 tokens (incl. terminal EOS), ~6M input tokens, same 70/15/15 mix and category weights, counts by length bucket (2049–4096 / 4097–8192), shuffled into the windows; ~9 % of the scored tokens |
| per-sample loss | mean over 2 047 targets | mean over the sample's scored targets |
| mu, LCB, source scores | plain means over sequences | weighted by scored tokens; bootstrap resamples whole windows/documents and recomputes the weighted ratio |
| early stopping | futility rule | off whenever documents are in the sample (always, live) |
| numerics | default kernels | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |

How the parity code follows it (all through the validator's own functions):
- `load_live_config` reproduces each live `config_version` only with the 6M long-document config built from the
  pinned index manifest (`doc_index/manifest.json`); e.g. TEXT `0a2840c9…` (old `b5318eaf…`). New sample ids follow.
- `build_sample` calls `sources.sample_eval_sequences` (windows + `extend_with_long_documents`). The index is read
  as thousands of 64 KiB ranges (pages cached under `eval_cache/shards/doc_index`). Since the pin moved to
  validator `5118c32` (2026-10-13) window rows and document fragments are HTTP range reads too
  (`sources.open_shard` -> `token_shards.NpyShard`, ~0.3 GB per sample instead of ~85 GB of whole shards);
  `install_fetch_retry` adds a slow outer retry over the reader's own 429 backoff. A cached sample is reused only if
  its documents are complete (`documents_complete` in meta.json: document tokens = the planned per-category total
  and >= 95 % of the 6M budget); samples built while documents were skipped are listed in
  `samples/index_incomplete.json` (dirs kept) and rebuilt. Loss caches stay keyed by the scoring-code commit
  `14a9bb2` (`upstream.SCORING_CODE_COMMIT`). Variable-length samples are stored flat (`tokens.npy` +
  `offsets.npy`, `Sequences`) with `scored_tokens.npy` and `documents.json` (dataset, category, length, bucket,
  fragments).
- Scoring calls `engine.compute_per_sequence_loss` (masking inside: FA4 gets explicit fragment boundaries, eager gets
  explicit masks; unequal lengths are packed into one row). **`--attn flash_attention_4` is the default and the only
  practical backend** (sm_90/100/103); `fa4sim` has no document masking. Batches follow `scoring_batch_end`.
- `verdict.duel` passes `scored_tokens` to `paired_bootstrap_verdict`; also prints `component_scores` (windows vs
  documents) and `long_document_scores` by length bucket — check both: gains on windows alone may not carry.
- Trainer (`cpt_fsdp.py --doc-mask 1`, default) trains on the same objective: positions restart at EOS (the MiMo
  forward's mask builders then isolate fragments), fragment starts are not targets, losses per scored token.
- Checks: `scripts/verify/verify_doc_mask_eval.py` (window = its fragments scored alone, through the FA4 pipeline)
  and `scripts/verify/verify_doc_mask_train.py` (trainer layout = validator layout; isolation; backward).
- The `fn=` diagnostics (`calibrate`, `tokendiag`, `routing`, `headtune`) still run unmasked 2048-token rows.

## What is reused from the validator (not re-implemented)

Pinned in `src/parity/upstream.py`: `unarbos/teutonic@f5bfc3f` and the validator's
transformers fork `tony-dendrite/transformers@9e8a181` (v5.15.1 + FA4 asymmetric-V fix).

| Step | Validator function used |
|---|---|
| live config → settings | `configuration.evaluation_config_version` (we re-derive the live `config_version` hash exactly) |
| shard plan per seed | `configuration.pretokenized_dataset_request` (category-stratified, 24 categories) |
| sequence sampling | `sources.sample_pretokenized_sequences` (same RNG order, vocab filter, global shuffle) |
| per-sequence CE | `engine.compute_per_sequence_loss` (bf16 logits, 1024-position lm_head chunks, sum/2047) |
| verdict | `policy.paired_bootstrap_verdict` (B=10000, alpha=0.001, bootstrap seed 0) |
| early stopping | `early_stopping.challenger_futility_decision` (live params private; defaults assumed) |
| model construction | `engine.load_model_config`, `patch_mimo_masking_compat`, attention config validation |

Ours (numerically equivalent plumbing): layer-sharded loading over 8×32 GB GPUs, 8-stage
pipelining, grouped MoE via `torch._grouped_mm` (bit-identical per expert to the
validator's per-expert GEMMs), FA4-faithful attention (`fa4sim`) because FA4 kernels do
not run on sm_120, merge recipes, loss cache, risk diagnostics.

## Verified
- Live `config_version` reproduced from public manifests (proves shards_per_dataset=4 + category rules).
- A past duel's exact sample can be rebuilt: its seed is the block containing the miner's
  `r2ready:v1` commitment (`src/parity/chain.py`). For duel `81bcc8fa` (#29 vs #28) the plan
  reproduced all 27 published `shards_used`.
- Grouped MoE vs validator `grouped_mimo_moe`: within run-to-run noise (atomic `index_add_`).
- With `torch.use_deterministic_algorithms(True)` two runs are bit-identical (no hidden bugs);
  default nondeterminism is ~1.6e-3 per sequence (sd) → ~1e-5 on a 30k mean. The validator has it too.
- Batch 4 vs batch 1: no bias (mean diff −1e-6 over 400 sequences).
- Replay of duel #29 vs #28 vs published numbers: see UPDATE.md.

## Usage

```bash
source .venv-eval/bin/activate                      # scripts/setup/setup_eval_venv.sh
export TEUTONIC_MODELS_DIRS=/opt/teutonic/models:/root/teutonic_models
python scripts/data/download_king.py --reign 29 --output /opt/teutonic/models/<digest>   # or --digest

python scripts/eval/parity_eval.py config --drift            # live config + upstream drift check
python scripts/eval/parity_eval.py replay --challenge-id latest   # reproduce the last duel exactly
python scripts/eval/parity_eval.py sample --sample tag:A --sample tag:B
python scripts/eval/parity_eval.py duel --king reign:29 \
    --challenger configs/merge/extrap29_a0.5.json --sample tag:A --sample tag:B
python scripts/eval/parity_eval.py --limit 8000 duel ...      # quick screen on first 8k sequences
```

Samples: `tag:<name>` (fresh sample, synthetic seed), `seed:<block_hash>:<hotkey>`,
`duel:<challenge_id>` (exact past duel), `id:<sample_id>` (cached).
Models: a directory, `reign:<n>`, `digest:<sha256>`, `1.5:reign:29,-0.5:reign:28`, or a recipe JSON
(`configs/merge/*.json`; per-tensor-regex `overrides` supported). Merges are combined per tensor
in fp32 and rounded to bf16 once — the same code (`src/parity/weights._fill`) that
`scripts/merge/write_merge.py` uses to materialize a submission, so what you score is what you ship.

Losses are cached per (sample, model identity, scorer) in `eval_cache/losses/`: the king is scored
once per sample; every candidate afterwards costs one pass (~28 min per 30k sequences at 17 seq/s).
`scripts/eval/run_queue.sh` runs a job file unattended (`eval_cache/queue.txt`).

## Reading a verdict
- `mu_hat`, `lcb`, `accepted` — exactly the validator's statistics on that sample.
- `se_iid` — bootstrap-style standard error; `se_shard_jackknife` — delete-one-shard jackknife
  (sequences from one shard are correlated, so a *different* validator draw moves mu_hat by more
  than `se_iid` suggests). `robust_lcb = mu_hat − 3.09·max(se_iid, se_jack)`.
- `source_scores` / `category_scores` — regressions on any source are a red flag even if mu_hat is fine.
- `early_stop` — would the validator's futility check stop this challenger (assumed parameters).

## Submission rule of thumb (delta = 0.003)
Submit only if, on ≥2 independent fresh samples (full 30k each): `robust_lcb > 0.003` on each,
mu_hat ≥ ~0.0035–0.004, and no source with a clearly negative mu_hat. Then materialize with
`write_merge.py`, run `scripts/verify/verify_submission.py`, and re-score the written directory on
one sample to confirm it reproduces the recipe's losses.
