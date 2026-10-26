# Teutonic MoE (MiMo-V2 CPT)

## Quick start on a new box (no LLM needed)
Five scripts in the repo root run the whole pipeline; each prints what it does and is safe to re-run.

```bash
git clone https://github.com/wilfrid51/MoE.git /root/workspace/MoE && cd /root/workspace/MoE
HF_TOKEN=hf_xxx bash setup.sh        # venv (.venv-eval), /data/teutonic layout, env.sh, helper scripts, HF token, pm2
bash download.sh 10                  # the 10 most recently crowned public kings + every reigning king + T1; keeps
                                     #   downloading new kings as they are crowned (pm2 "download_kings"); 10 = your choice
bash merge.sh                        # merge campaign over everything downloaded: every method, all four competitions
                                     #   scored per candidate; best per competition in /data/teutonic/search/public.json
bash train.sh <merge label> <main|math|code|text>   # write that merge and train it for the competition (also accepts
                                     #   a checkpoint dir or a model name; DRY=1 prints the plan; STEPS / LR / ... override)
bash dashboard.sh                    # http://<box>:8787 (training, merge panel, screens)
```
`merge.sh stop` / `dashboard.sh stop` stop them; `pm2 ls` lists every job, `pm2 logs <name>` follows one.
Details: [docs/START_MERGE_B300.md](docs/START_MERGE_B300.md). Submitting (hotkeys, `ready`) stays manual: a merge that
passes the full validator-size check is written to `/data/teutonic/submit/<comp>_<label>`, uploaded to HF and listed in
`/data/teutonic/search/public_ready.json`.


Tooling for Bittensor Subnet 3 (Teutonic) challengers. **Start with `UPDATE.md`** (running log: what was
done, why, results) and these current entry points:

| What | Where |
|---|---|
| Validator-parity eval (reproduces live verdicts to ~1e-5), merges, head tuning, routing stats | `scripts/eval/parity_eval.py`, `src/parity/`, [docs/PARITY_EVAL.md](docs/PARITY_EVAL.md) |
| GPU job queue for evals | `scripts/eval/queue_ctl.sh start\|pause\|resume\|status`, `eval_cache/queue.txt` |
| Merge recipes (linear/TIES/DARE/DARE-TIES/SLERP/orthogonal/routing-aware expert merges) | `configs/merge/*.json`, `scripts/merge/make_expert_recipe.py` |
| **Multi-competition campaign (MAIN/MATH/CODE/TEXT, Oct 2026): log, recipe book, handoff** | [41.md](41.md), [docs/HANDOFF_K41.md](docs/HANDOFF_K41.md), `scripts/merge/sweep41.py` |
| **King ledger: each king, our best against it, submissions, bars** | [42.md](42.md) |
| Write + verify a submission | `scripts/merge/write_merge.py`, `scripts/verify/verify_submission.py`, [docs/SUBMIT.md](docs/SUBMIT.md) |
| Full-parameter CPT for 8×B300 (new trainer) | `src/cpt/`, `configs/cpt/`, `scripts/train/launch_cpt.sh`, [docs/CPT.md](docs/CPT.md) |
| Why the old `src/train` stack can't win as configured | [docs/TRAINING_REVIEW.md](docs/TRAINING_REVIEW.md) |

Environments: `scripts/setup/setup_eval_venv.sh` (eval, any GPU with torch ≥ 2.8) and
`scripts/setup/setup_cpt_b300.sh` (training on B300). The sections below describe the older stack.

## King #31 merge campaign (2026-09-26)

Goal: beat king #31 (`teutonic-II-110B-A7B-5ac02e46`) by merging the latest public kings. Full recipe → score
log: **[31.md](31.md)** (regenerated from the loss cache by `scripts/merge/update_31_md.py`).

| candidate | method | mu_hat vs #31 (800 seqs, tag:A) |
|---|---|---:|
| `L1_coef_g060` | #31u + Dm with learned per-tensor/per-expert coefficients (one gradient step) | **+0.00185** |
| `K_dm100` | #31u + 1.0·Dm (Dm = k30u − #30u = 0.17·D26 + 0.13·D25, module-scaled) | +0.00166 |
| `K_dm075` / `K_dm125` / `K_dm150` | same, scaled 0.75 / 1.25 / 1.5 | +0.00162 / +0.00145 / +0.00113 |
| `task_ap075` | #31u + 0.75·(CPT − k30u) | +0.00022 |
| `K_s31p20` | extrapolate #31's own step by 0.2 | −0.00006 |

The crown needs ≈ +0.0036 (LCB > 0.003 on the validator's 30k draw), so the best merge reaches about half of it.

What was built:
- **Unpermuting kings**: #30 and #31 ship with a function-preserving hidden-dim shuffle.
  `scripts/merge/perm_from_embed.py` recovers the permutation from embedding columns, and
  `scripts/merge/unpermute_checkpoint.py` maps a king into the common basis (#30u, #31u).
- **Delta geometry**: `scripts/merge/delta_geometry.py` gives per-module norms and cosines of task vectors. #31's step
  touches only layers 16–47 + lm_head and is nearly orthogonal to the public-king merge step (cos 0.17).
- **Fast sweep server**: `scripts/merge/sweep31.py` keeps #31u on the GPUs and the task vectors in pinned host
  memory, then rebuilds each candidate in ~1.5 min instead of ~11 min. Merge ops live in the hot-reloaded
  `scripts/merge/sweep31_ops.py`: task arithmetic, per-regex / per-tensor coefficients, TIES, DARE, DARE-TIES,
  breadcrumbs, trim, clip, sign-agree, orthogonal projection, and live `addvec`. Queue items are made by
  `scripts/merge/queue31.py`.
- **Learned merge coefficients** (AdaMerging-style): `grad` queue items backpropagate the eval objective on a
  separate sample and record `<dL/dW_t, vec_t>` for every tensor and expert. `scripts/merge/grad_report31.py`
  summarizes a probe, and `scripts/merge/grad_step31.py` turns probes into per-tensor coefficient recipes.
- `scripts/merge/rescore31.py` re-derives old rows after the king cache was re-scored at the challengers' batch
  size. MoE kernel rounding depends on the batch size, and a batch-2 king cache biased mu_hat by about −1e-4.

Reproduce the best candidate (its exact linear recipe is the `ship` block of the recipe JSON):

```bash
source /opt/teutonic/env.sh
python - <<'EOF'
import json; r = json.load(open("configs/merge/k31/L1_coef_g060.json"))
json.dump(r["ship"], open("/tmp/L1_coef_g060.ship.json", "w"))
EOF
python scripts/merge/write_merge.py --recipe /tmp/L1_coef_g060.ship.json --out /opt/teutonic/submit/L1_coef_g060
python scripts/verify/verify_submission.py --model-dir /opt/teutonic/submit/L1_coef_g060 --king-dir /opt/teutonic/hf/king31u
```

Conclusion: the only public training step outside #31's ancestry is the D26/D25 pair. Its gains on
code/math/pretrain/sft are paid back on txt360 as its scale grows, and learned coefficients add about +0.0002 per
step before the noisy gradient's trust region runs out. Beating #31 needs new training signal (e.g. CPT started from
this merge), not more merge arithmetic.

---

Distributed continued-pretrain stack for Bittensor Subnet 3 (Teutonic): FSDP2, hybrid Muon+AdamW, FA4, MXFP8, DCP, offline paired eval.

## Layout

```text
configs/          # by hardware: b300/  4090/  pro6000/  smoke/
scripts/
  setup/          # venv, FA4, NVML, GPU detect
  train/          # launch + gate runners
  data/           # king download, pools, mini king build
  eval/           # offline eval wrappers
  merge/          # model soup / merge experiments
  verify/         # unit + design checks
src/
  train/          # python -m src.train
  model/ moe/ data/ optim/ checkpoint/ eval/ dist/ submit/
pipeline_models/  # depth-shrink modeling sources
docs/             # TEST.md, TESTPLAN.md, prompts/
```

Runtime artifacts stay at repo root (gitignored): `king/`, `pipeline_model/`, `checkpoints/`, `shard_cache/`, `sequence_pool*/`.

## Quickstarts

### Environment (hardware-aware)

Uses the vendored GPU_check installer policy (`scripts/setup/bootstrap_ml.py`) plus MoE FA4/MXFP8 pins:

```bash
bash scripts/setup/setup_env.sh --diagnose-only   # inventory + plan only
bash scripts/setup/setup_env.sh                   # diagnose → install → check_env
# or force:
bash scripts/setup/setup_env.sh --profile b300
bash scripts/setup/setup_env.sh --profile 4090
bash scripts/setup/check_env.sh                   # re-verify after activate
```

See [docs/ML_ENV_GUIDE.md](docs/ML_ENV_GUIDE.md) for the underlying selection policy.

### 1×B300 (TESTPLAN gates)

```bash
bash scripts/setup/setup_env.sh --profile b300 && source .venv/bin/activate
python scripts/data/download_king.py --output ./king --workers 8 > logs/download_king.log 2>&1 &
bash scripts/train/run_b300_1gpu_tests.sh          # GATE=C|D|E|F|G for one gate
```

See [docs/TESTPLAN.md](docs/TESTPLAN.md).

### 8×4090 mini (shrink ~8.59B dress rehearsal)

```bash
bash scripts/setup/setup_env.sh --profile 4090 && source .venv/bin/activate
BRINGUP=1 bash scripts/train/run_4090_mini_cpt.sh
# real data: configs/4090/mini_real.yaml — see docs/TEST.md
```

### 8×B300 production

```bash
bash scripts/setup/setup_env.sh --profile b300 && source .venv/bin/activate
CONFIG=configs/b300/train.yaml bash scripts/train/launch.sh
```

Train entrypoint everywhere:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=N \
  -m src.train --config configs/b300/train.yaml
```

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/TESTPLAN.md](docs/TESTPLAN.md) | 1×B300 tick-list before 8× |
| [docs/TEST.md](docs/TEST.md) | Append-only 4090/B300 test log |
| [docs/TRAINING_CONTROL.md](docs/TRAINING_CONTROL.md) | Train/eval/export controls |
| [docs/ML_ENV_GUIDE.md](docs/ML_ENV_GUIDE.md) | Hardware-aware env installer policy |
| [docs/prompts/](docs/prompts/) | Historical build prompts |
