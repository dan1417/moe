# Handoff — Teutonic (SN3) merge campaign, state at 2026-10-08 07:55 UTC

Everything needed to continue on another machine. Ledgers: [42.md](../42.md) (per-competition king ledger +
auto leaderboards — read this first), [41.md](../41.md) (chronological log, results table, exact recipe book),
[CURRENT_BEST.md](../CURRENT_BEST.md), [TRAINING_PLAN_TEXT.md](TRAINING_PLAN_TEXT.md) (the route beyond merging).

## 1. State of play

| competition | king | ours vs king (screen) | status |
|---|---|---|---|
| CODE | **ours**: `teutonic-II-110B-A7B-001` = `t1_d36050_pe015`, digest `7e4aa8238859533ac2cc63bc214e5d78e1113f59fe4c2c3a38211328ab4b08cd`, uid 110, hotkey `sn3auto002` (wallet `autotao`) | – | crowned 2026-10-07 20:18 UTC, defended once (5Ek5Ko rejected). Incentive 0.1481 ≈ the 0.15 CODE share (the gap is two small validators with stale weights). |
| MAIN | #42 `67643e860f86edf5def81436b630216d23c892c05b0c9503000cfa155d375665` (5Ek5Ko, uid 182) = T1 + their step + 0.25·our CODE delta | ≈ 0 | flat; needs +0.0036 |
| TEXT | T1 `95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0` (5DALwjE4, uid 129) | best `ck3_d41_025_pmom015` +0.00195 (confirmed on the exact text sample: +0.00189, lcb30k +0.00177) | needs +0.0026; merge ceiling reached |
| MATH | vacant (challengers duel #42 on the math mix) | ≈ 0 | flat; needs +0.0026 |

Conclusions that should not be re-derived (all numbers in 42.md): every merge family (task arithmetic, momentum,
module splits, TIES/SLERP/consensus/low-rank, routing-aware per-expert coefficients, learned coefficients from gradient
probes, head tuning, router-bias γ, logit temperature, norm matching) sits at +0.0013…+0.0020 on TEXT; the #40→#41 step
is a txt360-qa specialist whose collateral damage lives in the same experts; **router-only fine-tuning gives nothing**
(sigmoid top-k routing: the gradient only re-balances already-selected experts). Open at handoff: the dense-path CPT run
`cpt_shared_a` (shared expert + layer-0 MLP + norms, 1000 steps, started 07:18, step 200 val 1.14965 vs 1.14970).

## 2. Rules (hard-won, do not relax)
1. One upload per registration prefix: a second upload into a live prefix (even a rename) exceeds the 250 GB quota → prefix wipe → hotkey lost.
   Run `python scripts/submit/prefix_check.py` before every `ready`.
2. Recommend a hotkey only with ≥ +0.0006 above the bar on the *exact* competition sample (`confirm41`); the validator's draw ran 0.0004 below our local estimate.
3. Coldkey never on the box; the box holds the ed25519 hotkey only. Logs are sed-redacted; never print secrets.
4. Every GPU job on the current box must `source /data/teutonic/env.sh` (LD_PRELOAD P2P shim — box-specific, see §5).
5. Every recipe goes to markdown (41.md recipe book is auto-generated from `configs/merge/k41/`; 42.md prose rows are manual).
6. Do not poll/report on-chain incentive unless asked.

## 3. What is on this box (`/data/teutonic`, overlay disk; nothing here is in git)
| path | what | size |
|---|---|---|
| `models/1f1efb68…` | #41 (sweep server `--base`) | 206 GB |
| `models/7deb3bb4…` | #40 | 206 GB |
| `models/95ef7a26…` | T1 (TEXT king) | 206 GB |
| `models/CK3_b41` | #42 un-permuted into our basis (`configs/merge/k41/perm/perm_CK3.json`) | 206 GB |
| `submit/t1_d36050_pe015` | our CODE king, as submitted (`CK1` in models.json) | 206 GB |
| `submit/t1_d36050` | first CODE attempt (lost to the prefix wipe) | 206 GB |
| `vec/d36_35q … d40_39q, d39_38q, d38_37q, p40_36q, p32_29q, d33_32q, d29_28q, d28_27q, d27_26q, d24_23q` | int8 row-quantized king-to-king deltas (D36 … D38, P40_36 …), T1/#41 basis, `DELTA.json` each | 103 GB each |
| `vec/T_cons41, T_lr50, T_q` | consensus / low-rank / plain-delta forms of T = T1 − #41 | 103 GB each |
| `vec/routers/` | trained router snapshots (`rt_smoke`, `rt_text_a`) | 133 MB |
| `pools/text_train` (6144 seqs), `pools/math_train` (49k seqs) | diverse training pools mirroring the text / math mixes, eval shards excluded | 49 MB / 386 MB |
| `eval_cache/samples` | mix tag:A `ec50cbc7-cf035b46…` (the screen), mix tag:B `ec50cbc7-7f03354e…` (held-out), code `9109dbf8-…`, math `a4981bea-ffba01c4…`, text exact `b5318eaf-e83a2099…` (+ `b5318eaf-13d2d158…`), MAIN `e4a9f831-…` | 1.4 GB |
| `eval_cache/losses`, `eval_cache/routing` | every scored (sample, model) loss vector; per-split and per-category routing of T1 / #42 | small |
| `*.sh` in `/data/teutonic` | orchestration triggers (`queue_d39d38.sh`, `queue_rx.sh`, `pick_rt_base.sh`, `build_kings_chain.sh`, …) — all finished | – |

**Copy priority** if only some of it can move: `eval_cache/` (7.8 GB, makes every old score reusable), `vec/routers/`,
`pools/`, then `models/CK3_b41` + `models/95ef7a26…` + `models/1f1efb68…` + `models/7deb3bb4…` (the four the best TEXT
recipe needs), then `vec/d36_35q` (CODE king ingredient). Everything else is re-downloadable / re-buildable (§5).

## 4. Running at handoff
- Sweep server pid 1070703: `python scripts/merge/sweep41.py --base $K41 --sample tag:A --main-sample tag:A --limit 2400 --main-limit 0`
  (log `/data/teutonic/logs/sweep41.log`; stop with `touch configs/merge/k41/queue/STOP`). Currently executing the plugin
  `cpt_shared_a` (train_router41, ~2 h); then the queue (46 items, snapshot in `configs/merge/k41/queue_pending/`,
  priority order: D39/D38 step combos, router swaps `tx_best_rt*`, norm-matched `tx_*_mn`, dense-heavy `tx_d41d*`, then
  older low-value items).
- Pending user action: Hugging Face upload of `rt_smoke` (`scripts/submit/hf_upload_rt_smoke.sh`, needs the user's HF
  token; the session's permission classifier blocked creating the public gated repo). Rotate that token afterwards.
- 94 local commits are **not pushed** (no git credentials on the box): `git push origin main` from a machine with access,
  or `git bundle create moe.bundle origin/main..main` here and apply it there.

## 5. Rebuild on a new machine
```bash
git clone https://github.com/wilfrid51/MoE.git && cd MoE          # or apply the bundle from §4
bash scripts/setup/setup_eval_venv.sh                               # torch ≥ 2.8 + CUDA, validator f5bfc3f, transformers fork, safetensors, huggingface_hub
python -m venv .venv-miner && .venv-miner/bin/pip install bittensor  # only for chain queries / submission
mkdir -p /data/teutonic/{models,vec,logs,eval_cache,submit,pools}
cat > /data/teutonic/env.sh <<'EOF'
cd /path/to/MoE && source .venv-eval/bin/activate
export TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache TEUTONIC_MODELS_DIRS=/data/teutonic/models PYTHONUNBUFFERED=1
export K41=/data/teutonic/models/1f1efb68b4a03d4b54304562597b34a71aa9b58d916757dfc83821df0b6d61c4
export T1=/data/teutonic/models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0
EOF
# P2P test — add `export LD_PRELOAD=<build of nop2p>` to env.sh ONLY if this prints False (RTX 6000 Ada box bug):
python -c "import torch;a=torch.arange(10,device='cuda:0');print(torch.equal(a.to('cuda:1').cpu(),a.cpu()))"
# kings (any published model): download → index → permutation vs T1 → un-permute → register in configs/merge/k41/models.json
bash scripts/merge/ingest_king41.sh 95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0 T1raw   # T1 first (reference)
bash scripts/merge/ingest_king41.sh 1f1efb68b4a03d4b54304562597b34a71aa9b58d916757dfc83821df0b6d61c4 K41
bash scripts/merge/ingest_king41.sh 7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c K40
bash scripts/merge/ingest_king41.sh 67643e860f86edf5def81436b630216d23c892c05b0c9503000cfa155d375665 CK3
# deltas (≈ 55 min each from R2): e.g. D36
python scripts/merge/remote_delta40.py --a 36 --b 35 --ref $T1 --out /data/teutonic/vec/d36_35q --int8
# samples (skip if eval_cache/ was copied)
for c in mix code math text main; do python scripts/eval/parity_eval.py --competition $c sample --sample tag:A; done
python scripts/eval/parity_eval.py --competition mix sample --sample tag:B
```
Fix the paths in `configs/merge/k41/{models,vectors,kings}.json` (models: K41, T1, K40, CK3, CK1 = our CODE king dir,
delta dirs; kings: main/math → CK3, code → CK1, text → T1). Hardware: 8 GPUs × ≥ 40 GB (layer-sharded, batch 8) or
one ≥ 230 GB GPU; the box also needs ~1.5 TB for the four models + a few deltas.

## 6. Resume
```bash
source /data/teutonic/env.sh
cp configs/merge/k41/queue_pending/*.json configs/merge/k41/queue/
nohup python scripts/merge/sweep41.py --base $K41 --sample tag:A --main-sample tag:A --limit 2400 --main-limit 0 > /data/teutonic/logs/sweep41.log 2>&1 &
python scripts/merge/queue41.py --base none --vec CK3=1,D41=0.25 --ov experts:CK3=1,D41=0.25,P=0.15 --label <name> --main-limit 800   # new candidates
python scripts/merge/report41.py        # regenerates 41.md tables + recipe book and the 42.md leaderboards
```
Tooling map: `queue41.py` (vector notation, module overrides, transforms), `build_vec41.py` (consensus / low-rank / par-perp
deltas), `routing41.py` + `route_recipe41.py` (per-split or per-category expert routing → per-expert coefficients),
`grad41.py` + `grad_step41.py` (gradient probes → learned coefficients), `train_router41.py` (router / dense-path training
plugin), `confirm41.py` (validator verdict on the exact sample), `catdiag41.py` (per-category gains), `write_merge.py`
(recipe → submission dir), `scripts/submit/prefix_check.py`, `scripts/verify/verify_submission.py`.

## 7. Submission procedure (what worked for the CODE crown)
1. `write_merge.py --recipe configs/merge/k41/<label>.json --out /data/teutonic/submit/<label>` → `verify_submission.py`.
2. User registers a fresh hotkey on the secure device; the box gets only the hotkey (ed25519) for upload/ready.
3. Upload once with the official tooling; name `teutonic-II-110B-A7B-<suffix>`; never touch the prefix again.
4. `prefix_check.py` (size < 250 GB, no in-flight parts, manifest matches) → `ready --competition <c>` → watch
   `https://teutonic.ai/competitions/<c>/dashboard.json` (queue → current_eval → history; ~30 min per duel).
