# 1×B300 TEST results — 2026-09-06

Host: **1× NVIDIA B300 SXM6 AC** (~268 GiB), driver 580.126.09, CUDA API 13.0  
Env: **`ml-env-cpt`** (not broken `.venv` TE meta-package)  
Workload: Teutonic SN3 CPT dress rehearsal on **depth-shrink full-width MiMo-V2** (real MoE: 256 experts, top-8)

```
torch=2.11.0+cu130
FA4=4.0.0b28
TE=2.18.0 (pytorch + core_cu13, source-built against this torch)
attn=flash_attention_4
```

E (soup / LoRA / resume) was **skipped** this session per request.

---

## Bottom line (throughput)

**We can train** — gates A–D and F (plus fixed C3) prove the stack runs.

**On 1× it is very slow for full-depth CPT.** F shows tok/s falls roughly with depth. Naive scale to ~45–48L @ seq 256 ≈ **~5–8 tok/s on 1×**. Even ideal **8× → ~50 tok/s** would need **months** for 1B tokens — **not** ~1 week (1 week needs ~1.6k+ tok/s sustained).

| Scenario | Ballpark | 1B tokens |
|----------|----------|-----------|
| 1× @ ~6 tok/s (deep extrapolation) | very slow | ~years |
| Ideal 8× @ ~50 tok/s from that | still slow | ~**7–8 months** |
| Need for ~1 week | ~**1650+ tok/s** | measure on real 8× batch/seq |

**Production path remains 8× B300** (`configs/b300/train.yaml`: mbs=4, accum=4, seq=2048 ≈ 262K tokens/step). **Measure step time there** before planning CPT schedules. Do **not** full-train 110B on 1×.

---

## Gate summary

| Gate | Pass? | Notes |
|------|-------|-------|
| A Env | **PASS** | FA4 + TE MXFP8 Linear OK on B300 |
| B Shrink | **PASS** | `pipeline_model` ~8.59B, 4L, 256 experts |
| C1 eager BF16 | **PASS** | ~83–96 tok/s @ seq 256 |
| C2 FA4 BF16 | **PASS** | ~73–98 tok/s @ seq 256 |
| C3 FA4 MXFP8 | **PASS** (after fix) | ~68–79 tok/s @ seq 256 |
| C5 Winner | **FA4 + MXFP8** | Production target; BF16 = correctness baseline |
| D Real data | **PASS*** | 20 steps, FA4+BF16 (before C3 fix); all 3 sources |
| E Soup/LoRA | **SKIP** | Deferred |
| F Layer ladder | **PASS** | L3/L4/L6/L8 MoE speed table below |
| G 110B | **NOT RUN** | Disk too tight for ~200+ GiB king |

\*Re-run D with MXFP8 after the fix when convenient.

---

## These shrink models **are** MoE

Full-width Teutonic/MiMo recipe; only **depth** is cut.

| Layers | Total params | Activated ≈ / token | MoE pattern |
|--------|--------------|---------------------|-------------|
| 3 | 6.11B | **~1.42B** | `[0,1,1]` (2 MoE) |
| 4 | 8.59B | **~1.56B** | `[0,1,1,1]` (3 MoE) |
| 6 | 13.55B | **~1.84B** | 5 MoE |
| 8 | 18.51B | **~2.12B** | 7 MoE |

All: **256** routed experts, **top-8**. Total ≫ activated (sparse MoE). Full 110B ≈ same idea at 48L (~7B active / token).

---

## C3 failure → fix

**Symptom:** MXFP8 blow-ups on expert batches like `dims=[7, 3072]`, then AC recompute `uint8` vs `bfloat16`.

**Causes:**
1. MoE experts get irregular token counts; MXFP8 needs leading dim **÷ 32**.
2. TE autocast ended **before** `.backward()`.

**Fixes:**
- `src/model/core.py`: pad expert / shared-expert batches to multiple of 32 in MoE `aligned_forward`.
- `src/train/cli.py`: keep `low_precision_context` through loss + `.backward()`.

---

## C speed path (seq=256, mbs=1, 5 steps, synthetic)

| Config | low_precision | Steady tok/s | lm_loss finite? |
|--------|---------------|--------------|-----------------|
| C1 eager | none | ~83–96 | yes |
| C2 FA4 | none | ~73–98 | yes |
| C3 FA4 | **mxfp8** | ~68–79 | yes |

On this tiny shrink + short seq, MXFP8 is not faster than BF16. Still the B300 production target; re-measure @ seq **2048** / 8×.

Logs: `logs/b300_1gpu_C{1,2,3}_*.log`.

---

## D real Teutonic data (before MXFP8 fix)

- Pool `(20, 2048)` uint32; sources: dclm / finewebedu / automathtext (not synthetic)
- FA4 + **BF16** override: `configs/b300/_runtime_1gpu_mini_real_bf16.yaml`
- Steady tok/s @ seq **2048**: ~**110–128**
- Log: `logs/b300_1gpu_D_real.log`

---

## F layer ladder (FA4 + MXFP8, seq=256, mbs=1)

| Layers | ~params | Activated ≈ | Steady tok/s | Train | Ckpt |
|--------|---------|--------------|--------------|-------|------|
| 3 | 6.11B | 1.42B | **~111–132** | PASS | PASS |
| 4 | 8.59B | 1.56B | **~68–79** | PASS | PASS |
| 6 | 13.55B | 1.84B | **~44–52** | PASS | FAIL (disk mid-save) |
| 8 | 18.51B | 2.12B | **~32–37** | PASS | PASS (overlay) |

**Trend:** deeper → slower (~3× from L3→L8). Extrapolate to ~45–48L @ same smoke ≈ **~5–8 tok/s on 1×** (not an 8× CPT forecast).

### Ops notes

- Prefer large scratch on overlay (`/var/moe_*`) — `/root` filled during F.
- Launcher prefers `ml-env-cpt` over `.venv`.
- New helper: `scripts/setup/optimize_ml_env.py` (workload-aware CPT env).

---

## Winner settings (8× starting point — still must measure)

```
attn=flash_attention_4
low_precision=mxfp8
seq=2048
mbs=4                 # production train.yaml; 1× smokes used 1
grad_accum=4
ac=selective
gpus_per_node=8
```

**Not certified:** 8× NCCL/IB, full 110B train, MXFP8 @ 2048 on mini, soup/LoRA (E skipped).

---

## How to re-run

```bash
source ml-env-cpt/training/bin/activate
export PYTHONPATH=$PWD TEUTONIC_ATTN=flash_attention_4 GPUS_PER_NODE=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$PWD/.cache/fa4_cute

bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_speed_mxfp8.yaml
GATE=F bash scripts/train/run_b300_1gpu_tests.sh
```
