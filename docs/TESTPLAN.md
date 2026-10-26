# 1×B300 TESTPLAN — tick while testing

**Host:** 1× NVIDIA B300 (~288GB)  
**Primary model:** depth-shrink full-width MiMo-V2 (~8.59B, 4 layers) = `pipeline_model`  
**Secondary:** real 110B king = `./king` (eval / load only unless memory is absurdly free)  
**Rule:** keep every gate short. Prefer 3–10 train steps. Long downloads run in background.

Tick boxes as you go. Record numbers in the **Results scrap** at the bottom (or append to `docs/TEST.md`).

### Patched artifacts (ready to run)

| Piece | Path |
|-------|------|
| Setup (cu130 + FA4 + TE) | `scripts/setup/setup_env.sh` (or `setup_b300_venv.sh`) |
| Env diagnose / plan | `scripts/setup/teutonic_stack.py` + `bootstrap_ml.py` |
| Env verify | `scripts/setup/check_env.sh` |
| Single launch helper | `scripts/train/run_b300_1gpu.sh` |
| Ordered gates | `scripts/train/run_b300_1gpu_tests.sh` (`GATE=A\|B\|C\|D\|E\|F\|download\|G\|all`) |
| Speed C1/C2/C3 | `configs/b300/1gpu_speed_{eager,fa4,mxfp8}.yaml` |
| Real data D | `configs/b300/1gpu_mini_real.yaml` |
| Soup E1 | `configs/b300/1gpu_soup.yaml` |
| LoRA E2 | `configs/b300/1gpu_lora.yaml` |
| 110B eval G | `configs/b300/1gpu_eval.yaml` |
| Layer ladder | `scripts/data/prepare_pipeline_model.py --num-layers {3,4,6,8}` |

```bash
bash scripts/setup/setup_b300_venv.sh && source .venv/bin/activate
mkdir -p logs && python scripts/data/download_king.py --output ./king --workers 8 > logs/download_king.log 2>&1 &
bash scripts/train/run_b300_1gpu_tests.sh          # A–F
# GATE=C bash scripts/train/run_b300_1gpu_tests.sh # one gate
```

---

## Answers (read once)

### 1) Is NCCL / FSDP proven on 8×4090?

**Yes — on 8×4090 with the ~8.59B shrink king.**

Already PASS there: NCCL, FSDP2 (per-expert), sharded Muon, DCP save/resume, real Teutonic data CPT.

**Not proven yet:** same stack on **B300**, with **FA4**, **MXFP8**, or **110B**, or multi-GPU **IB/NCCL on B300**.

### 2) Why use the shrink (~8.3–8.59B) model?

**You should.** That is the right 1×B300 workhorse.

| Use shrink (~8.59B) for | Use real 110B for |
|-------------------------|-------------------|
| Env / FA4 / MXFP8 / speed | Download + load |
| Real-data train smoke | Offline eval smoke |
| Layer-count speed ladder | Peak HBM telemetry |
| Soup + LoRA + DCP path | (train only if load is easy) |

Shrink keeps **110B width** (experts, hidden, heads) so MoE/FSDP/Muon bugs still show up, without waiting on full-depth train.

### Does 1×B300 env scale to 8×B300?

**Software yes** (same torch / FA4 / TE / configs → set `gpus_per_node=8`).  
**Hardware collectives no** — still need a short 8× NCCL/FSDP bring-up later.

### Does proving speed on 1× mean 8× is fine?

**Kernels yes** (FA4 + MXFP8 + no silly eager path).  
**End-to-end tok/s no** — 8× adds shard/collectives; re-measure briefly on 8×.

---

## Time budget (target ~1 day rent)

| Block | Budget | What |
|-------|--------|------|
| A Env | 30–60 min | CUDA / FA4 / TE |
| B Shrink ready | 20–40 min | build mini + units |
| C Speed path | 45–90 min | eager vs FA4, BF16 vs MXFP8 |
| D Real-data mini CPT | 30–60 min | 10–20 steps |
| E Soup + LoRA + DCP | 45–90 min | short |
| F Layer ladder | 30–60 min | 3L vs 4L (+ optional deeper) |
| G 110B eval | download bg + 1–2 h | load + `--eval-only` smoke |
| **Skip / defer** | — | long burn-in, full eval corpus, 8× NCCL |

Start **G download in background** as soon as network works (after A0).

---

## A — Environment (must match MiMo / production path)

Goal: one venv you can reuse on 8×B300.

- [ ] **A0** Box OK: `nvidia-smi` shows B300, ~288GB, NVML healthy
- [ ] **A1** Disk: ≥1TB free (`df -h .`) — king + cache + ckpts
- [ ] **A2** Network: HTTPS to `teutonic.ai` + GitHub + R2 OK
- [ ] **A3** Clone/pull `main`, note commit: `git log -1 --oneline`
- [ ] **A4–A7** Hardware-aware B300 env (diagnose → FA4/TE → check):
  ```bash
  bash scripts/setup/setup_env.sh --profile b300
  source .venv/bin/activate
  bash scripts/setup/check_env.sh
  # diagnose only: bash scripts/setup/setup_env.sh --diagnose-only
  ```
- [ ] **A8** Launch defaults (also set by `run_b300_1gpu.sh`):
  ```bash
  export TEUTONIC_ATTN=flash_attention_4 GPUS_PER_NODE=1
  export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
  export FLASH_ATTENTION_CUTE_DSL_CACHE_DIR=$PWD/.cache/fa4_cute
  ```

**A pass:** FA4 + TE available on this GPU image. Same install path is what 8× will use.

**Start 110B download in background now (feeds G):**
```bash
mkdir -p logs
python scripts/data/download_king.py --output ./king --workers 8 > logs/download_king.log 2>&1 &
```

---

## B — Shrink king + sanity (fast)

- [ ] **B1–B4** `GATE=B bash scripts/train/run_b300_1gpu_tests.sh` (builds `./pipeline_model`, verifier, unit smoke)
- [ ] **B2** Confirm `pipeline_model/king_info.json`: ~8.5B+, layers=4, experts=256

**B pass:** mini king loads; contracts green.

---

## C — Speed matching (prove the fast path)

```bash
GATE=C bash scripts/train/run_b300_1gpu_tests.sh
# logs/b300_1gpu_C1_eager.log  C2_fa4.log  C3_mxfp8.log
```

- [ ] **C1** `configs/b300/1gpu_speed_eager.yaml` — eager + BF16 (s/step, tok/s, peak GiB)
- [ ] **C2** `configs/b300/1gpu_speed_fa4.yaml` — FA4 + BF16 (+ first-step JIT time)
- [ ] **C3** `configs/b300/1gpu_speed_mxfp8.yaml` — FA4 + MXFP8 strict (no fallback)
- [ ] **C4** Optional: bump `smoke_seq_len` to 512/1024 on mxfp8 — 3 steps (stop at OOM)
- [ ] **C5** Freeze winner (expect FA4+MXFP8) for D–F

**C pass:** FA4 + MXFP8 work on this GPU + shrink MoE + this pipeline.

---

## D — Real Teutonic data on shrink (short CPT)

```bash
GATE=D bash scripts/train/run_b300_1gpu_tests.sh
# configs/b300/1gpu_mini_real.yaml → 20 steps, pool ./sequence_pool_b300_1gpu
```

- [ ] **D1** Pool built under `sequence_pool_b300_1gpu/`
- [ ] **D2** Pool shape `[N, 2048]` uint32; all three manifest sources
- [ ] **D3** Train finishes (`logs/b300_1gpu_D_real.log`)
- [ ] **D4** Log is **not** synthetic smoke batches
- [ ] **D5** `lm_loss` finite; note tok/s + peak GiB
- [ ] **D6** `checkpoints/b300_1gpu_mini_real/` + `COMMITTED`

**D pass:** real-data path + fast stack work on 1×B300.

---

## E — Soup, LoRA, resume (short)

```bash
GATE=E bash scripts/train/run_b300_1gpu_tests.sh
# E1: configs/b300/1gpu_soup.yaml + --trigger-decay
# E2: configs/b300/1gpu_lora.yaml
# E3: resume from b300_1gpu_mini_real if D ran
```

### E1 Soup (checkpoint averaging)

- [ ] **E1a** Soup train saved ≥2 stable DCPs under `checkpoints/b300_1gpu_soup/`
- [ ] **E1b** `--trigger-decay` runs (`logs/b300_1gpu_E1_soup_decay.log`)
- [ ] **E1c** Log shows averaging N checkpoints (not “No stable checkpoints”)
- [ ] **E1d** Post-soup steps: loss finite

**E1 pass:** soup path works on MoE DCP.

### E2 LoRA

- [ ] **E2a** LoRA 5-step smoke PASS (`logs/b300_1gpu_E2_lora.log`)
- [ ] **E2b** Log shows LoRA enabled; loss finite; no hang

**E2 pass:** LoRA wiring OK on B300 stack.

### E3 DCP resume (quick)

- [ ] **E3a** Resume from a DCP dir; runs next steps
- [ ] **E3b** (Nice) first resumed `lm_loss` vs uninterrupted

---

## F — Layer-count speed ladder (shrink)

`--num-layers` now allows **3, 4, 6, 8**.

```bash
GATE=F bash scripts/train/run_b300_1gpu_tests.sh
LAYERS_EXTRA=6 GATE=F bash scripts/train/run_b300_1gpu_tests.sh   # optional
```

- [ ] **F1** `pipeline_model_L3` built
- [ ] **F2** `pipeline_model_L4` built
- [ ] **F3** 5 steps each FA4+MXFP8 (`logs/b300_1gpu_F_L*.log`)
- [ ] **F4** Fill table:

| Layers | ~params | s/step | tok/s | peak GiB |
|--------|---------|--------|-------|----------|
| 3 | | | | |
| 4 | | | | |
| (opt) 6/8 | | | | |

- [ ] **F5** Optional `LAYERS_EXTRA=6` or `8` — **stop if >20 min**

**F pass:** clear scaling trend with depth.

---

## G — Real 110B on this device (eval-focused)

Do **not** block A–F on this. Download in background.

```bash
GATE=download bash scripts/train/run_b300_1gpu_tests.sh   # if not already bg
GATE=G bash scripts/train/run_b300_1gpu_tests.sh          # prefetch + eval-n 48
```

- [ ] **G1** `./king` complete + checksums OK
- [ ] **G2** Load-only: peak HBM GiB, headroom vs 288
- [ ] **G3** Forward smoke optional if load near OOM
- [ ] **G4** Offline eval smoke → `results/b300_110b_eval_smoke48.json`
- [ ] **G5** Metrics finite
- [ ] **G6** **Skip** full 110B train on 1× (world=1 does not shard)

**G pass:** “we can eval real 110B on this B300.”

---

## H — Extra gates you might miss (still keep short)

- [ ] **H1** FA4 JIT cache reuse: second run first-step << first run (cache dir works)
- [ ] **H2** No CPU-offload thrash (`fsdp_cpu_offload=false` unless OOM)
- [ ] **H3** Gradients / z / aux losses finite on a real-data step
- [ ] **H4** `nvidia-smi` during train: GPU busy, not stalled
- [ ] **H5** Write results into `docs/TEST.md` before killing the rent
- [ ] **H6** Note exact torch / FA4 / TE versions for 8× image parity

### Explicitly NOT on this 1× rent (do on 8× later)

- [ ] ~~8× NCCL / IB~~
- [ ] ~~Per-expert FSDP collectives at world=8~~
- [ ] ~~110B sharded train burn-in~~
- [ ] ~~Full WSD / multi-hour CPT~~
- [ ] ~~Full eval corpus (all shards)~~ — smoke only here

---

## Go / no-go → rent 8×B300

**GO** if all of these are ticked:

- [ ] A5–A7 FA4 + TE/MXFP8 OK
- [ ] C5 winner settings chosen (FA4+MXFP8 preferred)
- [ ] D3–D5 real-data mini CPT OK
- [ ] E1 soup path OK
- [ ] E2 LoRA smoke OK
- [ ] G1–G5 110B eval smoke OK (or G2 proves load + documented eval blocker)

**NO-GO / fix first** if:

- FA4 install fails on this image
- MXFP8 refused / silent fallback
- Mini real-data train hangs or NaNs
- 110B cannot load at all for eval

**First hour on 8× (after GO):** 5-step NCCL bring-up → 10-step 110B shard train → DCP once → then longer burn-in.

---

## Suggested command skeleton (1×)

```bash
source .venv/bin/activate
# one-off:
bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_mini_real.yaml
# layer override:
KING_DIR=./pipeline_model_L3 OUTPUT_DIR=./checkpoints/b300_1gpu_L3 \
  bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_speed_mxfp8.yaml
# soup trigger:
bash scripts/train/run_b300_1gpu.sh --config configs/b300/1gpu_soup.yaml \
  --resume checkpoints/b300_1gpu_soup/step_XXXXXXX --trigger-decay
```

---

## Results scrap (fill while testing)

| Gate | Pass? | s/step | tok/s | peak GiB | Notes |
|------|-------|--------|-------|----------|-------|
| C1 eager BF16 | | | | | |
| C2 FA4 BF16 | | | | | |
| C3 FA4 MXFP8 | | | | | |
| D real 10–20 | | | | | |
| E1 soup | | | | | |
| E2 LoRA | | | | | |
| F3 L3 | | | | | |
| F3 L4 | | | | | |
| G2 110B load | | — | — | | |
| G4 eval-n48 | | — | — | | |

**Torch / FA4 / TE versions:**

```
torch=
FA4=
TE=
commit=
```

**Winner train settings for 8× starting point:**

```
attn=
low_precision=
seq=
mbs=
ac=
```
