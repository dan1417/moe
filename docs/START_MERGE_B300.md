# Run the pipeline on a new box (no LLM needed)

```bash
git clone https://github.com/wilfrid51/MoE.git /root/workspace/MoE && cd /root/workspace/MoE
HF_TOKEN=hf_xxx bash setup.sh
bash download.sh 10
bash merge.sh
bash dashboard.sh
# later, with a merge label from the campaign (or a checkpoint dir / model name):
bash train.sh pub_xxxxxxxxxx math
```
Requirements: 8 GPUs with ≥ 180 GB each (B300), ≥ 1.5 TB RAM, ≥ 4 TB NVMe at `/data` (each model is 206 GB), a fast uplink,
Python 3.12, an HF token that can read the gated `Valhalla-ai` repos. Every script is safe to re-run.

| script | what it does | how long |
|---|---|---|
| `setup.sh` | `.venv-eval` (CUDA 13 torch, the validator's transformers fork + package, flash-attn-4, bitsandbytes), `/data/teutonic` layout + `env.sh`, helper scripts into `/data/teutonic/logs`, HF token, pm2; links `/root/workspace/MoE` if cloned elsewhere | 5–15 min |
| `download.sh [N] [--once]` | the N most recently crowned public kings (default 10) + every reigning king + T1 (the basis), each downloaded with 8 processes (~1 GB/s), hidden permutation undone, registered; then checks every 5 min and downloads each new king (pm2 `download_kings`) | ~5 min per model |
| `merge.sh [anchor] \| stop` | the merge campaign (`scripts/merge/public_merge.py`, pm2 `public_merge`): raw table → probes toward / away from every model → per-competition search with TIES / DARE / DELLA / EMR / breadcrumbs / Model Stock / SLERP / per-layer scales, each candidate scored on MAIN / MATH / CODE / TEXT vs the reigning kings (1 200 seqs each, ~2 min). Near the bar it confirms automatically (4 800 seqs → full A/B/C) and writes + uploads a passing merge | runs until converged |
| `train.sh <model> <competition>` | writes the merge (if a label), stops the campaign, builds the competition's data pools if missing, launches training with the best settings so far (TEXT: c11 recipe, 8 192-token long-document mix, lr 6e-6; MAIN / MATH / CODE: 4 096 × 2 048 tokens per step, lr 3e-6), plus auto-resume guard, checkpoint screens vs the reigning kings and retention. `DRY=1` prints the plan; `STEPS`, `LR`, `BATCHES`, `EVERY`, `EMA`, `FREEZE`, `FORCE=1` override | pools ~3 min each, then training |
| `dashboard.sh \| stop` | dashboard on port 8787 (current training run, merge panel with every candidate, screens) | – |

## Where results are
* `/data/teutonic/search/public.json` — `best` per competition: score, the merge label (`pub_…`, the argument for
  `train.sh`), recipe path; `probes`: what every public model does; `confirm`: larger-sample checks.
* `/data/teutonic/search/public_ready.json` — merges that passed the full validator-size check (written to
  `/data/teutonic/submit/<comp>_<label>`, uploaded to HF). **Submitting (hotkeys, `ready`) is manual.**
* Training: `/data/teutonic/ckpt/<run>/step-N` checkpoints, screens in `/data/teutonic/logs/pm2_main_watch.log`.

## Control
```bash
pm2 ls                         # every job
pm2 logs public_merge          # merge rounds, probes, new bests
pm2 logs download_kings        # downloads, new kings
pm2 logs launch_<run>          # training
bash merge.sh stop             # pause merging (resume: bash merge.sh)
python scripts/merge/cold_archiver.py --promote M30   # bring an archived king back from /root/teutonic_cold
```
A 13 TB-class volume at `/root` (≥ 5 TB free) is used as cold storage for models leaving `/data` (`cold_archiver`).
