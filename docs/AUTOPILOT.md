# Autopilot: train → screen → confirm → stage, merge on every new king

Two commands run the whole loop on an 8× B300 box (after `setup.sh` / `download.sh`):

```bash
bash train.sh math [start]     # autopilot for one competition (pm2 autopilot_math); runs until you stop it
bash merge.sh math             # merge mode now (a running autopilot does it at its next check)
bash train.sh status           # kings + phase / run / LR / best screens / SUBMIT-READY per autopilot
bash train.sh stop math        # stop the autopilot and its trainer (checkpoint + exact state first)
```

`start` (first run only) is a checkpoint dir, a `models.json` name (`MA2`) or a merge label. Without it the autopilot
uses `search/train_inits.json` (written by merge mode), else the reigning king. Restarting the autopilot (pm2 restart,
box reboot with `pm2 resurrect`) adopts the running lineage from `/data/teutonic/autopilot/<comp>/state.json`.

## What one autopilot does

```
            ┌──────────────── new king (target competition, or MAIN) ────────────────┐
            ▼                                                                         │
  MERGE MODE (merge4train.py)                                                         │
  stop trainer (ckpt + exact state) → soups / transplants of our training delta       │
  onto the king, "ours as is", public best → choose on an independent sample          │
            │ start                                                                   │
            ▼                                                                         │
  TRAIN (cpt_fsdp.py)  ── every 10 steps: EMA val → LR controller (CONTROL.json) ─────┤
            │          ── every 20 steps: checkpoint + exact state                    │
            │          ── crash / hang / OOM / NaN → resume the exact state           │
            ▼ every 40 steps                                                          │
  SCREEN all 4 competitions vs their kings (2 400 seqs, sample A) ───────────────────┘
            │ mu ≥ bar + 0.0002 and LCB > bar
            ▼
  CONFIRM on full samples B and C → LCB > bar on both → STAGE (random hidden permutation,
  verify_submission, HF backup) → search/autopilot_ready.json + SUBMIT_READY event.  You submit.
```

### Training (scripts/train/cpt_fsdp.py)
- **Exact state** every `--state-every` steps (`<run>/state-N`: fp32 masters, 8-bit Adam moments, EMA, schedule and
  control state; ~100 GB per rank, ~1 min). A resume continues the same run dir with the same step numbering and data
  order and is bit-identical to an uninterrupted run (smoke test: resumed steps reproduce the losses to 0.0).
  Before this, every crash or LR change restarted from a bf16 checkpoint with fresh Adam moments and a re-warmup.
- **Live control** (`<run>/CONTROL.json`, acknowledged in `CONTROL.ack.json`): `lr_scale`, `anneal` (cosine to the
  floor over K steps), one-shot `save` / `state` / `stop`. No restart is needed to change the LR.
- **Step guard**: a non-finite gradient or |g| > 5× the recent median skips the optimizer step; three non-finite steps
  in a row exit (the autopilot resumes the newest state at half the LR).

### LR policy (autopilot `lr_decision`)
Default since 2026-10-14: **`wsd`** = hands off. The recipe is 5DALwjE4's, read off their consecutive kings
(docs/ANALYSIS_2026-10-14.md): lr 5e-6 constant (warmup 100, no decay), AdamW weight decay 0.1, only routers frozen,
0.5M-token steps (`16x2` x 2 048, TEXT / MAIN `4x2` x 8 192; ~10 s per step, 8-9k updates a day), EMA 0.99, exact
state every 200 steps, screens every 1 000 steps on 4 800 seqs, 10 000-step segments (~5 B tokens, ~1.2 days). The
controller never raises, lowers or anneals; it only rolls back a blow-up (val > best + `rollback_rise` 0.02) to the
best exact state at half the scale. The screens judge; the next segment continues from the best screened checkpoint.

`POLICY=ema` restores the previous adaptive controller (judge the paired EMA-val trend over >= 60 steps: improving ->
keep; flat -> anneal 40 steps, screen, restore; rising -> x0.6; > 3e-4 above the best -> roll back) for the old
small-step runs (`LR=5e-6 WD=0 BATCHES=16x16`).

`REQUEST_RELAUNCH` (a file in `/data/teutonic/autopilot/<comp>/`) stops the trainer cleanly and resumes its exact
state with the current configuration (e.g. pools built since the launch); `REQUEST_MERGE` asks for a merge round.

### Screens and confirmation
Every 40 steps the trainer pauses (~85 GB per GPU stays allocated) and the checkpoint is scored on MAIN, MATH, CODE and
TEXT against each competition's reigning king (resolved strictly by `scripts/box/kings.py`: a king that is not ingested
is never guessed). A MATH run can win CODE; t007 (a MAIN-run checkpoint) took CODE that way. Confirmation uses the
full samples B and C, which played no part in the selection.

### Merge mode (scripts/merge/merge4train.py)
Candidates vs the competition's king K and, for a specialist, also on the newest MAIN king X when it is a different
model (the MATH / CODE / TEXT kings have been X re-submitted with a few Adam steps of noise; a new X beats the old
specialist kings by 0.0025-0.0045 on their mixes): `ours as is` (O), soups `(1-a)K + aO` / `(1-a)X + aO`
(a = 0.3/0.5/0.7), transplants `K + b(O - S)` / `X + b(O - S)` (S = O's segment start: our pure training delta,
b = 0.5/1.0), the public campaign's best for K, X as is (a training start only), soups `0.85 X + 0.15 (another
miner's newest king)` (a training start only; what 5DALwjE4's M44 was), and, when cos(O - S, X - parent) > 0.25 (our
delta replays the king's own step, which extrapolating measured negative), the orthogonalized transplant
`X + b ((O - S) ⟂ (X - parent))` (`delta_orthogonal_pair` in src/parity/weights.py). The cosine is logged every round. "Ours" = the newest and best
checkpoints of EVERY competition's autopilot lineage (a TEXT lineage is an ingredient for MATH / CODE / MAIN). Round 2
refines the best family's coefficient. All are affine and content-addressed. The top 2 on sample A get an independent
sample B screen; the choice is the best on B (near-ties: a submittable candidate first, then the least extrapolated).
If "ours as is" wins, training resumes from its exact state with nothing reset. A merge that already clears the bar on
B is confirmed on full B and C and staged like a training checkpoint, **but only when it carries a material
contribution of ours** (soup a >= 0.2, transplant b >= 0.5, ours as is) or is a merge of several public models: a king
or X as is is never staged (copies are against the subnet's rules and are not our work).

### Retention
Keeps the newest two checkpoints, the best checkpoint per competition (vs the current king), staged ones and every
segment start; finished segments keep only those. States: the trainer's newest two, plus the best-val and anneal
bases as hard links.

## Files
| what | where |
|---|---|
| lineage state / dashboard status | `/data/teutonic/autopilot/<comp>/state.json`, `status.json` |
| autopilot log | `/data/teutonic/logs/autopilot_<comp>.log` (`pm2 logs autopilot_<comp>`) |
| screen / merge scorer output | `logs/autopilot_<comp>_screens.log`, `logs/autopilot_<comp>_merge.log`, `logs/merge4train.log` |
| trainer | `ckpt/<comp>_ap_<MMDDHHMM>/` (train_log.jsonl, step-N, state-N, CONTROL.json), `logs/run_<run>.sh`, `logs/pm2_<run>.out.log` |
| staged submissions | `search/autopilot_ready.json`, `submit/ap_<label>/`, `logs/stage_<label>.log` |
| merge mode result | `search/merge4train_<comp>.json`, `search/train_inits.json` |

## Overrides
`STEPS LR WD DENSE WARMUP BATCHES FREEZE EMA SCREEN_EVERY SAVE_EVERY STATE_EVERY SCALE_MAX FOLLOW_MAIN POLICY DECAY_FRAC ROLLBACK_RISE` (env for train.sh),
`DRY=1` prints the plan. The previous fixed-recipe launcher is `LEGACY=1 bash train.sh <start> <comp>`.

## Submitting a staged winner
Check `logs/stage_<label>.log` for `STAGED` and the verify output, then follow docs/SUBMIT.md with the staged dir
(one submission per hotkey, irreversible; re-check the king right before `ready`).
