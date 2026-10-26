# Box orchestration scripts (B300 spot box, 2026-10-09/10)

Copies of the pm2 job scripts that ran on the box from `/data/teutonic/logs/` (they reference `/data/teutonic/...` paths and
`/data/teutonic/env.sh`; secrets are read from files, never stored here). Kept in git so the campaign can be rebuilt on a
new box. Chronology and results: `42.md`.

- merge campaigns: `campaign43_w*.sh` (vs #43), `text_kt2*.sh` / `all_kings.sh` (new TEXT king KT2), `code_*.sh` (new CODE
  king KC2), `soup_full.sh` (full-sample TEXT check)
- submissions: `write_*.sh` (CPU write of a recipe), `permute_*.sh` (random hidden permutation + verify), `submit_ckpt.py`
- data: `build_*_pools*.sh` (CODE / MATH / MAIN training pools)
- training: `main_train.sh` + `run_main_upper_ema_m1.sh` (MAIN run from the best merge), `main_watch.sh` (step-50 screens)
- ops: `upload_best.sh` (HF backups), `render_dashboard.sh`, `ingest_*.sh` (new kings)
