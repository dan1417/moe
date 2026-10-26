#!/usr/bin/env python3
"""After king_merge's decision on the new MAIN king KMAIN_c7ef5487 (uid 107, 5DAL): can it -- or our best merge with it --
take MATH from the MATH king KMATH_ddd9e626 (bar 0.002), and does it threaten our crowns CODE (t007 = m1/step-150) and
TEXT (t005 = soup_k43_c1_050)? Waits until the (switched) MAIN trainer is stepping, pauses it with PAUSE ownership,
screens 4 800 seqs of sample A, resumes.

  MATH (vs KMATH): KMAIN raw, KMAIN + 0.25 / 0.5 x (KMATH - #43), soup KMAIN / KMATH 0.5
  CODE (vs t007), TEXT (vs t005): KMAIN raw
"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/data/teutonic/logs")
import king_merge as km  # noqa: E402

R = km.REPO / "configs/merge/kmain_c7ef5487"
LOG = km.LOGS / "pm2_king_merge.log"


def decision() -> str | None:
    t = LOG.read_text(errors="replace")
    i = t.rfind("BEST MAIN after KMAIN_c7ef5487")
    if i < 0:
        return None
    tail = t[i:]
    for key in ("switched to ", "keep training", "no MAIN reference", "write_merge failed"):
        if key in tail:
            return key.strip()
    return None


while (d := decision()) is None:
    time.sleep(30)
km.log(f"cross check: king_merge decided ({d})")
while True:  # the newest MAIN run must be stepping (the switched run loads ~10 min before its first step)
    run = sorted((km.D / "ckpt").glob("main_upper_ema_m*"), key=lambda p: p.stat().st_mtime)[-1]
    tl = run / "train_log.jsonl"
    if tl.exists() and tl.read_text().strip():
        break
    time.sleep(30)
own = False
try:
    while (run / "PAUSE").exists():
        time.sleep(10)
    (run / "PAUSE").touch(); own = True
    while not ((run / "PAUSED").exists() and (run / "PAUSED").stat().st_mtime >= (run / "PAUSE").stat().st_mtime):
        time.sleep(5)
    km.log(f"cross check: trainer {run.name} paused")
    common = "PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py --campaign newking --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 0"
    grep = "2>&1 | grep -E '^=== |  (MAIN|MATH|CODE|TEXT) |Traceback|Error'"
    math_recipes = [R / "kmain_c7ef5487_raw.json", R / "w2_kmain_c7ef5487_plus_kmathd_050_x050.json", R / "kmain_c7ef5487_plus_kmathd_050.json",
                    R / "soup_kmain_c7ef5487_kmath_050.json"]
    rc, out = km.sh(f"{common} {' '.join(f'--recipe {r}' for r in math_recipes)} --king math=KMATH_ddd9e626:0.002 --comps math {grep}")
    print(out, flush=True)
    rc, out = km.sh(f"{common} --recipe {R / 'kmain_c7ef5487_raw.json'} --king code=T007:0.002 --king text=T005:0.002 --comps code,text {grep}")
    print(out, flush=True)
finally:
    if own:
        (run / "PAUSE").unlink(missing_ok=True)
        km.log("cross check: training resumed")
