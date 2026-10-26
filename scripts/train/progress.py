#!/usr/bin/env python3
"""Compact progress report for the node's training runs (reads history.jsonl; no GPU needed).

  python scripts/train/progress.py            # all runs under /workspace/teutonic/runs
  python scripts/train/progress.py line_a     # one run
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUNS = Path("/workspace/teutonic/runs")
DELTA = 0.003
TARGET = 0.0036  # our own-sample bar (see docs/CPT.md)


def fmt_eta(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds > 5400 else f"{seconds / 60:.0f} min"


def report(run: Path) -> None:
    history = run / "history.jsonl"
    if not history.exists():
        print(f"{run.name}: no history yet")
        return
    records = [json.loads(line) for line in history.read_text().splitlines() if line.strip()]
    trains = [r for r in records if "loss" in r]
    vals = [r for r in records if "val" in r and r.get("step", 0) > 0 and "lr" in r]
    start = next((r for r in records if r.get("step") == 0 and "val" in r), None)
    cfg = json.loads(json.dumps(__import__("yaml").safe_load((run / "config.yaml").read_text()))) if (run / "config.yaml").exists() else {}
    total = int(cfg.get("schedule", {}).get("total_steps", 0))
    done = (run / "DONE").exists()
    print(f"=== {run.name} {'(DONE)' if done else ''}")
    if start and "start_vs_king" in start:
        s = start["start_vs_king"]
        print(f"  start body vs king #30 on val: mu_hat {s['mu_hat']:+.5f} (se {s['se']:.5f})  "
              + "  ".join(f"{k[:5]} {v:+.4f}" for k, v in s.items() if k not in ("mu_hat", "se", "win_rate")))
    if trains:
        t = trains[-1]
        recent = trains[-5:]
        tps = sum(r["tok_per_s"] for r in recent) / len(recent)
        steps_left = max(total - t["step"] - 1, 0)
        sec_per_step = sum(r.get("sec_per_step", 0) for r in recent) / len(recent) or 1
        print(f"  step {t['step'] + 1}/{total}  loss {t['loss']:.4f}  lr {t['lr']:.2e}  grad_norm {t['grad_norm']:.3f}  "
              f"{tps:,.0f} tok/s  {t['tokens_seen'] / 1e9:.3f} B tokens seen  "
              f"gpu peak {t.get('gpu_gb', {}).get('peak', '?')} GB  ETA to cap {fmt_eta(steps_left * sec_per_step)}")
    for v in vals[-3:]:
        parts = [f"  val@{v['step']}: loss {v['val']['all']:.5f}"]
        for key, label in (("vs_king", "raw vs king"), ("ema_vs_king", "EMA vs king"), ("vs_start", "raw vs start"), ("ema_vs_start", "EMA vs start")):
            if key in v:
                parts.append(f"{label} {v[key]['mu_hat']:+.5f}")
        print("  ".join(parts))
    if vals and "ema_vs_king" in vals[-1]:
        e = vals[-1]["ema_vs_king"]
        worst = min((k, x) for k, x in e.items() if k not in ("mu_hat", "se", "win_rate"))
        need = TARGET - e["mu_hat"]
        print(f"  EMA vs king now {e['mu_hat']:+.5f} ± {e['se']:.5f} (target {TARGET:+.4f}, bar {DELTA:+.4f}); "
              f"worst source {worst[0]} {worst[1]:+.5f}; {'READY TO CONFIRM with parity eval' if need <= 0 else f'need {need:+.5f} more'}")
    exports = sorted(Path(cfg.get("export_dir", run / "exports")).glob("step*")) if cfg else []
    if exports:
        print("  exports: " + ", ".join(e.name + (" [KEEP]" if (e / "KEEP").exists() else "") for e in exports))
    for trigger in ("COOLDOWN", "STOP", "EXPORT"):
        if (run / trigger).exists():
            print(f"  trigger armed: {trigger}")


def main() -> None:
    names = sys.argv[1:]
    runs = [RUNS / n for n in names] if names else sorted(p for p in RUNS.iterdir() if p.is_dir())
    print(time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    for run in runs:
        report(run)
    for mount in ("/dev/shm", "/mnt/ram", "/"):
        usage = shutil.disk_usage(mount)
        print(f"  {mount}: {usage.used / 1e9:.0f} GB used, {usage.free / 1e9:.0f} GB free")
    try:
        mem = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.splitlines()[1].split()
        print(f"  RAM: {mem[2]} GB used, {mem[6]} GB available of {mem[1]} GB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
