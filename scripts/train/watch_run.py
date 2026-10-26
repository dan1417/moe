#!/usr/bin/env python3
"""Emit one compact line per new eval / export / terminal state of a training run (for Monitor or a terminal).

  python scripts/train/watch_run.py line_a2          # polls every 20 s; never exits on its own
Reads history.jsonl (written by rank 0), the export dir and the pm2 status — no log grepping.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

RUNS = Path("/workspace/teutonic/runs")


def fmt(v: dict) -> str:
    parts = [f"{k[:5]} {x:+.4f}" for k, x in v.items() if k not in ("mu_hat", "se", "win_rate")]
    return f"{v['mu_hat']:+.5f}±{v['se']:.5f} [{' '.join(parts)}]"


def pm2_status(name: str) -> str:
    try:
        out = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=20).stdout
        for proc in json.loads(out[out.index("["):]):
            if proc.get("name") == name:
                return proc.get("pm2_env", {}).get("status", "?")
    except Exception:
        return "?"
    return "absent"


def main() -> None:
    name = sys.argv[1]
    run = RUNS / name
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    export_dir = Path(cfg.get("export_dir") or run / "exports")
    seen_lines = len((run / "history.jsonl").read_text().splitlines()) if (run / "history.jsonl").exists() else 0
    seen_exports = {p.name for p in export_dir.glob("step*")} if export_dir.exists() else set()
    last_status = None
    print(f"watching {name}: {seen_lines} history lines, exports {sorted(seen_exports)}", flush=True)
    while True:
        lines = (run / "history.jsonl").read_text().splitlines() if (run / "history.jsonl").exists() else []
        for line in lines[seen_lines:]:
            rec = json.loads(line)
            if "val" in rec and rec.get("step", 0) > 0 and "lr" in rec:
                msg = f"EVAL step {rec['step']} lr {rec['lr']:.1e}"
                for key, label in (("ema_vs_king", "EMA vs king"), ("vs_king", "raw vs king"), ("ema_vs_start", "EMA vs start"), ("vs_start", "raw vs start")):
                    if key in rec:
                        msg += f" | {label} {fmt(rec[key]) if 'king' in key else format(rec[key]['mu_hat'], '+.5f')}"
                print(msg, flush=True)
            elif "loss" in rec and rec["step"] % 100 == 0:
                print(f"step {rec['step']} loss {rec['loss']:.4f} lr {rec['lr']:.1e} gnorm {rec['grad_norm']:.3f} "
                      f"{rec.get('tok_per_s', 0):,} tok/s" + (f" noise {rec['noise']}" if "noise" in rec else ""), flush=True)
        seen_lines = len(lines)
        exports = {p.name for p in export_dir.glob("step*")} if export_dir.exists() else set()
        for new in sorted(exports - seen_exports):
            if (export_dir / new / "model.safetensors.index.json").exists():
                print(f"EXPORT ready {export_dir / new}", flush=True)
                seen_exports.add(new)
        for gone in sorted(seen_exports - exports):
            print(f"export pruned {gone}", flush=True)
            seen_exports.discard(gone)
        status = pm2_status(name)
        if (run / "DONE").exists() and status != last_status:
            print(f"DONE {(run / 'DONE').read_text().strip()} (pm2 {status})", flush=True)
        elif status != last_status and status not in ("online", "?"):
            print(f"PM2 STATUS {name}: {status} — check pm2 logs {name}", flush=True)
        last_status = status
        time.sleep(20)


if __name__ == "__main__":
    main()
