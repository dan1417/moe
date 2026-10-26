#!/usr/bin/env python3
"""Score every checkpoint of a cpt_fsdp.py run on the TEXT screen vs the king, pausing the trainer for the GPUs.

  score_ckpts.py --run /data/teutonic/ckpt/text_lora_a --king $T1 [--limit 2400] [--only step-200,step-400] [--once]

For each new <run>/step-N (or final) with a complete index: touch <run>/PAUSE, wait for <run>/PAUSED (the trainer frees its
activation cache and idles), run `parity_eval.py --competition text duel` (text-competition sample tag:A, first --limit
sequences; king losses cached) and `--competition mix score` (so catdiag41 can read per-category gains), register the
checkpoint in configs/merge/k41/models.json as <run>_<step>, append the verdict to <run>/scores.json, remove PAUSE.
Re-renders the dashboard (scripts/train/dashboard.py) after every score. Safe to restart: scored checkpoints are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def complete(ckpt: Path) -> bool:
    return (ckpt / "model.safetensors.index.json").exists() and (ckpt / "config.json").exists()


def ckpt_order(name: str) -> int:
    m = re.match(r"step-(\d+)$", name)
    return int(m.group(1)) if m else 10**9


def run(cmd: list[str], log: Path) -> str:
    with log.open("a") as h:
        h.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=REPO)
        h.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[:3]} failed ({proc.returncode}); see {log}")
    return proc.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--king", required=True)
    ap.add_argument("--limit", type=int, default=2400)
    ap.add_argument("--only", default=None, help="comma list of checkpoint names to score (default: all)")
    ap.add_argument("--no-mix", action="store_true", help="skip the mix-sample scoring (catdiag input)")
    ap.add_argument("--once", action="store_true", help="score what exists and exit (default: keep polling)")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--dashboard", type=Path, default=Path("/data/teutonic/dashboard.html"))
    ap.add_argument("--dashboard-args", default="", help="extra args for dashboard.py (e.g. '--steps 800 --name X')")
    args = ap.parse_args()
    out = args.run
    scores_path, log = out / "scores.json", out / "score_ckpts.log"
    only = set(args.only.split(",")) if args.only else None
    while True:
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else []
        done = {s["ckpt"] for s in scores if s.get("status") == "done"}
        todo = sorted((d.name for d in (out.iterdir() if out.exists() else []) if d.is_dir() and complete(d) and d.name not in done
                       and (only is None or d.name in only)), key=ckpt_order)
        if not todo:
            if args.once:
                break
            time.sleep(args.poll)
            continue
        name = todo[0]
        ckpt = out / name
        label = f"{out.name}_{name.replace('step-', 's')}"
        scores = [s for s in scores if s["ckpt"] != name] + [{"ckpt": name, "status": "scoring"}]
        scores_path.write_text(json.dumps(sorted(scores, key=lambda s: ckpt_order(s["ckpt"])), indent=1))
        (out / "PAUSE").touch()
        t0 = time.time()
        while not (out / "PAUSED").exists() and not (out / "final").exists() and time.time() - t0 < 7200:
            time.sleep(5)  # the trainer pauses after its current step (or has finished: then the GPUs are free anyway)
        try:
            res = run([PY, "scripts/eval/parity_eval.py", "--competition", "text", "--limit", str(args.limit), "duel",
                       "--king", args.king, "--challenger", str(ckpt), "--sample", "tag:A"], log)
            m = re.search(r"^saved (.+\.json)$", res, re.M)
            payload = json.loads(Path(m.group(1)).read_text())
            v = payload["results"][0]["verdict"]
            src = v.get("source_scores") or {}
            splits = {k: (val.get("mu_hat") if isinstance(val, dict) else val) for k, val in src.items()}
            if not args.no_mix:
                run([PY, "scripts/eval/parity_eval.py", "--competition", "mix", "--limit", str(args.limit), "score",
                     "--model", str(ckpt), "--sample", "tag:A"], log)
            registry = REPO / "configs/merge/k41/models.json"  # optional: lets the per-category diagnostics find the checkpoint
            if registry.exists():
                models = json.loads(registry.read_text())
                models[label] = str(ckpt)
                registry.write_text(json.dumps(models, indent=1))
            entry = {"ckpt": name, "status": "done", "label": label, "text_mu": v.get("mu_hat"), "text_se": v.get("se_iid"),
                     "lcb": v.get("lcb"), "win_rate": v.get("win_rate"), "splits": splits, "n": v.get("n") or args.limit,
                     "categories": v.get("category_scores"), "result_file": m.group(1), "note": ""}
        except Exception as exc:  # noqa: BLE001
            entry = {"ckpt": name, "status": "pending", "note": f"scoring failed: {exc}"[:200]}
        finally:
            (out / "PAUSE").unlink(missing_ok=True)
        scores = json.loads(scores_path.read_text())
        scores = [s for s in scores if s["ckpt"] != name] + [entry]
        scores_path.write_text(json.dumps(sorted(scores, key=lambda s: ckpt_order(s["ckpt"])), indent=1))
        print(f"{time.strftime('%H:%M:%S')} {name}: {entry.get('status')} TEXT mu {entry.get('text_mu')} lcb {entry.get('lcb')} "
              f"splits {entry.get('splits')} ({time.time() - t0:.0f}s)", flush=True)
        subprocess.run([PY, "scripts/train/dashboard.py", "--run", str(out), "--out", str(args.dashboard)] + args.dashboard_args.split(), cwd=REPO)
        if args.once and len(todo) == 1:
            break


if __name__ == "__main__":
    main()
