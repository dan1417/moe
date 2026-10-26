#!/usr/bin/env python3
"""Regenerate the results table for the current king inside MERGE.md from the loss cache.

  merge_report.py --king reign:30 --prefix u30_ --sample tag:A [--sample tag:B]
The table is written between <!-- K30_RESULTS --> and <!-- /K30_RESULTS --> markers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts" / "eval"))
import numpy as np  # noqa: E402

from parity_eval import resolve_recipe, resolve_sample  # noqa: E402
from src.parity.cache import LossCache, scorer_signature  # noqa: E402
from src.parity.sample import load_live_config  # noqa: E402
from src.parity.verdict import duel  # noqa: E402

SOURCES = ["code-reasoning", "math-reasoning", "pretrain-behaviors", "sft-reasoning", "txt360-v2"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", default="reign:30")
    ap.add_argument("--prefix", default="u30_")
    ap.add_argument("--sample", action="append", default=None)
    ap.add_argument("--marker", default="K30_RESULTS")
    ap.add_argument("--doc", default=str(REPO / "MERGE.md"))
    args = ap.parse_args()
    live = load_live_config()
    cache, sig = LossCache(), scorer_signature("fa4sim", 1024)
    king = resolve_recipe(args.king)
    descs = {}
    for p in (REPO / "configs" / "merge").glob(f"{args.prefix}*.json"):
        d = json.loads(p.read_text()); descs[d.get("label", p.stem)] = (d.get("mode", "linear"), d.get("description", ""))
    rows = []
    for spec in args.sample or ["tag:A"]:
        sample = resolve_sample(spec, live)
        kl = cache.load(sample.sample_id, king.identity(), sig, sample.n)
        for e in cache.entries(sample.sample_id):
            label = (e.get("recipe") or {}).get("label") or ""
            if not label.startswith(args.prefix) or e["signature"] != sig:
                continue
            c = cache.load(sample.sample_id, e["model_id"], sig, sample.n)
            both = np.isfinite(kl) & np.isfinite(c)
            n = int(np.argmin(both)) if not both.all() else sample.n
            if n < 500:
                continue
            m = np.zeros(sample.n, bool); m[:n] = True
            v = duel(np.where(m, kl, np.nan), np.where(m, c, np.nan), sample, delta=live.delta_threshold)
            mode, desc = descs.get(label, ("?", ""))
            rows.append((v["mu_hat"], label, mode, desc, spec, n, v))
    rows.sort(key=lambda r: -r[0])
    lines = [f"_Regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from the loss cache; mu_hat = mean(king CE − candidate CE), positive = beats the king; need ≈ +0.0036._",
             "", "| candidate | method | what it does | sample | seqs | **mu_hat** | LCB | win % | code | math | pretrain | sft | txt360 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for mu, label, mode, desc, spec, n, v in rows:
        s = v["source_scores"]
        lines.append(f"| `{label}` | {mode} | {desc} | {spec} | {n} | **{mu:+.5f}** | {v['lcb']:+.5f} | {100 * v['win_rate']:.0f} | "
                     + " | ".join(f"{s[k]['mu_hat']:+.4f}" if k in s else "–" for k in SOURCES) + " |")
    if not rows:
        lines.append("| (no results yet) | | | | | | | | | | | | |")
    doc = Path(args.doc).read_text()
    start, end = f"<!-- {args.marker} -->", f"<!-- /{args.marker} -->"
    if start not in doc:
        raise SystemExit(f"marker {start} not found in {args.doc}")
    pre, rest = doc.split(start, 1); _, post = rest.split(end, 1)
    Path(args.doc).write_text(pre + start + "\n" + "\n".join(lines) + "\n" + end + post)
    print(f"{len(rows)} rows written to {args.doc}")


if __name__ == "__main__":
    main()
