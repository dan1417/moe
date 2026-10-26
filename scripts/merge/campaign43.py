#!/usr/bin/env python3
"""Merge campaign against the four competitions' kings: every recipe is loaded ONCE and screened on all four masked
validator samples (MAIN/MATH vs the MAIN king, TEXT vs the TEXT king, CODE vs the CODE king).

  python scripts/merge/campaign43.py --recipe configs/merge/k43/a.json --recipe ... [--limit 2400] [--sample tag:A]

With --pingpong 1 (default) recipes run through src.parity.mppool.ReplicaPool: two sets of persistent replica
processes on the same 4 x 2 GPU groups, one set scoring recipe i while the other loads recipe i+1 into the free half
of the GPU memory, so the 8 GPUs never idle on a load (load ~65-150 s, a 4 x 2 400 screen ~100-140 s). Kings are
scored once (loss cache); fully cached items are skipped without a load.
Results are saved in parity_eval's duel format (eval_cache/results/*_duel.json), so the dashboard's Evaluation panel
lists every recipe per competition. Scoring is validator 14a9bb2 (document-masked, token-weighted).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

import numpy as np  # noqa: E402

import parity_eval as pe  # noqa: E402
from src.parity.sample import load_live_config  # noqa: E402
from src.parity.verdict import duel as duel_verdict  # noqa: E402

MODELS = json.loads((REPO / "configs/merge/k41/models.json").read_text())
# competition -> (king model name in models.json, bar)
KINGS = {"main": ("K43", 0.003), "math": ("K43", 0.002), "code": ("CK1", 0.002), "text": ("T1", 0.002)}
log = logging.getLogger("campaign43")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", action="append", required=True, help="recipe JSON (parity_eval format) or a model dir")
    ap.add_argument("--comps", default="main,math,code,text")
    ap.add_argument("--sample", default="tag:A", help="one or more sample specs, comma-separated (tag:A,tag:B,tag:C)")
    ap.add_argument("--limit", type=int, default=2400)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--replica-procs", type=int, default=1)
    ap.add_argument("--deterministic", type=int, default=1)
    ap.add_argument("--pingpong", type=int, default=1, help="two persistent replica sets: load the next recipe while scoring")
    ap.add_argument("--king", action="append", default=[], help="override a king: comp=NAME:bar (NAME in models.json), "
                    "e.g. text=KT2:0.002 for a new TEXT king")
    ap.add_argument("--campaign", default="k43", help="campaign tag stored with the results (dashboard grouping)")
    args = ap.parse_args()
    for spec in args.king:
        comp, rest = spec.split("=", 1)
        name, bar = rest.split(":")
        KINGS[comp] = (name, float(bar))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.deterministic:
        import os
        import torch
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
    comps = [c for c in args.comps.split(",") if c]
    live = {c: load_live_config(competition=c) for c in comps}
    specs = [x for x in args.sample.split(",") if x]
    # slots: one (competition, sample) pair per screened sample; every recipe is loaded once and scored on all slots
    slots = [(c, pe.resolve_sample(spec, live[c])) for c in comps for spec in specs]
    tag = lambda c, sample: c.upper() if len(specs) == 1 else f"{c.upper()}/{getattr(sample, 'label', '') or sample.sample_id[-4:]}"  # noqa: E731
    ns = argparse.Namespace(attn="flash_attention_4", lm_head_chunk=1024, deterministic=args.deterministic, gpus=None,
                            replicas=args.replicas, batch_size=args.batch_size, replica_procs=args.replica_procs)
    scorer = pe.Scorer(ns)
    kings = {c: pe.resolve_recipe(MODELS[KINGS[c][0]]) for c in comps}
    recipes = [(spec, pe.resolve_recipe(spec)) for spec in args.recipe]

    def report(spec, recipe, losses, seconds):
        desc = json.loads(Path(spec).read_text()).get("description", "") if spec.endswith(".json") else ""
        rows, results = [], []
        for (c, sample), chall in zip(slots, losses):
            king = kings[c]
            king_losses = scorer.cache.load(sample.sample_id, king.identity(), scorer.signature, sample.n)
            upto = min(args.limit, sample.n)
            mask = np.zeros(sample.n, dtype=bool)
            mask[:upto] = True
            v = duel_verdict(np.where(mask, king_losses, np.nan), np.where(mask, chall, np.nan), sample, delta=KINGS[c][1])
            results.append({"challenger": recipe.describe(), "king": king.describe(), "sample": sample.sample_id, "verdict": v})
            src = {k: round(x["mu_hat"], 5) for k, x in v["source_scores"].items()}
            rows.append(f"  {tag(c, sample):<7} vs {KINGS[c][0]:<4} mu {v['mu_hat']:+.6f}  lcb {v['lcb']:+.6f}  bar {KINGS[c][1]}  {src}")
        print(f"\n=== {recipe.label} ({seconds:.0f}s) {desc}\n" + "\n".join(rows), flush=True)
        print("saved", pe.save_result("duel", {"delta": None, "attn": "flash_attention_4", "campaign": args.campaign, "results": results}), flush=True)

    if not args.pingpong:
        for name in sorted({KINGS[c][0] for c in comps}):
            group = [c for c in comps if KINGS[c][0] == name]
            log.info("king %s on %s", name, group)
            scorer.losses(kings[group[0]], [sample for c, sample in slots if c in group], args.limit)
        for spec, recipe in recipes:
            t0 = time.time()
            report(spec, recipe, scorer.losses(recipe, [sample for _, sample in slots], args.limit), time.time() - t0)
        return

    import torch
    from src.parity.mppool import ReplicaPool

    def wanted(recipe, comps_):
        out, results = [], []
        for c, sample in slots:
            if c not in comps_:
                continue
            cached = scorer.cache.load(sample.sample_id, recipe.identity(), scorer.signature, sample.n)
            missing = [i for i in range(min(args.limit, sample.n)) if not np.isfinite(cached[i])]
            results.append(cached)
            if missing:
                out.append((sample, cached, missing))
        return out, results

    # kings first (one item per king model, on the competitions it rules), then the recipes in the given order
    jobs = []  # (kind, spec, recipe, results-per-comp)
    items = []
    for name in sorted({KINGS[c][0] for c in comps}):
        group = [c for c in comps if KINGS[c][0] == name]
        w, _ = wanted(kings[group[0]], group)
        if w:
            jobs.append(("king", name, kings[group[0]], None))
            items.append((kings[group[0]], w))
        else:
            log.info("king %s on %s: cached", name, group)
    # fully cached recipes are reported once every king is scored (a new king has no cached losses yet)
    cached_reports = []
    for spec, recipe in recipes:
        w, results = wanted(recipe, comps)
        if w:
            jobs.append(("recipe", spec, recipe, results))
            items.append((recipe, w))
        else:
            cached_reports.append((spec, recipe, results))
    n_kings, kings_done = sum(1 for j in jobs if j[0] == "king"), [0]

    def flush_cached():
        while cached_reports:
            spec, recipe, results = cached_reports.pop(0)
            report(spec, recipe, results, 0.0)

    if n_kings == 0:
        flush_cached()
    if not items:
        return
    gpus = list(range(torch.cuda.device_count()))
    per = len(gpus) // args.replicas
    groups = [gpus[r * per:(r + 1) * per] for r in range(args.replicas)]
    pool = ReplicaPool(groups, attn="flash_attention_4", lm_head_chunk=1024, batch_size=args.batch_size,
                       deterministic=bool(args.deterministic))

    def on_save(i, sample, cached, elapsed):
        recipe = items[i][0]
        scorer.cache.save(sample.sample_id, recipe.identity(), scorer.signature, cached,
                          {"recipe": recipe.describe(), "score_seconds": round(elapsed, 1), "replicas": len(groups),
                           "replica_procs": True, "pingpong": True})

    def on_done(i, load_s, score_s):
        kind, spec, recipe, results = jobs[i]
        if kind == "king":
            log.info("king %s scored (load %.0fs, score %.0fs)", spec, load_s or 0, score_s)
            kings_done[0] += 1
            if kings_done[0] == n_kings:
                flush_cached()
        else:
            log.info("%s: load %.0fs (overlapped), score %.0fs", recipe.label, load_s or 0, score_s)
            report(spec, recipe, results, score_s)

    try:
        pool.run(items, on_save, on_done)
    finally:
        pool.close()

if __name__ == "__main__":
    main()
