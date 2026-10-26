#!/usr/bin/env python3
"""Re-derive the verdict of every cached challenger in scores.jsonl against the current king cache.

The first campaign31 screens compared batch-8 challengers with king losses cached at batch 2
(MoE kernel rounding depends on the batch). Once sweep31 has re-scored the king at batch 8,
this rewrites those rows' summary/sources from the cached per-sequence losses (the original
summary is kept under `summary_orig`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

import numpy as np  # noqa: E402

SCORES = Path("/opt/teutonic/logs/scores.jsonl")
KING = "/opt/teutonic/hf/king31u"


def main() -> None:
    import parity_eval as pe
    from src.parity.cache import LossCache, scorer_signature
    from src.parity.sample import load_live_config
    from src.parity.verdict import duel as duel_verdict

    live = load_live_config()
    sample = pe.resolve_sample("tag:A", live)
    cache = LossCache()
    sig = scorer_signature("flash_attention_4", 1024)
    king_id = pe.resolve_recipe(KING).identity()
    king = cache.load(sample.sample_id, king_id, sig, sample.n)
    rows = [json.loads(line) for line in SCORES.read_text().splitlines() if line.strip()]
    changed = 0
    for row in rows:
        if row.get("engine") == "sweep31" or row.get("summary") == "FAIL" or "summary_orig" in row:
            continue
        recipe_path = row.get("recipe") or str(REPO / "configs/merge/k31" / f"{row['name']}.json")
        chall = cache.load(sample.sample_id, pe.resolve_recipe(recipe_path).identity(), sig, sample.n)
        limit = int(row.get("limit") or 800)
        mask = np.zeros(sample.n, dtype=bool)
        mask[:limit] = True
        k, c = np.where(mask, king, np.nan), np.where(mask, chall, np.nan)
        if not (np.isfinite(k[:limit]).all() and np.isfinite(c[:limit]).all()):
            print("skip (missing losses)", row["name"])
            continue
        v = duel_verdict(k, c, sample, delta=live.delta_threshold)
        row["summary_orig"] = row["summary"]
        row["summary"] = (f"n_sequences={v['n_sequences']}  avg_king_loss={v['avg_king_loss']}  "
                          f"avg_challenger_loss={v['avg_challenger_loss']}  mu_hat={v['mu_hat']}  lcb={v['lcb']}  "
                          f"delta={v.get('delta')}  accepted={v.get('accepted')}")
        row["sources"] = {name: s["mu_hat"] for name, s in v["source_scores"].items()}
        row["king_batch"] = 8
        changed += 1
        print(f"{row['name']:<24} {row['summary_orig'].split('mu_hat=')[1].split()[0]:>10} -> {v['mu_hat']:+.6f}")
    SCORES.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print("rewrote", changed, "rows")


if __name__ == "__main__":
    main()
