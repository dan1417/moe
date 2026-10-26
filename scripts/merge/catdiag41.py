#!/usr/bin/env python3
"""Per-category gains of scored candidates on the mix screening sample (from the loss cache, no GPU).

The split manifests label every shard with a category (code: reasoning, code-thinking-v0, data-science, ...);
each mix-sample sequence is mapped to its shard's category and the paired gain vs a reference is reported per
split and category, with the split's category weight, so ingredients can be targeted at a competition.

  catdiag41.py --ref K41 T1_raw t1_d36050 [...]        (labels from configs/merge/k41/<label>.json, or model names)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

import numpy as np  # noqa: E402

DIR = REPO / "configs/merge/k41"
SPLIT_MANIFESTS = Path("/data/teutonic/eval_cache/manifests_split")


def shard_categories() -> dict[str, str]:
    out = {}
    for split in ("math", "code", "text"):
        m = json.loads((SPLIT_MANIFESTS / f"{split}.json").read_text())
        for s in m["shards"]:
            out[(s.get("url") or s.get("key")).rsplit("/", 1)[-1]] = s["category"]
    return out


def category_weights() -> dict[str, dict[str, float]]:
    return {split: json.loads((SPLIT_MANIFESTS / f"{split}.json").read_text())["category_weights"]
            for split in ("math", "code", "text")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="K41")
    ap.add_argument("--sample", default="tag:A")
    ap.add_argument("--attn", default="flash_attention_4")
    ap.add_argument("labels", nargs="+")
    args = ap.parse_args()
    import parity_eval as pe
    from src.parity.cache import LossCache, scorer_signature
    from src.parity.sample import load_live_config
    from src.parity.weights import Recipe

    sample = pe.resolve_sample(args.sample, load_live_config(competition="mix"))
    cache, sig = LossCache(), scorer_signature(args.attn, 1024)
    models = json.loads((DIR / "models.json").read_text())

    def losses(label: str) -> np.ndarray:
        if label in models:
            recipe = Recipe.single(models[label])
        else:
            recipe = pe.resolve_recipe(str(DIR / f"{label}.json"))
        return cache.load(sample.sample_id, recipe.identity(), sig, sample.n)

    ref = losses(args.ref)
    cats_of = shard_categories()
    names = sample.meta["shard_names"]
    order = [s["name"] for s in sample.meta["sources"]]
    cats = np.asarray([cats_of.get(names[order[g]][s], "?") for g, s in sample.provenance[:, :2].tolist()])
    weights = category_weights()
    for label in args.labels:
        c = losses(label)
        ok = np.isfinite(ref) & np.isfinite(c)
        d = ref - c
        print(f"\n=== {label} vs {args.ref} (n={int(ok.sum())})")
        for split in ("math", "code", "text"):
            m = ok & (sample.sources == split)
            print(f"  {split:<5} n={int(m.sum()):4d} mu {d[m].mean():+.5f}")
            for cat, w in sorted(weights[split].items(), key=lambda kv: -kv[1]):
                mc = m & (cats == cat)
                if mc.sum():
                    se = d[mc].std(ddof=1) / np.sqrt(mc.sum()) if mc.sum() > 1 else 0.0
                    print(f"      {cat:<26} w {w:.3f} n={int(mc.sum()):4d} mu {d[mc].mean():+.5f} ± {se:.5f}")


if __name__ == "__main__":
    main()
