#!/usr/bin/env python3
"""Generate the wave-2 method sweep around a chosen (a, b) for king K and queue it.

  wave2.py --king 30 --a 0.17 --b 0.13 [--routing eval_cache/routing/<k30 stem>.npy]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

D26, D25 = "/opt/teutonic/deltas/d26_22", "/opt/teutonic/deltas/d25_23"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", type=int, required=True)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--b", type=float, required=True)
    ap.add_argument("--limit", type=int, default=4000)
    args = ap.parse_args()
    K, a, b = f"reign:{args.king}", args.a, args.b
    base_terms = [{"model": K, "coef": 1.0}, {"model": D26, "coef": a}, {"model": D25, "coef": b}]
    sparse_terms = [{"model": K, "coef": 1.0}, {"model": D26, "coef": 0.0}, {"model": D25, "coef": 0.0}]
    R = {}
    def lin(name, desc, overrides):
        R[name] = {"description": desc, "terms": base_terms,
                   "overrides": [{"pattern": p, "coefs": [1.0, round(a * m, 4), round(b * m, 4)]} for p, m in overrides]}
    # sparse / trimmed variants of the stack
    R[f"k{args.king}_dties_d0.5"] = {"description": "TIES-trim each delta to top-50% |x|, x1.2 to compensate", "terms": sparse_terms,
                                     "mode": "delta_ties", "mode_args": {"betas": [a * 1.2, b * 1.2], "density": 0.5}}
    R[f"k{args.king}_dcrumbs"] = {"description": "breadcrumbs: keep |x| in top 60% but drop top 1%", "terms": sparse_terms,
                                  "mode": "delta_ties", "mode_args": {"betas": [a * 1.2, b * 1.2], "density": 0.6, "top_drop": 0.01}}
    R[f"k{args.king}_ddare_p0.3"] = {"description": "DARE drop 30% with rescale on each delta", "terms": sparse_terms,
                                     "mode": "delta_dare", "mode_args": {"betas": [a, b], "drop": 0.3}}
    # per-module and per-depth scaling
    lin(f"k{args.king}_experts1.3", "routed-expert deltas x1.3", [(r"\.mlp\.experts\.", 1.3)])
    lin(f"k{args.king}_experts0.7", "routed-expert deltas x0.7", [(r"\.mlp\.experts\.", 0.7)])
    lin(f"k{args.king}_attn1.3", "attention deltas x1.3", [(r"\.self_attn\.", 1.3)])
    lin(f"k{args.king}_attn0.7", "attention deltas x0.7", [(r"\.self_attn\.", 0.7)])
    lin(f"k{args.king}_head1.3", "lm_head + shared-expert deltas x1.3", [(r"^lm_head\.|\.shared_experts\.", 1.3)])
    lin(f"k{args.king}_head0.7", "lm_head + shared-expert deltas x0.7", [(r"^lm_head\.|\.shared_experts\.", 0.7)])
    lin(f"k{args.king}_upper1.3", "layers 27-44 x1.3, 0-26 x0.7", [(r"^model\.layers\.(2[7-9]|3[0-9]|4[0-4])\.", 1.3), (r"^model\.layers\.([0-9]|1[0-9]|2[0-6])\.", 0.7)])
    lin(f"k{args.king}_lower1.3", "layers 0-26 x1.3, 27-44 x0.7", [(r"^model\.layers\.([0-9]|1[0-9]|2[0-6])\.", 1.3), (r"^model\.layers\.(2[7-9]|3[0-9]|4[0-4])\.", 0.7)])
    lines = []
    for name, r in R.items():
        json.dump({"label": name, **r}, open(f"configs/merge/{name}.json", "w"), indent=1)
        lines.append(f"--attn fa4sim --limit {args.limit} duel --king {K} --challenger configs/merge/{name}.json --sample tag:A")
    with open("eval_cache/queue.txt", "a") as f:
        f.write(f"# --- wave 2 vs #{args.king} around ({a}, {b}) ---\n" + "\n".join(lines) + "\n")
    print("queued", len(lines), "wave-2 jobs")


if __name__ == "__main__":
    main()
