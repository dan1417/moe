#!/usr/bin/env python3
"""Routing-aware per-expert merge coefficients for the specialist competitions (MoE-specific merge).

Each routed expert e in layer l gets an affinity to a target split s:
    a[l, e] = (share of expert e's tokens that come from split s) / (split s's share of all tokens)
from `routing41` counts on the equal-thirds mix sample (a = 1: neutral, > 1: the expert specializes in s).
Every vector's coefficient is ramped linearly between `lo` (affinity <= a_lo) and `hi` (affinity >= a_hi):
    coef_v(l, e) = lo_v + (hi_v - lo_v) * clip((a - a_lo) / (a_hi - a_lo), 0, 1)
Non-expert tensors use --dense; routers keep the base model (frozen in this lineage).

  route_recipe41.py --routing /data/teutonic/eval_cache/routing/route_K41.npy --split code --label rc_t \
      --dense T=1.0 --expert T=0.8:1.6 --a-lo 0.6 --a-hi 1.8
  (T=0.8:1.6: experts that code tokens avoid get 0.8·T, code-specialised experts 1.6·T)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
import queue41 as Q  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True, type=Path)
    ap.add_argument("--split", required=True, help="group name in the routing file: math/code/text, or a category when the routing was taken with by=category")
    ap.add_argument("--label", required=True)
    ap.add_argument("--base", default="K41", help="model name or 'none'")
    ap.add_argument("--dense", required=True, help="VEC=c,... for all non-expert tensors")
    ap.add_argument("--expert", required=True, help="VEC=lo:hi,... ramped on affinity to --split")
    ap.add_argument("--minus", default=None, help="second group: ramp on affinity(split) - affinity(minus) + 1 (experts the split uses MORE than this group)")
    ap.add_argument("--router", default=None, help="VEC=c,... for the routers (default: base only, or --dense when --base none)")
    ap.add_argument("--a-lo", default="0.6", help="affinity at which the ramp starts; qX = X-quantile of the affinities")
    ap.add_argument("--a-hi", default="1.8", help="affinity at which the ramp saturates; qX = quantile")
    ap.add_argument("--priority", type=float, default=50)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--main-limit", type=int, default=None)
    args = ap.parse_args()
    if args.base == "none":
        args.base = None
    table = np.load(args.routing)  # [layers, splits, experts]
    meta = json.loads(args.routing.with_suffix(".json").read_text())
    layers, splits = meta["layers"], meta["splits"]
    s = splits.index(args.split)
    total_share = table[:, s, :].sum() / table.sum()
    per = table.sum(axis=1)
    share = np.where(per > 0, table[:, s, :] / np.maximum(per, 1), total_share)
    affinity = share / total_share
    if args.minus:
        u = splits.index(args.minus)
        ts_u = table[:, u, :].sum() / table.sum()
        affinity = affinity - np.where(per > 0, table[:, u, :] / np.maximum(per, 1), ts_u) / ts_u + 1.0
    q = lambda t: float(np.quantile(affinity, float(t[1:]))) if str(t).startswith('q') else float(t)  # noqa: E731
    a_lo, a_hi = q(args.a_lo), q(args.a_hi)
    ramp = np.clip((affinity - a_lo) / (a_hi - a_lo), 0.0, 1.0)

    dense = Q.parse_vecs(args.dense)
    expert = {}
    for item in args.expert.split(","):
        k, v = item.split("=")
        lo, hi = (float(x) for x in v.split(":"))
        expert[k] = (lo, hi)
    vecs = sorted(set(dense) | set(expert))
    models, vectors = Q.registry()
    # one term per checkpoint: base + each vector's expansion; per-tensor coefficients are recomputed exactly
    base_item = Q.build(args.label, "", args.base, dict.fromkeys(vecs, 0.0), [], [], args.limit, args.main_limit,
                        args.priority, "routing-aware")
    paths = [t["model"] for t in base_item["terms"]]

    def coefs_for(vec_coefs: dict[str, float]) -> list[float]:
        total = Q.combine(args.base, vec_coefs, models, vectors)
        return [round(float(total.get(p, 0.0)), 8) for p in paths]

    default = coefs_for(dense)
    tensor_coefs = {}
    for li, layer in enumerate(layers):
        for e in range(table.shape[2]):
            vc = {v: (expert[v][0] + (expert[v][1] - expert[v][0]) * ramp[li, e]) if v in expert else dense.get(v, 0.0)
                  for v in vecs}
            c = coefs_for(vc)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensor_coefs[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"] = c
    spec = (f"{args.base or '0'} + dense[{args.dense}] experts[{args.expert}] ramped on {args.split}{' − ' + args.minus if args.minus else ''} affinity "
            f"[{a_lo:.2f}, {a_hi:.2f}]")
    item = {**base_item, "description": "routing-aware: " + spec, "vec_spec": spec,
            "terms": [{"model": p, "coef": c} for p, c in zip(paths, default)],
            "overrides": [{"pattern": Q.MODULES["router"], "coefs": coefs_for(
                Q.parse_vecs(args.router) if args.router else (dict.fromkeys(vecs, 0.0) if args.base else dense))}],
            "tensor_coefs": tensor_coefs, "routing": str(args.routing)}
    Q.QUEUE.mkdir(parents=True, exist_ok=True)
    (Q.QUEUE / f"{args.label}.json").write_text(json.dumps(item) + "\n")
    print(f"queued {args.label}: {spec}; {(ramp > 0.5).mean():.0%} of experts on the {args.split}-heavy side; "
          f"affinity range {affinity.min():.2f}..{affinity.max():.2f}")


if __name__ == "__main__":
    main()
