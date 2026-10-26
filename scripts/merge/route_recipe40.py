#!/usr/bin/env python3
"""Routing-aware per-expert merge coefficients (the k38 'rmom' recipe type, made source-aware).

Each routed expert gets a web affinity a = (its share of txt360 tokens) / (txt360's share of all tokens), from
`parity_eval.py routing` on king #40. Vectors that help code/math/pretrain but hurt txt360 (D36) go mostly into
web-light experts; vectors that help txt360 (P40_36, 5DALwjE4's path) mostly into web-heavy experts:

  coef_v(expert) = lo_v + (hi_v - lo_v) * clip((a - a_lo) / (a_hi - a_lo), 0, 1)     (linear ramp in affinity)

Non-expert tensors use --dense (e.g. D36=0.5,P40_36=0). Output: a sweep40 queue item with tensor_coefs.

  route_recipe40.py --routing eval_cache/routing/<stem>.npy --label r_a \
      --dense D36=0.5,P40_36=0 --expert D36=0.7:0.3,P40_36=0.0:0.25 --a-lo 0.5 --a-hi 1.5
  (D36=0.7:0.3 means 0.7 on web-light experts ramping to 0.3 on web-heavy ones)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
QUEUE = REPO / "configs/merge/k40/queue"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--dense", required=True, help="VEC=c,... for all non-expert tensors")
    ap.add_argument("--expert", required=True, help="VEC=web_light:web_heavy,...")
    ap.add_argument("--a-lo", type=float, default=0.5)
    ap.add_argument("--a-hi", type=float, default=1.5)
    ap.add_argument("--web-source", default="txt360-v2")
    ap.add_argument("--priority", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=800)
    args = ap.parse_args()
    import queue40

    table = np.load(args.routing)  # [layers, sources, experts]
    meta = json.loads(args.routing.with_suffix(".json").read_text())
    layers, sources = meta["layers"], meta["sources"]
    w = sources.index(args.web_source)
    total_share = table[:, w, :].sum() / table.sum()
    per = table.sum(axis=1)  # [layers, experts]
    share = np.where(per > 0, table[:, w, :] / np.maximum(per, 1), total_share)
    affinity = share / total_share
    ramp = np.clip((affinity - args.a_lo) / (args.a_hi - args.a_lo), 0.0, 1.0)  # 0 = web-light, 1 = web-heavy

    dense = {k: float(v) for k, v in (x.split("=") for x in args.dense.split(","))}
    expert = {}
    for item in args.expert.split(","):
        k, v = item.split("=")
        lo, hi = (float(x) for x in v.split(":"))
        expert[k] = (lo, hi)
    vecs = sorted(set(dense) | set(expert))
    paths = {v: next(iter(queue40.expand(v))) for v in vecs}
    for v, p in paths.items():
        if not isinstance(p, str):
            raise SystemExit(f"{v} has no delta checkpoint; build it first")
    terms = [{"model": queue40.K40, "coef": 1.0}] + [{"model": paths[v], "coef": dense.get(v, 0.0)} for v in vecs]
    tensor_coefs = {}
    for li, layer in enumerate(layers):
        for e in range(table.shape[2]):
            coefs = [1.0] + [expert[v][0] + (expert[v][1] - expert[v][0]) * ramp[li, e] if v in expert else dense.get(v, 0.0)
                             for v in vecs]
            coefs = [round(float(c), 6) for c in coefs]
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensor_coefs[f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight"] = coefs
    desc = (f"routing-aware: dense {args.dense}; experts {args.expert} ramped on txt360 affinity "
            f"[{args.a_lo}, {args.a_hi}] (web-light → web-heavy)")
    recipe = {"label": args.label, "description": desc, "method": "routing-aware", "terms": terms,
              "overrides": [{"pattern": r"\.mlp\.gate\.", "coefs": [1.0] + [0.0] * len(vecs)}],
              "tensor_coefs": tensor_coefs, "routing": str(args.routing), "limit": args.limit, "priority": args.priority}
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / f"{args.label}.json").write_text(json.dumps(recipe) + "\n")
    frac = (ramp > 0.5).mean()
    print(f"queued {args.label}: {desc}; {frac:.0%} of experts on the web-heavy side; affinity range "
          f"{affinity.min():.2f}..{affinity.max():.2f}")


if __name__ == "__main__":
    main()
