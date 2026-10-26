#!/usr/bin/env python3
"""Learned merge coefficients from a grad40 probe (first order, AdaMerging-style), as linear recipes.

The probe at W0 = sum_k a_k K_k (default: king #40) gives g_{v,t} = <dL/dW_t, V_{v,t}> for task vectors
V_v = K_a - K_b and every tensor t (each routed expert projection separately). With
W(t) = W0(t) + sum_v delta_{v,t} V_{v,t}, the mean per-sequence CE changes by ~ sum g_{v,t} delta_{v,t}
(mu_hat by the negative). Steps (eta set so the first-order predicted gain hits each --gains target):

  coef    delta = -eta * g                (steepest descent in coefficient space)
  weight  delta = -eta * g / |V|^2        (steepest descent in weight space, projected)
  sign    delta = -eta * sign(g)          (only where the split-half SNR is high)

Grouping (--group): tensor (each tensor / expert projection), expert (gate/up/down of one routed expert
share one coefficient), module (one coefficient per layer x {attn, experts, shared, norms, ...}).
Noisy groups are shrunk by max(0, 1 - 1/SNR^2) using the probe's two half-sample estimates.
Recipes go to the sweep40 queue as `tensor_coefs` over the kings involved.

  grad_step40.py G40_a --vecs D40,D36 --group expert --gains 0.0005,0.001,0.002 --prefix LA
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
GRAD_DIR = Path("/root/teutonic_grad")
QUEUE = REPO / "configs/merge/k40/queue"
EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts\.\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def module_key(name: str) -> str:
    m = re.match(r"^model\.layers\.(\d+)\.(.*)$", name)
    if not m:
        return name.split(".weight")[0]
    layer, rest = m.groups()
    if ".mlp.experts." in name:
        kind = "experts"
    elif "shared_experts" in rest:
        kind = "shared"
    elif "self_attn" in rest:
        kind = "attn"
    elif "layernorm" in rest:
        kind = "norm"
    else:
        kind = "mlp"
    return f"L{layer}.{kind}"


def group_of(name: str, mode: str) -> str:
    if mode == "tensor":
        return name
    if mode == "expert":
        m = EXPERT.match(name)
        return f"{m.group(1)}.*" if m else name
    return module_key(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--vecs", default=None, help="comma list (default: all vectors in the probe)")
    ap.add_argument("--group", choices=["tensor", "expert", "module"], default="expert")
    ap.add_argument("--mode", choices=["coef", "weight", "sign"], default="coef")
    ap.add_argument("--gains", default="0.0005,0.001,0.002")
    ap.add_argument("--max-delta", type=float, default=2.0)
    ap.add_argument("--no-snr-shrink", dest="snr_shrink", action="store_false")
    ap.add_argument("--base", default=None, help="JSON {reign: coef} of the probe point (default {40: 1})")
    ap.add_argument("--prefix", default="L")
    ap.add_argument("--priority", type=float, default=60)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import queue40

    probe = json.loads((GRAD_DIR / f"{args.probe}.json").read_text())
    vecs = args.vecs.split(",") if args.vecs else list(probe["vecs"])
    base = {int(k): float(v) for k, v in json.loads(args.base).items()} if args.base else {40: 1.0}
    pairs = {v: tuple(probe["vecs"][v]) for v in vecs}
    half = probe.get("dots_half1")
    # pool per group: g (full), g1 (first half, x2 -> half-sample estimate), vnorm2
    G, G1, VN, members = {}, {}, {}, {}
    for v in vecs:
        for name, g in probe["dots"][v].items():
            key = (v, group_of(name, args.group))
            G[key] = G.get(key, 0.0) + g
            VN[key] = VN.get(key, 0.0) + probe["vnorm2"][v].get(name, 0.0)
            if half and v in half:
                G1[key] = G1.get(key, 0.0) + 2.0 * half[v].get(name, 0.0)
            members.setdefault(key, []).append(name)
    direction = {}
    snr_stats = []
    for key, g in G.items():
        shrink = 1.0
        if half and args.snr_shrink and key in G1:
            a, b = G1[key], 2.0 * g - G1[key]  # two half-sample estimates of g
            se = abs(a - b) / 2.0
            snr2 = (g / se) ** 2 if se > 0 else 1e9
            shrink = max(0.0, 1.0 - 1.0 / snr2)
            snr_stats.append(snr2)
        if args.mode == "coef":
            d = -g
        elif args.mode == "weight":
            d = -g / VN[key] if VN[key] > 0 else 0.0
        else:
            d = -math.copysign(1.0, g) if shrink > 0.5 else 0.0
        direction[key] = d * shrink
    if snr_stats:
        s = np.array(snr_stats)
        ks = [k for k in G if k in G1]
        corr = np.corrcoef([G1[k] for k in ks], [2 * G[k] - G1[k] for k in ks])[0, 1]
        print(f"groups {len(G)}; split-half corr {corr:+.3f}; median SNR^2 {np.median(s):.2f}; "
              f"share SNR^2>1 {np.mean(s > 1):.2f}; >4 {np.mean(s > 4):.2f}")
    for v in vecs:
        tot = sum(g for (vv, _), g in G.items() if vv == v)
        print(f"  {v}: dL/dc (uniform) {tot:+.4e}  -> c += 1 changes mu_hat by {-tot:+.5f} (first order)")
    unit = sum(-G[k] * d for k, d in direction.items())
    if unit <= 0:
        raise SystemExit("direction does not decrease the loss to first order")
    for target in [float(x) for x in args.gains.split(",")]:
        eta = target / unit
        deltas, clipped = {}, 0
        for key, d in direction.items():
            x = eta * d
            if abs(x) > args.max_delta:
                x, clipped = math.copysign(args.max_delta, x), clipped + 1
            deltas[key] = x
        pred = sum(-G[k] * x for k, x in deltas.items())
        # per tensor: reign coefficients = base + sum_v delta_{v,t} * (e_a - e_b)
        reigns = sorted(set(base) | {r for ab in pairs.values() for r in ab}, key=lambda r: (r != 40, -r))
        default = [base.get(r, 0.0) for r in reigns]
        tensor_coefs = {}
        for (v, grp), x in deltas.items():
            if x == 0.0:
                continue
            a, b = pairs[v]
            for name in members[(v, grp)]:
                row = tensor_coefs.setdefault(name, list(default))
                row[reigns.index(a)] += x
                row[reigns.index(b)] -= x
        tensor_coefs = {k: [round(c, 7) for c in row] for k, row in tensor_coefs.items()}
        label = f"{args.prefix}_{args.group[0]}{args.mode[0]}_g{int(round(target * 1e5)):03d}"
        stats = {v: np.array([x for (vv, _), x in deltas.items() if vv == v]) for v in vecs}
        desc = (f"learned coefs from {args.probe} ({args.group}/{args.mode}, SNR-shrink={args.snr_shrink and bool(half)}); "
                f"1st-order pred +{pred:.4f}; " + ", ".join(f"δ{v} {s.mean():+.3f}±{s.std():.3f}" for v, s in stats.items()))
        recipe = {"label": label, "description": desc, "method": "learned-coef", "probe": args.probe,
                  "terms": [{"model": queue40.king_path(r), "coef": c} for r, c in zip(reigns, default)],
                  "tensor_coefs": tensor_coefs, "limit": args.limit, "priority": args.priority}
        print(f"{label}: eta {eta:.3e} pred {pred:+.5f} clipped {clipped} tensors {len(tensor_coefs)} | {desc}")
        if not args.dry_run:
            QUEUE.mkdir(parents=True, exist_ok=True)
            (QUEUE / f"{label}.json").write_text(json.dumps(recipe) + "\n")


if __name__ == "__main__":
    main()
