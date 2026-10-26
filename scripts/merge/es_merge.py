#!/usr/bin/env python3
"""Evolution-strategy merge search, round 2 (user 21:25 JST Oct 11: "blends can't help currently, go ES to the next round").

Round 1 (merge_search3) found every fixed blend within screen noise of the new CODE king KCODE_0c681d78 (5Ek5Ko v56): MAIN
+0.001094, TEXT +0.001119 by itself; SLERP 0.3 toward our TEXT checkpoint TX75 gave TEXT +0.001148. Instead of fixed recipes,
search the continuous space around it with an antithetic evolution strategy, one search per target competition (MAIN, TEXT):

  genome   W = KCODE + sum_i c_i V_i, per-part scales on the vector sum (attention, routed experts, layers 0-14 / 15-29 / 30-44)
  vectors  toTX (TX75 - KCODE: toward our TEXT checkpoint; linear stand-in for the SLERP), toKMA (KMATH - KCODE), S_T, sCODE5,
           DC1, dM1T (merge_search3.VECS); |c| < 0.03 counts as 0 (each vector is a 206 GB model read per candidate)
  update   per generation and target: 4 antithetic pairs theta +- sigma*eps (eps ~ N(0, I) over the genome), every candidate
           scored on all four competitions (1 200 seqs of sample A, merge_search3.score: one load, both replica sets);
           theta += lr * mean_i(w_i * sigma * eps_i) with w_i = (f+ - f-) / max|f+ - f-| (rank-free, noise-robust);
           sigma *= 0.85 per generation; the best candidate ever is kept per competition (elitism)

  python scripts/merge/es_merge.py --name es1 [--generations 6]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merge_search3 as ms  # noqa: E402

KC = "KCODE_0c681d78"
ms.VECS["toTX"] = ("TX75", KC, 0.2, 0.05)
ms.VECS["toKMA"] = ("KMATH_ddd9e626", KC, 0.2, 0.05)
# the vectors with any signal in round 1 (CK1D, dKC2, dM3 only ever lowered the scores on KCODE); per-dimension sigma
COEFS = {"toTX": 0.15, "toKMA": 0.1, "S_T": 0.5, "sCODE5": 0.15, "DC1": 0.15, "dM1T": 0.25}
SCALES = {"s_attn": 0.25, "s_exp": 0.25, "s_lo": 0.25, "s_mid": 0.25, "s_hi": 0.25}
TARGETS = {"main": {"toTX": 0.0}, "text": {"toTX": 0.3}}


def point(theta):
    # |c| < 0.03 counts as 0: every vector costs a 206 GB model read per candidate
    return {"base": KC, "mode": "lin", "c": {k: round(theta.get(k, 0.0), 3) for k in COEFS if abs(theta.get(k, 0.0)) >= 0.03},
            **{k: round(min(2.0, max(0.0, theta.get(k, 1.0))), 3) for k in SCALES}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    ms.OUT.mkdir(parents=True, exist_ok=True)
    state_path = ms.D / "search" / f"{args.name}.json"
    theta = {t: {**{k: 0.0 for k in COEFS}, **{k: 1.0 for k in SCALES}, **init} for t, init in TARGETS.items()}
    sigma = {t: 1.0 for t in TARGETS}
    best = {}  # comp -> (mu, key, point)
    log = ms.log
    centers = [point(theta[t]) for t in TARGETS]
    for k, mu in ms.score(centers).items():
        for c, v in mu.items():
            if c not in best or v > best[c][0]:
                best[c] = (v, k, next(p for p in centers if ms.key(p) == k))
    for t in TARGETS:
        log(f"ES {t.upper()} start {ms.key(point(theta[t]))}: {t.upper()} {ms.score([point(theta[t])])[ms.key(point(theta[t]))].get(t, float('nan')):+.6f}")
    history = []
    for gen in range(1, args.generations + 1):
        cands, pairs = {}, {t: [] for t in TARGETS}
        for t in TARGETS:
            for _ in range(args.pairs):
                eps = {k: rng.gauss(0, 1) * s * sigma[t] for k, s in {**COEFS, **SCALES}.items()}
                plus = point({k: theta[t][k] + eps[k] for k in theta[t]})
                minus = point({k: theta[t][k] - eps[k] for k in theta[t]})
                pairs[t].append((eps, ms.key(plus), ms.key(minus)))
                cands[ms.key(plus)], cands[ms.key(minus)] = plus, minus
        log(f"ES generation {gen}: {len(cands)} candidates (sigma " + ", ".join(f"{t} {sigma[t]:.2f}" for t in TARGETS) + ")")
        mus = ms.score(list(cands.values()))
        for k, mu in mus.items():
            for c, v in mu.items():
                if c not in best or v > best[c][0]:
                    best[c] = (v, k, cands[k])
                    log(f"  new best {c.upper()} {v:+.6f}: {k}")
        for t in TARGETS:
            diffs = [(eps, mus.get(kp, {}).get(t), mus.get(km, {}).get(t)) for eps, kp, km in pairs[t]]
            diffs = [(e, fp - fm) for e, fp, fm in diffs if fp is not None and fm is not None]
            if not diffs:
                continue
            scale = max(abs(d) for _, d in diffs) or 1.0
            step = {k: args.lr * sum(e[k] * d / scale for e, d in diffs) / len(diffs) for k in theta[t]}
            theta[t] = {k: theta[t][k] + step[k] for k in theta[t]}
            sigma[t] *= 0.85
            center = point(theta[t])
            log(f"  {t.upper()} centre -> {ms.key(center)}")
        history.append({"gen": gen, "theta": {t: point(theta[t]) for t in TARGETS}, "best": {c: [b[0], b[1]] for c, b in best.items()}})
        state_path.write_text(json.dumps({"best": {c: {"mu": b[0], "key": b[1], "recipe": str(ms.recipe(b[2])[1])} for c, b in best.items()},
                                          "history": history}, indent=1))
    log("ES DONE: " + "  ".join(f"{c.upper()} {b[0]:+.6f} ({b[1]})" for c, b in best.items()))


if __name__ == "__main__":
    main()
