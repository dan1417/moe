#!/usr/bin/env python3
"""Damped-Newton (natural-gradient) merge coefficients from a fisher40 probe.

X[m, k] = sum of d[m, v, t] over the tensors t of group k (group = vector x module / expert / tensor), so X[m]
is the micro-batch gradient in coefficient space. g = mean_m X, F = micro * T * Cov_m(X) ~ Hessian.

  delta(lam) = -(F + lam * mean(diag F) * I)^-1 g,   first-order + quadratic change  dL = g.delta + delta.F.delta/2

lam is chosen by 2-fold cross-validation over micro-batches (fit g, F on one half, score dL on the other with
the other half's g and F), so the step is judged on gradient noise it did not see. Recipes for the chosen and
neighbouring lam (and an optional overall scale) go to the sweep40 queue as tensor_coefs over the kings.

  grad_newton40.py F40_a --group module --prefix N1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
GRAD_DIR = Path("/root/teutonic_grad")
QUEUE = REPO / "configs/merge/k40/queue"
T = 2047


def group_of(name: str, mode: str) -> str:
    import grad_step40
    return grad_step40.group_of(name, mode)


def newton(g: np.ndarray, F: np.ndarray, lam: float) -> np.ndarray:
    k = len(g)
    damp = lam * max(np.trace(F) / k, 1e-30)
    return -np.linalg.solve(F + damp * np.eye(k), g)


def dL(delta: np.ndarray, g: np.ndarray, F: np.ndarray) -> float:
    return float(g @ delta + 0.5 * delta @ F @ delta)


def stats(X: np.ndarray, micro: int) -> tuple[np.ndarray, np.ndarray]:
    g = X.mean(axis=0)
    Xc = X - g
    F = micro * T * (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    return g, F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--group", choices=["module", "expert", "tensor", "layer", "vector"], default="module")
    ap.add_argument("--lams", default="0.01,0.03,0.1,0.3,1,3,10,30")
    ap.add_argument("--scales", default="1.0", help="overall multipliers applied to the chosen step")
    ap.add_argument("--pick", type=int, default=3, help="queue the best N lam values by CV")
    ap.add_argument("--prefix", default="N")
    ap.add_argument("--priority", type=float, default=35)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    import queue40

    z = np.load(GRAD_DIR / f"{args.probe}.npz", allow_pickle=False)
    D, names, vecs, pairs, micro = z["D"].astype(np.float64), list(z["names"]), list(z["vecs"]), z["pairs"], int(z["micro"])
    M = D.shape[0]

    def gkey(name: str) -> str:
        if args.group == "vector":
            return "all"
        if args.group == "layer":
            m = re.match(r"^model\.layers\.(\d+)\.", name)
            return f"L{m.group(1)}" if m else name.split(".weight")[0]
        return group_of(name, args.group)

    groups: dict[tuple[int, str], list[int]] = {}
    for j in range(len(vecs)):
        for c, name in enumerate(names):
            groups.setdefault((j, gkey(name)), []).append(c)
    gkeys = list(groups)
    X = np.zeros((M, len(gkeys)))
    for k, (j, _) in enumerate(gkeys):
        X[:, k] = D[:, j, groups[gkeys[k]]].sum(axis=1)
    g, F = stats(X, micro)
    print(f"{args.probe}: M={M} micro={micro} groups={len(gkeys)} | uniform slopes " +
          ", ".join(f"{v} {g[[k for k, (j, _) in enumerate(gkeys) if j == jj]].sum():+.3e}" for jj, v in enumerate(vecs)))
    # 2-fold CV over micro-batches (interleaved halves)
    rng = np.random.default_rng(0)
    perm = rng.permutation(M)
    A, B = perm[: M // 2], perm[M // 2:]
    gA, FA = stats(X[A], micro)
    gB, FB = stats(X[B], micro)
    rows = []
    for lam in [float(x) for x in args.lams.split(",")]:
        dA, dB = newton(gA, FA, lam), newton(gB, FB, lam)
        cv = 0.5 * (dL(dA, gB, FB) + dL(dB, gA, FA))
        d = newton(g, F, lam)
        rows.append((lam, cv, dL(d, g, F), d))
        print(f"  lam {lam:>6g}: CV dL {cv:+.6f} (mu {-cv:+.6f}) | in-sample dL {dL(d, g, F):+.6f} | |delta|max {np.abs(d).max():.3f} "
              f"mean|delta| {np.abs(d).mean():.4f}")
    best = sorted(rows, key=lambda r: r[1])[: args.pick]
    def as_key(x):
        return None if x in ("None", None) else (int(x) if str(x).lstrip("-").isdigit() else str(x))
    vpairs = [(as_key(a), as_key(b)) for a, b in pairs]
    keys = {40} | {x for ab in vpairs for x in ab if x is not None}
    reigns = sorted(keys, key=lambda r: (r != 40, isinstance(r, str), -r if isinstance(r, int) else 0, str(r)))
    default = [1.0 if r == 40 else 0.0 for r in reigns]
    for lam, cv, ins, d in best:
        if cv >= 0:
            print(f"  lam {lam:g}: CV predicts no gain; not queued")
            continue
        for scale in [float(x) for x in args.scales.split(",")]:
            tensor_coefs: dict[str, list[float]] = {}
            for k, (j, key) in enumerate(gkeys):
                x = scale * d[k]
                if x == 0.0:
                    continue
                a, b = vpairs[j]
                for c in groups[(j, key)]:
                    row = tensor_coefs.setdefault(names[c], list(default))
                    row[reigns.index(a)] += x
                    if b is not None:
                        row[reigns.index(b)] -= x
            tensor_coefs = {n: [round(x, 7) for x in row] for n, row in tensor_coefs.items()}
            label = f"{args.prefix}_{args.group[0]}_l{lam:g}".replace(".", "p") + (f"_s{scale:g}".replace(".", "p") if scale != 1.0 else "")
            desc = (f"damped Newton from {args.probe} ({args.group}, {len(gkeys)} coefs, lam {lam:g}, scale {scale:g}); "
                    f"CV mu +{-cv * scale:.5f}, in-sample mu +{-ins:.5f}; vecs {','.join(vecs)}")
            recipe = {"label": label, "description": desc, "method": "newton-coef", "probe": args.probe,
                      "terms": [{"model": r if isinstance(r, str) else queue40.king_path(r), "coef": c} for r, c in zip(reigns, default)],
                      "tensor_coefs": tensor_coefs, "limit": args.limit, "priority": args.priority}
            print(f"-> {label}: {desc}")
            if not args.dry_run:
                QUEUE.mkdir(parents=True, exist_ok=True)
                (QUEUE / f"{label}.json").write_text(json.dumps(recipe) + "\n")


if __name__ == "__main__":
    main()
