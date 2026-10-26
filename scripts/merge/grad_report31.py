#!/usr/bin/env python3
"""Summarize a sweep31 gradient probe: per-group first-order signals and split-half reliability.

For each vector v and group G: sum_t in G of g_t (= dL/dc along vec_t, per unit coefficient) —
negative means adding more of v to that group lowers the loss. The split-half correlation of
per-tensor g_t across probe chunks says how much of the per-tensor structure is signal.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

GRAD_DIR = Path("/opt/teutonic/logs/grad31")
GROUPS = [
    ("experts L0-15", r"layers\.([0-9]|1[0-5])\.mlp\.experts\."),
    ("experts L16-29", r"layers\.(1[6-9]|2\d)\.mlp\.experts\."),
    ("experts L30-44", r"layers\.(3\d|4\d)\.mlp\.experts\."),
    ("attn L0-15", r"layers\.([0-9]|1[0-5])\.self_attn\."),
    ("attn L16-44", r"layers\.(1[6-9]|[2-4]\d)\.self_attn\."),
    ("shared/dense", r"shared_experts|layers\.0\.mlp\."),
    ("lm_head", r"^lm_head"),
    ("embed", r"embed_tokens"),
    ("norms", r"norm\.weight$"),
]


def group(name: str) -> str:
    for label, pattern in GROUPS:
        if re.search(pattern, name):
            return label
    return "other"


def main() -> None:
    probe = json.loads((GRAD_DIR / f"{sys.argv[1]}.json").read_text())
    chunks = probe.get("dots_chunks", [])
    k = len(chunks)
    print(f"probe {probe['label']} at {probe['at'].get('label')} on {probe['sample']} n={probe['n']} "
          f"mean CE {probe['mean_ce']:.5f} chunks {k}")
    for v in probe["vecs"]:
        dots = probe["dots"][v]
        names = sorted(dots)
        g = np.array([dots[n] for n in names])
        print(f"\n== {v}: {len(names)} tensors, total dL/dc = {g.sum():+.4e} "
              f"(adding +0.1*{v} everywhere: predicted mu_hat {-0.1 * g.sum():+.5f})")
        if k >= 2:
            parts = np.array([[c[v].get(n, 0.0) for n in names] for c in chunks]) * k
            half_a, half_b = parts[: k // 2].mean(0), parts[k // 2:].mean(0)
            corr = float(np.corrcoef(half_a, half_b)[0, 1])
            se = parts.std(0, ddof=1) / math.sqrt(k)
            snr = np.abs(g) / np.maximum(se, 1e-30)
            print(f"   split-half corr of per-tensor g: {corr:+.3f}; tensors with |g|/se > 2: {(snr > 2).mean():.1%}, > 3: {(snr > 3).mean():.1%}")
        by = {}
        for n, x in zip(names, g):
            by.setdefault(group(n), []).append(x)
        print(f"   {'group':<16}{'n':>7}{'sum g':>12}{'+0.1 pred':>11}{'frac g<0':>10}")
        for label, _ in GROUPS + [("other", "")]:
            if label in by:
                arr = np.array(by[label])
                print(f"   {label:<16}{arr.size:>7}{arr.sum():>+12.3e}{-0.1 * arr.sum():>+11.5f}{(arr < 0).mean():>10.1%}")
        order = np.argsort(g)
        print("   most negative (want more):", ", ".join(f"{names[i].replace('model.layers.', 'L')}={g[i]:+.2e}" for i in order[:5]))
        print("   most positive (want less):", ", ".join(f"{names[i].replace('model.layers.', 'L')}={g[i]:+.2e}" for i in order[-5:]))


if __name__ == "__main__":
    main()
