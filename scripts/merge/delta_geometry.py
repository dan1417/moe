#!/usr/bin/env python3
"""Per-module norms and cosines between task vectors built from same-basis checkpoints.

  delta_geometry.py --model k31=/opt/teutonic/hf/king31u --model k30=/opt/teutonic/hf/king30u \
      --model m=/opt/teutonic/hf/k30u --delta S31=k31-k30 --delta Dm=m-k30 [--expert-stride 4]

Prints ||delta|| per group (relative to ||W|| of the first model in each delta) and the
cosine between every pair of deltas, per group and overall.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open

GROUPS = [
    ("experts", r"\.mlp\.experts\."),
    ("shared", r"\.shared_experts\."),
    ("attn", r"\.self_attn\."),
    ("router", r"\.mlp\.gate\."),
    ("lm_head", r"^lm_head\."),
    ("embed", r"embed_tokens"),
    ("norm", r"norm\.weight$"),
]


def group_of(name: str) -> str:
    for label, pattern in GROUPS:
        if re.search(pattern, name):
            return label
    return "other"


def depth_of(name: str) -> str:
    m = re.search(r"layers\.(\d+)\.", name)
    if not m:
        return "-"
    layer = int(m.group(1))
    return "L00-15" if layer < 16 else ("L16-29" if layer < 30 else "L30-47")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="alias=path")
    ap.add_argument("--delta", action="append", required=True, help="name=a-b")
    ap.add_argument("--expert-stride", type=int, default=4, help="keep every Nth expert index")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    models = dict(item.split("=", 1) for item in args.model)
    deltas = {}
    for item in args.delta:
        name, expr = item.split("=", 1)
        a, b = expr.split("-")
        deltas[name] = (a, b)
    index = {alias: json.loads((Path(p) / "model.safetensors.index.json").read_text())["weight_map"]
             for alias, p in models.items()}
    names = sorted(set.intersection(*(set(v) for v in index.values())))

    def keep(name: str) -> bool:
        m = re.search(r"\.experts\.(\d+)\.", name)
        return m is None or int(m.group(1)) % args.expert_stride == 0

    names = [n for n in names if keep(n)]
    handles: dict[tuple[str, str], object] = {}

    def tensor(alias: str, name: str) -> torch.Tensor:
        path = Path(models[alias]) / index[alias][name]
        with safe_open(str(path), framework="pt", device="cpu") as h:
            return h.get_tensor(name).float()

    acc = defaultdict(float)  # (bucket, key) -> value
    torch.set_num_threads(2)

    def work(name: str):
        needed = sorted({x for pair in deltas.values() for x in pair})
        w = {alias: tensor(alias, name) for alias in needed}
        vec = {d: (w[a] - w[b]).flatten() for d, (a, b) in deltas.items()}
        out = []
        for bucket in (group_of(name), f"{group_of(name)}@{depth_of(name)}", "ALL"):
            ref = next(iter(deltas.values()))[0]
            out.append((bucket, "W2", float(w[ref].flatten().pow(2).sum())))
            for d, v in vec.items():
                out.append((bucket, f"{d}.{d}", float(v.dot(v))))
            for d1, d2 in itertools.combinations(vec, 2):
                out.append((bucket, f"{d1}.{d2}", float(vec[d1].dot(vec[d2]))))
        return out

    with ThreadPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(work, names)):
            for bucket, key, val in res:
                acc[(bucket, key)] += val
    buckets = sorted({b for b, _ in acc}, key=lambda b: (b != "ALL", b))
    dn = list(deltas)
    header = f"{'bucket':<22}" + "".join(f"{'|'+d+'|/|W|':>12}" for d in dn) + "".join(
        f"{'cos('+a+','+b+')':>16}" for a, b in itertools.combinations(dn, 2))
    print(header)
    table = {}
    for b in buckets:
        w2 = acc[(b, "W2")]
        row = [(acc[(b, f"{d}.{d}")] / w2) ** 0.5 if w2 else 0 for d in dn]
        cos = []
        for d1, d2 in itertools.combinations(dn, 2):
            den = (acc[(b, f"{d1}.{d1}")] * acc[(b, f"{d2}.{d2}")]) ** 0.5
            cos.append(acc[(b, f"{d1}.{d2}")] / den if den else 0.0)
        table[b] = {"rel": dict(zip(dn, row)), "cos": dict(zip([f"{a},{c}" for a, c in itertools.combinations(dn, 2)], cos))}
        print(f"{b:<22}" + "".join(f"{x:>12.5f}" for x in row) + "".join(f"{x:>16.3f}" for x in cos))
    if args.json:
        args.json.write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
