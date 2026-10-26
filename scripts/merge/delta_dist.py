#!/usr/bin/env python3
"""Distance between two checkpoints on a fixed tensor subset (layers 5 / 20 / 35, embeddings, lm_head), CPU only.

Used to size the probe dose of a far-away king: the subset's ||X - A|| / ||Y - A|| tracks the full-model ratio closely
and reads ~3 % of each model (a few GB) instead of 206 GB.

  python scripts/merge/delta_dist.py --a <dir> --b <dir> [--b <dir> ...]      -> JSON {b: dist} on stdout
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

SUBSET = re.compile(r"^model\.layers\.(5|20|35)\.|embed_tokens|lm_head")


def index(root):
    return json.loads((Path(root) / "model.safetensors.index.json").read_text())["weight_map"]


def tensors(root, names):
    wm, out, by_file = index(root), {}, {}
    for n in names:
        by_file.setdefault(wm[n], []).append(n)
    for f, ns in by_file.items():
        with safe_open(str(Path(root) / f), framework="pt", device="cpu") as h:
            for n in ns:
                out[n] = h.get_tensor(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", action="append", required=True)
    args = ap.parse_args()
    torch.set_num_threads(32)
    names = sorted(n for n in index(args.a) if SUBSET.search(n))
    ref = tensors(args.a, names)
    out = {}
    for b in args.b:
        other = tensors(b, names)
        sq = sum(float((other[n].float() - ref[n].float()).pow(2).sum()) for n in names)
        out[b] = sq ** 0.5
    print(json.dumps(out))


if __name__ == "__main__":
    main()
