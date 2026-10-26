#!/usr/bin/env python3
"""Compare two safetensors checkpoints tensor by tensor.

Reports bit-identical tensors, tensors that differ (max |diff|, relative L2), and
keys present in only one side. Use for export round-trips, written merges, or to see
which modules a king changed.

Usage: compare_checkpoints.py A_dir B_dir [--only REGEX] [--show 20] [--allow-missing]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def open_all(model_dir: Path) -> dict[str, tuple[Path, str]]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    return {key: (model_dir / filename, key) for key, filename in index.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--only", default=None, help="regex filter on tensor names")
    parser.add_argument("--show", type=int, default=20)
    parser.add_argument("--allow-missing", action="store_true", help="compare the key intersection only")
    args = parser.parse_args()
    a, b = open_all(args.a), open_all(args.b)
    pattern = re.compile(args.only) if args.only else None
    keys = sorted(k for k in set(a) & set(b) if pattern is None or pattern.search(k))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    handles: dict[Path, object] = {}

    def load(ref: tuple[Path, str]) -> torch.Tensor:
        path, key = ref
        if path not in handles:
            handles[path] = safe_open(str(path), framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    identical, differing = 0, []
    for key in keys:
        x, y = load(a[key]), load(b[key])
        if x.shape != y.shape or x.dtype != y.dtype:
            differing.append((key, float("inf"), float("inf")))
            continue
        if torch.equal(x, y):
            identical += 1
            continue
        xf, yf = x.float(), y.float()
        diff = (xf - yf).abs().max().item()
        rel = ((xf - yf).norm() / (yf.norm() + 1e-12)).item()
        differing.append((key, diff, rel))
    print(f"compared {len(keys)} tensors: {identical} bit-identical, {len(differing)} differ")
    for key, diff, rel in sorted(differing, key=lambda item: -item[2])[: args.show]:
        print(f"  {key}: max|diff| {diff:.3e}  rel L2 {rel:.3e}")
    if only_a or only_b:
        print(f"only in A: {len(only_a)} {only_a[:3]}  only in B: {len(only_b)} {only_b[:3]}")
    mismatch = bool(only_a or only_b) and not args.allow_missing
    return 1 if (differing or mismatch) else 0


if __name__ == "__main__":
    sys.exit(main())
