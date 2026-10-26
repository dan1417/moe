#!/usr/bin/env python3
"""Task-vector checkpoint D = A - B from two local checkpoints in the same hidden basis, int8 row-quantized.

  python scripts/merge/local_delta40.py /opt/teutonic/hf/r36_b40 /opt/teutonic/hf/r35_b40 /opt/teutonic/vec/d36_35q
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delta_quant40 import quantize  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(4)
    ia = json.loads((args.a / "model.safetensors.index.json").read_text())["weight_map"]
    ib = json.loads((args.b / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(ia) == set(ib)
    names = sorted(ia)
    shards = [names[i:i + 1100] for i in range(0, len(names), 1100)]
    args.out.mkdir(parents=True, exist_ok=True)
    started, weight_map = time.time(), {}
    sq = [0.0]

    def one(k: int) -> None:
        fname = f"delta-{k + 1:05d}-of-{len(shards):05d}.safetensors"
        for n in shards[k]:
            weight_map[n] = fname
        if (args.out / fname).exists():
            return
        out, scales = {}, {}
        for n in shards[k]:
            with safe_open(str(args.a / ia[n]), framework="pt", device="cpu") as ha, \
                 safe_open(str(args.b / ib[n]), framework="pt", device="cpu") as hb:
                d = ha.get_tensor(n).float() - hb.get_tensor(n).float()
            sq[0] += float(d.pow(2).sum())
            out[n], s = quantize(d)
            if s is not None:
                scales[n] = s
        save_file(scales, str(args.out / fname.replace(".safetensors", ".scales.safetensors")))
        save_file(out, str(args.out / (fname + ".tmp")), metadata={"format": "pt"})
        (args.out / (fname + ".tmp")).rename(args.out / fname)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, range(len(shards))))
    total = sum(p.stat().st_size for p in args.out.glob("delta-*.safetensors"))
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total, "quantized": "int8-row"},
                                                                        "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    (args.out / "DELTA.json").write_text(json.dumps({"a": str(args.a), "b": str(args.b), "basis": "king #40",
                                                     "norm": sq[0] ** 0.5, "quantized": "int8 rows, fp32 absmax/127 row scales"}, indent=1))
    print(f"wrote {args.out}: {total / 1e9:.1f} GB, |D| {sq[0] ** 0.5:.2f}, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
