#!/usr/bin/env python3
"""Quantize a task-vector checkpoint (remote_delta40.py output) to int8 rows + fp32 row scales (half the size).

2-D tensors are stored as I8 with absmax/127 row scales in '<shard>.scales.safetensors'; 1-D tensors stay BF16.
src/parity/weights.py reads I8 tensors back as fp32 (q * scale), so recipes use the quantized delta unchanged.
Row-quantization error is ~1% of the row rms of the delta (well below bf16 rounding of the merged weight).

  python scripts/merge/delta_quant40.py /root/teutonic_vec/d33_32 /root/teutonic_vec/d33_32q [--delete-src]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantize(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if t.dim() < 2:
        return t.to(torch.bfloat16).contiguous(), None
    f = t.float().reshape(t.shape[0], -1)
    scale = f.abs().amax(dim=1).clamp_min(1e-30) / 127.0
    q = torch.round(f / scale[:, None]).clamp_(-127, 127).to(torch.int8)
    return q.view(t.shape).contiguous(), scale.contiguous()


def convert_shard(src: Path, dst: Path) -> None:
    out, scales = {}, {}
    with safe_open(str(src), framework="pt", device="cpu") as h:
        for name in h.keys():
            q, s = quantize(h.get_tensor(name))
            out[name] = q
            if s is not None:
                scales[name] = s
    save_file(scales, str(dst.with_suffix(".scales.safetensors.tmp")))
    dst.with_suffix(".scales.safetensors.tmp").rename(dst.with_suffix(".scales.safetensors"))
    save_file(out, str(dst) + ".tmp", metadata={"format": "pt"})
    Path(str(dst) + ".tmp").rename(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--delete-src", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    args.dst.mkdir(parents=True, exist_ok=True)
    todo = [f for f in files if not (args.dst / f).exists()]
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda f: convert_shard(args.src / f, args.dst / f), todo))
    total = sum((args.dst / f).stat().st_size + (args.dst / f).with_suffix(".scales.safetensors").stat().st_size for f in files)
    index["metadata"] = {"total_size": total, "quantized": "int8-row"}
    (args.dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=1))
    meta = json.loads((args.src / "DELTA.json").read_text()) if (args.src / "DELTA.json").exists() else {}
    meta["quantized"] = "int8 rows, fp32 absmax/127 row scales"
    (args.dst / "DELTA.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {args.dst}: {len(files)} shards, {total / 1e9:.1f} GB")
    if args.delete_src:
        shutil.rmtree(args.src)


if __name__ == "__main__":
    sys.exit(main())
