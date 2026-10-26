#!/usr/bin/env python3
"""Materialize a merge recipe as a submission-ready checkpoint directory.

Every tensor is combined by ``src.parity.weights._fill`` -- the exact code path the
parity evaluator used to score the recipe -- so the written weights are the weights
that were evaluated. Contract files are copied byte-for-byte from the first recipe
term and checked against chain.toml; no manifest.json or MTP tensors are written.

Usage:
  write_merge.py --recipe configs/merge/extrap29_a0.5.json --out /opt/teutonic/submit/extrap29_a0.5
  (then: scripts/verify/verify_submission.py --model-dir <out> --king-dir <king>)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "eval"))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from src.parity import upstream  # noqa: E402
from src.parity.weights import _fill, _Reader  # noqa: E402

log = logging.getLogger("write_merge")
CONTRACT_FILES = tuple(upstream.chain_config.GENESIS_CONTRACT_FILES)
SHARD_BYTES = 5 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", required=True, help="recipe JSON or spec understood by parity_eval.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="comma-separated GPU ids, or 'cpu'")
    parser.add_argument("--workers", type=int, default=16, help="worker threads in cpu mode")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    from parity_eval import resolve_recipe  # resolves reign:/digest: through TEUTONIC_MODELS_DIRS

    recipe = resolve_recipe(args.recipe)
    out = Path(args.out)
    if out.exists():
        if not args.force:
            raise SystemExit(f"{out} exists; pass --force to overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    reader = _Reader(recipe)
    base_index = json.loads((recipe.terms[0].path / "model.safetensors.index.json").read_text())
    names = list(base_index["weight_map"])
    for index, weight_map in enumerate(reader.maps):
        if set(weight_map) != set(names):
            raise SystemExit(f"term {index} has a different tensor set than term 0")
    if any("mtp" in name.lower() for name in names):
        raise SystemExit("recipe base contains MTP tensors")

    # shapes/dtypes from term-0 headers; group into ~5 GB output shards in index order
    from safetensors import safe_open

    meta = {}
    for filename in sorted(set(base_index["weight_map"].values())):
        with safe_open(str(recipe.terms[0].path / filename), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                sliced = handle.get_slice(name)
                meta[name] = (tuple(sliced.get_shape()), sliced.get_dtype())
    shards, current, size = [], [], 0
    for name in names:
        shape, _ = meta[name]
        nbytes = 2 * int(torch.tensor(shape).prod().item()) if shape else 2
        if current and size + nbytes > SHARD_BYTES:
            shards.append(current)
            current, size = [], 0
        current.append(name)
        size += nbytes
    if current:
        shards.append(current)

    if args.gpus == "cpu":
        devices, workers = [torch.device("cpu")], args.workers
        torch.set_num_threads(8)
    else:
        devices = [torch.device(f"cuda:{int(g)}") for g in args.gpus.split(",")]
        workers = len(devices)
    started = time.time()
    weight_map: dict[str, str] = {}
    total = len(shards)

    def write_shard(position: int) -> int:
        device = devices[position % len(devices)]
        tensors = {}
        for name in shards[position]:
            shape, dtype = meta[name]
            if dtype != "BF16":
                raise RuntimeError(f"{name}: expected BF16, got {dtype}")
            target = torch.empty(shape, dtype=torch.bfloat16, device=device)
            with torch.no_grad():
                _fill(target, name, reader, recipe)
            tensors[name] = target.cpu()
        filename = f"model-{position + 1:05d}-of-{total:05d}.safetensors"
        save_file(tensors, str(out / filename), metadata={"format": "pt"})
        for name in tensors:
            weight_map[name] = filename
        return sum(t.numel() * 2 for t in tensors.values())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(write_shard, range(total)))
    index = {"metadata": {"total_size": written}, "weight_map": dict(sorted(weight_map.items()))}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))

    source_dir = recipe.terms[0].path
    for name in CONTRACT_FILES:
        shutil.copyfile(source_dir / name, out / name)
    (out.parent / f"{out.name}.recipe.json").write_text(json.dumps(recipe.describe(), indent=1))
    log.info(
        "wrote %s: %d tensors in %d shards, %.1f GB, %.0fs (recipe saved next to it, not inside)",
        out, len(weight_map), total, written / 1e9, time.time() - started,
    )
    if os.path.exists(out / "manifest.json"):
        raise SystemExit("refusing: manifest.json inside submission dir")


if __name__ == "__main__":
    main()
