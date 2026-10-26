#!/usr/bin/env python3
"""Long-context training pool from existing diverse pools (05:00 JST Oct 12): the TEXT score is ~8 % long documents
(2,049-8,192 tokens, where we lead the king by ~+0.005 per token) and ~92 % 2,048-token windows; training only ever saw
2,048-token rows. build_diverse_pool.py reads blocks of `block` (32) consecutive rows of a shard - a contiguous token stream
cut into 2,048-token rows, documents continue across rows - and then shuffles all rows with
default_rng(blake2b("diverse:<tag>") + 1). This undoes the shuffle, splits the rows back into their per-source runs and blocks,
and joins every 4 consecutive rows of a block into one 8,192-token row (real contiguous text; the trainer's document masking
handles the EOS boundaries inside).

  build_long_pool.py --pools /data/teutonic/pools/text_cpt_v18,... --out /data/teutonic/pools/text_long_v1 [--join 4]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def unshuffle(pool: Path):
    meta = json.loads((pool / "meta.json").read_text())
    tok = np.load(pool / "tokens.npy", mmap_mode="r")
    src = np.load(pool / "sources.npy", allow_pickle=True)
    seed = int.from_bytes(hashlib.blake2b(f"diverse:{meta['tag']}".encode(), digest_size=8).digest(), "little")
    order = np.random.default_rng(seed + 1).permutation(len(src))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))   # saved[i] = concat[order[i]] -> concat[j] = saved[inv[j]]
    return meta, tok, src, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--join", type=int, default=4)
    args = ap.parse_args()
    rows_out, src_out, n_in = [], [], 0
    for pool in map(Path, args.pools.split(",")):
        meta, tok, src, inv = unshuffle(pool)
        block = int(meta["block"]); assert block % args.join == 0
        concat_src = src[inv]
        # per-source runs in build order; blocks of `block` rows from each run's start
        starts = [0] + [i for i in range(1, len(concat_src)) if concat_src[i] != concat_src[i - 1]] + [len(concat_src)]
        for a, b in zip(starts[:-1], starts[1:]):
            for blk in range(a, b - args.join + 1, block):
                end = min(blk + block, b)
                for g in range(blk, end - args.join + 1, args.join):
                    idx = inv[g:g + args.join]
                    rows_out.append(np.concatenate([np.asarray(tok[i]) for i in idx]))
                    src_out.append(concat_src[g])
        n_in += len(src)
        print(f"{pool.name}: {len(src)} rows -> {len(rows_out)} long rows so far", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    tokens = np.stack(rows_out).astype(np.uint32)
    perm = np.random.default_rng(12345).permutation(len(tokens))   # mix sources / pools for the trainer
    np.save(args.out / "tokens.npy", tokens[perm]); np.save(args.out / "sources.npy", np.array(src_out).astype(str)[perm])  # plain unicode: the trainer loads with allow_pickle=False
    counts = {s: int(c) for s, c in zip(*np.unique(np.array(src_out), return_counts=True))}
    (args.out / "meta.json").write_text(json.dumps({"tag": args.out.name, "kind": "long_from_blocks", "join": args.join,
        "n_sequences": int(len(tokens)), "seq_len": int(tokens.shape[1]), "n_tokens": int(tokens.size), "source_counts": counts,
        "config_version": meta["config_version"], "from_pools": args.pools.split(",")}, indent=1))
    print(f"wrote {args.out}: {tokens.shape} {counts}", flush=True)


if __name__ == "__main__":
    main()
