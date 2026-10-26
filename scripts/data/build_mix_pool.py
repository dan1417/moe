#!/usr/bin/env python3
"""Mixed long-context pool (07:25 JST Oct 12): c8 / c9 on contiguous 8,192-token rows raise the TEXT long-document component
(+0.0054 -> +0.0074) but erode the 2,048-token windows (+0.00087 -> +0.00072, 92 % of the scored tokens): a window usually starts
mid-document with no context (a cold start), which 2,048-token rows train in every row and 8,192-token rows only in every fourth.
Half the rows here are contiguous 8,192-token text (text_long_v1), half are four independent 2,048-token windows separated by
the validator's document EOS (151645: the trainer's document masking restarts positions and attention after it, so every window
is a cold start again; it replaces the last token of windows 1-3).

  build_mix_pool.py --long /data/teutonic/pools/text_long_v1 --windows /data/teutonic/pools/text_cpt_v1,... --out ... [--n 200000]
"""
import argparse
import json
from pathlib import Path

import numpy as np

EOS = 151645


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, type=Path)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=200000, help="rows of each kind")
    args = ap.parse_args()
    rng = np.random.default_rng(20261012)
    lt = np.load(args.long / "tokens.npy", mmap_mode="r"); ls = np.load(args.long / "sources.npy")
    li = np.sort(rng.choice(len(lt), size=min(args.n, len(lt)), replace=False))
    long_rows, long_src = np.asarray(lt[li]), ls[li]
    wt = [np.load(Path(p) / "tokens.npy", mmap_mode="r") for p in args.windows.split(",")]
    ws = [np.load(Path(p) / "sources.npy") for p in args.windows.split(",")]
    sizes = np.array([len(t) for t in wt]); total = int(sizes.sum())
    pick = rng.choice(total, size=args.n * 4, replace=False)
    pool_of = np.searchsorted(np.cumsum(sizes), pick, side="right"); local = pick - np.concatenate([[0], np.cumsum(sizes)[:-1]])[pool_of]
    win = np.empty((args.n, 8192), dtype=np.uint32); win_src = []
    for r in range(args.n):
        segs = []
        for k in range(4):
            j = 4 * r + k
            seg = np.array(wt[pool_of[j]][local[j]], dtype=np.uint32)
            if k < 3:
                seg[-1] = EOS
            segs.append(seg)
        win[r] = np.concatenate(segs)
        win_src.append(ws[pool_of[4 * r]][local[4 * r]])
        if r % 50000 == 0:
            print(f"window rows {r}/{args.n}", flush=True)
    tokens = np.concatenate([long_rows, win]); sources = np.concatenate([long_src.astype(str), np.array(win_src).astype(str)])
    perm = rng.permutation(len(tokens))
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "tokens.npy", tokens[perm]); np.save(args.out / "sources.npy", sources[perm])
    counts = {s: int(c) for s, c in zip(*np.unique(sources, return_counts=True))}
    (args.out / "meta.json").write_text(json.dumps({"tag": args.out.name, "kind": "mix_long_and_windows", "n_long": int(len(long_rows)),
        "n_window_rows": int(args.n), "n_sequences": int(len(tokens)), "seq_len": 8192, "n_tokens": int(tokens.size), "source_counts": counts,
        "eos": EOS, "from_long": str(args.long), "from_windows": args.windows.split(","),
        "config_version": json.loads((args.long / "meta.json").read_text())["config_version"]}, indent=1))
    print(f"wrote {args.out}: {tokens.shape} {counts}", flush=True)


if __name__ == "__main__":
    main()
