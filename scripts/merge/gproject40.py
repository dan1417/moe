#!/usr/bin/env python3
"""Project a saved grad40 gradient (G, written with --save-g) onto more task vectors, on CPU or GPU.

Adds dots[v][tensor] = <G_t, (K_a - K_b)_t> and vnorm2[v][tensor] to the probe JSON for every vector given
(vectors already in the JSON are recomputed only with --force). No split-half for these vectors.

  python scripts/merge/gproject40.py G40_a --vecs D36=36-35,P39_36=39-36 [--device cuda:0]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/merge"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

GRAD_DIR = Path("/root/teutonic_grad")
G_DIR = Path("/opt/teutonic/grad")
STACK = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--vecs", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import grad40
    from src.parity.weights import Recipe, _Reader

    torch.set_num_threads(max(1, args.threads // 2))
    path = GRAD_DIR / f"{args.probe}.json"
    probe = json.loads(path.read_text())
    gdir = G_DIR / f"{args.probe}_G"
    vecs = grad40.parse_vecs(args.vecs)
    files = sorted(gdir.glob("*.safetensors"))
    for v, (a, b) in vecs.items():
        if v in probe["dots"] and not args.force:
            print(f"{v} already projected; skip")
            continue
        reader = _Reader(Recipe.parse(f"1:{grad40.king_path(a)},-1:{grad40.king_path(b)}"))
        dots, vn = {}, {}
        started = time.time()

        def work(item):
            key, g = item
            m = STACK.match(key)
            names = ([f"model.layers.{m.group(1)}.mlp.experts.{e}.{m.group(2)}.weight" for e in range(g.shape[0])]
                     if m else [key])
            out = []
            for idx, name in enumerate(names):
                ta = reader.handle(0, name).get_tensor(name).to(args.device).float()
                tb = reader.handle(1, name).get_tensor(name).to(args.device).float()
                d = ta.sub_(tb)
                gg = (g[idx] if m else g).to(args.device).float()
                out.append((name, float((gg * d).sum()), float(d.pow(2).sum())))
            return out

        for f in files:
            with safe_open(str(f), framework="pt", device="cpu") as h:
                items = [(k, h.get_tensor(k)) for k in h.keys()]
            with ThreadPoolExecutor(args.threads) as pool:
                for rows in pool.map(work, items):
                    for name, d, n2 in rows:
                        dots[name], vn[name] = d, n2
        probe["dots"][v], probe["vnorm2"][v] = dots, vn
        probe["vecs"][v] = [a, b]
        path.write_text(json.dumps(probe))
        print(f"projected {args.probe} on {v} = r{a} - r{b}: {len(dots)} tensors, total <G,V> {sum(dots.values()):+.4e} "
              f"({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
