#!/usr/bin/env python3
"""King lineage with hidden-dim permutations undone (HTTP range reads, no full download).

Kings #30/#31 (and maybe later ones) ship with a function-preserving shuffle of the hidden axis.
For every reign this fetches a slice of the embedding (rows ROWS), recovers the hidden permutation
against a reference reign by nearest-column cosine, un-permutes a set of sample tensors and then
reports distances, likely parents and cosines between steps.

  python scripts/merge/lineage40.py --reigns 29-40 --ref 29 --out eval_cache/lineage40.json
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BASE = "https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256"
DASH = "https://teutonic.ai/dashboard.json"
EMB_ROWS = (20000, 22048)
HEAD_ROWS = (20000, 21024)
SAMPLE = [
    "model.layers.1.self_attn.qkv_proj.weight",
    "model.layers.10.self_attn.o_proj.weight",
    "model.layers.20.self_attn.qkv_proj.weight",
    "model.layers.30.self_attn.o_proj.weight",
    "model.layers.40.self_attn.qkv_proj.weight",
    "model.layers.44.self_attn.qkv_proj.weight",
    "model.layers.4.mlp.experts.17.down_proj.weight",
    "model.layers.12.mlp.experts.77.up_proj.weight",
    "model.layers.20.mlp.experts.200.gate_proj.weight",
    "model.layers.28.mlp.experts.3.down_proj.weight",
    "model.layers.33.mlp.experts.99.up_proj.weight",
    "model.layers.43.mlp.experts.5.down_proj.weight",
    "model.layers.18.mlp.shared_experts.up_proj.weight",
    "model.layers.38.mlp.shared_experts.down_proj.weight",
    "model.layers.8.mlp.gate.weight",
    "model.layers.27.mlp.gate.weight",
    "model.layers.42.mlp.gate.weight",
    "model.layers.27.mlp.gate.e_score_correction_bias",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.30.input_layernorm.weight",
    "model.norm.weight",
]
COLS = ("embed_tokens.weight", "lm_head.weight", "qkv_proj.weight", "gate_proj.weight", "up_proj.weight", "mlp.gate.weight")
ROWS = ("o_proj.weight", "down_proj.weight")
VEC = ("layernorm.weight", "norm.weight")


class Remote:
    def __init__(self, digest: str):
        self.digest = digest
        self.s = requests.Session()
        self.index = self.s.get(f"{BASE}/{digest}/model.safetensors.index.json", timeout=120).json()["weight_map"]
        self.headers: dict = {}

    def _header(self, filename: str):
        if filename not in self.headers:
            url = f"{BASE}/{self.digest}/{filename}"
            raw = self.s.get(url, headers={"Range": "bytes=0-7"}, timeout=120).content
            (n,) = struct.unpack("<Q", raw)
            self.headers[filename] = (8 + n, json.loads(self.s.get(url, headers={"Range": f"bytes=8-{7 + n}"}, timeout=120).content))
        return self.headers[filename]

    def get(self, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        filename = self.index[name]
        off, header = self._header(filename)
        e = header[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        start, end = e["data_offsets"]
        shape = list(e["shape"])
        if rows is not None:
            rowbytes = 2 * int(np.prod(shape[1:]))
            start, end = start + rows[0] * rowbytes, start + rows[1] * rowbytes
            shape[0] = rows[1] - rows[0]
        url = f"{BASE}/{self.digest}/{filename}"
        body = self.s.get(url, headers={"Range": f"bytes={off + start}-{off + end - 1}"}, timeout=(30, 90)).content
        bits = np.frombuffer(body, dtype=np.uint16).astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)


def fetch(digest: str) -> dict[str, np.ndarray]:
    r = Remote(digest)
    out = {"emb": r.get("model.embed_tokens.weight", EMB_ROWS), "head": r.get("lm_head.weight", HEAD_ROWS)}
    for n in SAMPLE:
        out[n] = r.get(n)
    return out


def col_perm(src: np.ndarray, ref: np.ndarray) -> tuple[np.ndarray, float]:
    """perm[j] = ref column matching src column j."""
    a = ref / (np.linalg.norm(ref, axis=0, keepdims=True) + 1e-12)
    b = src / (np.linalg.norm(src, axis=0, keepdims=True) + 1e-12)
    dots = a.T @ b
    perm = dots.argmax(axis=0)
    return perm, float(dots.max(axis=0).min())


def unpermute(name: str, w: np.ndarray, inv: np.ndarray) -> np.ndarray:
    if name in ("emb", "head") or name.endswith(COLS):
        return w[:, inv]
    if name.endswith(ROWS):
        return w[inv, :]
    if name.endswith(VEC):
        return w[inv]
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reigns", default="29-40")
    ap.add_argument("--extra", default="", help="comma list of extra reigns (e.g. 25,26)")
    ap.add_argument("--ref", type=int, default=29)
    ap.add_argument("--out", default="eval_cache/lineage40.json")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.reigns.split("-"))
    chain = {int(r["reign_number"]): r for r in requests.get(DASH, timeout=60).json()["king_chain"]}
    reigns = sorted({r for r in range(lo, hi + 1)} | {int(x) for x in args.extra.split(",") if x} | {args.ref})
    with ThreadPoolExecutor(max_workers=6) as pool:
        W = dict(zip(reigns, pool.map(lambda r: fetch(chain[r]["model_digest"]), reigns)))
    ref = W[args.ref]["emb"]
    perms = {}
    print("hidden permutation vs reign", args.ref)
    for r in reigns:
        perm, mincos = col_perm(W[r]["emb"], ref)
        bij = len(set(perm.tolist())) == len(perm)
        fixed = int((perm == np.arange(len(perm))).sum())
        perms[r] = perm
        print(f"  reign {r:>2} {chain[r]['model_repo']:<40} fixed {fixed:>4}/{len(perm)} bijection {bij} min_cos {mincos:.5f}")
        inv = np.argsort(perm)
        W[r] = {k: unpermute(k, v, inv) for k, v in W[r].items()}
    # expert order check: router rows vs reference (layer 27)
    print("expert order (router rows nearest match vs ref, layer 8/27/42):")
    for r in reigns:
        msg = []
        for L in (8, 27, 42):
            g = W[r][f"model.layers.{L}.mlp.gate.weight"]
            g0 = W[args.ref][f"model.layers.{L}.mlp.gate.weight"]
            a = g0 / np.linalg.norm(g0, axis=1, keepdims=True)
            b = g / np.linalg.norm(g, axis=1, keepdims=True)
            m = (a @ b.T).argmax(axis=0)
            msg.append(f"L{L} fixed {int((m == np.arange(len(m))).sum())}")
        print(f"  reign {r:>2}: " + ", ".join(msg))
    names = ["emb", "head"] + SAMPLE
    vec = {r: np.concatenate([W[r][n].ravel() for n in names]) for r in reigns}
    dist = {(a, b): float(np.linalg.norm(vec[a] - vec[b])) for a in reigns for b in reigns}
    print("\npairwise L2 distance (unpermuted sample tensors)")
    print("      " + "".join(f"{b:>8}" for b in reigns))
    for a in reigns:
        print(f"{a:>5} " + "".join(f"{dist[(a, b)]:>8.3f}" for b in reigns))
    parents = {a: min((b for b in reigns if b < a), key=lambda b: dist[(a, b)]) for a in reigns if a > min(reigns)}
    print("\nlikely parent (closest earlier king):")
    for a, p in parents.items():
        print(f"  {a} ({chain[a]['model_repo']}, {chain[a]['coldkey'][:6]}) <- {p} ({chain[p]['coldkey'][:6]}) dist {dist[(a, p)]:.3f}")
    print("\nper-tensor relative step |W_a - W_p| / |W_p| (sample tensors):")
    rel = {}
    for a, p in parents.items():
        rel[f"{a}<-{p}"] = {n: float(np.linalg.norm(W[a][n] - W[p][n]) / (np.linalg.norm(W[p][n]) + 1e-12)) for n in names}
        print(f"  {a:>2}<-{p:>2}: " + " ".join(f"{v:.1e}" for v in rel[f'{a}<-{p}'].values()))
    print("\ncosine between steps (a - parent_a):")
    steps = {a: vec[a] - vec[p] for a, p in parents.items()}
    ks = list(steps)
    print("       " + "".join(f"{b:>7}" for b in ks))
    for a in ks:
        print(f"  {a:>3}  " + "".join(f"{float(steps[a] @ steps[b] / (np.linalg.norm(steps[a]) * np.linalg.norm(steps[b]) + 1e-12)):>7.2f}" for b in ks))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "reigns": reigns, "ref": args.ref, "parents": parents, "names": names,
        "perm_identity": {r: bool((perms[r] == np.arange(len(perms[r]))).all()) for r in reigns},
        "dist": {f"{a}-{b}": d for (a, b), d in dist.items()}, "relative_step": rel,
    }, indent=1))


if __name__ == "__main__":
    main()
