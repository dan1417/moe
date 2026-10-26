#!/usr/bin/env python3
"""Build a task-vector checkpoint D = K_a - K_b in king #40's hidden basis straight from R2 (no king downloads).

Each king's hidden permutation vs #40 is recovered from its embedding (nearest-column cosine, bijection check);
then every tensor of both kings is range-read, re-indexed into #40's basis, subtracted in fp32 and stored as
bf16 in ~5 GB safetensors shards with an index. A recipe uses it like any checkpoint:
  {"terms": [{"model": K40, "coef": 1.0}, {"model": "/root/teutonic_vec/d33_32", "coef": 0.3}]}

  python scripts/merge/remote_delta40.py --a 33 --b 32 --out /root/teutonic_vec/d33_32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/merge"))

import numpy as np  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

import lineage40 as L  # noqa: E402

log = logging.getLogger("remote_delta40")
K40 = Path("/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c")
COLS = ("embed_tokens.weight", "lm_head.weight", "qkv_proj.weight", "gate_proj.weight", "up_proj.weight", "mlp.gate.weight")
ROWS = ("o_proj.weight", "down_proj.weight")
VEC = ("layernorm.weight", "norm.weight")


class Local:
    """A local checkpoint already in #40's basis (identity permutation)."""

    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.root / self.index[name]), framework="pt", device="cpu") as h:
            return h.get_tensor(name)


def remap(name: str, w: torch.Tensor, inv: torch.Tensor | None) -> torch.Tensor:
    if inv is None:
        return w
    if name.endswith(COLS):
        return w[:, inv]
    if name.endswith(ROWS):
        return w[inv, :]
    if name.endswith(VEC):
        return w[inv]
    return w


class R2:
    """Range reads of one king's tensors; one lineage40.Remote (HTTP session) per thread, shared headers."""

    def __init__(self, digest: str):
        self.digest = digest
        self.proto = L.Remote(digest)
        self.local = threading.local()

    def remote(self) -> "L.Remote":
        r = getattr(self.local, "r", None)
        if r is None:
            r = L.Remote.__new__(L.Remote)
            r.digest, r.index, r.headers, r.s = self.digest, self.proto.index, self.proto.headers, requests.Session()
            self.local.r = r
        return r

    def get(self, name: str) -> torch.Tensor:
        # R2 drops / stalls connections now and then (IncompleteRead, read timeouts): retry for a long time with a
        # fresh session each time, so one bad tensor never kills a 50-minute build.
        for attempt in range(20):
            try:
                arr = self.remote().get(name)
                return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16)
            except Exception as exc:  # noqa: BLE001
                if attempt == 19:
                    raise
                if attempt >= 2:
                    log.warning("%s: retry %d after %s", name, attempt + 1, str(exc)[:80])
                self.local.r = None
                time.sleep(min(60, 3 * (attempt + 1)))
        raise AssertionError


def perm_vs_40(remote: R2) -> torch.Tensor:
    with safe_open(str(K40 / json.loads((K40 / "model.safetensors.index.json").read_text())["weight_map"]["model.embed_tokens.weight"]),
                   framework="pt", device="cpu") as h:
        ref = h.get_tensor("model.embed_tokens.weight").float().cuda()
    src = remote.get("model.embed_tokens.weight").float().cuda()
    a = ref / ref.norm(dim=0, keepdim=True).clamp_min(1e-12)
    b = src / src.norm(dim=0, keepdim=True).clamp_min(1e-12)
    best, arg = torch.empty(b.shape[1], device="cuda"), torch.empty(b.shape[1], dtype=torch.long, device="cuda")
    for s in range(0, b.shape[1], 256):
        v, i = (a.T @ b[:, s:s + 256]).max(dim=0)
        best[s:s + 256], arg[s:s + 256] = v, i
    perm = arg.cpu()
    if len(set(perm.tolist())) != len(perm) or best.min().item() < 0.98:
        raise SystemExit(f"permutation recovery failed (unique {len(set(perm.tolist()))}, min cos {best.min().item():.4f})")
    log.info("perm: fixed points %d, min cos %.5f", int((perm == torch.arange(len(perm))).sum()), best.min().item())
    return torch.argsort(perm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=int, required=True)
    ap.add_argument("--b", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--int8", action="store_true", help="store 2-D tensors as int8 rows + fp32 scales (delta_quant40)")
    ap.add_argument("--a-local", type=Path, default=None, help="read king a from this local dir (already in #40 basis)")
    ap.add_argument("--b-local", type=Path, default=None, help="read king b from this local dir (already in #40 basis)")
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference king dir defining the hidden basis (default #40; #41 and T1 share #40's basis)")
    args = ap.parse_args()
    global K40
    if args.ref is not None:
        K40 = args.ref
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    chain = {int(r["reign_number"]): r for r in requests.get(L.DASH, timeout=60).json()["king_chain"]}
    def side(n, local):
        if local is None:
            r = R2(chain[n]["model_digest"])
            return r, perm_vs_40(r)
        return Local(local), None
    A, inv_a = side(args.a, args.a_local)
    B, inv_b = side(args.b, args.b_local)
    index = json.loads((K40 / "model.safetensors.index.json").read_text())["weight_map"]
    names = sorted(index)
    shards, cur, size = [], [], 0
    for n in names:  # group by layer order into ~5 GB shards
        cur.append(n)
        size += 1
        if size >= 1100:
            shards.append(cur)
            cur, size = [], 0
    if cur:
        shards.append(cur)
    args.out.mkdir(parents=True, exist_ok=True)
    weight_map, started, done = {}, time.time(), [0]
    lock = threading.Lock()
    stats = {"sq": 0.0, "nz": 0}

    def one(name: str) -> tuple[str, torch.Tensor]:
        wa = remap(name, A.get(name), inv_a).float()
        wb = remap(name, B.get(name), inv_b).float()
        d = (wa - wb).to(torch.bfloat16).contiguous()
        with lock:
            done[0] += 1
            stats["sq"] += float(d.float().pow(2).sum())
            stats["nz"] += int(d.count_nonzero())
            if done[0] % 2000 == 0:
                log.info("%d/%d tensors (%.0fs)", done[0], len(names), time.time() - started)
        return name, d

    with ThreadPoolExecutor(args.workers) as pool:
        for k, shard in enumerate(shards):
            fname = f"delta-{k + 1:05d}-of-{len(shards):05d}.safetensors"
            if (args.out / fname).exists():
                for n in shard:
                    weight_map[n] = fname
                continue
            tensors = dict(pool.map(one, shard))
            if args.int8:
                from delta_quant40 import quantize
                scales = {}
                for n in list(tensors):
                    tensors[n], sc = quantize(tensors[n])
                    if sc is not None:
                        scales[n] = sc
                save_file(scales, str(args.out / fname.replace(".safetensors", ".scales.safetensors")))
            save_file(tensors, str(args.out / (fname + ".tmp")), metadata={"format": "pt"})
            (args.out / (fname + ".tmp")).rename(args.out / fname)
            for n in shard:
                weight_map[n] = fname
    total = sum((args.out / f).stat().st_size for f in set(weight_map.values()))
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                        "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    (args.out / "DELTA.json").write_text(json.dumps({"a": args.a, "b": args.b, "basis": str(K40),
                                                     "a_digest": chain[args.a]["model_digest"], "b_digest": chain[args.b]["model_digest"],
                                                     "norm": stats["sq"] ** 0.5, "nonzero": stats["nz"]}, indent=1))
    log.info("wrote %s: %d tensors, %.1f GB, |D| %.2f, %d nonzero, %.0fs", args.out, len(weight_map), total / 1e9,
             stats["sq"] ** 0.5, stats["nz"], time.time() - started)


if __name__ == "__main__":
    main()
