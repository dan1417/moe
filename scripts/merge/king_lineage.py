#!/usr/bin/env python3
"""Infer the king lineage and update directions from a few tensors per king.

Fetches only selected tensors from each public king (HTTP range reads of the
safetensors payloads), then reports pairwise distances, each king's closest earlier
king (its likely parent) and cosine similarities between consecutive updates.
Positive cosines between successive updates suggest extrapolating along the latest
update may help; near-orthogonal updates suggest merging siblings is the better bet.

Usage: king_lineage.py [--reigns 20-29] [--out eval_cache/lineage.json]
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

from src.parity.sample import DASHBOARD_URL, fetch_json  # noqa: E402

BASE = "https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256"
SAMPLE_TENSORS = [
    "model.layers.1.self_attn.qkv_proj.weight",
    "model.layers.12.self_attn.qkv_proj.weight",
    "model.layers.23.self_attn.o_proj.weight",
    "model.layers.40.self_attn.qkv_proj.weight",
    "model.layers.6.mlp.experts.17.down_proj.weight",
    "model.layers.20.mlp.experts.200.gate_proj.weight",
    "model.layers.33.mlp.experts.99.up_proj.weight",
    "model.layers.43.mlp.experts.5.down_proj.weight",
    "model.layers.18.mlp.shared_experts.up_proj.weight",
    "model.layers.27.mlp.gate.weight",
    "model.layers.27.mlp.gate.e_score_correction_bias",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
]


def fetch_tensor(session: requests.Session, digest: str, weight_map: dict, name: str, headers: dict) -> np.ndarray:
    filename = weight_map[name]
    if filename not in headers:
        url = f"{BASE}/{digest}/{filename}"
        raw = session.get(url, headers={"Range": "bytes=0-7"}, timeout=120).content
        (length,) = struct.unpack("<Q", raw)
        header = json.loads(session.get(url, headers={"Range": f"bytes=8-{7 + length}"}, timeout=120).content)
        headers[filename] = (8 + length, header)
    offset, header = headers[filename]
    entry = header[name]
    if entry["dtype"] != "BF16":
        raise RuntimeError(f"{name}: unexpected dtype {entry['dtype']}")
    start, end = entry["data_offsets"]
    url = f"{BASE}/{digest}/{filename}"
    body = session.get(url, headers={"Range": f"bytes={offset + start}-{offset + end - 1}"}, timeout=300).content
    bits = np.frombuffer(body, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32).reshape(entry["shape"])


def fetch_king(digest: str) -> dict[str, np.ndarray]:
    session = requests.Session()
    index = session.get(f"{BASE}/{digest}/model.safetensors.index.json", timeout=120).json()["weight_map"]
    headers: dict = {}
    return {name: fetch_tensor(session, digest, index, name, headers) for name in SAMPLE_TENSORS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reigns", default="18-29")
    parser.add_argument("--out", default="eval_cache/lineage.json")
    args = parser.parse_args()
    low, high = (int(x) for x in args.reigns.split("-"))
    chain = {int(r["reign_number"]): r for r in fetch_json(DASHBOARD_URL)["king_chain"]}
    reigns = [r for r in range(low, high + 1) if r in chain]
    with ThreadPoolExecutor(max_workers=6) as pool:
        weights = dict(zip(reigns, pool.map(lambda r: fetch_king(chain[r]["model_digest"]), reigns)))

    def flat(r: int) -> np.ndarray:
        return np.concatenate([weights[r][n].ravel() for n in SAMPLE_TENSORS])

    vectors = {r: flat(r) for r in reigns}
    dist = {(a, b): float(np.linalg.norm(vectors[a] - vectors[b])) for a in reigns for b in reigns}
    print("pairwise L2 distance over sampled tensors")
    print("      " + "".join(f"{b:>9}" for b in reigns))
    for a in reigns:
        print(f"{a:>5} " + "".join(f"{dist[(a, b)]:>9.3f}" for b in reigns))
    parents = {}
    for a in reigns[1:]:
        earlier = [b for b in reigns if b < a]
        parents[a] = min(earlier, key=lambda b: dist[(a, b)])
    print("\nlikely parent (closest earlier king):")
    for a, p in parents.items():
        print(f"  reign {a} ({chain[a]['model_repo']}) <- reign {p}  dist {dist[(a, p)]:.3f}")
    print("\ncosine between successive updates (reign_a - parent_a) vs (parent_a - parent_of_parent):")
    for a, p in parents.items():
        if p in parents:
            d1 = vectors[a] - vectors[p]
            d0 = vectors[p] - vectors[parents[p]]
            cos = float(d1 @ d0 / (np.linalg.norm(d1) * np.linalg.norm(d0) + 1e-12))
            print(f"  {a}<-{p} vs {p}<-{parents[p]}: cos {cos:+.3f}  |d1| {np.linalg.norm(d1):.3f} |d0| {np.linalg.norm(d0):.3f}")
    per_tensor = {}
    for a, p in parents.items():
        per_tensor[f"{a}<-{p}"] = {
            n: float(np.linalg.norm(weights[a][n] - weights[p][n]) / (np.linalg.norm(weights[p][n]) + 1e-12))
            for n in SAMPLE_TENSORS
        }
    print("\nrelative update size per tensor (|W_a - W_parent| / |W_parent|):")
    for key, row in per_tensor.items():
        print(f"  {key:>8}: " + " ".join(f"{v:.1e}" for v in row.values()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "reigns": reigns, "parents": parents,
        "dist": {f"{a}-{b}": d for (a, b), d in dist.items()},
        "relative_update": per_tensor, "tensors": SAMPLE_TENSORS,
    }, indent=1))


if __name__ == "__main__":
    main()
