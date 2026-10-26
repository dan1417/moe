#!/usr/bin/env python3
"""Recover a hidden-dimension permutation by exact embedding-column match.

King #30 and king #31 were published with a function-preserving shuffle of the
hidden axis (embedding columns match an earlier king one-to-one, but the column
order is a derangement). This writes the same JSON shape as
configs/merge/k30_hidden_perm.json:

    perm[j] = reference-basis column that equals permuted column j

scripts/merge/unpermute_checkpoint.py then maps the permuted checkpoint back.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_tensor(root: Path, name: str) -> torch.Tensor:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    filename = index["weight_map"][name]
    with safe_open(str(root / filename), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def column_perm_exact(permuted: torch.Tensor, reference: torch.Tensor) -> list[int] | None:
    """Exact bf16 column match. Returns None when bits have drifted (merge / training)."""
    ref_bits = reference.contiguous().view(torch.uint16)
    perm_bits = permuted.contiguous().view(torch.uint16)
    lookup: dict[bytes, int] = {}
    for index in range(ref_bits.shape[1]):
        key = ref_bits[:, index].numpy().tobytes()
        if key in lookup:
            return None
        lookup[key] = index
    perm: list[int] = []
    for index in range(perm_bits.shape[1]):
        hit = lookup.get(perm_bits[:, index].numpy().tobytes())
        if hit is None:
            return None
        perm.append(hit)
    if len(set(perm)) != len(perm):
        return None
    return perm


def column_perm_cosine(permuted: torch.Tensor, reference: torch.Tensor) -> tuple[list[int], torch.Tensor]:
    """Nearest reference column by cosine. Works after a few-ulp merge/training drift."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ref = reference.float().to(device)
    src = permuted.float().to(device)
    ref = ref / ref.norm(dim=0, keepdim=True).clamp_min(1e-12)
    src = src / src.norm(dim=0, keepdim=True).clamp_min(1e-12)
    best = torch.empty(src.shape[1], device=device)
    arg = torch.empty(src.shape[1], dtype=torch.long, device=device)
    step = 256
    for start in range(0, src.shape[1], step):
        dots = ref.T @ src[:, start:start + step]
        values, indices = dots.max(dim=0)
        best[start:start + step] = values
        arg[start:start + step] = indices
    perm = arg.cpu().tolist()
    if len(set(perm)) != len(perm):
        raise SystemExit(
            f"cosine match is not a bijection: {len(set(perm))}/{len(perm)} unique, "
            f"min cosine {best.min().item():.6f} mean {best.mean().item():.6f}"
        )
    return perm, best.cpu()


def column_perm(permuted: torch.Tensor, reference: torch.Tensor) -> tuple[list[int], str, float, float]:
    """perm[j] = reference column matching permuted column j."""
    if permuted.shape != reference.shape:
        raise SystemExit(f"embed shape {tuple(permuted.shape)} vs {tuple(reference.shape)}")
    exact = column_perm_exact(permuted, reference)
    if exact is not None:
        return exact, "exact-bf16", 1.0, 1.0
    print("exact bf16 match failed; using cosine nearest-column", flush=True)
    perm, scores = column_perm_cosine(permuted, reference)
    return perm, "cosine", float(scores.min()), float(scores.mean())


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).norm().item()
    return diff / (b.float().norm().item() + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permuted", type=Path, required=True, help="shuffled checkpoint (king #31)")
    parser.add_argument("--reference", type=Path, required=True, help="hidden basis to match (k30u / #29)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    print("loading embeddings", flush=True)
    perm, how, min_cos, mean_cos = column_perm(
        load_tensor(args.permuted, "model.embed_tokens.weight"),
        load_tensor(args.reference, "model.embed_tokens.weight"),
    )
    inv = torch.argsort(torch.tensor(perm, dtype=torch.long))
    identity = sum(i == j for i, j in enumerate(perm))
    print(f"perm length {len(perm)} fixed-points {identity} via {how} min_cos {min_cos:.6f} mean_cos {mean_cos:.6f}", flush=True)
    if how == "cosine" and min_cos < 0.99:
        raise SystemExit(f"column cosine too low ({min_cos:.6f}); #31 may not share this hidden basis")

    checks = [
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.40.self_attn.qkv_proj.weight",
        "model.layers.27.mlp.gate.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    print("hidden-unpermute check vs reference (relL2; ~0 means that tensor was only shuffled)", flush=True)
    for name in checks:
        raw = load_tensor(args.permuted, name)
        ref = load_tensor(args.reference, name)
        if name.endswith(("qkv_proj.weight", "gate.weight", "embed_tokens.weight", "lm_head.weight", "gate_proj.weight", "up_proj.weight")):
            undone = raw[:, inv]
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            undone = raw[inv, :]
        elif name.endswith("norm.weight"):
            undone = raw[inv]
        else:
            print(f"  skip {name}")
            continue
        before = rel_l2(raw, ref)
        after = rel_l2(undone, ref)
        print(f"  {name}: relL2 before {before:.4f} after {after:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "source": str(args.permuted),
        "reference": str(args.reference),
        "meaning": "perm[j] = reference hidden index of permuted hidden index j; inv=argsort(perm) maps into the reference basis",
        "match": how,
        "min_cosine": min_cos,
        "mean_cosine": mean_cos,
        "perm": perm,
    }))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
