#!/usr/bin/env python3
"""Build task-vector (delta) checkpoints from local checkpoints in one hidden basis.

  V = A - B                                   --op diff
  V = rank-r SVD truncation of (A - B)        --op lowrank --energy 0.5 | --rank 64
      (per 2-D tensor; routed experts batched per layer/projection; 1-D tensors and embeddings kept whole)
      Keeps the dominant directions of a training step and drops its high-rank noise (TSV/STAR-style
      spectral denoising).
  V = (A - B) where sign(A - B) == sign(C - D), else 0     --op consensus
      Trajectory sign-consensus: of the latest step, keep only the coordinates that moved the same way in the
      previous step (the persistent part of a training run; the rest is mostly minibatch noise).
  V = (A - B) projected on / orthogonal to (C - D) per tensor (per expert for routed experts)   --op par | perp

Output: safetensors shards (+ int8 row scales with --int8) with an index and DELTA.json, readable by
src/parity/weights.py like any checkpoint (recipes add c * V).

  build_vec41.py --op diff --a T1 --b K41 --out /data/teutonic/vec/T_q --int8
  build_vec41.py --op lowrank --energy 0.5 --a T1 --b K41 --out /data/teutonic/vec/T_lr50 --int8
  build_vec41.py --op consensus --a T1 --b K41 --c K41 --d K40 --out /data/teutonic/vec/T_cons --int8
(model names from configs/merge/k41/models.json, or paths)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
from delta_quant40 import quantize  # noqa: E402

log = logging.getLogger("build_vec41")
EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class Ckpt:
    def __init__(self, root: Path):
        self.root = root
        self.index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.local = threading.local()

    def get(self, name: str) -> torch.Tensor:
        handles = getattr(self.local, "h", None)
        if handles is None:
            handles = self.local.h = {}
        fname = self.index[name]
        if fname not in handles:
            handles[fname] = safe_open(str(self.root / fname), framework="pt", device="cpu")
        h = handles[fname]
        t = h.get_tensor(name)
        if t.dtype == torch.int8:  # int8 delta input
            with safe_open(str((self.root / fname).with_suffix(".scales.safetensors")), framework="pt", device="cpu") as s:
                scale = s.get_tensor(name).float()
            t = t.float().mul_(scale.view(-1, *([1] * (t.dim() - 1))))
        return t


def resolve(spec: str) -> Path:
    models = json.loads((REPO / "configs/merge/k41/models.json").read_text())
    return Path(models.get(spec, spec))


def lowrank(d: torch.Tensor, energy: float | None, rank: int | None, chunk: int = 32) -> tuple[torch.Tensor, float, int]:
    """Chunked over the batch dim (bounded GPU memory: ~2 GB per 32 expert matrices)."""
    if d.dim() == 3 and d.shape[0] > chunk:
        parts = [_lowrank(d[i:i + chunk], energy, rank) for i in range(0, d.shape[0], chunk)]
        w = [p[0].shape[0] for p in parts]
        return (torch.cat([p[0] for p in parts]), sum(p[1] * n for p, n in zip(parts, w)) / sum(w),
                sum(p[2] * n for p, n in zip(parts, w)) / sum(w))
    return _lowrank(d, energy, rank)


def _lowrank(d: torch.Tensor, energy: float | None, rank: int | None) -> tuple[torch.Tensor, float, int]:
    """Batched rank truncation of d [..., m, n]; returns (approx, kept energy fraction, mean rank).

    Uses the eigendecomposition of the small Gram matrix (d d^T or d^T d) instead of an SVD: the same
    leading singular subspace, but batched eigh is fast and robust where cuSOLVER's batched SVD falls back
    to a slow path. The discarded small singular values are never needed precisely.
    """
    m, n = d.shape[-2:]
    left = m <= n
    gram = d @ d.mT if left else d.mT @ d
    evals, vecs = torch.linalg.eigh(gram.double())
    evals, vecs = evals.flip(-1).clamp_min(0).float(), vecs.flip(-1).float()  # descending
    tot = evals.sum(-1, keepdim=True).clamp_min(1e-30)
    if rank is not None:
        keep = torch.zeros_like(evals, dtype=torch.bool)
        keep[..., :rank] = True
    else:
        frac = evals.cumsum(-1) / tot
        keep = torch.cat([torch.ones_like(frac[..., :1], dtype=torch.bool), frac[..., :-1] < energy], dim=-1)
    basis = vecs * keep.unsqueeze(-2)
    approx = basis @ (basis.mT @ d) if left else (d @ basis) @ basis.mT
    kept = float(((evals * keep).sum(-1) / tot.squeeze(-1)).mean())
    return approx, kept, float(keep.sum(-1).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=["diff", "lowrank", "consensus", "par", "perp"])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c")
    ap.add_argument("--d")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--energy", type=float, default=None)
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gpu-slots", type=int, default=1, help="concurrent GPU computations (memory)")
    ap.add_argument("--only", default=None, help="regex: tensors outside it get V = 0 (e.g. experts only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    A, B = Ckpt(resolve(args.a)), Ckpt(resolve(args.b))
    C = Ckpt(resolve(args.c)) if args.c else None
    D = Ckpt(resolve(args.d)) if args.d else None
    if args.op in ("consensus", "par", "perp") and not (C and D):
        raise SystemExit(f"--op {args.op} needs --c and --d")
    if args.op == "lowrank" and (args.energy is None) == (args.rank is None):
        raise SystemExit("--op lowrank needs exactly one of --energy / --rank")
    only = re.compile(args.only) if args.only else None
    dev = torch.device(args.device)
    names = sorted(A.index)
    # work units: one routed-expert projection of a layer (256 tensors, batched) or one other tensor
    units: dict[str, list[str]] = {}
    for n in names:
        m = EXPERT.match(n)
        key = f"{m.group(1)}.*.{m.group(3)}" if m else n
        units.setdefault(key, []).append(n)
    for key, members in units.items():
        if len(members) > 1:
            members.sort(key=lambda n: int(EXPERT.match(n).group(2)))
    keys = list(units)
    args.out.mkdir(parents=True, exist_ok=True)
    stats = {"sq": 0.0, "sq_src": 0.0, "kept": [], "rank": []}
    lock = threading.Lock()
    gpu_lock = threading.Semaphore(args.gpu_slots)
    started = time.time()

    def compute(key: str) -> dict[str, torch.Tensor]:
        members = units[key]
        if only and not only.search(members[0]):
            return {n: torch.zeros_like(A.get(n), dtype=torch.bfloat16) for n in members}
        a = torch.stack([A.get(n).float() for n in members])
        b = torch.stack([B.get(n).float() for n in members])
        d = a.sub_(b)
        del b
        with gpu_lock:
            dg = d.to(dev)
            src_sq = float(dg.pow(2).sum())
            if args.op == "lowrank" and dg.dim() == 3 and "embed_tokens" not in key and "lm_head" not in key:
                approx, kept, rank = lowrank(dg, args.energy, args.rank)
                with lock:
                    stats["kept"].append(kept)
                    stats["rank"].append(rank)
                dg = approx
            elif args.op in ("consensus", "par", "perp"):
                ref = torch.stack([C.get(n).float() for n in members]).sub_(torch.stack([D.get(n).float() for n in members])).to(dev)
                if args.op == "consensus":
                    dg.mul_((dg * ref) > 0)
                else:
                    dims = tuple(range(1, dg.dim()))
                    coef = (dg * ref).sum(dims, keepdim=True) / ref.pow(2).sum(dims, keepdim=True).clamp_min(1e-30)
                    par = ref.mul_(coef)
                    dg = par if args.op == "par" else dg.sub_(par)
            out = dg.to(torch.bfloat16).cpu()
            with lock:
                stats["sq"] += float(dg.pow(2).sum())
                stats["sq_src"] += src_sq
        return {n: out[i].contiguous() for i, n in enumerate(members)}

    shards, cur, size = [], [], 0
    for key in keys:
        cur.append(key)
        size += len(units[key])
        if size >= 1100:
            shards.append(cur)
            cur, size = [], 0
    if cur:
        shards.append(cur)
    weight_map = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for k, shard in enumerate(shards):
            fname = f"delta-{k + 1:05d}-of-{len(shards):05d}.safetensors"
            members = [n for key in shard for n in units[key]]
            if (args.out / fname).exists():
                weight_map.update({n: fname for n in members})
                continue
            tensors: dict[str, torch.Tensor] = {}
            for part in pool.map(compute, shard):
                tensors.update(part)
            if args.int8:
                scales = {}
                for n in list(tensors):
                    tensors[n], sc = quantize(tensors[n])
                    if sc is not None:
                        scales[n] = sc
                save_file(scales, str(args.out / fname.replace(".safetensors", ".scales.safetensors")))
            save_file(tensors, str(args.out / (fname + ".tmp")), metadata={"format": "pt"})
            (args.out / (fname + ".tmp")).rename(args.out / fname)
            weight_map.update({n: fname for n in members})
            log.info("shard %d/%d (%.0fs) |V|/|A-B| so far %.3f", k + 1, len(shards), time.time() - started,
                     (stats["sq"] / max(stats["sq_src"], 1e-30)) ** 0.5)
    total = sum((args.out / f).stat().st_size for f in set(weight_map.values()))
    (args.out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=1))
    info = {"op": args.op, "a": str(resolve(args.a)), "b": str(resolve(args.b)),
            "c": str(resolve(args.c)) if args.c else None, "d": str(resolve(args.d)) if args.d else None,
            "energy": args.energy, "rank": args.rank, "only": args.only, "int8": args.int8,
            "norm": stats["sq"] ** 0.5, "norm_src": stats["sq_src"] ** 0.5,
            "rel_norm": (stats["sq"] / max(stats["sq_src"], 1e-30)) ** 0.5,
            "mean_kept_energy": sum(stats["kept"]) / len(stats["kept"]) if stats["kept"] else None,
            "mean_rank": sum(stats["rank"]) / len(stats["rank"]) if stats["rank"] else None,
            "seconds": round(time.time() - started)}
    (args.out / "DELTA.json").write_text(json.dumps(info, indent=1))
    log.info("wrote %s: %s", args.out, json.dumps(info))


if __name__ == "__main__":
    main()
