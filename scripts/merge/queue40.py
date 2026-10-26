#!/usr/bin/env python3
"""Queue sweep40 candidates written as king #40 + task vectors, compiled to linear recipes over kings.

Vectors (all checkpoints are in king #40's hidden basis):
  D<n>       = r<n> - r<n-1>            (king n's own training step, e.g. D36 = #36 - #35)
  P<a>_<b>   = r<a> - r<b>              (path between two kings, e.g. P40_36)
  K<n>       = r<n>                     (raw checkpoint; use with care, coefficients should sum to 1)
Modules for overrides: experts, shared, attn, head, emb, norm, dense, router, or a raw regex.

  queue40.py --label s36_050 --desc "#40 + 0.5*D36" --vec D36=0.5
  queue40.py --label s36_e06 --vec D36=0.5 --ov experts:D36=0.6 --ov attn:D36=0.3
  queue40.py --label soup3 --base none --vec K40=0.5 --vec K39=0.3 --vec K38=0.2
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
QUEUE = REPO / "configs/merge/k40/queue"
K40 = "/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c"
ROOTS = [Path("/opt/teutonic/hf"), Path("/root/teutonic_hf")]
VEC_ROOTS = [Path("/opt/teutonic/vec"), Path("/root/teutonic_vec")]  # delta checkpoints: V_<dir>
MODULES = {
    "experts": r"\.mlp\.experts\.",
    "shared": r"\.mlp\.shared_experts\.",
    "attn": r"\.self_attn\.",
    "head": r"^lm_head\.",
    "emb": r"embed_tokens",
    "norm": r"norm\.weight$",
    "dense": r"^model\.layers\.0\.mlp\.(gate|up|down)_proj",
    "router": r"\.mlp\.gate\.",
}


def king_path(n: int) -> str:
    if n == 40:
        return K40
    for root in ROOTS:
        p = root / f"r{n}_b40"
        if (p / "UNPERMUTED.json").exists() or (p / ".b40_done").exists():  # complete, not mid-write
            return str(p)
    raise SystemExit(f"reign {n} not available in #40 basis (looked in {ROOTS})")


def expand(vec: str) -> dict:
    """Vector name -> {reign or delta-checkpoint path: coefficient}."""
    if vec.startswith("V_"):
        for root in VEC_ROOTS:
            d = root / vec[2:]
            if (d / "DELTA.json").exists():
                return {str(d): 1.0}
        raise SystemExit(f"delta checkpoint {vec[2:]} not complete in {VEC_ROOTS}")
    if m := re.fullmatch(r"D(\d+)", vec):
        n = int(m.group(1))
        for root in VEC_ROOTS:  # prefer the int8 step checkpoint (remote/local_delta40) when it exists
            if (root / f"d{n}_{n - 1}q" / "DELTA.json").exists():
                return {str(root / f"d{n}_{n - 1}q"): 1.0}
        return {n: 1.0, n - 1: -1.0}
    if m := re.fullmatch(r"P(\d+)_(\d+)", vec):
        a, b = int(m.group(1)), int(m.group(2))
        for root in VEC_ROOTS:
            if (root / f"p{a}_{b}q" / "DELTA.json").exists():
                return {str(root / f"p{a}_{b}q"): 1.0}
        return {a: 1.0, b: -1.0}
    if m := re.fullmatch(r"K(\d+)", vec):
        return {int(m.group(1)): 1.0}
    raise SystemExit(f"bad vector {vec}")


def parse_vecs(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        name, c = item.split("=")
        out[name] = float(c)
    return out


def combine(base: bool, vecs: dict[str, float]) -> dict[int, float]:
    total: dict[int, float] = {40: 1.0} if base else {}
    for name, c in vecs.items():
        for r, s in expand(name).items():
            total[r] = total.get(r, 0.0) + c * s
    return total


def build(label: str, desc: str, base: bool, vecs: dict[str, float], ovs: list[tuple[str, dict[str, float]]],
          limit: int, priority: float, method: str) -> dict:
    default = combine(base, vecs)
    per = [(pat, combine(base, {**vecs, **ov})) for pat, ov in ovs]
    reigns = sorted({r for d in [default] + [p for _, p in per] for r in d},
                    key=lambda r: (r != 40, isinstance(r, str), -r if isinstance(r, int) else 0, str(r)))
    path = lambda r: r if isinstance(r, str) else king_path(r)  # noqa: E731
    terms = [{"model": path(r), "coef": round(default.get(r, 0.0), 10)} for r in reigns]
    overrides = [{"pattern": pat, "coefs": [round(d.get(r, 0.0), 10) for r in reigns]} for pat, d in per]
    return {"label": label, "description": desc, "method": method, "vecs": vecs,
            "vec_overrides": [{"pattern": p, "vecs": o} for p, o in ovs],
            "terms": terms, "overrides": overrides, "limit": limit, "priority": priority}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--base", default="k40", choices=["k40", "none"])
    ap.add_argument("--vec", action="append", default=[])
    ap.add_argument("--ov", action="append", default=[], help="module:VEC=c[,VEC=c]  (first match wins)")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--priority", type=float, default=100)
    ap.add_argument("--method", default="task-arith")
    args = ap.parse_args()
    vecs = parse_vecs(args.vec)
    ovs = []
    for item in args.ov:
        module, rest = item.split(":", 1)
        ovs.append((MODULES.get(module, module), parse_vecs(rest.split(","))))
    desc = args.desc or ("#40" if args.base == "k40" else "") + "".join(
        f" {'+' if c >= 0 else '-'} {abs(c):g}·{v}" for v, c in vecs.items()) + "".join(
        f" [{m}: " + ", ".join(f"{v}={c:g}" for v, c in o.items()) + "]" for m, o in
        ((next((k for k, v in MODULES.items() if v == p), p), o) for p, o in ovs))
    recipe = build(args.label, desc.strip(), args.base == "k40", vecs, ovs, args.limit, args.priority, args.method)
    QUEUE.mkdir(parents=True, exist_ok=True)
    path = QUEUE / f"{args.label}.json"
    path.write_text(json.dumps(recipe, indent=1) + "\n")
    time.sleep(0.01)
    print(f"queued {path.name}: {recipe['description']}")


if __name__ == "__main__":
    main()
