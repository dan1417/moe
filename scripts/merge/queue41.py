#!/usr/bin/env python3
"""Queue sweep41 candidates written as BASE + sum_i c_i * VECTOR_i, compiled to linear recipes.

Models (all in king #41's hidden basis) are named in configs/merge/k41/models.json, task vectors in
configs/merge/k41/vectors.json as {"T": ["T1", "K41"]} (= T1 - K41) or {"D36": "/data/teutonic/vec/d36_35q"}
(an int8/bf16 delta checkpoint). Module keys for overrides (first match wins) are listed in MODULES; a raw
regex works too. Transforms (post-combine ops) use pattern:op:k=v[,k=v].

  queue41.py --label t100 --vec T=1.0
  queue41.py --label t_e12_a08 --vec T=1.0 --ov experts:T=1.2 --ov attn:T=0.8
  queue41.py --label g09 --vec T=1.0 --tf 'e_score_correction_bias$:center_scale:g=0.9'
  queue41.py --label soup --base none --vec K41=0.5 --vec T1=0.5      (raw models as vectors)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "configs/merge/k41"
QUEUE = DIR / "queue"
MODULES = {
    "experts": r"\.mlp\.experts\.",
    "experts_gate": r"\.mlp\.experts\.\d+\.gate_proj",
    "experts_up": r"\.mlp\.experts\.\d+\.up_proj",
    "experts_down": r"\.mlp\.experts\.\d+\.down_proj",
    "shared": r"\.mlp\.shared_experts\.",
    "attn": r"\.self_attn\.",
    "attn_qkv": r"\.self_attn\.(q|k|v|qkv)_proj",
    "attn_o": r"\.self_attn\.o_proj",
    "sinks": r"\.self_attn\..*sink",
    "head": r"^lm_head\.",
    "emb": r"embed_tokens",
    "norm": r"norm\.weight$",
    "dense": r"^model\.layers\.0\.mlp\.(gate|up|down)_proj",
    "router": r"\.mlp\.gate\.",
    "router_w": r"\.mlp\.gate\.weight$",
    "router_b": r"e_score_correction_bias$",
}


def layer_regex(spec: str) -> str:
    """L16-47 -> regex matching model.layers.16 .. 47 (any tensor inside)."""
    lo, hi = (int(x) for x in spec[1:].split("-"))
    return r"^model\.layers\.(" + "|".join(str(i) for i in range(lo, hi + 1)) + r")\."


def module_regex(key: str) -> str:
    """'experts' or 'L16-47' or 'experts@L16-47' (both must match) or a raw regex."""
    if "@" in key:  # both parts must match somewhere in the name
        a, b = key.split("@", 1)
        return f"^(?=.*(?:{module_regex(a)}))(?=.*(?:{module_regex(b)}))"
    if key.startswith("L") and "-" in key and key[1].isdigit():
        return layer_regex(key)
    return MODULES.get(key, key)


def registry() -> tuple[dict, dict]:
    models = json.loads((DIR / "models.json").read_text())
    vectors = json.loads((DIR / "vectors.json").read_text()) if (DIR / "vectors.json").exists() else {}
    return models, vectors


def expand(vec: str, models: dict, vectors: dict) -> dict[str, float]:
    """Vector or model name -> {checkpoint path: coefficient}."""
    if vec in vectors:
        v = vectors[vec]
        if isinstance(v, str):  # delta checkpoint
            return {v: 1.0}
        plus, minus = v
        return {models[plus]: 1.0, models[minus]: -1.0}
    if vec in models:
        return {models[vec]: 1.0}
    raise SystemExit(f"unknown vector/model {vec!r} (models: {sorted(models)}, vectors: {sorted(vectors)})")


def parse_vecs(text: str) -> dict[str, float]:
    out = {}
    for item in text.split(","):
        if item:
            name, c = item.split("=")
            out[name] = float(c)
    return out


def combine(base: str | None, vecs: dict[str, float], models: dict, vectors: dict) -> dict[str, float]:
    total: dict[str, float] = {models[base]: 1.0} if base else {}
    for name, c in vecs.items():
        for path, s in expand(name, models, vectors).items():
            total[path] = total.get(path, 0.0) + c * s
    return total


def build(label: str, desc: str, base: str | None, vecs: dict[str, float], ovs: list[tuple[str, dict]],
          tfs: list[dict], limit: int | None, main_limit: int | None, priority: float, method: str) -> dict:
    models, vectors = registry()
    default = combine(base, vecs, models, vectors)
    per = [(pat, combine(base, {**vecs, **ov}, models, vectors)) for pat, ov in ovs]
    first = models[base] if base else None
    paths = sorted({p for d in [default] + [d for _, d in per] for p in d}, key=lambda p: (p != first, p))
    terms = [{"model": p, "coef": round(default.get(p, 0.0), 10)} for p in paths]
    overrides = [{"pattern": pat, "coefs": [round(d.get(p, 0.0), 10) for p in paths]} for pat, d in per]
    spec = (base or "0") + "".join(f" {'+' if c >= 0 else '-'} {abs(c):g}·{v}" for v, c in vecs.items())
    spec += "".join(f" [{p}: " + ", ".join(f"{v}={c:g}" for v, c in o.items()) + "]" for p, o in
                    ((next((k for k, v in MODULES.items() if v == pat), pat), o) for pat, o in ovs))
    spec += "".join(f" {{{t['pattern']} {t['op']} {t['args']}}}" for t in tfs)
    item = {"label": label, "description": desc or spec, "method": method, "vec_spec": spec, "vecs": vecs,
            "terms": terms, "overrides": overrides, "priority": priority}
    if tfs:
        item["transforms"] = tfs
    if limit:
        item["limit"] = limit
    if main_limit is not None:
        item["main_limit"] = main_limit
    return item


def parse_transform(text: str) -> dict:
    pattern, op, rest = text.rsplit(":", 2)
    args = {k: float(v) for k, v in (kv.split("=") for kv in rest.split(",") if kv)}
    return {"pattern": module_regex(pattern), "op": op, "args": args}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--base", default="K41", help="model name or 'none'")
    ap.add_argument("--vec", action="append", default=[], help="NAME=c (repeatable or comma-separated)")
    ap.add_argument("--ov", action="append", default=[], help="module:NAME=c[,NAME=c]  (first match wins)")
    ap.add_argument("--tf", action="append", default=[], help="pattern:op:k=v[,k=v]")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--main-limit", type=int, default=None)
    ap.add_argument("--priority", type=float, default=100)
    ap.add_argument("--method", default="task-arith")
    args = ap.parse_args()
    vecs: dict[str, float] = {}
    for v in args.vec:
        vecs.update(parse_vecs(v))
    ovs = []
    for item in args.ov:
        module, rest = item.split(":", 1)
        ovs.append((module_regex(module), parse_vecs(rest)))
    tfs = [parse_transform(t) for t in args.tf]
    base = None if args.base == "none" else args.base
    item = build(args.label, args.desc, base, vecs, ovs, tfs, args.limit, args.main_limit, args.priority, args.method)
    QUEUE.mkdir(parents=True, exist_ok=True)
    (QUEUE / f"{args.label}.json").write_text(json.dumps(item, indent=1) + "\n")
    time.sleep(0.01)
    print(f"queued {args.label}: {item['vec_spec']}")


if __name__ == "__main__":
    main()
