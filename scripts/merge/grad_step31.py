#!/usr/bin/env python3
"""Turn a sweep31 gradient probe into per-tensor merge-coefficient recipes (learned merge, first order).

The probe gives g_t = <dL/dW, vec_t> for every tensor t (each routed expert separately) at the
probe point W = base + sum_v c_v(t) vec_v. Moving tensor t's coefficient by delta_t changes the
mean per-sequence CE by ~ sum_t g_t delta_t, i.e. mu_hat by -sum_t g_t delta_t. Steps:

  coef    delta_t = -eta * g_t                     (steepest descent in coefficient space)
  weight  delta_t = -eta * g_t / |vec_t|^2         (steepest descent in weight space)
  sign    delta_t = -eta * sign(g_t)               (only tensors whose g_t is significant)

eta is chosen so that the first-order predicted gain equals each --gains target; delta is shrunk
by the split-half SNR of g_t (from the probe's per-chunk gradients) so noisy tensors move less.

  grad_step31.py GRAD_dm100 --mode coef --gains 0.0005,0.001,0.002 --prefix L1
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/merge"))
GRAD_DIR = Path("/opt/teutonic/logs/grad31")
QUEUE = REPO / "configs/merge/k31/queue"


def load_probes(labels: list[str]) -> dict:
    """One probe, or several at the same point combined (n-weighted mean gradient, chunks pooled)."""
    probes = [json.loads((GRAD_DIR / f"{label}.json").read_text()) for label in labels]
    if len(probes) == 1:
        return probes[0]
    at = probes[0]["at"]
    for p in probes[1:]:
        if {k: p["at"].get(k) for k in ("vecs", "overrides")} != {k: at.get(k) for k in ("vecs", "overrides")}:
            raise SystemExit("probes were taken at different points")
    total = sum(p["n"] for p in probes)
    vecs = probes[0]["vecs"]
    dots = {v: {} for v in vecs}
    for p in probes:
        for v in vecs:
            for name, g in p["dots"][v].items():
                dots[v][name] = dots[v].get(name, 0.0) + g * p["n"] / total
    # per-chunk gradients rescaled so that chunk * n_chunks estimates the full mean gradient
    chunks = []
    for p in probes:
        k = len(p.get("dots_chunks", []))
        for c in p.get("dots_chunks", []):
            chunks.append({v: {name: g * k for name, g in c[v].items()} for v in vecs})
    kk = len(chunks)
    chunks = [{v: {name: g / kk for name, g in c[v].items()} for v in vecs} for c in chunks]
    vnorm = {}
    for p in probes:
        for v, table in p.get("vnorm2", {}).items():
            vnorm.setdefault(v, {}).update(table)
    return {"label": "+".join(labels), "at": at, "vecs": vecs, "dots": dots, "dots_chunks": chunks, "vnorm2": vnorm,
            "n": total, "sample": probes[0]["sample"], "mean_ce": float(np.mean([p["mean_ce"] for p in probes]))}


EXPERT_PROJ = re.compile(r"^(model\.layers\.\d+\.mlp\.experts\.\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def tie_experts(probe: dict) -> dict:
    """Pool gate/up/down of each expert into one group key '<expert>.*'; expanded again when writing."""
    def pooled(table: dict) -> dict:
        out = {}
        for name, g in table.items():
            m = EXPERT_PROJ.match(name)
            key = f"{m.group(1)}.*" if m else name
            out[key] = out.get(key, 0.0) + g
        return out
    probe = dict(probe)
    probe["dots"] = {v: pooled(t) for v, t in probe["dots"].items()}
    probe["dots_chunks"] = [{v: pooled(t) for v, t in c.items()} for c in probe.get("dots_chunks", [])]
    probe["vnorm2"] = {v: pooled(t) for v, t in probe.get("vnorm2", {}).items()}
    return probe


def expand(name: str) -> list[str]:
    if name.endswith(".*"):
        return [f"{name[:-2]}.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")]
    return [name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probe")
    ap.add_argument("--mode", choices=["coef", "weight", "sign"], default="coef")
    ap.add_argument("--vecs", default="Dm,S31")
    ap.add_argument("--gains", default="0.0005,0.001,0.002")
    ap.add_argument("--max-delta", type=float, default=3.0, help="clip |delta_t|")
    ap.add_argument("--snr-shrink", action="store_true", default=True)
    ap.add_argument("--no-snr-shrink", dest="snr_shrink", action="store_false")
    ap.add_argument("--prefix", default="L")
    ap.add_argument("--priority", type=float, default=60)
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--tie-expert", action="store_true",
                    help="one coefficient per routed expert (gate/up/down pooled): fewer, less noisy parameters")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import sweep31_ops as ops

    probe = load_probes(args.probe.split("+"))
    if args.tie_expert:
        probe = tie_experts(probe)
    at = probe["at"]
    vecs = [v for v in args.vecs.split(",") if v in probe["dots"]]
    coefs = ops.coef_fn(at, probe["vecs"])
    n_chunks = len(probe.get("dots_chunks", []))
    out_tables: dict[str, dict] = {}
    for target in [float(x) for x in args.gains.split(",")]:
        deltas: dict[tuple[str, str], float] = {}
        directions = {}
        for v in vecs:
            dots = probe["dots"][v]
            vn = probe.get("vnorm2", {}).get(v, {})
            for name, g in dots.items():
                shrink = 1.0
                if args.snr_shrink and n_chunks >= 2:
                    parts = np.array([c[v].get(name, 0.0) for c in probe["dots_chunks"]]) * n_chunks
                    se = parts.std(ddof=1) / math.sqrt(n_chunks)
                    snr2 = (g / se) ** 2 if se > 0 else 1e9
                    shrink = max(0.0, 1.0 - 1.0 / snr2)
                if args.mode == "coef":
                    d = -g
                elif args.mode == "weight":
                    d = -g / vn[name] if vn.get(name) else 0.0
                else:
                    d = -math.copysign(1.0, g) if shrink > 0.5 else 0.0
                directions[(v, name)] = d * shrink
        pred_unit = sum(-probe["dots"][v][name] * d for (v, name), d in directions.items())
        if pred_unit <= 0:
            raise SystemExit("direction does not decrease the loss (to first order)")
        eta = target / pred_unit
        clipped = 0
        for key, d in directions.items():
            delta = eta * d
            if abs(delta) > args.max_delta:
                delta = math.copysign(args.max_delta, delta)
                clipped += 1
            deltas[key] = delta
        pred = sum(-probe["dots"][v][name] * dl for (v, name), dl in deltas.items())
        tensor_vecs: dict[str, dict[str, float]] = {}
        for (v, key), dl in deltas.items():
            if dl == 0.0:
                continue
            for name in expand(key):
                base = coefs(name)
                table = tensor_vecs.setdefault(name, {})
                table[v] = round(base.get(v, 0.0) + dl, 6)
        label = f"{args.prefix}_{args.mode}{'T' if args.tie_expert else ''}_g{int(round(target * 1e5)):03d}"
        stats = {v: np.array([dl for (vv, _), dl in deltas.items() if vv == v]) for v in vecs}
        desc = (f"learned per-tensor coefs from {args.probe} ({args.mode} step, SNR-shrunk={args.snr_shrink}); "
                f"first-order predicted +{pred:.4f}; " + ", ".join(
                    f"δ{v} mean {s.mean():+.3f} sd {s.std():.3f}" for v, s in stats.items() if s.size))
        recipe = {"label": label, "description": desc, "method": "learned-coef",
                  "vecs": at.get("vecs", {}), "overrides": at.get("overrides", []),
                  "tensor_vecs": tensor_vecs, "priority": args.priority, "limit": args.limit}
        out_tables[label] = recipe
        print(f"{label}: eta {eta:.3e} predicted {pred:+.5f} clipped {clipped} tensors {len(tensor_vecs)} | {desc}")
    if not args.dry_run:
        QUEUE.mkdir(parents=True, exist_ok=True)
        for label, recipe in out_tables.items():
            (QUEUE / f"{label}.json").write_text(json.dumps(recipe) + "\n")
            print("queued", label)


if __name__ == "__main__":
    main()
