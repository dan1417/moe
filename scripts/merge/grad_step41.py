#!/usr/bin/env python3
"""Turn a grad41 probe into learned per-tensor / per-expert merge-coefficient recipes (first order, AdaMerging-style).

The probe gives g_t = <dL/dW_t, V_t> for every probed tensor t and vector V at the point W = base recipe. Moving
tensor t's coefficient of V by delta_t changes the mean per-sequence CE by ~ g_t * delta_t (mu_hat by -g_t delta_t).
  coef    delta_t = -eta * g_t                 (steepest descent in coefficient space)
  weight  delta_t = -eta * g_t / |V_t|^2       (steepest descent in weight space)
  sign    delta_t = -eta * sign(g_t)           (only tensors whose g_t is significant)
eta is set so the first-order predicted gain equals each --gains target; each delta is shrunk by the tensor's
split-chunk SNR (shrink = max(0, 1 - 1/snr^2)) so noisy tensors move less; |delta| is clipped at --max-delta.

The recipe is the base recipe plus one extra term per vector (the delta checkpoint) whose coefficient is delta_t
per tensor (`tensor_coefs`, exact per-tensor coefficient lists) and 0 elsewhere — so what is scored is exactly
base + sum_t delta_t * V_t.

  grad_step41.py G_math_pe015 --vecs D36,T --gains 0.0003,0.0006,0.0012 --tie-expert
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "configs/merge/k41"
PROBES = DIR / "probes"
QUEUE = DIR / "queue"
sys.path.insert(0, str(REPO))
EXPERT_PROJ = re.compile(r"^(model\.layers\.\d+\.mlp\.experts\.\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


def load_probe(label: str) -> dict:
    with gzip.open(PROBES / f"{label}.json.gz", "rt") as f:
        return json.load(f)


def tie_experts(probe: dict) -> dict:
    """Pool gate/up/down of each expert into one key '<expert>.*' (fewer, less noisy coefficients)."""
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
    ap.add_argument("--vecs", default="D36,T")
    ap.add_argument("--vec-paths", default="D36=/data/teutonic/vec/d36_35q,T=/data/teutonic/vec/T_q")
    ap.add_argument("--gains", default="0.0003,0.0006,0.0012")
    ap.add_argument("--max-delta", type=float, default=2.0)
    ap.add_argument("--no-snr-shrink", dest="snr_shrink", action="store_false", default=True)
    ap.add_argument("--tie-expert", action="store_true")
    ap.add_argument("--prefix", default="L")
    ap.add_argument("--priority", type=float, default=-15)
    ap.add_argument("--main-limit", type=int, default=800)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from src.parity.weights import Recipe, Term

    probe = load_probe(args.probe)
    if args.tie_expert:
        probe = tie_experts(probe)
    base_raw = json.loads((DIR / f"{probe['at']}.json").read_text())
    vec_paths = dict(kv.split("=") for kv in args.vec_paths.split(","))
    vecs = [v for v in args.vecs.split(",") if v in probe["dots"]]
    base = Recipe(terms=[Term(Path(t["model"]), float(t["coef"])) for t in base_raw["terms"]],
                  overrides=[(o["pattern"], [float(c) for c in o["coefs"]]) for o in base_raw.get("overrides", [])])
    n_terms = len(base.terms)
    n_chunks = len(probe.get("dots_chunks", []))
    for target in [float(x) for x in args.gains.split(",")]:
        directions: dict[tuple[str, str], float] = {}
        shrinks = []
        for v in vecs:
            dots, vn = probe["dots"][v], probe.get("vnorm2", {}).get(v, {})
            for name, g in dots.items():
                shrink = 1.0
                if args.snr_shrink and n_chunks >= 2:
                    parts = np.array([c[v].get(name, 0.0) for c in probe["dots_chunks"]])
                    se = parts.std(ddof=1) / math.sqrt(n_chunks)
                    snr2 = (g / se) ** 2 if se > 0 else 1e9
                    shrink = max(0.0, 1.0 - 1.0 / snr2)
                shrinks.append(shrink)
                if args.mode == "coef":
                    d = -g
                elif args.mode == "weight":
                    d = -g / vn[name] if vn.get(name) else 0.0
                else:
                    d = -math.copysign(1.0, g) if shrink > 0.5 else 0.0
                directions[(v, name)] = d * shrink
        pred_unit = sum(-probe["dots"][v][name] * d for (v, name), d in directions.items())
        if pred_unit <= 0:
            raise SystemExit("direction does not decrease the loss to first order")
        eta = target / pred_unit
        deltas, clipped = {}, 0
        for key, d in directions.items():
            delta = eta * d
            if abs(delta) > args.max_delta:
                delta, clipped = math.copysign(args.max_delta, delta), clipped + 1
            deltas[key] = delta
        pred = sum(-probe["dots"][v][name] * dl for (v, name), dl in deltas.items())
        # recipe: base terms + one delta-checkpoint term per vector; per-tensor coefficient lists
        terms = list(base_raw["terms"]) + [{"model": vec_paths[v], "coef": 0.0} for v in vecs]
        overrides = [{"pattern": o["pattern"], "coefs": list(o["coefs"]) + [0.0] * len(vecs)} for o in base_raw.get("overrides", [])]
        tensor_coefs: dict[str, list[float]] = {}
        for (v, key), dl in deltas.items():
            if dl == 0.0:
                continue
            for name in expand(key):
                row = tensor_coefs.get(name) or (list(base.coefs_for(name)) + [0.0] * len(vecs))
                row[n_terms + vecs.index(v)] = round(row[n_terms + vecs.index(v)] + dl, 6)
                tensor_coefs[name] = row
        stats = {v: np.array([dl for (vv, _), dl in deltas.items() if vv == v]) for v in vecs}
        label = f"{args.prefix}_{args.probe}_{args.mode}{'T' if args.tie_expert else ''}_g{int(round(target * 1e5)):04d}"
        desc = (f"learned per-tensor coefs from probe {args.probe} on {probe['at']} ({args.mode} step, SNR-shrunk="
                f"{args.snr_shrink}, mean shrink {np.mean(shrinks):.2f}); first-order predicted +{pred:.4f}; "
                + ", ".join(f"δ{v} mean {s.mean():+.3f} sd {s.std():.3f} max |{np.abs(s).max():.2f}|" for v, s in stats.items() if s.size))
        item = {"label": label, "description": desc, "method": "learned-coef", "vec_spec": f"{probe['at']} + Σ_t δ_t(V) from {args.probe}",
                "probe": args.probe, "terms": terms, "overrides": overrides, "tensor_coefs": tensor_coefs,
                "main_limit": args.main_limit, "priority": args.priority}
        print(f"{label}: eta {eta:.3e} predicted {pred:+.5f} clipped {clipped} tensors {len(tensor_coefs)} | {desc}")
        if not args.dry_run:
            QUEUE.mkdir(parents=True, exist_ok=True)
            (QUEUE / f"{label}.json").write_text(json.dumps(item) + "\n")
            print("queued", label)


if __name__ == "__main__":
    main()
