#!/usr/bin/env python3
"""Recipes around a new MAIN king (user 17:00: "the most target is MAIN king"), written from configs/merge/k41/models.json
into configs/merge/<king lower>/. Screened on MAIN only (vs the new king itself), so every recipe is a blend that keeps
the new king's gain and adds what our models have on top of #43:

  M1_S175  our best MAIN training checkpoint (m1/step-175, +0.00241 vs #43)
  KMATH    the MATH king (5Ek5Ko, +0.00254 vs #43 on MAIN); 0.6 KMATH + 0.4 M1_S175 reached +0.00277 vs #43

  reference   M1_S175 as is (the running training's MAIN reference vs the new king; king_merge.training_best_main)
  soups       N / M1_S175 at 0.5 0.6 0.7, N / KMATH 0.5, three-way N / KMATH / M1_S175
  SLERP       N, M1_S175 at t = 0.3, 0.4
  task arith  N + a (M1_S175 - #43) + b (KMATH - #43), and TIES over the two task vectors (vec_ties)
  momentum    N + 0.5 (N - parent), when the parent is close (a targeted step)

  python scripts/merge/recipes_main.py --king KMAIN_c7ef5487     # prints the recipe paths in screening order
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
PROBES = ["model.layers.30.mlp.experts.17.down_proj.weight", "model.layers.40.self_attn.o_proj.weight",
          "model.layers.20.mlp.experts.200.up_proj.weight", "lm_head.weight", "model.layers.10.mlp.experts.5.gate_proj.weight"]
KMATH = next((k for k in reversed(list(M)) if k.startswith("KMATH_")), None)


def tensor(model, name):
    p = Path(M[model])
    idx = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(p / idx[name]), "pt") as f:
        return f.get_tensor(name).float()


def parent(king: str) -> tuple[str, dict]:
    cands = [b for b in ("K43", "KT2", "KC2", KMATH, "CK3", "M1_S175") if b and b in M and b != king]
    dist = {b: 0.0 for b in cands}
    for name in PROBES:
        k = tensor(king, name)
        for b in dist:
            dist[b] += float((k - tensor(b, name)).norm()) ** 2
    print(f"distance^2 from {king}:", {b: round(v, 2) for b, v in dist.items()}, file=sys.stderr)
    return min(dist, key=dist.get), dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", required=True, help="models.json name of the new MAIN king")
    king = ap.parse_args().king
    k = king.lower()
    out = REPO / "configs/merge" / k
    out.mkdir(parents=True, exist_ok=True)
    base, dist = parent(king)

    def write(label, desc, terms, **extra):
        if extra.get("mode", "linear") == "linear":
            merged = {}
            for name, coef in terms:
                merged[name] = merged.get(name, 0.0) + coef
            terms = [(n, c) for n, c in merged.items() if abs(c) > 1e-12]
        body = {"label": label, "description": desc, "group": k, "terms": [{"model": M[n], "coef": round(c, 6)} for n, c in terms], **extra}
        (out / f"{label}.json").write_text(json.dumps(body, indent=1))
        return out / f"{label}.json"

    N = king
    paths = [write("ref_m1s175", "our MAIN training checkpoint m1/step-175 as is (training reference vs the new MAIN king)", [("M1_S175", 1.0)])]
    for w in (0.5, 0.6, 0.7):
        paths.append(write(f"soup_{k}_m1s175_{int(round(w * 100)):03d}", f"{w:.1f} x {N} + {1 - w:.1f} x our m1/step-175",
                           [(N, w), ("M1_S175", 1 - w)]))
    for t in (0.3, 0.4):
        paths.append(write(f"slerp_{k}_m1s175_{int(round(t * 100)):03d}", f"SLERP({N}, our m1/step-175, t={t}) per tensor",
                           [(N, 1.0), ("M1_S175", 1.0)], mode="slerp", mode_args={"t": t}))
    paths.append(write(f"{k}_plus_m1d_050", f"{N} + 0.5 x (our m1/step-175 - #43)", [(N, 1.0), ("M1_S175", 0.5), ("K43", -0.5)]))
    if KMATH:
        paths += [
            write(f"soup_{k}_kmath_050", f"0.5 x {N} + 0.5 x the MATH king {KMATH}", [(N, 0.5), (KMATH, 0.5)]),
            write(f"soup3_{k}_050_kmath030_m1020", f"0.5 x {N} + 0.3 x {KMATH} + 0.2 x m1/step-175 (half the new king, half our best "
                  "KMATH soup)", [(N, 0.5), (KMATH, 0.3), ("M1_S175", 0.2)]),
            write(f"soup3_{k}_040_kmath030_m1030", f"0.4 x {N} + 0.3 x {KMATH} + 0.3 x m1/step-175", [(N, 0.4), (KMATH, 0.3), ("M1_S175", 0.3)]),
            write(f"{k}_plus_kmathd_050", f"{N} + 0.5 x ({KMATH} - #43)", [(N, 1.0), (KMATH, 0.5), ("K43", -0.5)]),
            write(f"{k}_plus_both_030", f"{N} + 0.3 x (m1/step-175 - #43) + 0.3 x ({KMATH} - #43)",
                  [(N, 1.0), ("M1_S175", 0.3), (KMATH, 0.3), ("K43", -0.6)]),
            write(f"{k}_plus_both_050", f"{N} + 0.5 x (m1/step-175 - #43) + 0.5 x ({KMATH} - #43)",
                  [(N, 1.0), ("M1_S175", 0.5), (KMATH, 0.5), ("K43", -1.0)]),
            # TIES over the two task vectors relative to #43, added to the new king (vec_ties: term 0 = king)
            write(f"vties_{k}_m1_kmath", f"{N} + TIES{{m1/step-175 - #43, {KMATH} - #43}} (density 0.5, lambda 0.5)",
                  [(N, 1.0), ("M1_S175", 1.0), (KMATH, 1.0), ("K43", 1.0)], mode="vec_ties",
                  mode_args={"vecs": [[1, 3, 1.0], [2, 3, 1.0]], "density": 0.5, "lambda": 0.5}),
        ]
    if base != "M1_S175" and dist[base] < 0.5 * min(v for b, v in dist.items() if b != base):  # a targeted step on a known parent
        paths.append(write(f"{k}_mom050", f"{N} + 0.5 x its step since {base}", [(N, 1.5), (base, -0.5)]))
    print(f"parent of {king}: {base}", file=sys.stderr)
    print(" ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
