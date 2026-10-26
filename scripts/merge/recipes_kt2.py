#!/usr/bin/env python3
"""Recipes around a newly crowned king (default KT2: 5DALwjE4, uid 83, TEXT, crowned over T1 at ~+0.0031; also used for
the new CODE king KC2 with --king KC2), written from configs/merge/k41/models.json into configs/merge/<king lower>/.

KT2's parent is measured first (per-tensor distance to #43 / #42 = CK3 / T1 on a few probe tensors), so that our steps
are stacked on the right base:
  O42 = C1_FINAL - CK3      our lineage since #42 (TEXT merges in C1_START + the 200-step c1 training)
  DC1 = C1_FINAL - C1_START our training step
On the old TEXT king T1 the leaders were k43_o42_100 (+0.00688), k43_dc1_100 (+0.00569) and #43 itself (+0.00394);
those are rescored against KT2 too.

  python scripts/merge/recipes_kt2.py [--king KC2]    # prints the recipe paths in screening order
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
OUT = REPO / "configs/merge/kt2"
KING = "KT2"
K43DIR = REPO / "configs/merge/k43"
PROBES = ["model.layers.30.mlp.experts.17.down_proj.weight", "model.layers.40.self_attn.o_proj.weight",
          "model.layers.20.mlp.experts.200.up_proj.weight", "lm_head.weight", "model.layers.10.mlp.experts.5.gate_proj.weight"]


def tensor(model, name):
    p = Path(M[model])
    idx = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(p / idx[name]), "pt") as f:
        return f.get_tensor(name).float()


def parent() -> str:
    dist = {b: 0.0 for b in ("K43", "CK3", "T1") if b != KING}
    for name in PROBES:
        kt2 = tensor(KING, name)
        for b in dist:
            dist[b] += float((kt2 - tensor(b, name)).norm()) ** 2
    print(f"distance^2 from {KING}:", {b: round(v, 2) for b, v in dist.items()}, file=sys.stderr)
    return min(dist, key=dist.get)


def write(label, desc, terms, **extra):
    if extra.get("mode", "linear") == "linear":
        merged = {}
        for name, coef in terms:
            merged[name] = merged.get(name, 0.0) + coef
        terms = [(n, c) for n, c in merged.items() if abs(c) > 1e-12]
    body = {"label": label, "description": desc, "group": KING.lower(),
            "terms": [{"model": M[n], "coef": round(c, 6)} for n, c in terms], **extra}
    (OUT / f"{label}.json").write_text(json.dumps(body, indent=1))
    return OUT / f"{label}.json"


def main():
    global KING, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", default="KT2", help="models.json name of the new king (KT2 = TEXT, KC2 = CODE)")
    KING = ap.parse_args().king
    OUT = REPO / "configs/merge" / KING.lower()
    k = KING.lower()
    OUT.mkdir(parents=True, exist_ok=True)
    base = parent()
    # our steps, relative to the lineage KT2 continues
    ours = [("C1_FINAL", 1.0), ("CK3", -1.0)] if base in ("K43", "CK3") else [("C1_FINAL", 1.0), ("T1", -1.0)]
    common = "CK3" if base in ("K43", "CK3") else "T1"
    dc1 = [("C1_FINAL", 1.0), ("C1_START", -1.0)]
    scale = lambda terms, k: [(n, c * k) for n, c in terms]  # noqa: E731
    paths = [
        write(f"{k}_raw", f"new king {KING} as is (parent {base}; reference)", [(KING, 1.0)]),
        write(f"{k}_o42_100", f"{KING} + 1.0 x our lineage since {common}", [(KING, 1.0)] + ours),
        write(f"orth_{k}_o42_100", f"{KING} + (our lineage since {common} minus its projection on {KING}'s path since {common})",
              [(KING, 1.0), ("C1_FINAL", 1.0), (common, 1.0)], mode="orthogonal", mode_args={"beta": 1.0}),
        write(f"{k}_o42_050", f"{KING} + 0.5 x our lineage since {common}", [(KING, 1.0)] + scale(ours, 0.5)),
        write(f"{k}_dc1_100", f"{KING} + 1.0 x our 200-step training step", [(KING, 1.0)] + dc1),
        write(f"{k}_dc1_050", f"{KING} + 0.5 x our training step", [(KING, 1.0)] + scale(dc1, 0.5)),
    ]
    if base == "K43":  # new king = #43 + 5DAL's step: also take only half of that step on top of our best #43 merge
        paths.append(write(f"k43_o42_{k}half", f"#43 + our lineage since #42 + 0.5 x 5DAL's step ({KING} - #43)",
                           [("K43", 0.5), (KING, 0.5)] + ours))
    # soups / SLERPs with our final model led TEXT (vs KT2) and CODE (vs CK1) on #43: the same with the new king
    paths += [
        write(f"soup_{k}_c1_050", f"linear soup 0.5 x {KING} + 0.5 x our final", [(KING, 0.5), ("C1_FINAL", 0.5)]),
        write(f"slerp_{k}_c1_040", f"SLERP({KING}, our final, t=0.4) per tensor", [(KING, 1.0), ("C1_FINAL", 1.0)],
              mode="slerp", mode_args={"t": 0.4}),
        write(f"slerp_{k}_c1_050", f"SLERP({KING}, our final, t=0.5) per tensor", [(KING, 1.0), ("C1_FINAL", 1.0)],
              mode="slerp", mode_args={"t": 0.5}),
        write(f"soup_{k}_c1_050_ck1_025", f"soup 0.5 x {KING} + 0.5 x our final + 0.25 x our CODE-king delta (CK1 - T1)",
              [(KING, 0.5), ("C1_FINAL", 0.5), ("CK1", 0.25), ("T1", -0.25)]),
    ]
    # our trained checkpoints (MAIN runs from the #43 soup): M1_S175 = best MAIN / CODE screen, M2_S50 = best MATH (full
    # A/B/C passed vs #43) -- blended with the new king (user 15:40: "merge the new king with my best model")
    if "M1_S175" in M:
        paths += [
            write(f"soup_{k}_m1s175_050", f"0.5 x {KING} + 0.5 x our MAIN run m1/step-175", [(KING, 0.5), ("M1_S175", 0.5)]),
            write(f"soup_{k}_m1s175_040", f"0.6 x {KING} + 0.4 x our MAIN run m1/step-175", [(KING, 0.6), ("M1_S175", 0.4)]),
            write(f"soup_{k}_m1s175_060", f"0.4 x {KING} + 0.6 x our MAIN run m1/step-175", [(KING, 0.4), ("M1_S175", 0.6)]),
            write(f"slerp_{k}_m1s175_050", f"SLERP({KING}, our m1/step-175, t=0.5) per tensor", [(KING, 1.0), ("M1_S175", 1.0)],
                  mode="slerp", mode_args={"t": 0.5}),
        ]
        if "M2_S50" in M:
            paths.append(write(f"soup_{k}_m2s50_050", f"0.5 x {KING} + 0.5 x our m2/step-50 (MATH)", [(KING, 0.5), ("M2_S50", 0.5)]))
        if base == "K43":  # the new king continues #43: put our whole delta vs #43 (merge + training) on top of it
            paths += [
                write(f"{k}_plus_m1s175d_100", f"{KING} + 1.0 x (our m1/step-175 - #43)", [(KING, 1.0), ("M1_S175", 1.0), ("K43", -1.0)]),
                write(f"{k}_plus_m1s175d_050", f"{KING} + 0.5 x (our m1/step-175 - #43)", [(KING, 1.0), ("M1_S175", 0.5), ("K43", -0.5)]),
                write(f"orth_{k}_m1s175", f"{KING} + (our m1/step-175 - #43, minus its projection on {KING} - #43)",
                      [(KING, 1.0), ("M1_S175", 1.0), ("K43", 1.0)], mode="orthogonal", mode_args={"beta": 1.0}),
            ]
    if base == "K43":  # 5DAL's step since #43 is small and targeted: extrapolate it
        paths += [
            write(f"{k}_mom050", f"{KING} + 0.5 x 5DAL's step since #43 ({KING} - #43)", [(KING, 1.5), ("K43", -0.5)]),
            write(f"{k}_mom100", f"{KING} + 1.0 x 5DAL's step since #43", [(KING, 2.0), ("K43", -1.0)]),
            write(f"soup_{k}mom050_c1_050", f"soup 0.5 x ({KING} + 0.5 x its step since #43) + 0.5 x our final",
                  [(KING, 0.75), ("K43", -0.25), ("C1_FINAL", 0.5)]),
        ]
    # the leaders so far, rescored against the new king
    paths += [K43DIR / f"{x}.json" for x in ("soup_k43_c1_050", "slerp_k43_c1_040", "orth_k43_o42_100", "k43_o42_100",
                                             "k43_o42_ck1_025", "k43_dc1_100", "k43_raw")]
    print(" ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
