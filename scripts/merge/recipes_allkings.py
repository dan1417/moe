#!/usr/bin/env python3
"""One merge round over all of today's kings (user 21:20: "there are several new kings, use them to merge"), written into
configs/merge/allkings/. Ingredients, as steps on top of the TEXT king KTEXT (5DAL uid 218 = the MAIN king KMAIN + a small
TEXT step: probe distance^2 0.47 vs ~46 for a king's step):

  S_T   = KTEXT - KMAIN            5DAL's TEXT step (KTEXT raw: MAIN +0.00022 vs KMAIN)
  dMATH = KMATH - #43              5Ek5Ko's MATH king step (KMAIN + 0.5 dMATH: MAIN +0.00041)
  DC1   = C1_FINAL - C1_START      our 200-step TEXT training step
  CK1D  = CK1 - T1                 our first CODE king's step (KTEXT + 0.5 DC1 + 0.25 CK1D: MAIN +0.00055, MATH +0.00080)

  python scripts/merge/recipes_allkings.py     # prints the recipe paths in screening order
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
OUT = REPO / "configs/merge/allkings"

KT, KM, KMATH = "KTEXT_9a39afee", "KMAIN_c7ef5487", "KMATH_ddd9e626"
S_T = [(KT, 1.0), (KM, -1.0)]
DMATH = [(KMATH, 1.0), ("K43", -1.0)]
DC1 = [("C1_FINAL", 1.0), ("C1_START", -1.0)]
CK1D = [("CK1", 1.0), ("T1", -1.0)]


def x(terms, k):
    return [(n, c * k) for n, c in terms]


def write(label, desc, terms):
    merged = {}
    for name, coef in terms:
        merged[name] = merged.get(name, 0.0) + coef
    body = {"label": label, "description": desc, "group": "allkings",
            "terms": [{"model": M[n], "coef": round(c, 6)} for n, c in merged.items() if abs(c) > 1e-12]}
    (OUT / f"{label}.json").write_text(json.dumps(body, indent=1))
    return OUT / f"{label}.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ours = x(DC1, 0.5) + x(CK1D, 0.25)
    base = [(KT, 1.0)]
    paths = [
        write("ak_kt_math050", "KTEXT + 0.5 dMATH", base + x(DMATH, 0.5)),
        write("ak_kt_ours_math025", "KTEXT + 0.5 DC1 + 0.25 CK1D + 0.25 dMATH", base + ours + x(DMATH, 0.25)),
        write("ak_kt_ours_math050", "KTEXT + 0.5 DC1 + 0.25 CK1D + 0.5 dMATH", base + ours + x(DMATH, 0.5)),
        write("ak_kt_mom100", "KTEXT + 1.0 S_T (5DAL's TEXT step once more)", base + S_T),
        write("ak_kt_mom200", "KTEXT + 2.0 S_T", base + x(S_T, 2.0)),
        write("ak_kt_mom100_ours_math025", "KTEXT + 1.0 S_T + 0.5 DC1 + 0.25 CK1D + 0.25 dMATH", base + S_T + ours + x(DMATH, 0.25)),
        write("ak_kt_ours2", "KTEXT + 1.0 DC1 + 0.5 CK1D (our steps x2)", base + x(ours, 2.0)),
        write("ak_kt_dc1_math050", "KTEXT + 0.5 DC1 + 0.5 dMATH (no CK1D)", base + x(DC1, 0.5) + x(DMATH, 0.5)),
    ]
    print(" ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
