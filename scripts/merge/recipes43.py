#!/usr/bin/env python3
"""First-wave merge recipes around the new MAIN king #43 (5DALwjE4-96), written from configs/merge/k41/models.json.

Ingredients (same hidden basis as T1/#41): K43, T1, CK1 (our CODE king), C1_START / C1_FINAL (our 200-step text run).
  T43 = K43 - T1          5DALwjE4's step since T1
  DC1 = C1_FINAL - C1_START   our training step
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
OUT = REPO / "configs/merge/k43"


def recipe(label, desc, terms):
    merged = {}
    for name, coef in terms:
        merged[M[name]] = merged.get(M[name], 0.0) + coef
    body = {"label": label, "description": desc,
            "terms": [{"model": path, "coef": round(c, 6)} for path, c in merged.items() if abs(c) > 1e-12]}
    (OUT / f"{label}.json").write_text(json.dumps(body, indent=1))
    return OUT / f"{label}.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        recipe("k43_raw", "new MAIN king #43 as is (reference on every competition)", [("K43", 1.0)]),
        recipe("c1_final", "our 200-step text run (final) as is", [("C1_FINAL", 1.0)]),
        recipe("k43_dc1_050", "#43 + 0.5 x our training step (final - start)", [("K43", 1.0), ("C1_FINAL", 0.5), ("C1_START", -0.5)]),
        recipe("k43_dc1_100", "#43 + 1.0 x our training step (final - start)", [("K43", 1.0), ("C1_FINAL", 1.0), ("C1_START", -1.0)]),
        recipe("c1_t43_050", "our final + 0.5 x 5DAL's new step (#43 - T1)", [("C1_FINAL", 1.0), ("K43", 0.5), ("T1", -0.5)]),
        recipe("c1_t43_100", "our final + 1.0 x 5DAL's new step (#43 - T1)", [("C1_FINAL", 1.0), ("K43", 1.0), ("T1", -1.0)]),
        recipe("soup_k43_c1", "plain soup 0.5 x #43 + 0.5 x our final", [("K43", 0.5), ("C1_FINAL", 0.5)]),
        recipe("k43_ours025", "#43 + 0.25 x our CODE-king delta (CK1 - T1)", [("K43", 1.0), ("CK1", 0.25), ("T1", -0.25)]),
    ]
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
