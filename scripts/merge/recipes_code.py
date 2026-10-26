#!/usr/bin/env python3
"""CODE recipes against the new CODE king KC2 (5DALwjE4-99, a light CODE fine-tune of #43), written into configs/merge/code/.

Screen vs KC2 so far (CODE A, 4 800): #43-based blends beat KC2-based ones (slerp_k43_c1_040 +0.00204, soup_k43_c1_050
+0.00190; slerp_kc2_c1_040 +0.00174, soup_kc2_c1_050 +0.00173), and a quarter of our CODE-king delta (CK1 - T1:
0.5 x D36 + experts momentum) on the KC2 soup adds +0.00017 (code split +0.0002). So: the #43 blends + that delta.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
OUT = REPO / "configs/merge/code"
CK1D = lambda k: [("CK1", k), ("T1", -k)]  # noqa: E731
DC1 = lambda k: [("C1_FINAL", k), ("C1_START", -k)]  # noqa: E731


def write(label, desc, terms):
    merged = {}
    for name, coef in terms:
        merged[name] = merged.get(name, 0.0) + coef
    body = {"label": label, "description": desc, "group": "code",
            "terms": [{"model": M[n], "coef": round(c, 6)} for n, c in merged.items() if abs(c) > 1e-12]}
    (OUT / f"{label}.json").write_text(json.dumps(body, indent=1))
    return OUT / f"{label}.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [
        write("soup_k43_c1_050_ck1_025", "soup 0.5 x #43 + 0.5 x our final + 0.25 x our CODE-king delta (CK1 - T1)",
              [("K43", 0.5), ("C1_FINAL", 0.5)] + CK1D(0.25)),
        write("soup_k43_c1_040_ck1_025", "0.6 x #43 + 0.4 x our final (SLERP t=0.4's linear twin) + 0.25 x CK1 delta",
              [("K43", 0.6), ("C1_FINAL", 0.4)] + CK1D(0.25)),
        write("soup_k43_c1_050_ck1_050", "soup 0.5 x #43 + 0.5 x our final + 0.5 x our CODE-king delta",
              [("K43", 0.5), ("C1_FINAL", 0.5)] + CK1D(0.5)),
        write("soup_k43_c1_040", "0.6 x #43 + 0.4 x our final (SLERP t=0.4's linear twin)", [("K43", 0.6), ("C1_FINAL", 0.4)]),
        write("soup_k43kc2_c1_050_ck1_025", "0.25 x #43 + 0.25 x KC2 + 0.5 x our final + 0.25 x CK1 delta (half of 5DAL's CODE step)",
              [("K43", 0.25), ("KC2", 0.25), ("C1_FINAL", 0.5)] + CK1D(0.25)),
        write("soup_k43_c1_050_dc1_050_ck1_025", "soup 0.5 x #43 + 0.5 x our final + 0.5 x our training step + 0.25 x CK1 delta",
              [("K43", 0.5), ("C1_FINAL", 0.5)] + DC1(0.5) + CK1D(0.25)),
        # soup_k43_c1_050_dc1_050_ck1_025 was the first to pass KC2's bar on the screen (+0.002249, LCB +0.002025): more
        # training step, and the 0.6 / 0.4 base (+0.0001 over 0.5 / 0.5 with the delta)
        write("soup_k43_c1_040_dc1_050_ck1_025", "0.6 x #43 + 0.4 x our final + 0.5 x our training step + 0.25 x CK1 delta",
              [("K43", 0.6), ("C1_FINAL", 0.4)] + DC1(0.5) + CK1D(0.25)),
        write("soup_k43_c1_050_dc1_100_ck1_025", "soup 0.5 x #43 + 0.5 x our final + 1.0 x our training step + 0.25 x CK1 delta",
              [("K43", 0.5), ("C1_FINAL", 0.5)] + DC1(1.0) + CK1D(0.25)),
        write("soup_k43_c1_040_dc1_100_ck1_025", "0.6 x #43 + 0.4 x our final + 1.0 x our training step + 0.25 x CK1 delta",
              [("K43", 0.6), ("C1_FINAL", 0.4)] + DC1(1.0) + CK1D(0.25)),
    ]
    # MAIN sweep (user 04:46: "adjust some parameters for a better MAIN model") around the MAIN leader
    # a x #43 + (1-a) x our final + b x training step + c x CK1 delta, leader (a, b, c) = (0.6, 0.5, 0.25): +0.002118
    for a, b, c in ((0.65, 0.5, 0.25), (0.7, 0.5, 0.25), (0.6, 0.75, 0.25), (0.6, 0.5, 0.0), (0.6, 0.5, 0.5), (0.65, 0.75, 0.25)):
        label = f"sw_a{int(a * 100):03d}_b{int(b * 100):03d}_c{int(c * 100):03d}"
        paths.append(write(label, f"MAIN sweep: {a} x #43 + {1 - a:.2f} x our final + {b} x our training step + {c} x CK1 delta",
                           [("K43", a), ("C1_FINAL", 1 - a)] + DC1(b) + CK1D(c)))
    print(" ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
