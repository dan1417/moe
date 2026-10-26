#!/usr/bin/env python3
"""Second-wave recipes around the MAIN king #43: non-linear combiners, module splits and momentum, written from
configs/merge/k41/models.json into configs/merge/k43/.

Lineage (measured 2026-10-10, per-tensor norms / cosines): #43 was trained from #42 (CK3), not from T1 --
|K43 - CK3| < |K43 - T1| on every probed tensor and cos(K43 - T1, CK3 - T1) = +0.3...+0.64. So the base for 5DAL's
new step is CK3:
  S43  = K43 - CK3          5DAL's step since #42 (3-5x larger than ours, touches every layer and lm_head)
  O42  = C1_FINAL - CK3     our lineage since #42 (TEXT merges in C1_START + the 200-step c1 training)
  DC1  = C1_FINAL - C1_START
cos(S43, O42) = +0.10...+0.28 and the signs agree on only 40-50 % of entries: the two steps are nearly independent,
so a linear stack should be close to additive, TIES election would mostly keep S43 (larger magnitudes win) and drop
about half of O42, and the orthogonal projection removes little. Wave 2 measures exactly that, on the same pair:
linear vs orthogonal vs sign-agree vs TIES (sign election, no trim) vs trimmed TIES vs SLERP, plus module splits,
experts-only momentum (the only extrapolation that ever worked: (T1 - #40) at 0.15-0.25) and completing our CODE
recipe (#42 carried 0.25 x (CK1 - T1); CK1 - T1 = 0.5 x D36 + experts 0.15 x (T1 - #40), the int8 D36 vector itself
is not on this box).

History this builds on (MERGE.md, 41.md, 42.md): linear stacking of sibling steps won (+0.0016 ceiling at the
time), delta_orthogonal / delta_agree were the best non-linear combiners (+0.00135 / +0.00102), trimmed TIES,
DARE, DARE-TIES and SLERP lost when the lineages were strongly correlated (cos 0.57-0.95, not the case here).
Dead ends not repeated: router-bias gamma, logit temperature, D36 > 0.5, D33, momentum > 0.25, consensus-T.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
OUT = REPO / "configs/merge/k43"
EXPERTS = r"\.mlp\.experts\."


def write(label, desc, group, terms, **extra):
    """terms: [(model name, coef)] in order (non-linear modes use the order as roles); duplicates are summed in
    linear mode only."""
    if extra.get("mode", "linear") == "linear":
        merged = {}
        for name, coef in terms:
            merged[name] = merged.get(name, 0.0) + coef
        terms = list(merged.items())
    body = {"label": label, "description": desc, "group": group,
            "terms": [{"model": M[name], "coef": round(c, 6)} for name, c in terms], **extra}
    (OUT / f"{label}.json").write_text(json.dumps(body, indent=1))
    return label


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pair = [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", 1.0)]  # [king, donor, base] for the 3-term modes
    labels = [
        # --- wave 1 found #43 + beta x DC1 still rising at beta = 1 (TEXT +0.00515 / +0.00569, MAIN +0.0010 / +0.00136) ---
        write("k43_dc1_150", "#43 + 1.5 x our training step (final - start)", "dose",
              [("K43", 1.0), ("C1_FINAL", 1.5), ("C1_START", -1.5)]),
        write("k43_dc1_200", "#43 + 2.0 x our training step (final - start)", "dose",
              [("K43", 1.0), ("C1_FINAL", 2.0), ("C1_START", -2.0)]),
        # between k43_dc1_100 (best MAIN) and k43_o42_100 (= + 1.0 x (C1_START - CK3), best MATH)
        write("k43_dc1_100_cs025", "#43 + 1.0 x our training step + 0.25 x our TEXT merge stack since #42 (C1_START - CK3)",
              "dose", [("K43", 1.0), ("C1_FINAL", 1.0), ("C1_START", -0.75), ("CK3", -0.25)]),
        write("k43_dc1_100_cs050", "#43 + 1.0 x our training step + 0.5 x our TEXT merge stack since #42 (C1_START - CK3)",
              "dose", [("K43", 1.0), ("C1_FINAL", 1.0), ("C1_START", -0.5), ("CK3", -0.5)]),
        # module doses (dense alone +0.00079 / +0.00097, experts alone +0.00046 / +0.00086, both +0.00116 / +0.0015)
        write("k43_o42_d125_e050", "#43 + our lineage since #42: 1.25 x on dense tensors, 0.5 x on routed experts", "module-split",
              [("K43", 1.0), ("C1_FINAL", 1.25), ("CK3", -1.25)], overrides=[{"pattern": EXPERTS, "coefs": [1.0, 0.5, -0.5]}]),
        write("k43_o42_d100_e050", "#43 + our lineage since #42: 1.0 x on dense tensors, 0.5 x on routed experts", "module-split",
              [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", -1.0)], overrides=[{"pattern": EXPERTS, "coefs": [1.0, 0.5, -0.5]}]),
        # SLERP(#43, our final, 0.5) led MAIN / MATH (+0.00144 / +0.00159): map t, compare the linear soup and our
        # pre-training merge
        write("slerp_k43_c1_030", "SLERP(#43, our final, t=0.3) per tensor", "non-linear",
              [("K43", 1.0), ("C1_FINAL", 1.0)], mode="slerp", mode_args={"t": 0.3}),
        write("slerp_k43_c1_040", "SLERP(#43, our final, t=0.4) per tensor", "non-linear",
              [("K43", 1.0), ("C1_FINAL", 1.0)], mode="slerp", mode_args={"t": 0.4}),
        write("slerp_k43_c1_060", "SLERP(#43, our final, t=0.6) per tensor", "non-linear",
              [("K43", 1.0), ("C1_FINAL", 1.0)], mode="slerp", mode_args={"t": 0.6}),
        write("slerp_k43_c1s_050", "SLERP(#43, our pre-training merge C1_START, t=0.5) per tensor", "non-linear",
              [("K43", 1.0), ("C1_START", 1.0)], mode="slerp", mode_args={"t": 0.5}),
        write("soup_k43_c1_050", "linear soup 0.5 x #43 + 0.5 x our final (SLERP's linear twin)", "transplant",
              [("K43", 0.5), ("C1_FINAL", 0.5)]),
        write("k43_dc1_100_mom010e", "#43 + 1.0 x our training step + experts-only momentum 0.10 x (K43 - CK3)", "momentum",
              [("K43", 1.0), ("C1_FINAL", 1.0), ("C1_START", -1.0), ("CK3", 0.0)],
              overrides=[{"pattern": EXPERTS, "coefs": [1.10, 1.0, -1.0, -0.10]}]),
        # --- the main bet: our lineage since #42 transplanted onto #43 (linear) ---
        write("k43_o42_100", "#43 + 1.0 x our lineage since #42 (C1_FINAL - CK3)", "transplant",
              [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", -1.0)]),
        write("k43_o42_125", "#43 + 1.25 x our lineage since #42 (1.0: MATH +0.0015, MAIN +0.00116)", "dose",
              [("K43", 1.0), ("C1_FINAL", 1.25), ("CK3", -1.25)]),
        write("k43_o42_050", "#43 + 0.5 x our lineage since #42", "transplant",
              [("K43", 1.0), ("C1_FINAL", 0.5), ("CK3", -0.5)]),
        write("c1_s43_050", "our final + 0.5 x 5DAL's step since #42 (K43 - CK3)", "transplant",
              [("C1_FINAL", 1.0), ("K43", 0.5), ("CK3", -0.5)]),
        # --- same pair, non-linear combiners ---
        write("orth_k43_o42_100", "#43 + 1.0 x (our lineage since #42, minus its projection on 5DAL's step), per tensor",
              "non-linear", pair, mode="orthogonal", mode_args={"beta": 1.0}),
        write("sign_k43_o42_100", "#43 + 1.0 x our lineage since #42 only where it moves the same way as 5DAL's step",
              "non-linear", pair, mode="sign_agree", mode_args={"beta": 1.0}),
        write("ties_k43_o42", "TIES (sign election, no trim): #42 + elected-sign sum of both steps since #42",
              "non-linear", pair, mode="dare_ties",
              mode_args={"weights": [1.0, 1.0], "densities": [1.0, 1.0], "lambda": 1.0, "normalize": False}),
        write("tiestrim_k43_o42_05", "#43 + 1.0 x top-50% (|x|) of our lineage since #42, per tensor (trimmed TIES)",
              "non-linear", pair, mode="ties", mode_args={"density": 0.5, "beta": 1.0}),
        write("slerp_k43_c1_050", "SLERP(#43, our final, t=0.5) per tensor (vs the linear soup of wave 1)",
              "non-linear", [("K43", 1.0), ("C1_FINAL", 1.0)], mode="slerp", mode_args={"t": 0.5}),
        write("k43_o42_100_mn", "#43 + 1.0 x our lineage since #42, every routed-expert matrix rescaled to #43's norm",
              "non-linear", [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", -1.0)],
              transforms=[{"pattern": EXPERTS, "op": "match_norm", "args": {"ref": 0}}]),
        # --- module splits of the transplant ---
        write("k43_o42e_100", "#43 + 1.0 x our lineage since #42 on routed experts only (dense = #43)", "module-split",
              [("K43", 1.0), ("C1_FINAL", 0.0), ("CK3", 0.0)], overrides=[{"pattern": EXPERTS, "coefs": [1.0, 1.0, -1.0]}]),
        write("k43_o42d_100", "#43 + 1.0 x our lineage since #42 on dense tensors only (attention, shared experts, "
              "routers, head; routed experts = #43)", "module-split",
              [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", -1.0)], overrides=[{"pattern": EXPERTS, "coefs": [1.0, 0.0, 0.0]}]),
        # --- experts-only momentum along 5DAL's own step ---
        write("k43_mom010e", "#43 + experts-only momentum 0.10 x (K43 - CK3)", "momentum",
              [("K43", 1.0), ("CK3", 0.0)], overrides=[{"pattern": EXPERTS, "coefs": [1.10, -0.10]}]),
        write("k43_mom020e", "#43 + experts-only momentum 0.20 x (K43 - CK3)", "momentum",
              [("K43", 1.0), ("CK3", 0.0)], overrides=[{"pattern": EXPERTS, "coefs": [1.20, -0.20]}]),
        # --- CODE / MATH ingredients on #43 ---
        write("k43_ck1_075", "#43 + 0.75 x our CODE-king delta (CK1 - T1): with #42's 0.25 the full CK1 recipe",
              "code-math", [("K43", 1.0), ("CK1", 0.75), ("T1", -0.75)]),
        write("k43_o42_ck1_025", "#43 + our lineage since #42 + 0.25 x (CK1 - T1): the grand stack for MAIN", "transplant",
              [("K43", 1.0), ("C1_FINAL", 1.0), ("CK3", -1.0), ("CK1", 0.25), ("T1", -0.25)]),
    ]
    print(" ".join(labels))


if __name__ == "__main__":
    main()
