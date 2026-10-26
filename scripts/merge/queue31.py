#!/usr/bin/env python3
"""Write sweep31 queue recipes for one wave of the king #31 merge campaign.

  queue31.py wave1        # Dm scale sweep + S31 diagnostics
  queue31.py --list       # show the waves

Vectors (see sweep31.py): Dm = k30u - #30u (public-king merge step, 0.17*D26 + 0.13*D25 with
experts x1.15 / attn,head x0.7), S31 = #31u - #30u (king #31's own step).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

QUEUE = Path(__file__).resolve().parents[2] / "configs/merge/k31/queue"

EXPERTS = r"\.mlp\.experts\."
ATTN = r"\.self_attn\."
HEAD = r"^lm_head\.|\.shared_experts\."
UPPER = r"model\.layers\.(1[6-9]|[2-4]\d)\."
LOWER = r"model\.layers\.([0-9]|1[0-5])\.|embed_tokens"


def recipe(label, description, vecs, overrides=(), *, priority=100, limit=800, method="task-arith"):
    return {"label": label, "description": description, "method": method, "vecs": vecs,
            "overrides": [{"pattern": p, "vecs": v} for p, v in overrides],
            "priority": priority, "limit": limit}


def wave1():
    out = []
    for i, c in enumerate((1.0, 0.75, 1.25, 0.5, 1.5)):
        out.append(recipe(f"K_dm{int(c*100):03d}", f"#31u + {c}·Dm (Dm = k30u − #30u: public-king merge 0.17·D26 + 0.13·D25)",
                          {"Dm": c}, priority=10 + i))
    out.append(recipe("K_dm100_s31m10", "#31u + 1.0·Dm − 0.1·S31 (give back a bit of #31's step where it overlaps)",
                      {"Dm": 1.0, "S31": -0.1}, priority=20))
    out.append(recipe("K_s31p20", "#31u + 0.2·S31 (extrapolate #31's own step; diagnostic)", {"S31": 0.2},
                      priority=30, method="extrapolate"))
    out.append(recipe("K_s31m20", "#31u − 0.2·S31 (partially undo #31's own step; diagnostic)", {"S31": -0.2},
                      priority=31, method="extrapolate"))
    out.append(recipe("K_dm100_upper", "#31u + 1.0·Dm on layers 16–47 + lm_head only (where S31 lives)",
                      {"Dm": 0.0}, [(UPPER, {"Dm": 1.0}), (r"^lm_head\.", {"Dm": 1.0})], priority=40, method="per-depth"))
    out.append(recipe("K_dm100_lower", "#31u + 1.0·Dm on layers 0–15 + embeddings only (untouched by S31)",
                      {"Dm": 0.0}, [(LOWER, {"Dm": 1.0})], priority=41, method="per-depth"))
    return out


def mrecipe(label, description, vecs, *, transforms=None, ties=None, overrides=(), priority=100, limit=800, method):
    r = recipe(label, description, vecs, overrides, priority=priority, limit=limit, method=method)
    if transforms:
        r["transforms"] = transforms
    if ties:
        r["ties"] = ties
    return r


def wave2():
    """Aggressive merge methods on the public-king step Dm (and TIES of the two kings over #30u)."""
    out = []
    for d, c in ((0.3, 1.0), (0.3, 1.5), (0.5, 1.0), (0.5, 1.3), (0.7, 1.1)):
        out.append(mrecipe(f"K_trim{int(d*100)}_c{int(c*100)}", f"#31u + {c}·trim_{d}(Dm): keep top-{int(d*100)}% |Δ| per tensor (TIES trim)",
                           {"Dm": c}, transforms={"Dm": [{"op": "trim", "density": d}]}, priority=110, method="TIES-trim"))
    for p_, c in ((0.5, 1.0), (0.9, 1.0), (0.5, 1.3)):
        out.append(mrecipe(f"K_dare{int(p_*100)}_c{int(c*100)}", f"#31u + {c}·DARE_p{p_}(Dm): drop {int(p_*100)}% of Δ at random, rescale",
                           {"Dm": c}, transforms={"Dm": [{"op": "dare", "p": p_}]}, priority=120, method="DARE"))
    out.append(mrecipe("K_bread_c100", "#31u + 1.0·breadcrumbs(Dm): drop top 1% and bottom 50% |Δ| per tensor",
                       {"Dm": 1.0}, transforms={"Dm": [{"op": "drop_top", "frac": 0.01}, {"op": "trim", "density": 0.5}]},
                       priority=125, method="breadcrumbs"))
    out.append(mrecipe("K_clip99_c100", "#31u + 1.0·clip_99%(Dm): clip Δ outliers at the per-tensor 99th percentile",
                       {"Dm": 1.0}, transforms={"Dm": [{"op": "clip", "q": 0.99}]}, priority=126, method="clip"))
    for c in (1.0, 1.5):
        out.append(mrecipe(f"K_signS31_c{int(c*100)}", f"#31u + {c}·Dm only where sign(Dm)=sign(S31) (all of Dm where #31 did not move)",
                           {"Dm": c}, transforms={"Dm": [{"op": "sign", "ref": "S31", "keep": "agree"}]}, priority=130, method="sign-agree"))
    out.append(mrecipe("K_signS31dis_c100", "#31u + 1.0·Dm only where it disagrees in sign with S31 (diagnostic)",
                       {"Dm": 1.0}, transforms={"Dm": [{"op": "sign", "ref": "S31", "keep": "disagree", "zero_ref": "drop"}]},
                       priority=135, method="sign-agree"))
    for c in (1.0, 1.3):
        out.append(mrecipe(f"K_orth_c{int(c*100)}", f"#31u + {c}·(Dm ⟂ S31): Dm minus its per-tensor projection on #31's step",
                           {"Dm": c}, transforms={"Dm": [{"op": "orth", "ref": "S31"}]}, priority=140, method="orthogonal"))
    for d, lam in ((1.0, 1.0), (0.5, 1.0), (0.5, 1.5), (0.3, 1.5)):
        out.append(mrecipe(f"K_ties_d{int(d*100)}_l{int(lam*100)}",
                           f"TIES over base #30u of #31u and k30u: #30u + {lam}·TIES(S31, Dm; density {d})",
                           {"S31": -1.0}, ties={"vecs": {"S31": 1.0, "Dm": 1.0}, "density": d, "lambda": lam},
                           priority=150, method="TIES"))
    out.append(mrecipe("K_dareties_p50_l15", "DARE-TIES over base #30u: #30u + 1.5·TIES(DARE_0.5(S31), DARE_0.5(Dm))",
                       {"S31": -1.0}, ties={"vecs": {"S31": 1.0, "Dm": 1.0}, "dare": 0.5, "lambda": 1.5},
                       priority=155, method="DARE-TIES"))
    out.append(mrecipe("K_ties_w2_d50", "#30u + TIES(2·S31, Dm; density 0.5), lambda 1 (king-weighted election)",
                       {"S31": -1.0}, ties={"vecs": {"S31": 2.0, "Dm": 1.0}, "density": 0.5, "lambda": 1.0},
                       priority=156, method="TIES"))
    return out


GROUPS = {
    "explo": r"model\.layers\.([0-9]|1[0-5])\.mlp\.experts\.",
    "exphi": r"model\.layers\.(1[6-9]|[2-4]\d)\.mlp\.experts\.",
    "attlo": r"model\.layers\.([0-9]|1[0-5])\.self_attn\.",
    "atthi": r"model\.layers\.(1[6-9]|[2-4]\d)\.self_attn\.",
    "shared": r"\.shared_experts\.|model\.layers\.0\.mlp\.",
    "head": r"^lm_head\.",
    "embed": r"embed_tokens",
    "norm": r"norm\.weight$",
}


def wave3(base_c: float = 1.0, mults=(0.0, 1.6), prefix="G"):
    """One group at a time: Dm coefficient base_c everywhere, x mult on that group."""
    out = []
    for g, pattern in GROUPS.items():
        for m in mults:
            c = round(base_c * m, 4)
            out.append(recipe(f"{prefix}_{g}_x{int(m*100):03d}", f"#31u + {base_c}·Dm, but {c}·Dm on {g} ({pattern})",
                              {"Dm": base_c}, [(pattern, {"Dm": c})], priority=200 + len(out), method="per-module"))
    return out


WAVES = {"wave1": wave1, "wave2": wave2, "wave3": wave3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wave", nargs="?")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--base-c", type=float, default=None, help="wave3: Dm coefficient to perturb around")
    args = ap.parse_args()
    if args.list or not args.wave:
        print("waves:", ", ".join(WAVES))
        return
    QUEUE.mkdir(parents=True, exist_ok=True)
    fn = WAVES[args.wave]
    for r in (fn(args.base_c) if args.base_c is not None else fn()):
        path = QUEUE / f"{r['label']}.json"
        if (QUEUE / "done" / path.name).exists():
            print("skip (done)", path.name)
            continue
        path.write_text(json.dumps(r, indent=1) + "\n")
        print("queued", path.name, r["priority"], r["description"])


if __name__ == "__main__":
    main()
