#!/usr/bin/env python3
"""Merge-method grid for the public campaign.

Linear probes (anchor ± donor) are the screen, not the search. The merges that
actually moved loss in this repo were king-step task arithmetic, experts-only
momentum, the module split experts×1.15 / attention×0.7 / head×0.7, SLERP,
TIES, DARE-TIES, DELLA, breadcrumbs, EMR, and orthogonal / sign-agreement
against a shared ancestor. This module proposes those methods from each
competition's own king, using every other model on disk as a donor and every
available king-to-king step as a task vector.

Points are dicts understood by merge_search4.norm / .recipe.
"""
from __future__ import annotations

import re

# shape that beat king #30: more of the donor on routed experts, less on attention and the head
MODULE = {"s_exp": 1.15, "s_attn": 0.7, "s_embed": 0.7}
_NAME = re.compile(r"^(M|TX|CD|MA)(\d+)$")


def predecessor(name: str) -> str | None:
    m = _NAME.fullmatch(name)
    if not m or int(m.group(2)) < 1:
        return None
    n = int(m.group(2)) - 1
    return f"M{n:02d}" if m.group(1) == "M" else f"{m.group(1)}{n}"


def toward(donor: str, king: str) -> str:
    return f"to{donor}_on_{king}"


def step_name(name: str) -> str:
    return f"step{name}"


def register(ms, avail: dict) -> list[str]:
    """Register donor and king-step task vectors. Returns step-vector names whose both ends are on disk."""
    steps = []
    for name in avail:
        pred = predecessor(name)
        if pred and pred in avail:
            ms.VECS.setdefault(step_name(name), (name, pred, 0.2, 0.05))
            steps.append(step_name(name))
    return steps


# Router-bias γ (k41): b' = mean(b) + g·(b − mean(b)) on e_score_correction_bias per layer.
# Sweet spot on TEXT king T1 was g≈0.9–1.1 (sharpen routing without retraining).
GAMMA_G = (0.9, 1.0, 1.05, 1.1, 1.15)


def gamma_on_king(king: str, gs: tuple[float, ...] = GAMMA_G) -> list[dict]:
    return [{"mode": "gamma", "base": king, "g": g} for g in gs]


def _lin(base: str, coefs: dict, **scales) -> dict:
    q = {"base": base, "c": dict(coefs), "mode": "lin"}
    q.update(scales)
    return q


def _bind(ms, donor: str, king: str) -> str:
    v = toward(donor, king)
    ms.VECS.setdefault(v, (donor, king, 0.2, 0.05))
    return v


def donor_methods_screen(ms, king: str, donor: str) -> list[dict]:
    """Cheap screen (5 recipes): enough to see if a donor deserves the full method grid.

    Sibling kings (same lineage) soup and extrapolate well; sparse election (TIES/DARE) loses on them (42.md), so the
    screen is soup / experts-only / module split / SLERP / ExPO-away.
    """
    v = _bind(ms, donor, king)
    return [
        _lin(king, {v: 0.25}, **MODULE),
        {**_lin(king, {v: 0.35}), "only": "experts"},
        _lin(king, {v: 0.35}),
        {"mode": "slerp", "a": king, "b": donor, "t": 0.5},
        _lin(king, {v: -0.15}),
    ]


def donor_methods(ms, king: str, donor: str, *, sparse: bool = True) -> list[dict]:
    """Full method grid for one donor (after a screen hit or a top probe). sparse=False skips TIES/DARE/DELLA/EMR/crumbs."""
    v = _bind(ms, donor, king)
    out = list(donor_methods_screen(ms, king, donor))
    out += [
        _lin(king, {v: 0.4}, **MODULE),
        {**_lin(king, {v: 0.5}), "only": "experts"},
        {**_lin(king, {v: 0.3}), "only": "dense"},
        _lin(king, {v: 0.2}),
        _lin(king, {v: 0.5}),
        _lin(king, {v: -0.25}),
        {"mode": "slerp", "a": king, "b": donor, "t": 0.35},
        {"mode": "slerp", "a": king, "b": donor, "t": 0.65},
    ]
    if not sparse:
        return out
    for mode, kw in (
        ("ties", {"density": 0.5}),
        ("ties", {"density": 0.25}),
        ("dare", {"drop": 0.5}),
        ("dare", {"drop": 0.3}),
        ("della", {"drop": 0.5, "eps": 0.1}),
        ("crumbs", {"density": 0.5}),
        ("emr", {"density": 0.5}),
    ):
        out.append({"base": king, "c": {v: 0.4}, "mode": mode, "lam": 1.0, **kw})
        out.append({"base": king, "c": {v: 0.4}, "mode": mode, "lam": 1.0, "only": "experts", **kw})
    return out


def lineage_methods(ms, king: str, donor: str, ancestor: str) -> list[dict]:
    """Keep only the part of the donor the king's own path from `ancestor` did not already learn."""
    if ancestor not in (king, donor):
        return (
            [{"mode": "ortho", "base": king, "donor": donor, "on": ancestor, "beta": b} for b in (0.25, 0.45, 0.7)]
            + [{"mode": "agree", "base": king, "donor": donor, "on": ancestor, "beta": b} for b in (0.3, 0.55)]
        )
    return []


def _reign(name: str) -> tuple[str, int]:
    m = _NAME.fullmatch(name)
    return (m.group(1), int(m.group(2))) if m else ("", 0)


def gradient_order(steps: list[str]) -> list[str]:
    """#36–#40 first. That chain's steps are what previously cut loss; then other MAIN steps, then the rest."""
    def key(v: str):
        prefix, n = _reign(v.removeprefix("step"))
        famous = 0 if prefix == "M" and 36 <= n <= 40 else 1
        return (famous, 0 if prefix == "M" else 1, -n, prefix)
    return sorted(steps, key=key)


def _famous_step(v: str) -> bool:
    p, n = _reign(v.removeprefix("step"))
    return p == "M" and 36 <= n <= 40


def gradient_methods(king: str, steps: list[str], *, focus: bool = True) -> list[dict]:
    """King + weighted sum of earlier kings' training steps (weight delta = that reign's gradient).

    A single step at 0.5, experts-only 0.5, and the module split are the doses that moved loss before.
    Stacks sum the #36–#40 chain when those checkpoints are on disk, newest step weighted most.
    """
    ordered = gradient_order(steps)
    out: list[dict] = []
    famous = [v for v in ordered if (lambda p, n: p == "M" and 36 <= n <= 40)(*_reign(v.removeprefix("step")))]
    famous = sorted(famous, key=lambda v: _reign(v.removeprefix("step"))[1])
    if len(famous) >= 2:
        weights = [0.5, 0.35, 0.25, 0.15, 0.1]
        coefs = {v: weights[i] for i, v in enumerate(reversed(famous[:5]))}
        out += [
            _lin(king, coefs, **MODULE),
            {**_lin(king, coefs), "only": "experts"},
            {"base": king, "c": dict(coefs), "mode": "ties", "density": 0.5, "lam": 1.0},
            {"base": king, "c": dict(coefs), "mode": "della", "drop": 0.5, "eps": 0.1, "lam": 1.0},
            {"base": king, "c": dict(coefs), "mode": "dare", "drop": 0.4, "lam": 1.0, "only": "experts"},
        ]
    main_steps = [v for v in ordered if _reign(v.removeprefix("step"))[0] == "M" and v.removeprefix("step") != king]
    main_steps = sorted(main_steps, key=lambda v: -_reign(v.removeprefix("step"))[1])
    if len(main_steps) >= 2:
        weights = [0.45, 0.3, 0.2, 0.12]
        coefs = {v: weights[i] for i, v in enumerate(main_steps[:4])}
        out.append(_lin(king, coefs, **MODULE))
        out.append({**_lin(king, {main_steps[0]: 0.5, main_steps[1]: 0.25}), "only": "experts"})
    for v in ordered:
        src = v.removeprefix("step")
        if src == king:
            # ExPO away from the king's own last step: [CD5] stepCD5=-0.1 led MATH and MAIN in the public pool
            for d in (-0.1, -0.15, -0.2, -0.3):
                out.append(_lin(king, {v: d}))
            out.append({**_lin(king, {v: -0.2}), "only": "experts"})
            out.append(_lin(king, {v: -0.15}, **MODULE))
            out.append(_lin(king, {v: 0.15}, **MODULE))
            continue
        if focus and not _famous_step(v):
            out.append(_lin(king, {v: 0.5}, **MODULE))
            out.append({**_lin(king, {v: 0.5}), "only": "experts"})
            out.append(_lin(king, {v: -0.15}))
            continue
        out.append(_lin(king, {v: 0.5}, **MODULE))
        out.append({**_lin(king, {v: 0.5}), "only": "experts"})
        out.append(_lin(king, {v: 0.25}))
        out.append(_lin(king, {v: -0.15}))
        if not focus:
            out.append({"base": king, "c": {v: 0.5}, "mode": "ties", "density": 0.5, "lam": 1.0})
            out.append({"base": king, "c": {v: 0.5}, "mode": "dare", "drop": 0.5, "lam": 1.0, "only": "experts"})
    return out


def soup_methods(ms, king: str, kings: list[str]) -> list[dict]:
    """Soups of the reigning kings (near siblings): uniform, king-weighted, Model Stock, and soup minus the newest step."""
    others = [k for k in kings if k != king]
    if len(others) < 2:
        return []
    vecs = {_bind(ms, d, king): 1.0 / (len(others) + 1) for d in others}
    out = [
        _lin(king, vecs),                                                   # uniform soup of all kings
        {**_lin(king, vecs), "only": "experts"},
        _lin(king, {v: w * 0.6 for v, w in vecs.items()}),                 # king-weighted soup
        _lin(king, vecs, **MODULE),
        {"mode": "stock", "fts": [king, *others], "on": king, "scale": 1.0},
        {"mode": "stock", "fts": [king, *others], "on": king, "scale": 0.6},
    ]
    step = step_name(king)
    if step in ms.VECS:
        out.append(_lin(king, {**vecs, step: -0.15}))                      # soup, then back off the newest step
    return out


def stock_methods(king: str, models: list[str], base: str = "TX1") -> list[dict]:
    # Model Stock's common ancestor is the base term; it must not also be listed as a fine-tune
    fam = [n for n in models if n not in (king, base)][:6]
    if len(fam) < 2:
        return []
    return [
        {"mode": "stock", "fts": fam, "on": king, "scale": 1.0},
        {"mode": "stock", "fts": fam[:4], "on": king, "scale": 0.5},
    ]


def pair_methods(ms, king: str, donors: list[str]) -> list[dict]:
    """Two-donor task arithmetic once single donors have been ranked."""
    vecs = [_bind(ms, d, king) for d in donors[:3] if d != king]
    out = []
    for i, a in enumerate(vecs):
        for b in vecs[i + 1:]:
            out.append(_lin(king, {a: 0.25, b: 0.2}, **MODULE))
            out.append({**_lin(king, {a: 0.3, b: 0.15}), "only": "experts"})
            out.append({"base": king, "c": {a: 0.25, b: 0.2}, "mode": "ties", "density": 0.5, "lam": 1.0})
            out.append({"base": king, "c": {a: 0.25, b: 0.2}, "mode": "emr", "density": 0.5, "lam": 1.0})
    return out


def refine(point: dict) -> list[dict]:
    """Local step around a point that already scored: coefficients, module scales, method hyperparameters."""
    import json
    mode = point.get("mode", "lin")
    if mode == "gamma":
        return [{**point, "g": round(point["g"] + d, 3)} for d in (-0.05, 0.05) if 0.75 < point["g"] + d < 1.35]
    if mode == "slerp":
        return [{**point, "t": round(point["t"] + d, 3)} for d in (-0.08, 0.08) if 0.05 < point["t"] + d < 0.95]
    if mode in ("ortho", "agree"):
        return [{**point, "beta": round(max(0.05, min(1.2, point["beta"] * f)), 3)} for f in (1.35, 0.7)]
    if mode == "stock":
        return [{**point, "scale": round(point.get("scale", 1.0) * f, 3)} for f in (1.4, 0.7)]
    if mode == "lin" and point.get("c"):
        out = []
        for k, w in list(point["c"].items()):
            for d in (0.08, -0.08):
                q = json.loads(json.dumps(point))
                q["c"][k] = round(w + d, 3)
                out.append(q)
        for sk, d in (("s_exp", 0.15), ("s_attn", -0.15), ("s_embed", -0.15), ("s_hi", 0.2), ("s_lo", -0.2)):
            q = json.loads(json.dumps(point))
            q[sk] = round(max(0.0, q.get(sk, 1.0) + d), 3)
            out.append(q)
        if not point.get("only"):
            out.append({**json.loads(json.dumps(point)), "only": "experts"})
        return out
    if mode not in ("lin", "stock", "slerp"):
        out = [{**point, "lam": round(point.get("lam", 1.0) * f, 3)} for f in (1.3, 0.75)]
        if mode in ("ties", "crumbs"):
            out += [{**point, "density": round(min(0.9, max(0.1, point.get("density", 0.5) * f)), 3)} for f in (0.6, 1.4)]
        if mode == "dare":
            out += [{**point, "drop": d} for d in (0.2, 0.6) if d != point.get("drop")]
        return out
    return []


def donors_in_point(q: dict) -> set[str]:
    """Public model aliases referenced as merge donors (steps, toward-vectors, SLERP b, stock fts)."""
    out: set[str] = set()
    for v in (q.get("c") or {}):
        if str(v).startswith("step"):
            out.add(str(v).removeprefix("step"))
        elif str(v).startswith("to") and "_on_" in str(v):
            out.add(str(v)[2:].split("_on_", 1)[0])
    if q.get("b"):
        out.add(q["b"])
    for n in q.get("fts") or ():
        out.add(n)
    if q.get("donor"):
        out.add(q["donor"])
    return {n for n in out if _NAME.fullmatch(n)}


def grid(ms, avail: dict, kings: dict, *, anchor: str | None = None, ancestor: str = "TX1",
         ranked: dict | None = None, focus: bool = True, expand_donors: set[str] | None = None) -> list[dict]:
    """Ordered candidates. `kings` is {competition: model name}. `ranked` is {king alias: [donor, ...]} by probe gain.

    With focus=True (default), most donors get a 5-recipe screen; full ~25-recipe grids run only for probe winners,
    cross-king specialists, and donors listed in expand_donors (screen recipes that beat the current best).
    """
    expand_donors = set(expand_donors or ())
    steps = register(ms, avail)
    # reigning kings first, then the anchor, then the rest of the near family, newest-looking names last
    preferred = list(dict.fromkeys([*kings.values(), *([anchor] if anchor else [])]))
    others = [n for n in avail if n not in preferred]
    donors_all = [n for n in preferred + others if n in avail]
    out = []
    seen = set()

    def add(q):
        # identity without hashing through recipe files: mode + the fields norm keeps
        mark = (q.get("mode", "lin"), q.get("base"), q.get("a"), q.get("b"), q.get("donor"), q.get("on"),
                q.get("only"), tuple(sorted((q.get("c") or {}).items())), q.get("t"), q.get("beta"),
                q.get("g"), q.get("gamma_g"), tuple(q.get("fts") or ()))
        if mark not in seen:
            seen.add(mark)
            out.append(q)

    for king in dict.fromkeys(kings.values()):
        if king not in avail:
            continue
        add({"base": king, "c": {}, "mode": "lin"})
        if not focus:  # router-bias gamma: +-0.00005 vs its own base in k41, listed as a dead end in 42.md
            for q in gamma_on_king(king):
                add(q)
        for q in soup_methods(ms, king, [k for k in kings.values() if k in avail]):
            add(q)
        # previous kings' training steps before any blend of the latest kings
        for q in gradient_methods(king, [s for s in steps if s.removeprefix("step") in avail], focus=focus):
            add(q)
        ranked_donors = [d for d in (ranked or {}).get(king, []) if d in avail and d != king]
        primary = list(dict.fromkeys(ranked_donors + [d for d in donors_all if d != king]))
        king_set = set(kings.values())
        specialists = [d for d in primary if d in king_set | ({anchor} if anchor else set()) | {ancestor}][:6]

        def donor_tier(donor: str) -> str:
            if donor in expand_donors or donor in ranked_donors[:2]:
                return "full"
            if donor in specialists or donor in ranked_donors[:4]:
                return "screen"
            return "skip"

        for donor in specialists:
            if donor not in avail or donor == king:
                continue
            tier = donor_tier(donor) if focus else "full"
            if tier == "skip":
                continue
            if tier == "full":
                for q in donor_methods(ms, king, donor, sparse=not focus):
                    add(q)
            else:
                for q in donor_methods_screen(ms, king, donor):
                    add(q)
            if (not focus or tier == "full") and ancestor in avail:
                for q in lineage_methods(ms, king, donor, ancestor):
                    add(q)
        for donor in primary:
            if donor in specialists or donor == king or donor not in avail:
                continue
            tier = donor_tier(donor) if focus else "screen"
            if tier == "full":
                for q in donor_methods(ms, king, donor, sparse=not focus):
                    add(q)
            elif tier == "screen":
                for q in donor_methods_screen(ms, king, donor):
                    add(q)
        family = [n for n in avail if _NAME.fullmatch(n)]
        if not focus or len(family) >= 4:
            for q in stock_methods(king, family, ancestor):
                add(q)
        if ranked_donors and (not focus or len(ranked_donors) >= 2):
            for q in pair_methods(ms, king, ranked_donors[:3]):
                add(q)
    return out
