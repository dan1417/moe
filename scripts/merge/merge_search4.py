#!/usr/bin/env python3
"""Joint merge search v3 (user 19:15 JST Oct 11: "merge with our checkpoints and the current top models, make the best model for
each competition, every possible way, not only linear"). As merge_search2, plus: each point has its own base king (MAIN / TEXT
on KTEXT, MATH on KMATH, CODE on the new CODE king KCODE_0c681d78 = 5Ek5Ko v56, uid 132), the new CODE king's step, our TEXT
training steps (c2 to abs 75, c3's layers 8-15 step) and SLERP between top models / our checkpoints.

(merge_search2's description follows.) Joint merge search for all four competitions, linear and non-linear (user 21:40: "why
only linear merges? and you don't calculate CODE and TEXT scores").

Every candidate is scored on MAIN (vs KMAIN, bar 0.003), MATH (vs KMATH, 0.002), CODE (vs our t007 = m1/step-150, 0.002)
and TEXT (vs KTEXT, 0.002), 1 200 seqs of sample A each, in one load. Each competition keeps its own current point and
moves to the best point for it among everything scored so far (all competitions share the pool).

A point = task-vector weights on top of KTEXT plus how they are combined (user 22:15: "also other best methods"):
  lin    W = KTEXT + sum_i c_i V_i, optionally per module: attention x s_attn, routed experts x s_exp (regex overrides)
  ties   W = KTEXT + TIES_i(c_i V_i): each V_i trimmed to its top-|x| `density` fraction, sign election, agreeing sum
  dare   W = KTEXT + TIES_i(DARE(c_i V_i)): random drop `drop`, 1/(1-drop) rescale, sign election
  crumbs ties with the largest 2 % of each V_i dropped too (model breadcrumbs)
  della  DELLA-TIES: drop with probability ranked by |x| (1-drop +- eps), rescale, sign election
  emr    EMR-merging's elected vector: elected sign x the largest agreeing |c_i V_i| (optionally TIES-trimmed)
  lin also takes per layer group scales (layers 0-14 / 15-29 / 30-44) on top of the module scales, and scales for the
         MoE routers, the embeddings / lm_head and the norms
  stock  Model Stock over kings fine-tuned from #43 (per-tensor ratio from their angle), optionally added onto KTEXT
Moves per round around each competition's point: every vector's weight +- its step; mode switches (lin -> ties 0.5 / 0.3,
dare 0.5, crumbs; ties density x0.6 / x1.5; dare drop 0.3 / 0.7; back to lin); module scales s_attn / s_exp +- step.
No gain for a competition halves its steps; done when every competition's steps are below their minimum.

  python scripts/merge/merge_search2.py --name all1 [--start DC1=0.5,CK1D=0.25]
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = Path("/data/teutonic")
M = json.loads((REPO / "configs/merge/k41/models.json").read_text())
K2 = "KTEXT2_6d6713a5"   # the new TEXT king (uid 130, coldkey 5Fxtier6), crowned 06:00 JST Oct 12
KT, KM, KMA, KC = "KTEXT_9a39afee", "KMAIN_c7ef5487", "KMATH_ddd9e626", "KCODE_0c681d78"
VECS = {  # name: (A, B, step, min step); "to*" = toward a model from the new king (linear interpolation when added to K2)
    "toC8": ("C8_S20", K2, 0.15, 0.05),       # toward our long-context checkpoint c8/step-20
    "toKC": (KC, K2, 0.15, 0.05),             # toward the CODE king (5Ek5Ko v56)
    "toKMA": (KMA, K2, 0.1, 0.05),            # toward the MATH king
    "toKT": (KT, K2, 0.15, 0.05),             # toward the old TEXT king
    "dK2T": (K2, KT, 0.25, 0.1),              # the new king's step over the old TEXT king
    "dK2C": (K2, KC, 0.25, 0.1),              # the new king's step over the CODE king
    "dTR": ("C8_S20", "START_TM", 0.5, 0.25), # our TEXT training (c5 -> c7 -> c8) on top of the ES merge
    "dES": ("START_TM", KC, 0.25, 0.1),       # our ES merge step over the CODE king
    "S_T": (KT, KM, 0.5, 0.25),               # 5DAL's earlier TEXT step
}
COMPS = {
    "main": (KM, 0.003, "e8ef9ae1", ("toC8", "toKC", "toKMA", "dK2T", "dK2C", "dTR", "dES", "S_T")),
    "math": (KMA, 0.002, "abfd40f3", ("toC8", "toKC", "toKMA", "dK2T", "dTR", "dES")),
    "code": (KC, 0.002, "7dcbbb5f", ("toC8", "toKC", "dK2T", "dK2C", "dTR")),
    "text": (K2, 0.002, "0a2840c9", ("toC8", "toKC", "toKMA", "toKT", "dK2T", "dK2C", "dTR", "dES")),
}
BASE = K2
STARTS = {
    "main": {"base": "C8_S20", "c": {}, "mode": "lin"},   # best MAIN so far (+0.001322 vs KMAIN)
    "math": {"base": "C8_S20", "c": {}, "mode": "lin"},
    "code": {"base": KC, "c": {}, "mode": "lin"},
    "text": {"base": K2, "c": {}, "mode": "lin"},
}
LIMIT, EPS, MAX_CANDS = 1200, 5e-5, 100
OUT = REPO / "configs/merge/search4"
LOCK = D / "gpu_screen.lock"
ATTN, EXP = r"self_attn\.", r"mlp\.experts\."
GROUPS = {"s_lo": r"layers\.([0-9]|1[0-4])\.", "s_mid": r"layers\.(1[5-9]|2[0-9])\.", "s_hi": r"layers\.(3[0-9]|4[0-4])\."}
SCALES = ("s_attn", "s_exp", "s_lo", "s_mid", "s_hi", "s_router", "s_embed", "s_norm")
# whole-model parts with their own scale (MoE routers, embeddings / head, norms); matched before the layer groups
PARTS = {"s_router": r"mlp\.gate\.", "s_embed": r"embed_tokens|lm_head", "s_norm": r"layernorm|^model\.norm\."}


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def norm(p):
    if p.get("mode") == "gamma":
        return {"mode": "gamma", "base": p["base"], "g": round(float(p["g"]), 4)}
    if p.get("mode") == "slerp":
        return {"mode": "slerp", "a": p["a"], "b": p["b"], "t": round(p["t"], 4)}
    if p.get("mode") in ("ortho", "agree"):
        # king + beta * donor-task, measured from a shared ancestor (`on`, usually TX1)
        return {"mode": p["mode"], "base": p["base"], "donor": p["donor"], "on": p.get("on", "TX1"),
                "beta": round(float(p.get("beta", 0.4)), 4)}
    q = {"c": {k: round(v, 4) for k, v in p.get("c", {}).items() if abs(v) > 1e-9}, "mode": p.get("mode", "lin"),
         "base": p.get("base", BASE)}
    if q["mode"] == "lin":
        for sk in SCALES:
            q[sk] = round(p.get(sk, 1.0), 4)
    elif q["mode"] == "ties":
        q["density"] = round(p.get("density", 0.5), 4)
    elif q["mode"] == "dare":
        q["drop"] = round(p.get("drop", 0.5), 4)
    elif q["mode"] == "crumbs":
        q["density"] = round(p.get("density", 0.5), 4)
    elif q["mode"] == "della":
        q["drop"], q["eps"] = round(p.get("drop", 0.5), 4), round(p.get("eps", 0.1), 4)
    elif q["mode"] == "emr":
        q["density"] = round(p.get("density", 1.0), 4)
    if q["mode"] == "stock":
        q["fts"], q["on"], q["scale"] = list(p["fts"]), p.get("on", KT), round(p.get("scale", 1.0), 4)
        q.pop("c", None)
        return q
    if p.get("only") in ("experts", "dense"):
        q["only"] = p["only"]
    if p.get("gamma_g") is not None:
        q["gamma_g"] = round(float(p["gamma_g"]), 4)
    if q["mode"] != "lin":
        q["lam"] = round(p.get("lam", 1.0), 4)
    return q


def key(p):
    p = norm(p)
    if p["mode"] == "gamma":
        return f"[{p['base']}] router-bias gamma g={p['g']:g}"
    if p["mode"] == "slerp":
        return f"slerp({p['a']}, {p['b']}, t={p['t']:g})"
    if p["mode"] == "stock":
        return f"model stock ({KT}; {', '.join(p['fts'])}) on {p['on']} x{p['scale']:g}"
    if p["mode"] == "ortho":
        return f"orthogonal({p['base']} + {p['beta']:g}*({p['donor']}-{p['on']}) perp ({p['base']}-{p['on']}))"
    if p["mode"] == "agree":
        return f"sign-agree({p['base']} + {p['beta']:g}*({p['donor']}-{p['on']}) where signs match)"
    s = ",".join(f"{k}={v:g}" for k, v in sorted(p["c"].items())) or "base"
    if p["base"] != BASE:
        s = f"[{p['base']}] " + s
    if p["mode"] == "lin":
        if p["s_attn"] != 1 or p["s_exp"] != 1:
            s += f" | lin attn x{p['s_attn']:g} experts x{p['s_exp']:g}"
        if any(p[g] != 1 for g in GROUPS):
            s += f" | layers 0-14 x{p['s_lo']:g} 15-29 x{p['s_mid']:g} 30-44 x{p['s_hi']:g}"
        if any(p[x] != 1 for x in PARTS):
            s += f" | router x{p['s_router']:g} embed/head x{p['s_embed']:g} norms x{p['s_norm']:g}"
    elif p["mode"] in ("ties", "crumbs"):
        s += f" | {p['mode']} density {p['density']:g}"
    elif p["mode"] == "della":
        s += f" | della-ties drop {p['drop']:g} eps {p['eps']:g}"
    elif p["mode"] == "emr":
        s += f" | emr elect-max density {p['density']:g}"
    elif p["mode"] == "dare":
        s += f" | dare-ties drop {p['drop']:g}"
    if p.get("only"):
        s += f" | only {p['only']}"
    if p.get("gamma_g") is not None:
        s += f" | router-bias gamma g={p['gamma_g']:g}"
    if p["mode"] != "lin" and p["lam"] != 1:
        s += f" lambda {p['lam']:g}"
    return s


def _gamma_transform(g: float) -> list[dict]:
    return [{"pattern": "e_score_correction_bias$", "op": "center_scale", "args": {"g": g}}]


def build(p):
    """(label, recipe body) of point p, nothing written (public_merge.py labels by this body)."""
    p = norm(p)
    label = "s4_" + hashlib.sha1(key(p).encode()).hexdigest()[:10]
    if p["mode"] == "gamma":
        base = p["base"]
        body = {"label": label, "description": key(p), "group": "search4", "method": "router-bias-gamma",
                "terms": [{"model": M[base], "coef": 1.0}], "transforms": _gamma_transform(p["g"])}
        return label, body
    if p["mode"] == "slerp":
        body = {"label": label, "description": key(p), "group": "search4", "terms": [{"model": M[p["a"]], "coef": 1.0},
                {"model": M[p["b"]], "coef": 1.0}], "mode": "slerp", "mode_args": {"t": p["t"]}}
        return label, body
    if p["mode"] in ("ortho", "agree"):
        body = {"label": label, "description": key(p), "group": "search4",
                "terms": [{"model": M[p["base"]], "coef": 1.0}, {"model": M[p["donor"]], "coef": 1.0},
                          {"model": M[p["on"]], "coef": 1.0}],
                "mode": "orthogonal" if p["mode"] == "ortho" else "sign_agree",
                "mode_args": {"beta": p["beta"]}}
        return label, body
    base = p.get("base", BASE)
    body = {"label": label, "description": f"{base} + {key(p).split('] ')[-1]}", "group": "search4"}
    if p["mode"] == "stock":
        models = [M[KT]] + [M[f] for f in p["fts"]]  # common base: the old TEXT king (K43 is gone; KCODE / c8 / the new king descend from KT)
        if p["on"] != KT:
            models.append(M[p["on"]])
        body.update({"description": key(p), "terms": [{"model": m, "coef": 1.0} for m in models], "mode": "model_stock",
                     "mode_args": {"base": 0, "models": list(range(1, 1 + len(p["fts"]))), "on": len(models) - 1 if p["on"] != KT else 0,
                                   "scale": p["scale"]}})
        return label, body
    if p["mode"] == "lin":
        models = [M[base]]
        for k in p["c"]:
            for n in VECS[k][:2]:
                if M[n] not in models:
                    models.append(M[n])

        def coefs(scale):
            c = {m: 0.0 for m in models}
            c[M[base]] = 1.0
            for k, w in p["c"].items():
                a, b = VECS[k][:2]
                c[M[a]] += w * scale
                c[M[b]] -= w * scale
            return [round(c[m], 6) for m in models]

        body["terms"] = [{"model": m, "coef": c} for m, c in zip(models, coefs(1.0))]
        # per (layer group x module) scale = group scale x module scale; first matching override wins
        over = [(pat, p[x]) for x, pat in PARTS.items() if p[x] != 1]
        for g, gpat in GROUPS.items():
            for mpat, ms in ((ATTN, p["s_attn"]), (EXP, p["s_exp"]), ("", 1.0)):
                sc = p[g] * ms
                if sc != 1:
                    over.append((gpat + mpat, sc))
        if p.get("only") == "experts":
            # routed experts keep the task vector; every other tensor stays at the base king
            over = [(r"mlp\.experts\.", 1.0), (r".", 0.0)]
        elif p.get("only") == "dense":
            over = [(r"mlp\.experts\.", 0.0)] + over
        if over:
            body["overrides"] = [{"pattern": pat, "coefs": coefs(sc)} for pat, sc in over]
        if p.get("gamma_g") is not None:
            body["method"] = "router-bias-gamma"
            body["transforms"] = _gamma_transform(p["gamma_g"])
    else:
        models = [M[base]]
        vecs = []
        for k, w in p["c"].items():
            idx = []
            for n in VECS[k][:2]:
                if M[n] not in models:
                    models.append(M[n])
                idx.append(models.index(M[n]))
            vecs.append([idx[0], idx[1], w])
        args = {"vecs": vecs, "lambda": p["lam"], "merge": "sum"}
        if p["mode"] in ("ties", "crumbs"):
            args["density"] = p["density"]
        if p["mode"] == "crumbs":
            args["top_drop"] = 0.02
        if p["mode"] == "dare":
            args["drop"] = p["drop"]
        if p["mode"] == "della":
            args["drop"], args["della"] = p["drop"], p["eps"]
        if p["mode"] == "emr":
            args["merge"] = "max"
            if p["density"] < 1:
                args["density"] = p["density"]
        if p.get("only") == "experts":
            args["exclude"] = [r"^(?!.*mlp\.experts\.).*"]
        elif p.get("only") == "dense":
            args["exclude"] = [r"mlp\.experts\."]
        body.update({"terms": [{"model": m, "coef": 1.0} for m in models], "mode": "vec_ties", "mode_args": args})
    return label, body


def recipe(p):
    label, body = build(p)
    path = OUT / f"{label}.json"
    path.write_text(json.dumps(body, indent=1))
    return label, path


def results(labels):
    """{label: {comp: mu}} at LIMIT seqs vs each competition's king on its sample A."""
    out = {}
    king_of = {Path(M[COMPS[c][0]]).name if not M[COMPS[c][0]].endswith(".json") else Path(M[COMPS[c][0]]).stem: c for c in COMPS}
    for f in sorted((D / "eval_cache/results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in data.get("results", []):
            lab, v = (r.get("challenger") or {}).get("label"), r.get("verdict") or {}
            if lab not in labels or v.get("n_sequences") != LIMIT:
                continue
            klabel = (r.get("king") or {}).get("label", "")
            for c, (king, _, prefix, _) in COMPS.items():
                if r.get("sample", "").startswith(prefix) and king_of.get(klabel) == c:
                    out.setdefault(lab, {})[c] = v["mu_hat"]
    return out


def score(points):
    rec = {key(p): recipe(p) for p in points}
    have = results({lab for lab, _ in rec.values()})
    todo = [path for lab, path in rec.values() if len(have.get(lab, {})) < len(COMPS)]
    if todo:
        while LOCK.exists() and time.time() - LOCK.stat().st_mtime < 4 * 3600:
            time.sleep(20)
        LOCK.write_text("merge_search4")
        try:
            training = bool(subprocess.run(["pgrep", "-f", "python.*cpt_fsdp"], capture_output=True, text=True).stdout.strip())
            kings = " ".join(f"--king {c}={k}:{bar}" for c, (k, bar, _, _) in COMPS.items())
            cmd = (f"cd {REPO} && source {D}/env.sh && PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py "
                   + " ".join(f"--recipe {p}" for p in todo)
                   + f" {kings} --comps {','.join(COMPS)} --campaign search4 --limit {LIMIT} --sample tag:A --replicas 4"
                   f" --batch-size 32 --pingpong {0 if training else 1}")
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            if p.returncode != 0:
                log("campaign43 failed: " + (p.stdout + p.stderr)[-800:])
        finally:
            LOCK.unlink(missing_ok=True)
        have = results({lab for lab, _ in rec.values()})
    return {k: have.get(lab, {}) for k, (lab, _) in rec.items()}


def moves(p, comp, st):
    """Neighbours of point p for competition comp with its steps st."""
    out = []
    for k in (COMPS[comp][3] if p.get("mode") != "stock" else ()):
        if st[k] < VECS[k][3] - 1e-9:
            continue
        for s in (st[k], -st[k]):
            q = json.loads(json.dumps(p)); q["c"][k] = q["c"].get(k, 0.0) + s
            out.append(q)
    mode, sm = p.get("mode", "lin"), st["mode"]
    if mode == "slerp":
        return [{**p, "t": round(p["t"] + d, 3)} for d in (-0.1, 0.1) if 0.05 < p["t"] + d < 0.95] if sm >= 0.25 else []
    if mode == "stock":
        return [{**p, "scale": round(p["scale"] * f, 3)} for f in (1.5, 0.67)] if sm >= 0.25 else []
    if mode == "lin":
        if sm >= 0.25:
            b = {"c": dict(p["c"]), "base": p.get("base", BASE)}
            out += [{**b, "mode": "ties", "density": 0.5}, {**b, "mode": "ties", "density": 0.3}, {**b, "mode": "dare", "drop": 0.5},
                    {**b, "mode": "crumbs", "density": 0.5}, {**b, "mode": "della", "drop": 0.5, "eps": 0.1}, {**b, "mode": "emr"}]
        for s_key in SCALES:
            if st[s_key] >= 0.125:
                for s in (st[s_key], -st[s_key]):
                    q = json.loads(json.dumps(p)); q[s_key] = max(0.0, q.get(s_key, 1.0) + s)
                    out.append(q)
    else:
        out.append({"c": dict(p["c"]), "mode": "lin", "base": p.get("base", BASE)})
        if mode in ("ties", "crumbs") and sm >= 0.25:
            for f in (0.6, 1.5):
                out.append({**p, "density": min(0.9, round(p["density"] * f, 3))})
        if mode == "dare" and sm >= 0.25:
            for dr in (0.3, 0.7):
                out.append({**p, "drop": dr})
        if mode == "della" and sm >= 0.25:
            out += [{**p, "drop": 0.3}, {**p, "drop": 0.7}, {**p, "eps": 0.2}]
        if mode == "emr" and sm >= 0.25:
            out += [{**p, "density": 0.5}, {**p, "density": 0.3}]
        if sm >= 0.25:  # trimming / dropping shrinks the vectors: scale the merged part
            for f in (1.5, 0.67):
                out.append({**p, "lam": round(p.get("lam", 1.0) * f, 3)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--start", default="DC1=0.5,CK1D=0.25")
    ap.add_argument("--max-rounds", type=int, default=15)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = D / "search" / f"{args.name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    cur = {c: json.loads(json.dumps(STARTS[c])) for c in COMPS}
    steps = {c: {**{k: VECS[k][2] for k in VECS}, "mode": 1.0, "s_attn": 0.5, "s_exp": 0.25, "s_lo": 0.5, "s_mid": 0.5, "s_hi": 0.5,
                 "s_router": 0.5, "s_embed": 0.5, "s_norm": 0.5} for c in COMPS}
    pool = {}  # key -> {"point": p, "mu": {comp: mu}}
    starts = {key(q): q for q in cur.values()}
    for k, mu in score(list(starts.values())).items():
        pool[k] = {"point": starts[k], "mu": mu}
    mu_cur = {c: pool[key(cur[c])]["mu"].get(c, -1) for c in COMPS}
    for c in COMPS:
        log(f"start {c.upper()}: {key(cur[c])}: " + "  ".join(f"{x.upper()} {pool[key(cur[c])]['mu'].get(x, float('nan')):+.6f}" for x in COMPS))
    state = {"base": BASE, "limit": LIMIT, "comps": {c: {"king": COMPS[c][0], "bar": COMPS[c][1]} for c in COMPS}}
    for rnd in range(1, args.max_rounds + 1):
        active = [c for c in COMPS if any(steps[c][k] >= VECS[k][3] - 1e-9 for k in COMPS[c][3]) or steps[c]["mode"] >= 0.25]
        if not active:
            log("every competition converged")
            break
        cands = {}
        if rnd == 1:  # user 05:50 JST Oct 12: the new TEXT king x our latest checkpoints x the top models, every merge method
            ex = (
                [{"base": K2, "c": {"toC8": t}, "mode": "lin"} for t in (0.2, 0.3, 0.5)]
                + [{"base": K2, "c": {"toKC": t}, "mode": "lin"} for t in (0.3, 0.5)]
                + [{"base": K2, "c": {"toC8": 0.3, "toKC": 0.2}, "mode": "lin"}, {"base": K2, "c": {"toC8": 0.2, "toKC": 0.2, "toKMA": 0.1}, "mode": "lin"}]
                + [{"base": K2, "c": {"dTR": t}, "mode": "lin"} for t in (0.5, 1.0)] + [{"base": K2, "c": {"dES": 0.5}, "mode": "lin"}]
                + [{"base": "C8_S20", "c": {"dK2T": t}, "mode": "lin"} for t in (0.5, 1.0)] + [{"base": "C8_S20", "c": {"dK2C": 0.5}, "mode": "lin"}]
                + [{"base": KC, "c": {"dK2T": 1.0}, "mode": "lin"}, {"base": KC, "c": {"dK2C": 0.5}, "mode": "lin"}]
                + [{"mode": "slerp", "a": K2, "b": b, "t": t} for b, t in (("C8_S20", 0.3), ("C8_S20", 0.5), (KC, 0.3), (KC, 0.5), ("C5_S25", 0.3))]
                + [{"base": K2, "c": {"toC8": 0.3, "toKC": 0.3, "dTR": 0.5}, "mode": m, **kw} for m, kw in (
                    ("ties", {"density": 0.3}), ("ties", {"density": 0.5}), ("dare", {"drop": 0.5}), ("della", {"drop": 0.5, "eps": 0.1}), ("emr", {}),
                    ("crumbs", {"density": 0.5}))]
                + [{"mode": "stock", "fts": [K2, KC, "C8_S20"], "on": KT}, {"mode": "stock", "fts": [KC, "C8_S20"], "on": K2},
                   {"mode": "stock", "fts": [K2, KC, "C8_S20", KM], "on": KT}]
                + [{"base": K2, "c": {"toC8": 0.3}, "mode": "lin", "s_lo": 0.5}, {"base": K2, "c": {"toC8": 0.3}, "mode": "lin", "s_exp": 1.25}]
            )
            extra = ex
            for q in extra:
                if key(q) not in pool:
                    cands.setdefault(key(q), q)
        for c in ["main"] + [x for x in active if x != "main"]:  # MAIN first (the main target), then the rest
            if c not in active:
                continue
            for q in moves(cur[c], c, steps[c]):
                if key(q) not in pool and key(q) not in cands and len(cands) < MAX_CANDS:
                    cands[key(q)] = q
        log(f"round {rnd}: {len(cands)} candidates (active: {', '.join(active)})")
        for k, mu in score(list(cands.values())).items():
            pool[k] = {"point": cands[k], "mu": mu}
        for c in active:
            ranked = sorted(((v["mu"][c], k) for k, v in pool.items() if c in v["mu"]), reverse=True)
            best_mu, best_k = ranked[0]
            if best_mu > mu_cur[c] + EPS:
                cur[c], mu_cur[c] = pool[best_k]["point"], best_mu
                log(f"  {c.upper()} -> {best_mu:+.6f}  {best_k}")
            else:
                steps[c] = {k: v / 2 for k, v in steps[c].items()}
                log(f"  {c.upper()} stays {mu_cur[c]:+.6f} (steps halved)")
        top = sorted(pool.items(), key=lambda kv: -kv[1]["mu"].get("main", -1))[:3]
        for k, v in top:
            log("    top MAIN: " + "  ".join(f"{c} {v['mu'].get(c, float('nan')):+.6f}" for c in COMPS) + f"  {k}")
        state["best"] = {c: {"point": key(cur[c]), "mu": mu_cur[c], "recipe": str(recipe(cur[c])[1]), "all": pool[key(cur[c])]["mu"]} for c in COMPS}
        state["pool"] = {k: v["mu"] for k, v in pool.items()}
        state_path.write_text(json.dumps(state, indent=1))
    log("SEARCH DONE: " + "  ".join(f"{c.upper()} {mu_cur[c]:+.6f} ({key(cur[c])})" for c in COMPS))


if __name__ == "__main__":
    main()
