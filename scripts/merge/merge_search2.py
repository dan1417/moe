#!/usr/bin/env python3
"""Joint merge search for all four competitions, linear and non-linear (user 21:40: "why only linear merges? and you don't
calculate CODE and TEXT scores").

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
BASE = "KTEXT_9a39afee"
VECS = {  # name: (A, B, step, min step)
    "S_T": ("KTEXT_9a39afee", "KMAIN_c7ef5487", 1.0, 0.25),   # 5DAL's TEXT step (KTEXT = KMAIN + S_T)
    "dMAIN": ("KMAIN_c7ef5487", "K43", 0.1, 0.05),            # 5DAL's MAIN step over #43
    "dMATH": ("KMATH_ddd9e626", "K43", 0.25, 0.1),            # 5Ek5Ko's MATH king step
    "DC1": ("C1_FINAL", "C1_START", 0.25, 0.1),               # our 200-step TEXT training step
    "CK1D": ("CK1", "T1", 0.25, 0.1),                         # our first CODE king's step
    "dKC2": ("KC2", "K43", 0.25, 0.1),                        # 5DAL's CODE king (uid 239) step
    "dKT2": ("KT2", "K43", 0.25, 0.1),                        # 5DAL's earlier TEXT king (uid 83) step
    "dM1T": ("M1_S175", "START_M1", 0.5, 0.25),               # our 175-step MAIN training step
    "dM3": ("M3_S25", "M1_S175", 1.0, 0.5),                   # m3's 25 reweighted steps
}
# competition: (king, bar, sample config prefix, vectors it moves)
COMPS = {
    "main": ("KMAIN_c7ef5487", 0.003, "e8ef9ae1", ("S_T", "dMAIN", "dMATH", "DC1", "CK1D", "dKC2", "dM1T", "dM3")),
    "math": ("KMATH_ddd9e626", 0.002, "abfd40f3", ("dMATH", "DC1", "CK1D", "S_T", "dM1T", "dM3")),
    "code": ("T007", 0.002, "7dcbbb5f", ("dKC2", "CK1D", "DC1", "dM1T", "dM3", "dMATH")),
    "text": ("KTEXT_9a39afee", 0.002, "0a2840c9", ("S_T", "DC1", "CK1D", "dMATH", "dKT2", "dM1T", "dM3")),
}
LIMIT, EPS, MAX_CANDS = 1200, 5e-5, 56
OUT = REPO / "configs/merge/search2"
LOCK = D / "gpu_screen.lock"
ATTN, EXP = r"self_attn\.", r"mlp\.experts\."
GROUPS = {"s_lo": r"layers\.([0-9]|1[0-4])\.", "s_mid": r"layers\.(1[5-9]|2[0-9])\.", "s_hi": r"layers\.(3[0-9]|4[0-4])\."}
SCALES = ("s_attn", "s_exp", "s_lo", "s_mid", "s_hi", "s_router", "s_embed", "s_norm")
# whole-model parts with their own scale (MoE routers, embeddings / head, norms); matched before the layer groups
PARTS = {"s_router": r"mlp\.gate\.", "s_embed": r"embed_tokens|lm_head", "s_norm": r"layernorm|^model\.norm\."}


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def norm(p):
    q = {"c": {k: round(v, 4) for k, v in p.get("c", {}).items() if abs(v) > 1e-9}, "mode": p.get("mode", "lin")}
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
        q["fts"], q["on"], q["scale"] = list(p["fts"]), p.get("on", "K43"), round(p.get("scale", 1.0), 4)
        q.pop("c", None)
        return q
    if q["mode"] != "lin":
        q["lam"] = round(p.get("lam", 1.0), 4)
    return q


def key(p):
    p = norm(p)
    if p["mode"] == "stock":
        return f"model stock (#43; {', '.join(p['fts'])}) on {p['on']} x{p['scale']:g}"
    s = ",".join(f"{k}={v:g}" for k, v in sorted(p["c"].items())) or "base"
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
    else:
        s += f" | dare-ties drop {p['drop']:g}"
    if p["mode"] != "lin" and p["lam"] != 1:
        s += f" lambda {p['lam']:g}"
    return s


def recipe(p):
    p = norm(p)
    label = "s2_" + hashlib.sha1(key(p).encode()).hexdigest()[:10]
    path = OUT / f"{label}.json"
    body = {"label": label, "description": f"{BASE} + {key(p)}", "group": "search2"}
    if p["mode"] == "stock":
        models = [M["K43"]] + [M[f] for f in p["fts"]]
        if p["on"] != "K43":
            models.append(M[p["on"]])
        body.update({"description": key(p), "terms": [{"model": m, "coef": 1.0} for m in models], "mode": "model_stock",
                     "mode_args": {"base": 0, "models": list(range(1, 1 + len(p["fts"]))), "on": len(models) - 1 if p["on"] != "K43" else 0,
                                   "scale": p["scale"]}})
        path.write_text(json.dumps(body, indent=1))
        return label, path
    if p["mode"] == "lin":
        models = [M[BASE]]
        for k in p["c"]:
            for n in VECS[k][:2]:
                if M[n] not in models:
                    models.append(M[n])

        def coefs(scale):
            c = {m: 0.0 for m in models}
            c[M[BASE]] = 1.0
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
        if over:
            body["overrides"] = [{"pattern": pat, "coefs": coefs(sc)} for pat, sc in over]
    else:
        models = [M[BASE]]
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
        body.update({"terms": [{"model": m, "coef": 1.0} for m in models], "mode": "vec_ties", "mode_args": args})
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
        LOCK.write_text("merge_search2")
        try:
            training = bool(subprocess.run(["pgrep", "-f", "python.*cpt_fsdp"], capture_output=True, text=True).stdout.strip())
            kings = " ".join(f"--king {c}={k}:{bar}" for c, (k, bar, _, _) in COMPS.items())
            cmd = (f"cd {REPO} && source {D}/env.sh && PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py "
                   + " ".join(f"--recipe {p}" for p in todo)
                   + f" {kings} --comps {','.join(COMPS)} --campaign search2 --limit {LIMIT} --sample tag:A --replicas 4"
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
    if mode == "stock":
        return [{**p, "scale": round(p["scale"] * f, 3)} for f in (1.5, 0.67)] if sm >= 0.25 else []
    if mode == "lin":
        if sm >= 0.25:
            out += [{"c": dict(p["c"]), "mode": "ties", "density": 0.5}, {"c": dict(p["c"]), "mode": "ties", "density": 0.3},
                    {"c": dict(p["c"]), "mode": "dare", "drop": 0.5}, {"c": dict(p["c"]), "mode": "crumbs", "density": 0.5},
                    {"c": dict(p["c"]), "mode": "della", "drop": 0.5, "eps": 0.1}, {"c": dict(p["c"]), "mode": "emr"}]
        for s_key in SCALES:
            if st[s_key] >= 0.125:
                for s in (st[s_key], -st[s_key]):
                    q = json.loads(json.dumps(p)); q[s_key] = max(0.0, q.get(s_key, 1.0) + s)
                    out.append(q)
    else:
        out.append({"c": dict(p["c"]), "mode": "lin"})
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
    start = {"c": {}, "mode": "lin"}
    for kv in filter(None, args.start.split(",")):
        k, v = kv.split("=")
        start["c"][k] = float(v)
    cur = {c: json.loads(json.dumps(start)) for c in COMPS}
    steps = {c: {**{k: VECS[k][2] for k in VECS}, "mode": 1.0, "s_attn": 0.5, "s_exp": 0.25, "s_lo": 0.5, "s_mid": 0.5, "s_hi": 0.5,
                 "s_router": 0.5, "s_embed": 0.5, "s_norm": 0.5} for c in COMPS}
    pool = {}  # key -> {"point": p, "mu": {comp: mu}}
    for k, mu in score([start]).items():
        pool[k] = {"point": start, "mu": mu}
    mu_cur = {c: pool[key(start)]["mu"].get(c, -1) for c in COMPS}
    log("start " + key(start) + ": " + "  ".join(f"{c.upper()} {mu_cur[c]:+.6f}" for c in COMPS))
    state = {"base": BASE, "limit": LIMIT, "comps": {c: {"king": COMPS[c][0], "bar": COMPS[c][1]} for c in COMPS}}
    for rnd in range(1, args.max_rounds + 1):
        active = [c for c in COMPS if any(steps[c][k] >= VECS[k][3] - 1e-9 for k in COMPS[c][3]) or steps[c]["mode"] >= 0.25]
        if not active:
            log("every competition converged")
            break
        cands = {}
        for c in ["main"] + [x for x in active if x != "main"]:  # MAIN first (the main target), then the rest
            if c not in active:
                continue
            for q in moves(cur[c], c, steps[c]):
                if key(q) not in pool and key(q) not in cands and len(cands) < MAX_CANDS:
                    cands[key(q)] = q
        if rnd == 1:  # user 22:10: "why don't you try ties or dare-ties?" -- over all the vectors at once
            allv = {"dMATH": 0.5, "DC1": 0.5, "CK1D": 0.25, "dKC2": 0.25, "dKT2": 0.25, "dM1T": 0.5, "dM3": 1.0}
            for q in ([{"c": dict(allv), "mode": "ties", "density": d} for d in (0.3, 0.5, 0.7)]
                      + [{"c": dict(allv), "mode": "dare", "drop": d} for d in (0.5, 0.7)]
                      + [{"c": dict(allv), "mode": "ties", "density": 0.5, "lam": 1.5}, {"c": dict(allv), "mode": "lin"},
                         {"c": dict(allv), "mode": "della", "drop": 0.5, "eps": 0.1}, {"c": dict(allv), "mode": "della", "drop": 0.7, "eps": 0.1},
                         {"c": dict(allv), "mode": "emr"}, {"c": dict(allv), "mode": "emr", "density": 0.5},
                         # Model Stock over the kings that fine-tune #43 (user 22:25: "no more possible merge method?")
                         {"mode": "stock", "fts": ["KMAIN_c7ef5487", "KMATH_ddd9e626", "KC2", "KT2"], "on": "K43"},
                         {"mode": "stock", "fts": ["KMATH_ddd9e626", "KC2", "KT2"], "on": "KTEXT_9a39afee"},
                         {"mode": "stock", "fts": ["KMATH_ddd9e626", "KC2", "KT2"], "on": "KTEXT_9a39afee", "scale": 2.0},
                         {"mode": "stock", "fts": ["KMATH_ddd9e626", "KC2", "KT2", "M1_S175"], "on": "KTEXT_9a39afee"}]):
                cands.setdefault(key(q), q)
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
