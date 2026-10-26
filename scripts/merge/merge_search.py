#!/usr/bin/env python3
"""Greedy coordinate search over task vectors on top of a king (user 21:25: "why don't you merge more aggressively").

  W = BASE + sum_i c_i (A_i - B_i)

Every round scores, for each vector i, the current point with c_i +- step_i (MAIN vs the MAIN king, 2 400 seqs of sample
A, both replica sets when no trainer runs), plus the sum of the previous round's two best moves. The best candidate
becomes the new point if it beats the current one by > EPS (paired: same king, same sequences); with no gain every step
halves; the search stops when all steps are below their minimum. Every scored point: /data/teutonic/search/<name>.json;
duel results in eval_cache/results (campaign "search").

Per competition (user 21:30: "duel with the new MAIN, MATH, TEXT kings, merge and merge to prepare the best model for each
competition"): --comp main (vs KMAIN, bar 0.003), math (vs KMATH, 0.002), text (vs KTEXT, 0.002); the three searches run
side by side and take turns on the GPUs (gpu_screen.lock).

  python scripts/merge/merge_search.py --comp main --name main1 [--start DC1=0.5,CK1D=0.25]
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
# competition: (king, bar, sample config prefix, vectors searched)
COMPS = {
    "main": ("KMAIN_c7ef5487", 0.003, "e8ef9ae1", ("S_T", "dMAIN", "dMATH", "DC1", "CK1D", "dKC2", "dM1T", "dM3")),
    "math": ("KMATH_ddd9e626", 0.002, "abfd40f3", ("dMATH", "DC1", "CK1D", "S_T", "dM1T", "dM3")),
    "text": ("KTEXT_9a39afee", 0.002, "0a2840c9", ("S_T", "DC1", "CK1D", "dMATH", "dKT2", "dM1T", "dM3")),
}
COMP = "main"
# name: (A, B, step, min step)
VECS = {
    "S_T": ("KTEXT_9a39afee", "KMAIN_c7ef5487", 1.0, 0.25),   # 5DAL's TEXT step (KTEXT = KMAIN + S_T)
    "dMAIN": ("KMAIN_c7ef5487", "K43", 0.1, 0.05),            # 5DAL's MAIN step over #43 (momentum)
    "dMATH": ("KMATH_ddd9e626", "K43", 0.25, 0.1),            # 5Ek5Ko's MATH king step
    "DC1": ("C1_FINAL", "C1_START", 0.25, 0.1),               # our 200-step TEXT training step
    "CK1D": ("CK1", "T1", 0.25, 0.1),                         # our first CODE king's step
    "dKC2": ("KC2", "K43", 0.25, 0.1),                        # 5DAL's CODE king (uid 239) step
    "dM1T": ("M1_S175", "START_M1", 0.5, 0.25),               # our 175-step MAIN training step
    "dKT2": ("KT2", "K43", 0.25, 0.1),                        # 5DAL's earlier TEXT king (uid 83) step
    "dM3": ("M3_S25", "M1_S175", 1.0, 0.5),                   # m3's 25 reweighted steps (code-/sft-reasoning x2; user 21:31)
}
LIMIT, EPS = 2400, 5e-5
OUT = REPO / "configs/merge/search"
LOCK = D / "gpu_screen.lock"


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def key(pt):
    return ",".join(f"{k}={pt[k]:g}" for k in VECS if abs(pt.get(k, 0)) > 1e-9) or "base"


def recipe(pt):
    label = "srch_" + hashlib.sha1(key(pt).encode()).hexdigest()[:10]
    terms = {M[BASE]: 1.0}
    for k, c in pt.items():
        if abs(c) > 1e-9:
            a, b = VECS[k][:2]
            terms[M[a]] = terms.get(M[a], 0.0) + c
            terms[M[b]] = terms.get(M[b], 0.0) - c
    path = OUT / f"{label}.json"
    path.write_text(json.dumps({"label": label, "description": f"{BASE} + {key(pt)}", "group": "search",
                                "terms": [{"model": m, "coef": round(c, 6)} for m, c in terms.items() if abs(c) > 1e-9]}, indent=1))
    return label, path


def king_label():
    return Path(M[COMPS[COMP][0]]).name


def results(labels):
    out = {}
    for f in sorted((D / "eval_cache/results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in data.get("results", []):
            lab, v = (r.get("challenger") or {}).get("label"), r.get("verdict") or {}
            if (lab in labels and (r.get("king") or {}).get("label") == king_label() and r.get("sample", "").startswith(COMPS[COMP][2])
                    and v.get("n_sequences") == LIMIT):
                out[lab] = v["mu_hat"]
    return out


def score(points):
    """mu (MAIN vs KMAIN) for every point; scores the missing ones in one campaign43 call under the GPU screen lock."""
    rec = {key(p): recipe(p) for p in points}
    have = results({lab for lab, _ in rec.values()})
    todo = [path for lab, path in rec.values() if lab not in have]
    if todo:
        while LOCK.exists() and time.time() - LOCK.stat().st_mtime < 4 * 3600:
            time.sleep(20)
        LOCK.write_text(f"merge_search {COMP}")
        try:
            training = bool(subprocess.run(["pgrep", "-f", "cpt_fsdp.py"], capture_output=True, text=True).stdout.strip())
            cmd = (f"cd {REPO} && source {D}/env.sh && PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py "
                   + " ".join(f"--recipe {p}" for p in todo)
                   + f" --king {COMP}={COMPS[COMP][0]}:{COMPS[COMP][1]} --comps {COMP} --campaign search_{COMP} --limit {LIMIT}"
                   " --sample tag:A --replicas 4"
                   f" --batch-size 32 --pingpong {0 if training else 1}")
            p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            if p.returncode != 0:
                log("campaign43 failed: " + (p.stdout + p.stderr)[-600:])
        finally:
            LOCK.unlink(missing_ok=True)
        have = results({lab for lab, _ in rec.values()})
    return {k: have.get(lab) for k, (lab, _) in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--comp", default="main", choices=list(COMPS))
    ap.add_argument("--start", default="DC1=0.5,CK1D=0.25")
    ap.add_argument("--max-rounds", type=int, default=12)
    args = ap.parse_args()
    global COMP, VECS
    COMP = args.comp
    VECS = {k: VECS[k] for k in COMPS[COMP][3]}
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = D / "search" / f"{args.name}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    cur = {k: 0.0 for k in VECS}
    for kv in filter(None, args.start.split(",")):
        k, v = kv.split("=")
        cur[k] = float(v)
    steps = {k: VECS[k][2] for k in VECS}
    state = {"base": BASE, "comp": COMP, "king": COMPS[COMP][0], "bar": COMPS[COMP][1], "limit": LIMIT, "points": {}, "path": []}
    mu_cur = score([cur])[key(cur)]
    log(f"{COMP.upper()} search vs {COMPS[COMP][0]} (bar {COMPS[COMP][1]}); start {key(cur)}: {mu_cur:+.6f}")
    state["path"].append({"point": key(cur), "mu": mu_cur})
    combo = None
    for rnd in range(1, args.max_rounds + 1):
        if all(steps[k] < VECS[k][3] - 1e-9 for k in VECS):
            log("all steps below their minimum: converged")
            break
        cands = {}
        for k in VECS:
            if steps[k] < VECS[k][3] - 1e-9:
                continue
            for s in (steps[k], -steps[k]):
                p = dict(cur); p[k] = round(p[k] + s, 4)
                cands[key(p)] = p
        if combo:
            cands[key(combo)] = combo
        log(f"round {rnd}: {len(cands)} candidates around {key(cur)} (steps {', '.join(f'{k} {v:g}' for k, v in steps.items())})")
        mus = score(list(cands.values()))
        for kk, mu in mus.items():
            state["points"][kk] = mu
        ranked = sorted(((mu, kk) for kk, mu in mus.items() if mu is not None), reverse=True)
        for mu, kk in ranked[:5]:
            log(f"  {mu:+.6f}  {kk}")
        if not ranked:
            log("no results; stopping")
            break
        best_mu, best_k = ranked[0]
        if best_mu > mu_cur + EPS:
            improving = [cands[kk] for mu, kk in ranked if mu > mu_cur + EPS]
            if len(improving) >= 2:  # next round also tries the two best moves together
                combo = dict(cur)
                for p in improving[:2]:
                    for k in VECS:
                        combo[k] = round(combo[k] + (p[k] - cur[k]), 4)
            else:
                combo = None
            cur, mu_cur = cands[best_k], best_mu
            log(f"round {rnd}: moved to {best_k} ({COMP.upper()} {best_mu:+.6f})")
        else:
            steps = {k: v / 2 for k, v in steps.items()}
            combo = None
            log(f"round {rnd}: no gain over {mu_cur:+.6f}; halving the steps")
        state["path"].append({"point": key(cur), "mu": mu_cur})
        state["best"] = {"point": key(cur), "mu": mu_cur, "recipe": str(recipe(cur)[1])}
        state_path.write_text(json.dumps(state, indent=1))
    log(f"SEARCH DONE: {COMP.upper()} best {key(cur)} {mu_cur:+.6f} (bar {COMPS[COMP][1]}) -> {recipe(cur)[1]}")


if __name__ == "__main__":
    main()
