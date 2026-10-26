#!/usr/bin/env python3
"""Merge mode: the best START for training against the current king of one competition (and a winner, if a merge
already clears the bar).

    python scripts/merge/merge4train.py --comp math --ours <ckpt>::<its segment start>::<label> [--ours ...] --out r.json
    python scripts/merge/merge4train.py --comp code --dry        # list the candidates, score nothing

Why not just the best public merge: a merge is scored once on 1 200 sequences and the maximum over hundreds of noisy
candidates is inflated by 2-3 standard errors; and the point training starts from should keep OUR training signal (the
crowns we took came from soups of a fresh king with our own trained checkpoint: t005 = 0.5 #43 + 0.5 c1, t007 = a
MAIN-run checkpoint). Candidates (all affine, coefficients summing to 1; K = the reigning king of --comp):
  ours_as_is        O                          (keep training our checkpoint; the autopilot then resumes it exactly)
  soup a            (1-a) K + a O              a in 0.3 / 0.5 / 0.7
  transplant b      K + b (O - S)              our pure training delta (O minus its segment start S) on the new king
  public_best       search/public.json's best for --comp, when it was scored against the same king
  king              K itself (mu 0 by definition; the fallback when nothing beats it)
Round 1 scores every candidate on --comp (and MAIN) on sample A (--limit); round 2 refines the best family's
coefficient; the top --shortlist then get an independent sample B screen, and the choice is the best on B (a
disjoint sample: no winner's curse). Choice >= bar + margin with LCB > bar on B -> full B and C confirmation -> staged
for submission (scripts/train/autopilot.py stage_checkpoint; nothing is submitted). The chosen merge is written to
/data/teutonic/ckpt/m4t_<comp>_<hash> (GPU write) and recorded in /data/teutonic/search/train_inits.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/box"))
sys.path.insert(0, str(REPO / "scripts/train"))
import kings as K  # noqa: E402

D = Path("/data/teutonic")
PY = str(REPO / ".venv-eval/bin/python")
RECIPES = REPO / "configs/merge/m4t"
SEARCH = D / "search"
LOGS = D / "logs"


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}", flush=True)


def canon(terms: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Sum the coefficients of identical models, drop zeros (K + b(O - S) with S = K becomes a soup)."""
    acc: dict[str, float] = {}
    for path, c in terms:
        key = str(Path(path).resolve()) if Path(path).is_symlink() else str(path)
        acc[key] = acc.get(key, 0.0) + c
    return [(p, round(c, 6)) for p, c in acc.items() if abs(c) > 1e-9]


def label_of(terms: list[tuple[str, float]]) -> str:
    """Content-addressed: the label is a hash of the exact model paths and coefficients (two different merges can never
    share a label, and a cached score always belongs to the model it was measured on)."""
    body = json.dumps(sorted([p, c] for p, c in terms))
    return "m4t_" + hashlib.sha1(body.encode()).hexdigest()[:10]


class Cand:
    def __init__(self, kind: str, desc: str, terms: list[tuple[str, float]], **meta):
        self.kind, self.desc, self.terms = kind, desc, canon(terms)
        self.label = label_of(self.terms)
        self.meta = meta
        self.scores: dict[str, dict] = {}  # "comp/sample" -> verdict summary

    @property
    def raw(self) -> str | None:
        """The model dir when the candidate is a single model as is."""
        return self.terms[0][0] if len(self.terms) == 1 and abs(self.terms[0][1] - 1) < 1e-9 else None

    def spec(self, comp: str) -> str:
        if self.raw:
            return self.raw
        path = RECIPES / comp / f"{self.label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"label": self.label, "description": self.desc, "group": "m4t",
                                        "terms": [{"model": p, "coef": c} for p, c in self.terms]}, indent=1))
        return str(path)

    def row(self) -> dict:
        return {"label": self.label, "kind": self.kind, "desc": self.desc, "terms": self.terms, "scores": self.scores, **self.meta}


def delta_cosine(O: str, S: str, X: str, P: str, tensors=("model.layers.30.self_attn.o_proj.weight",
                 "model.layers.20.mlp.experts.7.down_proj.weight", "model.layers.36.mlp.experts.100.gate_proj.weight",
                 "model.layers.30.mlp.shared_experts.down_proj.weight", "lm_head.weight")) -> float | None:
    """cos(O - S, X - P) over a few tensors: how much our delta (checkpoint O from start S) replays the king's own step
    (X from its parent P). High = merging it onto X extrapolates their step (measured negative, Sept 2026); the
    orthogonal part is what we add (OrthoCand)."""
    import torch
    from safetensors import safe_open

    def tensor(path, name):
        idx = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(Path(path) / idx[name]), "pt") as f:
            return f.get_tensor(name).float()
    try:
        num = den1 = den2 = 0.0
        for t in tensors:
            d = tensor(O, t) - tensor(S, t)
            sx = tensor(X, t) - tensor(P, t)
            num += float((d * sx).sum(dtype=torch.float64))
            den1 += float((d * d).sum(dtype=torch.float64))
            den2 += float((sx * sx).sum(dtype=torch.float64))
        return num / ((den1 * den2) ** 0.5) if den1 > 0 and den2 > 0 else None
    except Exception as exc:  # noqa: BLE001
        log(f"delta_cosine failed: {exc}")
        return None


class OrthoCand(Cand):
    """X + b ((O - S) orthogonal to X - P): our delta with its projection on the king's own step removed
    (src/parity/weights.py mode delta_orthogonal_pair, terms [king, parent, checkpoint, start])."""

    def __init__(self, X: dict, parent_path: str, o: dict, b: float):
        self.kind, self.desc = "transplant_orth", f"{X['name']} + {b:.2f} (ours {o['label']} - its start) orthogonal to {X['name']}'s own step"
        self.spec_dict = {"mode": "delta_orthogonal_pair", "mode_args": {"beta": b},
                          "terms": [{"model": X["path"], "coef": 1.0}, {"model": parent_path, "coef": 0.0},
                                    {"model": o["path"], "coef": b}, {"model": o["start"], "coef": -b}]}
        self.meta = {"ours": o["label"], "b": b, "O": o["path"], "start": o["start"], "base": X["path"], "base_name": X["name"],
                     "parent": parent_path, "orthogonal": True}
        self.terms = [(X["path"], 1.0), (o["path"], b), (o["start"], -b)]
        self.label = "m4t_" + hashlib.sha1(json.dumps({"orth": self.spec_dict}, sort_keys=True).encode()).hexdigest()[:10]
        self.scores = {}

    @property
    def raw(self):
        return None

    def spec(self, comp: str) -> str:
        path = RECIPES / comp / f"{self.label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(json.dumps({"label": self.label, "description": self.desc, "group": "m4t", **self.spec_dict}, indent=1))
        return str(path)


def default_ours(comp: str) -> list[str]:
    """Our training results from every competition's autopilot state (scripts/train/autopilot.py): the newest checkpoint
    of each current segment and the best screened checkpoint for `comp`, each with its segment start. A lineage trained
    for TEXT is a candidate ingredient for MATH / CODE / MAIN too (t007, a MAIN-run checkpoint, took CODE)."""
    out = []
    for st_path in sorted((D / "autopilot").glob("*/state.json")):
        st = json.loads(st_path.read_text())
        segs = {g["run"]: g for g in st.get("segments", [])}
        if st.get("segments"):
            g = st["segments"][-1]
            rd = D / "ckpt" / g["run"]
            cks = sorted((p for p in rd.glob("step-*") if (p / "SAVED").exists()), key=lambda p: int(p.name.split("-")[1]))
            cks += [rd / "final"] if (rd / "final/SAVED").exists() else []
            if cks:
                out.append(f"{cks[-1]}::{g['start']}::{g['run']}/{cks[-1].name}")
        b = (st.get("best") or {}).get(comp)
        if b and b["run"] in segs and Path(b["path"], "SAVED").exists():
            out.append(f"{b['path']}::{segs[b['run']]['start']}::{b['run']}/{b['ckpt']}")
    return list(dict.fromkeys(out))


def previous_king(comp: str, king: dict) -> tuple[str, str] | None:
    rows = json.loads((REPO / "configs/merge/public/public_models.json").read_text())
    older = sorted((r for r in rows if r.get("comp") == comp and (r.get("reign") or 0) < (king.get("reign") or 0)),
                   key=lambda r: r.get("reign") or 0, reverse=True)
    for r in older:
        try:
            return K.resolve(r["digest"])
        except K.KingNotReady:
            continue
    return None


def material(c: "Cand") -> bool:
    """A candidate we may stage for submission: it must carry our own training (a soup with >= 0.2 of our checkpoint, a
    transplant of >= 0.5 of our training delta, our checkpoint as is) or be a merge of several public models (the
    public campaign's best). A king as is, or the newest MAIN king as is, is a training START only: never a submission
    (the subnet rejects copies, and a renamed copy of someone's king is not our work)."""
    if c.kind == "ours_as_is":
        return True
    if c.kind == "soup":
        return float(c.meta.get("a", 0)) >= 0.2
    if c.kind in ("transplant", "transplant_orth"):
        return float(c.meta.get("b", 0)) >= 0.5
    return c.kind == "public_best"  # soup_other (two public kings, nothing of ours) is a training start only


def candidates(comp: str, king: dict, ours: list[dict], main_king: dict | None = None) -> list[Cand]:
    """vs the competition's king K (and, for a specialist, also on the newest MAIN king X when it is a different model):
    the current MATH / CODE / TEXT kings are the MAIN king re-submitted with a few Adam steps of noise (TX6 / MA2 / CD5
    vs M45: +-0.0002 on every mix, Oct 14), and a new MAIN king beats the old specialist kings by 0.0025-0.0045 on their
    mixes. So X + our delta is the natural specialist candidate as soon as X is ingested."""
    kp = king["path"]
    out = [Cand("king", f"{king['name']} as is (the reference)", [(kp, 1.0)])]
    bases = [(king["name"], kp, "")]
    if main_king and comp != "main" and main_king["digest"] != king["digest"]:
        mp = main_king["path"]
        out.append(Cand("main_king", f"{main_king['name']} as is (the newest MAIN king: a training start, never a submission)",
                        [(mp, 1.0)], path=mp))
        bases.append((main_king["name"], mp, "X"))
    for i, o in enumerate(ours):
        O, S, lab = o["path"], o["start"], o["label"]
        out.append(Cand("ours_as_is", f"ours {lab} as is", [(O, 1.0)], ours=lab, path=O, O=O))
        for bname, bp, tag in bases:
            alphas = (0.3, 0.5, 0.7) if (i == 0 and not tag) else (0.3, 0.5) if i == 0 else (0.5,)
            for a in alphas:
                out.append(Cand("soup", f"{1 - a:.2f} {bname} + {a:.2f} ours {lab}", [(bp, 1 - a), (O, a)], ours=lab, a=a,
                                base=bp, base_name=bname, O=O))
            if S and Path(S, "model.safetensors.index.json").exists() and str(Path(S).resolve()) != str(Path(O).resolve()):
                betas = (0.5, 1.0) if i == 0 else (1.0,)
                for b in betas:
                    out.append(Cand("transplant", f"{bname} + {b:.2f} (ours {lab} - its start)", [(bp, 1.0), (O, b), (S, -b)],
                                    ours=lab, b=b, start=S, base=bp, base_name=bname, O=O))
    # soups toward other miners' recent kings (5DALwjE4's M44 = M43 pulled 24 % toward 5Ek5Ko's M42): the newest
    # ingested king of each other coldkey, at a small dose; a training start only (nothing of ours in it)
    rows = json.loads((REPO / "configs/merge/public/public_models.json").read_text()) if (REPO / "configs/merge/public/public_models.json").exists() else []
    base_name, base_path = (main_king["name"], main_king["path"]) if main_king and comp != "main" and main_king["digest"] != king["digest"] else (king["name"], kp)
    owner_of = {r["name"]: str(r.get("owner") or r.get("coldkey") or "") for r in rows if r.get("name")}
    base_owner = owner_of.get(base_name, "")
    others: dict[str, tuple[str, str]] = {}
    for r in sorted(rows, key=lambda r: r.get("crowned_at") or "", reverse=True):
        own = str(r.get("owner") or r.get("coldkey") or "")
        if not own or own == base_owner or own in others:
            continue
        try:
            name, path = K.resolve(r["digest"])
        except K.KingNotReady:
            continue
        if path in (kp, base_path):
            continue
        others[own] = (name, path)
        if len(others) >= 2:
            break
    for own, (name, path) in others.items():
        a = 0.15
        out.append(Cand("soup_other", f"{1 - a:.2f} {base_name} + {a:.2f} {name} (another miner's king)", [(base_path, 1 - a), (path, a)],
                        other=name, a=a, base=base_path, base_name=base_name))
    pub = json.loads((SEARCH / "public.json").read_text()) if (SEARCH / "public.json").exists() else {}
    best = (pub.get("best") or {}).get(comp) or {}
    if best.get("recipe") and Path(best["recipe"]).exists() and best.get("king") == king["name"]:
        terms = [(t["model"], float(t["coef"])) for t in json.loads(Path(best["recipe"]).read_text()).get("terms", [])]
        if terms and all(Path(p, "model.safetensors.index.json").exists() for p, _ in terms):
            out.append(Cand("public_best", f"public campaign best {best.get('label')}: {best.get('key', '')[:80]}", terms))
    seen, uniq = set(), []
    for c in out:
        if c.label not in seen:
            seen.add(c.label)
            uniq.append(c)
    return uniq


def refine(best: Cand, king: dict, done: set[str]) -> list[Cand]:
    bp, bname = best.meta.get("base", king["path"]), best.meta.get("base_name", king["name"])
    out = []
    if best.kind == "soup":
        O = best.meta["O"]
        for a in (best.meta["a"] - 0.15, best.meta["a"] + 0.15):
            if 0.1 <= a <= 0.9:
                out.append(Cand("soup", f"{1 - a:.2f} {bname} + {a:.2f} ours {best.meta['ours']}", [(bp, 1 - a), (O, a)],
                                ours=best.meta["ours"], a=round(a, 3), base=bp, base_name=bname, O=O))
    elif best.kind == "transplant":
        O, S = best.meta["O"], best.meta["start"]
        for b in (best.meta["b"] - 0.25, best.meta["b"] + 0.25):
            if 0.25 <= b <= 1.5:
                out.append(Cand("transplant", f"{bname} + {b:.2f} (ours {best.meta['ours']} - its start)",
                                [(bp, 1.0), (O, b), (S, -b)], ours=best.meta["ours"], b=round(b, 3), start=S, base=bp,
                                base_name=bname, O=O))
    elif best.kind == "ours_as_is":
        for a in (0.85,):
            out.append(Cand("soup", f"{1 - a:.2f} {king['name']} + {a:.2f} ours {best.meta['ours']}", [(king["path"], 1 - a), (best.raw, a)],
                            ours=best.meta["ours"], a=a, base=king["path"], base_name=king["name"], O=best.raw))
    return [c for c in out if c.label not in done]


def score(cands: list[Cand], comps: list[str], kings: dict, sample: str, limit: int, tag: str, pingpong: int) -> None:
    """campaign43 over all candidates (each loaded once, scored on every competition); fills cand.scores."""
    todo = [c for c in cands if c.kind != "king"]
    if not todo:
        return
    specs = {c.spec(comps[0]): c for c in todo}
    cmd = [PY, "scripts/merge/campaign43.py", "--comps", ",".join(comps), "--limit", str(limit), "--sample", sample,
           "--campaign", tag, "--replicas", "4", "--batch-size", "32", "--pingpong", str(pingpong)]
    for c in comps:
        cmd += ["--king", f"{c}={kings[c]['name']}:{kings[c]['bar']}"]
    for s in specs:
        cmd += ["--recipe", s]
    log(f"scoring {len(todo)} candidates on {','.join(comps)} ({sample}, {limit} seqs)")
    env = {**os.environ, "TEUTONIC_EVAL_CACHE": str(D / "eval_cache"), "TEUTONIC_MODELS_DIRS": str(D / "models")}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=10 * 3600)
    (LOGS / "merge4train.log").open("a").write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n{r.stdout}\n{r.stderr[-8000:]}\n")
    files = re.findall(r"^saved (\S+_duel\.json)$", r.stdout, re.M)
    if r.returncode != 0 and not files:
        raise RuntimeError(f"campaign43 failed (rc {r.returncode}): {r.stderr[-400:]}")
    by_label = {c.label: c for c in todo}
    by_raw = {str(Path(c.raw)): c for c in todo if c.raw}
    samples = sample.split(",")
    for f in files:
        d = json.loads(Path(f).read_text())
        for i, res in enumerate(d["results"]):
            ch = res.get("challenger") or {}
            cand = by_label.get(ch.get("label"))
            if cand is None and len(ch.get("terms") or []) == 1:
                cand = by_raw.get(str(Path(ch["terms"][0]["model"])))
            if cand is None:
                continue
            comp, spec = comps[i // len(samples)], samples[i % len(samples)]  # campaign43 slots: for comp, for sample
            v = res["verdict"]
            cand.scores[f"{comp}/{spec}"] = {"mu": v["mu_hat"], "lcb": v["lcb"], "n": v.get("n"), "sample": res["sample"],
                                               "king": kings[comp]["name"]}
    missing = [c.label for c in todo if f"{comps[0]}/{samples[0]}" not in c.scores]
    if missing:
        log(f"WARNING: no {comps[0]} score for {missing} (see {LOGS}/merge4train.log)")


def mu(c: Cand, key: str) -> float:
    if c.kind == "king":
        return 0.0
    return (c.scores.get(key) or {}).get("mu", float("-inf"))


def write(c: Cand, comp: str) -> str:
    """The chosen start on disk (GPU write: identical to what was scored, also for stochastic modes)."""
    if c.raw:
        return c.raw
    out = D / "ckpt" / f"{c.label.replace("m4t_", f"m4t_{comp}_", 1)}"  # m4t_<comp>_<hash>
    if (out / "SAVED").exists():
        return str(out)
    log(f"writing {c.label} -> {out}")
    r = subprocess.run([PY, "scripts/merge/write_merge.py", "--recipe", c.spec(comp), "--out", str(out), "--force"], cwd=REPO,
                       capture_output=True, text=True, timeout=3 * 3600)
    if r.returncode != 0 or not (out / "model.safetensors.index.json").exists():
        raise RuntimeError(f"write_merge failed: {r.stderr[-400:]}")
    (out / "SAVED").write_text(json.dumps({"recipe": c.spec(comp), "label": c.label, "terms": c.terms, "t": time.time()}))
    return str(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp", required=True, choices=K.COMPS)
    ap.add_argument("--ours", action="append", default=[], help="<ckpt dir>::<its segment start>::<label> (start may be empty)")
    ap.add_argument("--limit", type=int, default=2400, help="round 1/2 screen size on sample A")
    ap.add_argument("--b-limit", type=int, default=4800, help="independent sample B screen of the shortlist")
    ap.add_argument("--shortlist", type=int, default=2)
    ap.add_argument("--margin", type=float, default=0.0002)
    ap.add_argument("--confirm-limit", type=int, default=40000)
    ap.add_argument("--no-write", action="store_true", help="do not write the chosen merge")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    comp = a.comp
    kings = K.reigning([comp])  # strict for the target; the other competitions are scored when their kings resolve
    for c in K.COMPS:
        if c == comp:
            continue
        try:
            kings.update(K.reigning([c]))
        except K.KingNotReady as exc:
            log(f"{c} king not ready ({exc}); not scored this round")
    comps = [comp] + [c for c in K.COMPS if c != comp and c in kings]
    king = kings[comp]
    ours = []
    from autopilot import strip_identity_files  # noqa: E402
    for spec in a.ours or default_ours(comp):
        path, start, label = (spec.split("::") + ["", ""])[:3]
        if Path(path, "model.safetensors.index.json").exists():
            if "/ckpt/" in str(Path(path).resolve()):
                strip_identity_files(Path(path))
            ours.append({"path": path, "start": start, "label": label or Path(path).name})
        else:
            log(f"skipping --ours {path}: not a complete checkpoint")
    cands = candidates(comp, king, ours, kings.get("main") if comp != "main" else None)
    # how much does each of our deltas replay the newest MAIN king's own step (X - its parent, the previous MAIN king)?
    X = kings.get("main") or king
    prev = previous_king("main", X)
    if prev and ours:
        for o in ours:
            if o["start"] and Path(o["start"], "model.safetensors.index.json").exists():
                c = delta_cosine(o["path"], o["start"], X["path"], prev[1])
                o["cos_king_step"] = c
                log(f"cos(ours {o['label']} - start, {X['name']} - {prev[0]}) = {'n/a' if c is None else round(c, 3)}")
                if c is not None and c > 0.25:
                    for b in (0.5, 1.0):
                        cands.append(OrthoCand(X, prev[1], o, b))
    log(f"{comp.upper()} vs {king['name']} ({king['repo']}, bar {king['bar']}): {len(cands)} candidates")
    for c in cands:
        log(f"  {c.label} {c.kind:12s} {c.desc}")
    if a.dry:
        return
    if subprocess.run(["pgrep", "-f", "[s]cripts/train/cpt_fsdp.py"], capture_output=True).stdout.strip():
        raise SystemExit("a trainer is on the GPUs: merge mode needs all 8 (the autopilot stops its trainer first)")
    tag = f"m4t_{comp}"
    keyA = f"{comp}/tag:A"
    score(cands, comps, kings, "tag:A", a.limit, tag, pingpong=1)
    ranked = sorted((c for c in cands if c.kind != "king"), key=lambda c: mu(c, keyA), reverse=True)
    if ranked:
        extra = refine(ranked[0], king, {c.label for c in cands})
        if extra:
            score(extra, comps, kings, "tag:A", a.limit, tag, pingpong=1)
            cands += extra
    ranked = sorted((c for c in cands if c.kind != "king" and mu(c, keyA) > float("-inf")), key=lambda c: mu(c, keyA), reverse=True)
    for c in ranked:
        log(f"  A {mu(c, keyA):+.6f}  {c.label} {c.desc}")
    short = ranked[:a.shortlist]
    keyB = f"{comp}/tag:B"
    if short:
        score(short, [comp], kings, "tag:B", a.b_limit, tag + "_B", pingpong=1)
    # the choice: best on the independent sample B; ties (5e-5) go to the candidate without extrapolation
    finals = [c for c in short if mu(c, keyB) > float("-inf")]
    king_c = next(c for c in cands if c.kind == "king")
    pick = king_c
    if finals:
        top = max(mu(c, keyB) for c in finals)
        near = [c for c in finals if mu(c, keyB) >= top - 5e-5]
        # extrapolation = total weight of negative coefficients (soups and "as is": 0; K + b(O - S): b)
        pick = min(near, key=lambda c: (not material(c), sum(-x for _, x in c.terms if x < 0), -mu(c, keyB)))
        if mu(pick, keyB) <= 0:
            log(f"nothing beats {king['name']} on sample B (best {top:+.6f}): the start is the king itself")
            pick = king_c
    result = {"comp": comp, "king": {k: king[k] for k in ("name", "digest", "repo", "bar")}, "t": time.time(),
              "candidates": [c.row() for c in cands], "ready": []}
    log(f"choice: {pick.label} ({pick.kind}) A {mu(pick, keyA):+.6f} B {mu(pick, keyB):+.6f} - {pick.desc}")
    # winners: for EVERY scored competition, the best material candidate on A that clears its bar + margin gets an
    # independent B screen (the shortlist's B screen is reused for `comp`), then full B and C, then staging (not submitted)
    from autopilot import stage_checkpoint  # noqa: E402
    for c in comps:
        kc, kA, kB = kings[c], f"{c}/tag:A", f"{c}/tag:B"
        cands_c = sorted((x for x in cands if material(x) and mu(x, kA) >= kc["bar"] + a.margin and (x.scores.get(kA) or {}).get("lcb", -1) > kc["bar"]),
                         key=lambda x: mu(x, kA), reverse=True)[:2]
        for x in cands_c:
            if kB not in x.scores:
                score([x], [c], kings, "tag:B", a.b_limit, tag + f"_{c}_B", pingpong=1)
            vb = x.scores.get(kB) or {}
            if not (vb and vb["mu"] >= kc["bar"] + a.margin and vb["lcb"] > kc["bar"]):
                log(f"{c}: {x.label} A {mu(x, kA):+.6f} but B {vb.get('mu', float('nan')):+.6f} / lcb {vb.get('lcb', float('nan')):+.6f}: no")
                continue
            x.scores.pop(kB, None)  # the 4 800-seq B screen is replaced by the full B score
            score([x], [c], kings, "tag:B,tag:C", a.confirm_limit, tag + f"_{c}_confirm", pingpong=1)
            conf = {sp: x.scores.get(f"{c}/{sp}") for sp in ("tag:B", "tag:C")}
            log(f"{c}: full confirmation of {x.label}: {conf}")
            if all(v is not None and v["lcb"] > kc["bar"] for v in conf.values()):
                path = write(x, comp)
                rec = stage_checkpoint(Path(path), c, {"comp": c, "king": kc["name"], "king_digest": kc["digest"],
                                                        "bar": kc["bar"], "full": conf, "merge": x.row()},
                                       label=f"{c}_{x.label}")
                result["ready"].append(rec)
                break  # one staged winner per competition per round
    path = None
    if pick.kind in ("king", "ours_as_is", "main_king"):
        path = pick.raw
    elif not a.no_write:
        path = write(pick, comp)
    result["choice"] = {"label": pick.label, "kind": pick.kind, "desc": pick.desc, "mu": mu(pick, keyB) if pick.kind != "king" else 0.0,
                        "mu_A": mu(pick, keyA) if pick.kind != "king" else 0.0, "path": path, "terms": pick.terms}
    inits = json.loads((SEARCH / "train_inits.json").read_text()) if (SEARCH / "train_inits.json").exists() else {}
    inits.setdefault("train_init", {})[comp] = {"king": king["name"], "label": pick.label, "kind": pick.kind, "path": path,
                                                "mu": result["choice"]["mu"], "by": "merge4train", "t": time.time()}
    (SEARCH / "train_inits.json").write_text(json.dumps(inits, indent=1))
    out = a.out or (SEARCH / f"merge4train_{comp}.json")
    out.write_text(json.dumps(result, indent=1, default=str))
    log(f"result -> {out}")


if __name__ == "__main__":
    main()
