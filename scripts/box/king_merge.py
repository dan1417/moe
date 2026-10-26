#!/usr/bin/env python3
"""New king -> merge (user 15:40: "when a new MATH model appears, merge it with my best model; two MAIN-targeted models
are queued, merge again so we have the best checkpoint ready").

Every 2 min: crown changes on MAIN / MATH / CODE / TEXT. For each new king:
  1. ingest it (R2 publishes the manifest before the shards: retried) -> models.json name K<COMP>_<digest8>
  2. recipes_kt2.py --king <name>: parent detection, the new king + our lineage / training step, soups and SLERPs with our
     final c1 model and with our trained checkpoints (m1/step-175 = best MAIN / CODE, m2/step-50 = best MATH)
  3. pause the MAIN trainer (PAUSE ownership), screen every recipe on MAIN and MATH vs the then-current kings (4 800 seqs
     of sample A; one replica set fits next to the paused trainer), resume.
Results land in eval_cache/results (campaign "newking") and on the dashboard's merge-campaign panel.
"""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

REPO = Path("/root/workspace/MoE")
D = Path("/data/teutonic")
LOGS = D / "logs"
MODELS = REPO / "configs/merge/k41/models.json"
URLS = {"main": "https://teutonic.ai/dashboard.json", **{c: f"https://teutonic.ai/competitions/{c}/dashboard.json" for c in ("math", "code", "text")}}
R2 = "https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256/{d}/manifest.json"
BARS = {"main": 0.003, "math": 0.002}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def sh(cmd: str, timeout: float | None = None) -> tuple[int, str]:
    p = subprocess.run(["bash", "-c", f"cd {REPO} && source {D}/env.sh && {cmd}"], text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def kings() -> dict:
    out = {}
    for comp, url in URLS.items():
        try:
            d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "teutonic-miner/1.0"}), timeout=60))
            out[comp] = ((d.get("king") or {}).get("king_digest") or "", (d.get("king") or {}).get("model_repo") or "")
        except Exception:
            pass
    return out


def name_for(digest: str) -> str | None:
    """models.json name of a model by digest: un-permuted kings live in <NAME>_b41 dirs that only carry the 8-char prefix
    (16:29: matching the full digest missed KMATH_ddd9e626_b41 and re-downloaded it)."""
    m = json.loads(MODELS.read_text())
    return next((k for k, v in m.items() if digest[:8] and digest[:8] in v), None)


def ingest(comp: str, digest: str) -> str | None:
    have = name_for(digest)
    if have:
        return have
    name = f"K{comp.upper()}_{digest[:8]}"
    for _ in range(90):  # up to ~30 min for the manifest to be published after the crown
        try:
            if urllib.request.urlopen(urllib.request.Request(R2.format(d=digest), headers={"User-Agent": "teutonic-miner/1.0"}), timeout=30).status == 200:
                break
        except Exception:
            pass
        time.sleep(20)
    for attempt in range(10):
        rc, out = sh(f"DL_WORKERS=64 nice -n 5 bash scripts/merge/ingest_king41.sh {digest} {name}")
        if rc == 0 and name_for(digest):
            log(f"ingested {comp} king {digest[:12]} as {name}")
            return name
        log(f"ingest attempt {attempt + 1} failed: {out.strip().splitlines()[-1][:160] if out.strip() else rc}")
        time.sleep(30)
    return None


def current_king_name(comp: str, ks: dict) -> str:
    digest = ks.get(comp, ("", ""))[0]
    if comp == "math" and not digest:  # vacant MATH crown: challengers face the MAIN king
        digest = ks.get("main", ("", ""))[0]
    return name_for(digest[:8]) or "K43"


def screen(recipes: list[str], ks: dict, comps: str = "main,math") -> None:
    runs = list((D / "ckpt").glob("main_upper_ema_m*")) + list((D / "ckpt").glob("text_upper_ema_c*"))
    run = Path(sorted(runs, key=lambda p: p.stat().st_mtime)[-1])
    training = bool(subprocess.run(["pgrep", "-f", f"cpt_fsdp.py.*--out {run}"], capture_output=True, text=True).stdout.strip())
    own = False
    lock = D / "gpu_screen.lock"  # one screen at a time on the GPUs (manual rounds take the same lock; 21:25)
    while lock.exists() and time.time() - lock.stat().st_mtime < 4 * 3600:
        time.sleep(30)
    lock.write_text("king_merge")
    try:
        if training:  # one replica set next to the paused trainer; with free GPUs the two-set ping-pong pool
            while (run / "PAUSE").exists():
                time.sleep(10)
            (run / "PAUSE").touch(); own = True
            while not ((run / "PAUSED").exists() and (run / "PAUSED").stat().st_mtime >= (run / "PAUSE").stat().st_mtime):
                time.sleep(5)
        log(f"{'trainer ' + run.name + ' paused' if training else 'no trainer running'}; screening {len(recipes)} recipes on {comps.upper()}")
        kargs = " ".join(f"--king {c}={current_king_name(c, ks)}:{BARS[c]}" for c in comps.split(","))
        rargs = " ".join(f"--recipe {r}" for r in recipes)
        rc, out = sh(f"PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py {rargs} {kargs} --comps {comps} --campaign newking "
                     f"--limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong {0 if training else 1} 2>&1 | grep -E '^=== |  (MAIN|MATH) |Traceback|Error'")
        print(out, flush=True)
    finally:
        lock.unlink(missing_ok=True)
        if own:
            (run / "PAUSE").unlink(missing_ok=True)
            log("training resumed")


def main_king_label() -> str:
    """Label (= model dir name) of the reigning MAIN king in the duel results: only screens against it are comparable."""
    m = json.loads(MODELS.read_text())
    return Path(m.get(current_king_name("main", kings()), m["K43"])).name


def best_main(labels: list[str]) -> tuple[str | None, float | None]:
    """Best MAIN screen (mu vs the reigning MAIN king) among `labels`, from the saved campaign results."""
    best = (None, None)
    king = main_king_label()
    for f in sorted((D / "eval_cache/results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("campaign") != "newking":
            continue
        for r in data.get("results", []):
            lab = (r.get("challenger") or {}).get("label")
            if lab in labels and r.get("sample", "").startswith("e8ef9ae1") and (r.get("king") or {}).get("label") == king:
                mu = (r.get("verdict") or {}).get("mu_hat")
                if mu is not None and (best[1] is None or mu > best[1]):
                    best = (lab, mu)
    return best


def wave2_main(name: str, by_label: dict, label: str, mu: float) -> list[str]:
    """MAIN king: around the best MAIN recipe of wave 1 -- the new king's share +-0.1 (soups), the added task vectors x0.5 /
    x1.5 (new king + deltas), SLERP t +-0.1, TIES lambda 0.3 / 0.8."""
    m = json.loads(MODELS.read_text())
    king_path = m[name]
    body = json.loads(by_label[label].read_text())
    mode, args = body.get("mode", "linear"), dict(body.get("mode_args", {}))
    terms = [(t["model"], float(t["coef"])) for t in body["terms"]]
    out_dir, made = by_label[label].parent, []

    def write(tag, desc, new_terms, new_args=None):
        lab = f"w2_{label}_{tag}"
        if mode == "linear":
            acc = {}
            for k_, c_ in new_terms:
                acc[k_] = acc.get(k_, 0.0) + c_
            new_terms = [(k_, c_) for k_, c_ in acc.items() if abs(c_) > 1e-12]
        (out_dir / f"{lab}.json").write_text(json.dumps({"label": lab, "description": desc, "group": "wave2",
            "terms": [{"model": k, "coef": round(c, 6)} for k, c in new_terms], **({"mode": mode, "mode_args": new_args} if mode != "linear" else {})},
            indent=1))
        made.append(str(out_dir / f"{lab}.json"))

    if mode == "slerp":
        for d in (-0.1, 0.1):
            if 0.05 < args["t"] + d < 0.95:
                write(f"t{int(round((args['t'] + d) * 100)):03d}", f"{label} at t={args['t'] + d:.2f}", terms, {**args, "t": round(args["t"] + d, 3)})
    elif mode == "vec_ties":
        for lam in (0.3, 0.8):
            write(f"lam{int(lam * 100):03d}", f"{label} with lambda {lam}", terms, {**args, "lambda": lam})
    elif mode == "linear":
        kc = sum(c for mdl, c in terms if mdl == king_path)
        rest = [(mdl, c) for mdl, c in terms if mdl != king_path]
        if abs(kc - 1.0) < 1e-6 and abs(sum(c for _, c in terms) - 1.0) < 1e-6 and rest:  # new king + task vectors
            for f in (0.5, 1.5, 2.0):
                write(f"x{int(f * 100):03d}", f"{label} with the added task vectors x{f}", [(king_path, 1.0)] + [(mdl, c * f) for mdl, c in rest])
            donors, bases = [(x, c) for x, c in rest if c > 0], [(x, c) for x, c in rest if c < 0]
            if len(donors) == 1 and len(bases) == 1:  # one task vector X - B: how it is added (17:50: KMATH - #43 on KMAIN +0.00041)
                (dx, dc), (bx, _) = donors[0], bases[0]
                beta = 2 * dc
                for tag, md, ma in (("orth", "orthogonal", {"beta": beta}), ("ties030", "ties", {"density": 0.3, "beta": beta}),
                                    ("sign", "sign_agree", {"beta": beta})):
                    lab = f"w2_{label}_{tag}"
                    (out_dir / f"{lab}.json").write_text(json.dumps({"label": lab, "description": f"{label}: {md} {ma} with x2 the dose",
                        "group": "wave2", "terms": [{"model": king_path, "coef": 1.0}, {"model": dx, "coef": 1.0}, {"model": bx, "coef": 1.0}],
                        "mode": md, "mode_args": ma}, indent=1))
                    made.append(str(out_dir / f"{lab}.json"))
                # the other public steps since #43 on top (5DAL's TEXT king KT2, the CODE king KC2)
                for other in ("KT2", "KC2"):
                    if other in m:
                        write(f"plus_{other.lower()}025", f"{label} + 0.25 x ({other} - #43)", terms + [(m[other], 0.25), (m["K43"], -0.25)])
        elif rest and abs(sum(c for _, c in terms) - 1.0) < 1e-6:  # soup: the new king's share +-0.1, the others rescaled
            other = sum(c for _, c in rest)
            for d in (0.1, -0.1):
                if 0.05 < kc + d < 0.95 and other - d > 0.05:
                    write(f"k{int(round((kc + d) * 100)):03d}", f"{label} with the new king at {kc + d:.2f}",
                          [(king_path, kc + d)] + [(mdl, c * (other - d) / other) for mdl, c in rest])
    log(f"wave 2 around {label} (MAIN {mu:+.6f}): {' '.join(Path(x).stem for x in made) or 'nothing to vary'}")
    return made


def wave2(name: str, recipes: list[str], comp: str = "math") -> list[str]:
    """Around the best MAIN recipe of wave 1: + 0.5 x our training step + 0.25 x CK1 delta (the additions that made our best
    #43 soup), + training step only, and the mix ratio +-0.1 for a two-model soup. A MAIN king: wave2_main."""
    by_label = {Path(r).stem: Path(r) for r in recipes}
    label, mu = best_main(list(by_label))
    if label is None:
        return []
    if comp == "main":
        return wave2_main(name, by_label, label, mu)
    m = json.loads(MODELS.read_text())
    path_of = {v: k for k, v in m.items()}
    body = json.loads(by_label[label].read_text())
    if body.get("mode", "linear") != "linear":
        log(f"wave 2: best MAIN recipe {label} ({mu:+.6f}) is {body['mode']}; using its linear twin terms is not defined -> skip")
        return []
    terms = [(t["model"], float(t["coef"])) for t in body["terms"]]
    out_dir = by_label[label].parent
    made = []

    def write(tag, desc, extra):
        merged = {}
        for model, c in terms + extra:
            merged[model] = merged.get(model, 0.0) + c
        lab = f"w2_{label}_{tag}"
        (out_dir / f"{lab}.json").write_text(json.dumps({"label": lab, "description": desc, "group": "wave2",
            "terms": [{"model": k, "coef": round(c, 6)} for k, c in merged.items() if abs(c) > 1e-12]}, indent=1))
        made.append(str(out_dir / f"{lab}.json"))

    dc1 = [(m["C1_FINAL"], 0.5), (m["C1_START"], -0.5)]
    ck1 = [(m["CK1"], 0.25), (m["T1"], -0.25)]
    write("dc1ck1", f"{label} + 0.5 x our training step + 0.25 x CK1 delta", dc1 + ck1)
    write("dc1", f"{label} + 0.5 x our training step", dc1)
    if len(terms) == 2 and abs(sum(c for _, c in terms) - 1.0) < 1e-6:
        (a_model, a), (b_model, b) = terms
        for d in (0.1, -0.1):
            if 0.05 < a + d < 0.95:
                terms_bak = list(terms)
                terms[:] = [(a_model, a + d), (b_model, b - d)]
                write(f"mix{int(round((a + d) * 100)):02d}", f"{path_of.get(a_model, a_model)} {a + d:.2f} / {path_of.get(b_model, b_model)} {b - d:.2f}", [])
                terms[:] = terms_bak
    log(f"wave 2 around {label} (MAIN {mu:+.6f}): {' '.join(Path(x).stem for x in made)}")
    return made


def training_best_main() -> float | None:
    """Best MAIN screen (vs the MAIN king) of the running training lineage's checkpoints (main_watch, campaign m1)."""
    plan = D / "ckpt/current_plan.json"
    cfg = json.loads(plan.read_text()) if plan.exists() else {"base": "main_upper_ema_m3", "start": str(D / "ckpt/main_upper_ema_m1/step-175")}
    base, start = cfg["base"], cfg.get("start", "")
    king = main_king_label()
    best = None
    for f in (D / "eval_cache/results").glob("*_duel.json"):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in data.get("results", []):
            terms = (r.get("challenger") or {}).get("terms") or []
            model = terms[0].get("model", "") if len(terms) == 1 else ""
            # the lineage's checkpoints, and its start (before the first screen the start is the reference)
            if (model and (f"/ckpt/{base}" in model or model.rstrip("/") == start.rstrip("/")) and r.get("sample", "").startswith("e8ef9ae1")
                    and (r.get("king") or {}).get("label") == king):
                mu = (r.get("verdict") or {}).get("mu_hat")
                if mu is not None and (best is None or mu > best):
                    best = mu
    return best


def maybe_switch(label: str, mu: float, recipe_path: str) -> None:
    """Train from the merge if it beats the running training on MAIN by more than 1e-4 (user 15:55)."""
    train_mu = training_best_main()
    if train_mu is None:
        log(f"no MAIN reference for the running training; not switching (best merge {label} {mu:+.6f})")
        return
    if mu <= train_mu + 1e-4:
        log(f"keep training: best merge {label} MAIN {mu:+.6f} vs training {train_mu:+.6f}")
        return
    out = D / f"ckpt/start_{label}"
    log(f"SWITCH: merge {label} MAIN {mu:+.6f} beats training {train_mu if train_mu is None else f'{train_mu:+.6f}'}; writing {out}")
    if not (out / "model.safetensors.index.json").exists():
        rc, txt = sh(f"nice -n 5 python scripts/merge/write_merge.py --recipe {recipe_path} --out {out} --gpus cpu --workers 32")
        if rc != 0:
            log(f"write_merge failed: {txt[-300:]}")
            return
    rc, txt = sh(f"bash {LOGS}/switch_training.sh {out} 'merge {label} MAIN {mu:+.6f} > training {train_mu}'")
    log(txt.strip()[-400:])


def main() -> None:
    seen = {c: d for c, (d, _) in kings().items()}
    import os
    for c in filter(None, os.environ.get("KM_FORCE", "").split(",")):  # treat these crowns as new (restart after a crown change)
        seen[c] = ""
    log(f"watching crowns: { {c: d[:8] for c, d in seen.items()} }")
    while True:
        ks = kings()
        for comp, (digest, repo) in ks.items():
            if not digest or digest == seen.get(comp):
                continue
            seen[comp] = digest
            log(f"NEW {comp.upper()} KING {repo} {digest[:12]}")
            if digest[:8] in ("82de662a", "d4e670ff"):  # our own crowns (t005 TEXT, t007 CODE)
                continue
            name = ingest(comp, digest)
            if not name:
                log(f"could not ingest {digest[:12]}; skipping its merges")
                continue
            gen = "recipes_main.py" if comp == "main" else "recipes_kt2.py"
            rc, out = sh(f".venv-eval/bin/python scripts/merge/{gen} --king {name}")
            # screen the most promising ~13 (each costs ~3.5 min of paused training): the king itself, our lineage / training
            # transplants, and the blends with our trained checkpoints; the #43 leaders are cached (instant)
            skip = ("_o42_050", "_dc1_", "half", "slerp_" + name.lower() + "_c1_050", "_c1_050_ck1", "_mom", "mom050_c1")
            recipes = [x for x in out.split() if x.endswith(".json") and not any(t in Path(x).stem for t in skip)]
            if rc != 0 or not recipes:
                log(f"recipe generation failed for {name}: {out[-300:]}")
                continue
            log(f"{len(recipes)} recipes around {name}: {' '.join(Path(r).stem for r in recipes)}")
            comps = "main" if comp == "main" else "main,math"
            screen(recipes, kings(), comps)
            w2 = wave2(name, recipes, comp)  # adaptive second wave around the best MAIN recipe (user 15:50: MAIN is the target)
            if w2:
                screen(w2, kings(), comps)
            allr = recipes + (w2 or [])
            label, mu = best_main([Path(r).stem for r in allr])
            if label is not None:
                log(f"BEST MAIN after {name}: {label} {mu:+.6f} (bar 0.003 vs the MAIN king)")
                maybe_switch(label, mu, next(r for r in allr if Path(r).stem == label))
        time.sleep(120)


if __name__ == "__main__":
    main()
