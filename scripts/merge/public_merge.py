#!/usr/bin/env python3
"""Merge campaign over EVERY public Teutonic model (user 2026-10-12 11:30 JST: "use all of the public models for model merging,
not only the current top models, previous kings too; every merge method; use the 8 B300 efficiently").

Ingredients: all 53 public kings (MAIN #0-#44, TEXT #1-#4, CODE #1-#4, MATH #1; configs/merge/public/public_models.json,
streamed in by ingest_public.py) and our TEXT-run checkpoints. Every candidate is scored on MAIN / MATH / CODE / TEXT vs that
competition's king (1 200 seqs of sample A each, one load; campaign43.py with both replica sets on the 8 GPUs).

  1 raw      every near-family model and checkpoint as is (the "every public model vs every king" table)
  2 probe    for every model X, the direction v_X = X - A (A = anchor = our best checkpoint): A + d v_X (toward X, soup /
             WiSE-FT) and A - d v_X (away from X: ExPO-style extrapolation). d = 0.3 for the near family; for the older
             kings d is scaled by distance (0.3 x |M43 - A| / |X - A| on a tensor subset, delta_dist.py), so each probe moves
             the same length. An old king whose probes gain nothing on any competition is deleted to make room for the next
             download; the rest stay as search directions.
  3 methods  per competition king: method_grid.py in focused mode — cheap 5-recipe screens per donor, full ~25-recipe
             grids only for probe winners and donors that beat the current best; #36–#40 gradient chain stays dense.
             refine() expands coefficients around whatever wins. Pass --focus 0 to brute-force the full grid.
  4 search   coordinate search from the best point over its best probe directions. No gain halves that competition's steps.
             The best point that beats its king is written to /data/teutonic/submit/train_<comp>_<label> for training.
  The loop runs until every public model has been probed and every competition has converged.

Bookkeeping that keeps cached numbers honest:
  * a recipe's label is pub_<hash of the body the scorer loads> (terms: model paths + coefficients, mode, mode args,
    overrides, transforms), so a point whose directions now mean something else (another anchor, another model path) is a
    new label, and every pool entry whose label no longer matches its point is re-scored before it is ranked again;
  * a stored result counts only on the sample campaign43 uses for that competition right now and vs its current king;
  * search/public.json records its anchor (and the anchor's path): restarting on another anchor is refused unless
    --reset-pool (the old state is kept next to it);
  * a king switch clears that competition's confirm / written state; a dethroned king whose successor is not ingested yet
    (public_kings_live.json "pending") gets no confirm runs and no train-init writes.

  python scripts/merge/public_merge.py [--anchor C10_S50] [--batch 64] [--focus 1] [--score-chunk 48] [--reset-pool]
"""
import argparse
import contextlib
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[2]))  # src.parity (sample ids), after the stdlib
import merge_search4 as ms  # noqa: E402  (recipe builder + moves for lin / ties / dare / della / emr / crumbs / stock / slerp)
import method_grid as mg  # noqa: E402  (per-king method search: module split, experts-only, TIES/DARE/DELLA, SLERP, orthogonal)

REPO, D = ms.REPO, ms.D
MODELS_JSON = REPO / "configs/merge/k41/models.json"
PUBLIC_JSON = REPO / "configs/merge/public/public_models.json"
PUBLIC = json.loads(PUBLIC_JSON.read_text())
STATE = D / "search" / "public.json"
LOCK = D / "gpu_screen.lock"
KM, KT, KMA, KC = "KMAIN_c7ef5487", "KTEXT_9a39afee", "KMATH_ddd9e626", "KCODE_0c681d78"
COMPS = {"main": (KM, 0.003), "math": (KMA, 0.002), "code": (KC, 0.002), "text": (KT, 0.002)}
OURS = {  # our TEXT-run checkpoints (descend from the ES merge on the CODE king)
    "C10_S50": "/data/teutonic/ckpt/text_upper_ema_c10/step-50", "C10C_S25": "/data/teutonic/ckpt/text_upper_ema_c10c/step-25",
    "C10C_S50": "/data/teutonic/ckpt/text_upper_ema_c10c/step-50", "C10C_F": "/data/teutonic/ckpt/text_upper_ema_c10c/final",
    "C9_S50": "/data/teutonic/ckpt/text_upper_ema_c9/step-50", "C11_S25": "/data/teutonic/ckpt/text_upper_ema_c11/step-25",
    "C11_S50": "/data/teutonic/ckpt/text_upper_ema_c11/step-50",
}
# near family: kept on disk for the whole campaign (the rest of the public kings are probed, then kept only if useful)
FAMILY = {"M45", "M44", "M43", "M42", "M41", "M40", "TX1", "TX2", "TX3", "TX4", "TX5", "TX6",
          "CD1", "CD2", "CD3", "CD4", "CD5", "MA1", "MA2"}
# king changes mid-campaign: {"text": "TX5"} switches that competition's king once the model is ingested; every scored
# recipe is then re-scored against it from the loss cache (no reload). download_kings.py lists a competition there only
# while its live king is ingested; public_kings_live.json "pending" names the live kings that are not ingested yet.
KINGS_FILE = D / "search" / "public_kings.json"
KINGS_LIVE = D / "search" / "public_kings_live.json"
# priority candidates (scored first in the next round, no restart): {"vecs": {name: [A, B]}, "points": [point, ...]}
INJECT = D / "search" / "public_inject.json"
LIMIT, EPS, GAIN_KEEP, TOPK, KEEP_OLD = 1200, 5e-5, 1e-4, 6, 2
CAMPAIGN = "public"
CAMPAIGN_LOG = D / "logs" / "pm2_campaign_public.log"
NAMES = {r["name"]: f"{r['comp'].upper()} #{r['reign']} {r['repo'] or r['digest'][:8]}" for r in PUBLIC}


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def kings_file():
    """{comp: name} from public_kings.json (only competitions; other keys and non-string values are ignored)."""
    return {c: n for c, n in read_json(KINGS_FILE, {}).items() if c in COMPS and isinstance(n, str)}


def pending_kings():
    """{comp: digest} of live kings that are not ingested yet (their competition still has its previous king here)."""
    return {c: d for c, d in (read_json(KINGS_LIVE, {}).get("pending") or {}).items() if c in COMPS}


def reload_public():
    """download.sh appends kings as they are crowned: pick them up every round; near family = crowned in the last 7 days."""
    global PUBLIC
    try:
        PUBLIC = json.loads(PUBLIC_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return
    NAMES.update({r["name"]: f"{r['comp'].upper()} #{r['reign']} {r['repo'] or r['digest'][:8]}" for r in PUBLIC})
    cutoff = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 7 * 86400))
    FAMILY.update(r["name"] for r in PUBLIC if (r.get("crowned_at") or "") >= cutoff)
    FAMILY.update(kings_file().values())


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def models():
    return json.loads(MODELS_JSON.read_text())


COLD_QUEUE = D / "search" / "cold_queue.jsonl"


def to_cold(name, path):
    """Move a model off /data to the 13 TB cold volume (cold_archiver.py does the copy + verify + removal in the
    background); user 2026-10-12: use the attached volume instead of deleting models."""
    with open(COLD_QUEUE, "a") as fh:
        fh.write(json.dumps({"name": name, "src": path}) + "\n")


def cold_paths():
    """/data paths queued for (or being moved to) the cold volume: gone for merging even while the copy runs."""
    out = set()
    try:
        lines = COLD_QUEUE.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            out.add(str(Path(json.loads(line)["src"])))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def available():
    """alias -> path of every ingredient fully on disk right now (public kings by alias, our checkpoints); models queued
    for the cold volume are left out."""
    m, out, cold = models(), {}, cold_paths()
    for r in PUBLIC:
        for path in ([m[r["name"]]] if r["name"] in m else []) + [p for p in m.values() if Path(p).name == r["digest"] or r["digest"][:8] in Path(p).name]:
            if Path(path, "INGESTED").exists() and str(Path(path)) not in cold:
                out[r["name"]] = path
                break
    for name, path in OURS.items():
        if Path(path, "SAVED").exists() and Path(path, "model.safetensors.index.json").exists():
            out[name] = path
    return out


def sync_models(avail):
    """Register aliases (and the kings' campaign names) in models.json so merge_search4's builder resolves them."""
    m = models()
    alias = {"M44": KM, "TX4": KT, "MA1": KMA, "CD4": KC}
    changed = False
    for name, path in avail.items():
        for n in (name, alias.get(name)):
            if n and m.get(n) != path:
                m[n], changed = path, True
    if changed:
        subprocess.run(["flock", str(D / "models_json.lock"), "python3", "-c",
                        "import json,sys; p=sys.argv[1]; m=json.load(open(p)); m.update(json.loads(sys.argv[2])); json.dump(m, open(p,'w'), indent=1)",
                        str(MODELS_JSON), json.dumps(m)], check=True)
    ms.M = models()


_SEED = {}


def _seed_material(tag):
    """src.parity.sample's seed material of a synthetic sample tag (imported once; the same constants if it fails)."""
    if not _SEED:
        try:
            from src.parity.sample import LOCAL_HOTKEY, synthetic_block_hash
            _SEED.update(hotkey=LOCAL_HOTKEY, block_hash=synthetic_block_hash)
        except Exception:  # noqa: BLE001 - src/parity mid-edit / import error
            _SEED.update(hotkey="local-parity-eval",
                         block_hash=lambda t: "0x" + hashlib.sha256(f"teutonic-parity-sample:{t}".encode()).hexdigest())
    return f"block_hash={_SEED['block_hash'](tag)}|hotkey={_SEED['hotkey']}"


def sample_ids(comp, specs="tag:A"):
    """{spec: sample id} campaign43 uses for these tag specs on comp's live config right now (the eval cache's sample
    index, same key as src.parity.sample.build_sample); a spec whose sample is not built yet is missing. A result stored
    on any other sample (an older live config) is not a number of this campaign."""
    cache = D / "eval_cache"
    try:
        cv = json.loads((cache / ("live_manifest.json" if comp == "main" else f"live_manifest_{comp}.json")).read_text())["config_version"]
        index = json.loads((cache / "samples" / "index.json").read_text())
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return {}
    out = {}
    for spec in [s for s in specs.split(",") if s.startswith("tag:")]:
        sid = index.get(hashlib.sha256(f"{cv}|{_seed_material(spec[4:])}".encode()).hexdigest()[:16])
        if sid:
            out[spec] = sid
    return out


def king_label(name):
    """The king label campaign43 stores for models.json entry `name` (directory name, or a recipe's own label)."""
    path = ms.M.get(name)
    if not path:
        return None
    if path.endswith(".json"):
        return (read_json(path, {}) or {}).get("label") or Path(path).stem
    return Path(path).name


def _duel_results():
    for f in sorted((D / "eval_cache/results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        yield from data.get("results", [])


def results(labels):
    """{label: {comp: mu}} at LIMIT seqs vs each competition's current king, on the sample campaign43 uses for that
    competition now (a stored result with the right label on another sample or vs another king does not count)."""
    want = {}
    for c, (k, _) in COMPS.items():
        sid, kl = sample_ids(c).get("tag:A"), king_label(k)
        if sid and kl:
            want[sid] = (c, kl)
    out = {}
    if not want or not labels:
        return out
    for r in _duel_results():
        lab, v = (r.get("challenger") or {}).get("label"), r.get("verdict") or {}
        if lab in labels and v.get("n_sequences") == LIMIT:
            hit = want.get(r.get("sample"))
            if hit and (r.get("king") or {}).get("label") == hit[1]:
                out.setdefault(lab, {})[hit[0]] = v["mu_hat"]
    return out


def build(p):
    """(label, body) of point p without writing anything: merge_search4's recipe body, labelled pub_<hash of what the
    scorer loads> (terms = model paths + coefficients, mode, mode args, overrides, transforms; not the description)."""
    _, body = ms.build(p)
    material = {k: v for k, v in body.items() if k not in ("label", "description", "group")}
    body["label"] = "pub_" + hashlib.sha1(json.dumps(material, sort_keys=True).encode()).hexdigest()[:10]
    body["group"] = CAMPAIGN
    # readable description: probes say what they probe; every public model is named by competition, reign and repo
    used = [n for n in NAMES if f"to{n}=" in body.get("description", "") or f"[{n}]" in body.get("description", "")
            or n in p.get("fts", []) or n in (p.get("a"), p.get("b"), p.get("on"), p.get("base"))]
    c = p.get("c", {})
    if p.get("mode") == "lin" and len(c) == 1 and p.get("base") == ms.BASE and not any(p.get(k, 1.0) != 1.0 for k in ms.SCALES):
        (v, w), = c.items()
        body["description"] = (f"probe: {ms.BASE} {'+' if w > 0 else '-'} {abs(w):g}·({v[2:]} - {ms.BASE}) = "
                               + ("toward " if w > 0 else "away from (ExPO) ") + v[2:])
    elif p.get("mode") == "lin" and not c:
        body["description"] = f"raw: {p.get('base')} as is"
    elif p.get("mode") == "gamma":
        g = p.get("g")
        body["description"] = (f"{p.get('base')} router-bias γ: b' = mean + {g:g}·(b − mean) per layer "
                               f"(e_score_correction_bias)")
    if used:
        body["description"] += " | " + "; ".join(f"{n} = {NAMES[n]}" for n in used)
    return body["label"], body


def label_of(p):
    """pub_ label of point p, None when it cannot be built any more (a model or direction it uses is gone)."""
    try:
        return build(p)[0]
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def recipe(p):
    """Write point p's recipe as configs/merge/public/<label>.json (rewritten only when it changed)."""
    label, body = build(p)
    path = ms.OUT / f"{label}.json"
    text = json.dumps(body, indent=1)
    try:
        same = path.read_text() == text
    except OSError:
        same = False
    if not same:
        path.write_text(text)
    return label, path


def stochastic(body):
    """DARE / DELLA masks come from a device RNG: a CPU write is not the model campaign43 scored on the GPUs."""
    args = body.get("mode_args") or {}
    try:
        drop = float(args.get("drop", 0) or 0)
    except (TypeError, ValueError):
        drop = 1.0
    return body.get("mode") in ("dare", "dare_ties", "delta_dare") or drop > 0


def trainer_running():
    return bool(subprocess.run(["pgrep", "-f", "[s]cripts/train/cpt_fsdp.py"], capture_output=True).stdout.strip())


def scorer_running():
    return bool(subprocess.run(["pgrep", "-f", "[s]cripts/merge/campaign43.py"], capture_output=True).stdout.strip())


@contextlib.contextmanager
def gpu_lock():
    """gpu_screen.lock around one scorer run, released in a finally (it used to stay behind after every round). Another
    holder's lock is honoured for 4 h (the convention of every tool that takes it); a stale one of ours is taken over."""
    announced = False
    while True:
        try:
            holder, age = LOCK.read_text().strip(), time.time() - LOCK.stat().st_mtime
        except OSError:
            break
        if holder == CAMPAIGN or age >= 4 * 3600:
            break
        if not announced:
            log(f"waiting for {LOCK} (held by {holder!r}, {age / 60:.0f} min old)")
            announced = True
        time.sleep(20)
    LOCK.write_text(CAMPAIGN)
    try:
        yield
    finally:
        try:
            if LOCK.read_text().strip() == CAMPAIGN:
                LOCK.unlink()
        except OSError:
            pass


def score(points, score_chunk: int = 48):
    rec = {ms.key(p): recipe(p) for p in points}
    labels = {lab for lab, _ in rec.values()}
    have = results(labels)
    todo = [str(path) for lab, path in rec.values() if len(have.get(lab, {})) < len(COMPS)]
    if todo:
        with gpu_lock():
            # a scorer left running by a restarted orchestrator finishes first (its results are then cached)
            while scorer_running():
                time.sleep(20)
            have = results(labels)
            todo = [str(path) for lab, path in rec.values() if len(have.get(lab, {})) < len(COMPS)]
            kings = " ".join(f"--king {c}={k}:{bar}" for c, (k, bar) in COMPS.items())
            t0, done = time.time(), 0
            chunk = max(1, score_chunk)
            for off in range(0, len(todo), chunk):
                batch = todo[off:off + chunk]
                cmd = (f"cd {REPO} && source {D}/env.sh && PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py "
                       + " ".join(f"--recipe {p}" for p in batch)
                       + f" {kings} --comps {','.join(COMPS)} --campaign {CAMPAIGN} --limit {LIMIT} --sample tag:A --replicas 4"
                       " --batch-size 32 --pingpong 1")
                with open(CAMPAIGN_LOG, "a") as fh:  # the dashboard's merge panel reads the scoring phase from pm2_campaign*.log
                    p = subprocess.run(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT)
                if p.returncode != 0:
                    log(f"campaign43 failed rc={p.returncode}: " + CAMPAIGN_LOG.read_text(errors="replace")[-1500:])
                    break
                done += len(batch)
            if todo:
                log(f"scored {done}/{len(todo)} recipes in {(time.time() - t0) / 60:.1f} min (chunks of {chunk})")
        have = results(labels)
    return {k: have.get(lab, {}) for k, (lab, _) in rec.items()}


FULL_SAMPLES = {"main": "tag:A,tag:C", "math": "tag:A,tag:B,tag:C", "code": "tag:A,tag:B,tag:C", "text": "tag:A,tag:B,tag:C"}
READY = D / "search" / "public_ready.json"


def run43(path, comp, campaign, samples, limit):
    """One recipe on one competition vs its king (larger samples than the search screen); waits for the GPUs.
    Returns campaign43's exit code (non-zero is logged)."""
    k, bar = COMPS[comp]
    cmd = (f"cd {REPO} && source {D}/env.sh && PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py --recipe {path}"
           f" --king {comp}={k}:{bar} --comps {comp} --campaign {campaign} --sample {samples} --limit {limit}"
           " --replicas 4 --batch-size 32 --pingpong 1")
    with gpu_lock():
        while scorer_running():
            time.sleep(20)
        with open(CAMPAIGN_LOG, "a") as fh:
            rc = subprocess.run(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        log(f"campaign43 ({campaign}: {comp} {samples} x{limit}) failed rc={rc}: "
            + CAMPAIGN_LOG.read_text(errors="replace")[-1000:])
    return rc


def verdicts(label, comp, n_min, samples="tag:A"):
    """{sample: (mu, lcb, accepted, n)} of `label` vs comp's current king on the current samples of `samples`, scored on
    >= n_min sequences (newest wins)."""
    king, sids = king_label(COMPS[comp][0]), set(sample_ids(comp, samples).values())
    out = {}
    if not king or not sids:
        return out
    for r in _duel_results():
        v = r.get("verdict") or {}
        if (r.get("challenger") or {}).get("label") == label and (r.get("king") or {}).get("label") == king \
                and r.get("sample") in sids and (v.get("n_sequences") or 0) >= n_min:
            out[r["sample"]] = (v.get("mu_hat"), v.get("lcb"), bool(v.get("accepted")), v.get("n_sequences"))
    return out


DIST_DIR = D / "search" / "public_dist"
_dist_jobs = {}


def dist(anchor, anchor_path, name, path):
    """|X - anchor| on delta_dist's tensor subset, computed in the background (never blocks the GPU loop); None until ready."""
    f = DIST_DIR / f"{anchor}__{name}.json"
    if f.exists():
        try:
            return json.loads(f.read_text())[path]
        except (KeyError, json.JSONDecodeError):
            f.unlink(missing_ok=True)
    job = _dist_jobs.get(name)
    if job is None and sum(j.poll() is None for j in _dist_jobs.values()) < 3:
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        cmd = (f"{REPO}/.venv-eval/bin/python {REPO}/scripts/merge/delta_dist.py --a {anchor_path} --b {path} > {f}.tmp"
               f" && mv {f}.tmp {f}")
        _dist_jobs[name] = subprocess.Popen(["nice", "-n", "10", "bash", "-c", cmd])
    elif job is not None and job.poll() not in (None, 0):
        _dist_jobs.pop(name)  # failed: retry next round
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="C10_S50")
    ap.add_argument("--batch", type=int, default=64,
                    help="max candidates scored per round (focused grid: ~120–250 pending with 23 kings, not thousands)")
    ap.add_argument("--score-chunk", type=int, default=48,
                    help="recipes per campaign43 subprocess (keeps command length and failure blast radius sane)")
    ap.add_argument("--focus", type=int, default=1,
                    help="1=screen donors then expand winners (default); 0=score every method×parameter combo")
    ap.add_argument("--delete-old", type=int, default=1, help="delete probed old kings without gain (disk for the next downloads)")
    ap.add_argument("--reset-pool", action="store_true",
                    help="search/public.json was built on another anchor: keep it as public.json.<anchor>.<time> and start over")
    args = ap.parse_args()
    ms.OUT = REPO / "configs/merge/public"
    ms.OUT.mkdir(parents=True, exist_ok=True)
    ms.LIMIT, ms.KT, ms.BASE = LIMIT, "TX1", args.anchor  # Model Stock base = T1 (every family member descends from it)
    A = args.anchor
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    # every probe direction to<X> = X - anchor and every cached number was built on the stored anchor: never reuse them
    # under another one
    old_anchor, old_path = st.get("anchor"), st.get("anchor_path")
    new_path = models().get(A) or OURS.get(A)
    moved = bool(old_path and new_path and old_path != new_path)
    if st.get("pool") and old_anchor and (old_anchor != A or moved):
        what = (f"anchor {old_anchor}" + (f" ({old_path})" if old_path else "")
                + f", not {A}" + (f" ({new_path})" if new_path else ""))
        if not args.reset_pool:
            raise SystemExit(f"{STATE} holds {len(st['pool'])} recipes built on {what}. Restart with --anchor {old_anchor}, "
                             f"or add --reset-pool to keep that state as {STATE.name}.{old_anchor}.<time> and start over on {A}")
        bak = STATE.with_name(f"{STATE.name}.{old_anchor}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}")
        shutil.copy2(STATE, bak)
        log(f"--reset-pool: {len(st['pool'])} recipes built on {what} kept in {bak}; starting over on {A}")
        st = {"kings": st.get("kings"), "written": st.get("written", {})}
    pool = st.get("pool", {})          # key -> {"point": p, "mu": {comp: mu}, "label": pub_<body hash>}
    probes = st.get("probes", {})      # model -> {"dose": d, "dist": x, "plus": key, "minus": key, "gain": {comp: g}, "deleted": bool}
    cur = st.get("cur", {})            # comp -> key of its current point
    steps = st.get("steps", {c: {} for c in COMPS})
    rot = st.get("rot", {c: 0 for c in COMPS})
    ref_dist = st.get("ref_dist")
    confirm = st.get("confirm", {})    # "comp:key" -> {"label": ..., "screen": {...}, "full": {...}, "status": ...}
    expand_donors = set(st.get("expand_donors", []))  # donors promoted from screen hits -> full method grid
    samples = st.setdefault("samples", {})  # comp -> tag:A sample id the pool's numbers are on
    written = st.setdefault("written", {})
    for c, w in written.items():  # entries from before king tracking: the king of that time; a write that died with us
        w.setdefault("king", (st.get("kings") or {}).get(c))
        if w.get("status") == "writing":
            out = Path(w.get("out", ""))
            if (out / "model.safetensors.index.json").exists():
                w["status"] = "ready"
            elif not subprocess.run(["pgrep", "-f", f"[w]rite_merge.py.*--out {out}"], capture_output=True).stdout.strip():
                w["status"] = "interrupted"
    METHODS = [{"mode": "ties", "density": 0.5}, {"mode": "dare", "drop": 0.5}, {"mode": "della", "drop": 0.5, "eps": 0.1},
               {"mode": "emr"}, {"mode": "crumbs", "density": 0.5}, {"mode": "ties", "density": 0.3}, {"mode": "dare", "drop": 0.3}]
    SC = [("s_attn", 0.25), ("s_exp", 0.25), ("s_attn", -0.25), ("s_exp", -0.25), ("s_lo", 0.25), ("s_hi", 0.25), ("s_lo", -0.25),
          ("s_hi", -0.25), ("s_mid", 0.25), ("s_router", -0.5), ("s_embed", 0.5), ("s_norm", 0.5), ("s_mid", -0.25), ("s_embed", -0.5)]
    said = set()

    def say_once(key, msg):
        if key not in said:
            said.add(key)
            log(msg)

    def save():
        STATE.parent.mkdir(parents=True, exist_ok=True)
        best, train = {}, {}
        for c in COMPS:
            if cur.get(c) in pool:
                p = pool[cur[c]]
                try:
                    lab, rpath = recipe(p["point"])
                except (KeyError, IndexError, TypeError, ValueError):
                    continue  # its models / directions are gone; the next ranking moves the competition off it
                mu = p["mu"].get(c)
                best[c] = {"key": cur[c], "mu": mu, "all": p["mu"], "recipe": str(rpath), "label": lab, "bar": COMPS[c][1],
                           "king": COMPS[c][0], "sample": samples.get(c)}  # merge4train.py uses "best" only vs the same king
                train[c] = {"king": COMPS[c][0], "mu": mu, "beats_king": bool(mu is not None and mu > 0),
                            "label": lab, "recipe": str(rpath), "key": cur[c], "all": p["mu"], "bar": COMPS[c][1]}
        tmp = STATE.with_name(STATE.name + ".tmp")
        tmp.write_text(json.dumps({"anchor": A, "anchor_path": ms.M.get(A) or OURS.get(A), "limit": LIMIT,
                                   "kings": {c: COMPS[c][0] for c in COMPS}, "samples": samples, "best": best,
                                   "train_init": train, "cur": cur, "steps": steps, "rot": rot, "ref_dist": ref_dist,
                                   "probes": probes, "confirm": confirm, "written": written,
                                   "expand_donors": sorted(expand_donors),
                                   **{k: v for k, v in st.items() if k.startswith("specials")}, "pool": pool}, indent=1))
        tmp.replace(STATE)
        (D / "search" / "train_inits.json").write_text(json.dumps(
            {"anchor": A, "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "train_init": train,
             "written": written}, indent=1))

    def add(points):
        merged = {}
        for i in range(0, len(points), args.score_chunk):
            chunk = points[i:i + args.score_chunk]
            part = score(chunk, score_chunk=args.score_chunk)
            merged.update(part)
            for p in chunk:
                k = ms.key(p)
                if part.get(k):
                    pool[k] = {"point": p, "mu": part[k], "label": label_of(p)}
            save()
        return merged

    def point_models(p):
        """Every model name point p loads; None when one of its directions is not registered any more."""
        names = {p.get("a"), p.get("b"), p.get("on"), p.get("donor")} | set(p.get("fts") or [])
        if p.get("mode") == "stock":
            names.add(ms.KT)  # merge_search4 builds every Model Stock recipe on KT (TX1)
        elif p.get("mode") != "slerp":
            names.add(p.get("base", A))
        for v in p.get("c", {}):
            if v not in ms.VECS:
                return None
            names |= set(ms.VECS[v][:2])
        return {n for n in names if n}

    def alive(p):
        """Every model of p is on disk (not archived, not queued for the cold volume, not deleted) and buildable."""
        names = point_models(p)
        return names is not None and all(n in avail and not probes.get(n, {}).get("deleted") for n in names)

    def valid_keys():
        """Pool entries that may be ranked: alive, and scored as exactly the recipe they build now."""
        return {k for k, v in pool.items() if v.get("label") and alive(v["point"]) and v["label"] == label_of(v["point"])}

    def in_use():
        return {x for c in COMPS if cur.get(c) in pool for x in (point_models(pool[cur[c]]["point"]) or ())}

    def forget(comp, why):
        """comp's numbers changed meaning (new king / new sample): drop its mu, its confirm runs and its written init."""
        for v in pool.values():
            v["mu"].pop(comp, None)
        gone = [k for k in confirm if k.startswith(comp + ":")]
        for k in gone:
            confirm.pop(k)
        w = written.pop(comp, None)
        if gone or w:
            log(f"  {comp.upper()}: {why}: cleared {len(gone)} confirm entries"
                + (f" and the train init {w.get('label')} ({w.get('status')}, vs {w.get('king')})" if w else ""))
        cur.pop(comp, None)

    writer = {"proc": None, "comp": None, "label": None}

    def maybe_write():
        """Write the best above-king merge for one competition (CPU, or the GPUs for a DARE / DELLA recipe) so training
        can start from it. Only an improvement over a write against the SAME king counts as one."""
        proc = writer["proc"]
        if proc is not None and proc.poll() is None:
            return
        if proc is not None:
            w = written.get(writer["comp"]) or {}
            if w.get("label") == writer["label"]:
                ok = proc.returncode == 0 and (Path(w["out"]) / "model.safetensors.index.json").exists()
                w["status"] = "ready" if ok else f"failed rc={proc.returncode}"
            if proc.returncode != 0:
                log(f"write_merge exited {proc.returncode} (see /data/teutonic/logs/write_train_*.log)")
        writer.update(proc=None, comp=None, label=None)
        pend = pending_kings()
        valid = valid_keys()
        for c in ("text", "code", "math", "main"):
            if cur.get(c) not in valid:
                continue
            if c in pend:
                say_once(f"pend-write-{c}-{pend[c]}", f"  {c.upper()}: live king {pend[c][:12]} not ingested yet: no train-init write vs the dethroned {COMPS[c][0]}")
                continue
            mu = pool[cur[c]]["mu"].get(c, -1)
            w = written.get(c) or {}
            same_king = w.get("king") == COMPS[c][0]
            prev = w.get("mu", -1) if same_king and w.get("status") in ("ready", "writing") else -1
            if mu < 1.5e-4 or mu < prev + 1e-4:
                continue
            label, path = recipe(pool[cur[c]]["point"])
            if same_king and w.get("label") == label and w.get("status") in ("ready", "writing"):
                continue
            out = D / "submit" / f"train_{c}_{label}"
            if (out / "model.safetensors.index.json").exists():
                written[c] = {"mu": mu, "label": label, "out": str(out), "status": "ready", "king": COMPS[c][0]}
                log(f"  train init {c.upper()} already written: {out} ({mu:+.6f})")
                continue
            gpus = "cpu"
            if stochastic(json.loads(path.read_text())):
                if trainer_running():
                    say_once(f"stoch-{label}", f"  train init {c.upper()} {label} is a DARE/DELLA recipe: it needs a GPU write "
                             "(a CPU write draws other masks) and a trainer holds the GPUs; waiting")
                    continue
                gpus = "0,1,2,3,4,5,6,7"
            log(f"  WRITE train init {c.upper()} {label} mu {mu:+.6f} vs {COMPS[c][0]} -> {out} (--gpus {gpus})")
            cmd = (f"cd {REPO} && source {D}/env.sh && ionice -c 3 nice -n 15 python scripts/merge/write_merge.py "
                   f"--recipe {path} --out {out} --gpus {gpus} --workers 16")
            logf = D / "logs" / f"write_train_{c}_{label}.log"
            writer.update(proc=subprocess.Popen(["bash", "-c", cmd], stdout=open(logf, "w"), stderr=subprocess.STDOUT),
                          comp=c, label=label)
            written[c] = {"mu": mu, "label": label, "out": str(out), "status": "writing", "log": str(logf), "king": COMPS[c][0]}
            return

    if st.get("kings"):
        for c, name in st["kings"].items():
            if c in COMPS:
                COMPS[c] = (name, COMPS[c][1])
    reload_public()
    avail0 = available()
    sync_models(avail0)
    for c, name in kings_file().items():
        if not pool and name in avail0:   # fresh campaign: start against the reigning kings directly
            COMPS[c] = (name, COMPS[c][1])
    for n, pr in probes.items():  # probe directions must exist before any recipe is rebuilt (restart)
        if not pr.get("deleted"):
            ms.VECS.setdefault(f"to{n}", (n, A, pr["dose"] / 2, pr["dose"] / 8))
    if INJECT.exists():
        for vname, (va, vb) in json.loads(INJECT.read_text()).get("vecs", {}).items():
            ms.VECS.setdefault(vname, (va, vb, 0.25, 0.06))

    def vec(name):  # direction toward model `name` from the anchor
        return f"to{name}"

    avail = avail0
    rnd = 0
    while True:
        rnd += 1
        reload_public()
        avail = available()
        sync_models(avail)
        mg.register(ms, avail)
        for v in pool.values():
            for name in v["point"].get("c", {}):
                if name in ms.VECS:
                    continue
                if name.startswith("step"):
                    src, pred = name.removeprefix("step"), mg.predecessor(name.removeprefix("step"))
                    if pred and src in avail and pred in avail:
                        ms.VECS[name] = (src, pred, 0.2, 0.05)
                elif name.startswith("to") and "_on_" in name:
                    donor, king = name[2:].split("_on_", 1)
                    if donor in avail and king in avail:
                        ms.VECS[name] = (donor, king, 0.2, 0.05)
        if A not in avail:
            fallback = next((n for n in ("TX6", "M45", "TX5", "TX1", "M44") if n in avail), None)
            if pool or fallback is None:
                log(f"anchor {A} not on disk" + (f"; {len(pool)} pool recipes are built on it: waiting (no silent anchor switch; "
                                                 f"restart with another --anchor and --reset-pool to start over)" if pool else " yet"))
                save()
                time.sleep(120)
                continue
            log(f"anchor {A} not on disk and the pool is empty; searching from {fallback}")
            A, ms.BASE = fallback, fallback
        for comp, name in kings_file().items():
            if name in avail and COMPS[comp][0] != name:
                log(f"{comp.upper()} king -> {name} ({NAMES.get(name, '')}): re-scoring {len(pool)} recipes against it from the loss cache")
                COMPS[comp] = (name, COMPS[comp][1])
                forget(comp, f"king {name}")
                # skip recipes whose models are queued for / being moved to the cold volume (the copy removes /data's)
                ok = [v["point"] for v in pool.values() if alive(v["point"])]
                for k, mu in score(ok).items():
                    if k in pool and mu:
                        pool[k]["mu"] = mu
                st["kings"] = {c: COMPS[c][0] for c in COMPS}
        # the live eval config of a competition changed (new sample id): its numbers are on the old sample
        for comp in COMPS:
            sid = sample_ids(comp).get("tag:A")
            if sid and samples.get(comp) and samples[comp] != sid:
                log(f"{comp.upper()} eval sample {samples[comp]} -> {sid} (live config changed): re-scoring {len(pool)} recipes on it")
                forget(comp, f"sample {sid}")
                samples[comp] = sid
                ok = [v["point"] for v in pool.values() if alive(v["point"])]
                for k, mu in score(ok).items():
                    if k in pool and mu:
                        pool[k]["mu"] = mu
            elif sid:
                samples[comp] = sid
        # entries scored as another recipe than the one they build now (labels hash the recipe body; entries from before
        # that have no label): re-scored before they are ranked again (identical bodies come straight from the loss cache)
        stale = {k: v for k, v in pool.items() if alive(v["point"]) and v.get("label") != label_of(v["point"])}
        if stale:
            log(f"re-checking {len(stale)} pool recipes whose label does not match what they build now "
                "(labels hash the recipe body: anchor, directions, model paths)")
            res = score([v["point"] for v in stale.values()], score_chunk=args.score_chunk)
            fixed = 0
            for k, v in stale.items():
                if len(res.get(k) or {}) == len(COMPS):
                    v["mu"], v["label"] = res[k], label_of(v["point"])
                    fixed += 1
            log(f"  {fixed}/{len(stale)} re-scored; the rest stay out of the ranking until they are")
            save()
        valid = valid_keys()
        for c in COMPS:
            if c in cur and cur[c] not in valid:
                log(f"  {c.upper()}: current point {cur[c]} is gone or unverified; it moves to the best valid point")
                cur.pop(c)
        cands = {}
        # 1 raw: every near-family model and checkpoint as is
        for name in sorted(avail):
            if name in FAMILY or name in OURS:
                q = {"base": name, "c": {}, "mode": "lin"}
                if ms.key(q) not in pool:
                    cands[ms.key(q)] = q
        # 2 probes for models not probed yet (near family first, then old kings newest first)
        new = [n for n in avail if n != A and n not in probes]
        new.sort(key=lambda n: (n not in FAMILY and n not in OURS, n))
        far = [n for n in new if n not in FAMILY and n not in OURS]
        if ref_dist is None and "M43" in avail:
            ref_dist = dist(A, avail[A], "M43", avail["M43"])
            if ref_dist:
                log(f"reference distance |M43 - {A}| = {ref_dist:.3f} (tensor subset)")
        for n in new:
            x = None
            if n in far:
                x = dist(A, avail[A], n, avail[n]) if ref_dist else None
                if x is None:
                    continue  # distance still being measured
            d = 0.3 if x is None else round(min(0.3, 0.3 * ref_dist / max(x, 1e-9)), 4)
            plus, minus = {"base": A, "c": {vec(n): d}, "mode": "lin"}, {"base": A, "c": {vec(n): -d}, "mode": "lin"}
            probes[n] = {"dose": d, "dist": x, "plus": ms.key(plus), "minus": ms.key(minus), "points": [plus, minus]}
        for n, pr in probes.items():  # directions (re-registered on a restart) and probes not scored yet
            if not pr.get("deleted"):
                ms.VECS.setdefault(vec(n), (n, A, pr["dose"] / 2, pr["dose"] / 8))
                for q in pr["points"]:
                    if ms.key(q) not in pool:
                        cands[ms.key(q)] = q
        # 3 search moves per competition from its current point over its TOPK directions
        anchor_key = ms.key({"base": A, "c": {}, "mode": "lin"})
        for c in COMPS:
            gains = sorted(((pr["gain"][c], n) for n, pr in probes.items() if pr.get("gain") and not pr.get("deleted")), reverse=True)
            top = [vec(n) for g, n in gains[:TOPK] if g > 0 and vec(n) in ms.VECS]
            if c not in cur or cur[c] not in valid:
                continue
            p = pool[cur[c]]["point"]
            if p.get("mode") in ("stock", "slerp", "ortho", "agree", "gamma"):
                continue  # method_grid.refine owns the local step for these
            top += [v for v in p.get("c", {}) if v not in top and v in ms.VECS]  # refine the best point's own vectors too
            s = steps[c]
            tries = []
            for v in top:
                s.setdefault(v, ms.VECS[v][2])
                if s[v] >= ms.VECS[v][3] - 1e-9:
                    for sign in (1, -1):
                        q = json.loads(json.dumps(p)); q["c"][v] = round(q["c"].get(v, 0.0) + sign * s[v], 4)
                        tries.append(q)
            if p.get("mode", "lin") == "lin" and p["c"]:
                m = METHODS[rot[c] % len(METHODS)]
                tries.append({"base": p.get("base", A), "c": dict(p["c"]), **m, "lam": 1.0})
                sk, ds = SC[rot[c] % len(SC)]
                q = json.loads(json.dumps(p)); q[sk] = max(0.0, q.get(sk, 1.0) + ds)
                tries.append(q)
            elif p.get("mode") != "lin":
                for f in (1.3, 0.75):
                    tries.append({**p, "lam": round(p.get("lam", 1.0) * f, 3)})
                tries.append({"c": dict(p["c"]), "mode": "lin", "base": p.get("base", A)})
            rot[c] += 1
            for q in tries:
                if ms.key(q) not in pool and len(cands) < args.batch * 8:
                    cands.setdefault(ms.key(q), q)
        # one-off method candidates once the family is in: Model Stock over the family (base T1) and SLERP toward the kings
        if not st.get("specials_done") and all(n in avail for n in ("M43", "TX2", "CD2", "M42")):
            fam = [n for n in ("M44", "TX4", "MA1", "CD4", "M43", "M42", "TX2") if n in avail]
            for q in ([{"mode": "stock", "fts": fam, "on": A, "scale": 1.0}, {"mode": "stock", "fts": fam + [A], "on": "TX1", "scale": 1.0},
                       {"mode": "stock", "fts": ["M44", "TX4", "MA1", "CD4", A], "on": A, "scale": 1.0}]
                      + [{"mode": "slerp", "a": A, "b": b, "t": t} for b in ("TX4", "M44", "MA1", "CD4") for t in (0.3, 0.5)]):
                cands.setdefault(ms.key(q), q)
            st["specials_done"] = True
        # every new king (public_kings.json): SLERP toward it from the anchor and from the other kings, Model Stock with it
        for comp, kname in kings_file().items():
            done_key = f"specials_{kname}"
            if st.get(done_key) or kname not in avail:
                continue
            prev = [x for x in ("TX5", "M44", "TX4", "CD4", "MA1") if x in avail and x != kname]
            for q in ([{"mode": "slerp", "a": A, "b": kname, "t": t} for t in (0.3, 0.5, 0.7)]
                      + [{"mode": "slerp", "a": kname, "b": b, "t": 0.3} for b in prev[:3]]
                      + [{"mode": "stock", "fts": [kname] + prev[:3] + [A], "on": kname, "scale": 1.0},
                         {"mode": "stock", "fts": [kname, A], "on": kname, "scale": 1.0}]):
                cands.setdefault(ms.key(q), q)
            st[done_key] = True
        st.setdefault("specials_TX5", st.get("tx5_done", False))
        # injected candidates: first in this round
        injected = set()
        if INJECT.exists():
            try:
                inj = json.loads(INJECT.read_text())
            except json.JSONDecodeError:
                inj = {}
            for vname, (va, vb) in inj.get("vecs", {}).items():
                ms.VECS[vname] = (va, vb, 0.25, 0.06)
            for q in inj.get("points", []):
                if alive(q) and ms.key(q) not in pool:
                    cands[ms.key(q)] = q
                    injected.add(ms.key(q))
        # method search from each competition's reigning king (M45 / MA2 / … on disk, not models.json ids)
        reigning = {c: COMPS[c][0] for c in COMPS}
        ranked = {reigning[c]: [n for g, n in sorted(
            ((pr["gain"].get(c, -1.0), n) for n, pr in probes.items() if pr.get("gain") and not pr.get("deleted")),
            reverse=True) if g > 0 and n in avail][:6] for c in COMPS}
        method_keys = set()
        for c in COMPS:
            kname = reigning[c]
            if cur.get(c) in valid and kname in avail:
                for q in mg.refine(pool[cur[c]]["point"]):
                    if alive(q):
                        method_keys.add(ms.key(q))
                        if ms.key(q) not in pool:
                            cands.setdefault(ms.key(q), q)
        for q in mg.grid(ms, avail, reigning, anchor=A,
                         ancestor="TX1" if "TX1" in avail else A, ranked=ranked,
                         focus=bool(args.focus), expand_donors=expand_donors):
            if not alive(q):
                continue
            method_keys.add(ms.key(q))
            if ms.key(q) not in pool:
                cands.setdefault(ms.key(q), q)
        # only candidates whose every model is on disk and whose every direction is registered (no KeyError mid-build)
        todo = [q for k, q in cands.items() if k not in pool and alive(q)]
        if not todo:
            pending = [r["name"] for r in PUBLIC if r["name"] not in probes]
            if not pending:
                log("CAMPAIGN DONE: every public model probed, every competition converged")
                maybe_write()
                save()
                break
            log(f"waiting for downloads ({len(pending)} public models not probed yet: {', '.join(pending[:8])} ...)")
            maybe_write()
            save()
            time.sleep(120)
            continue
        # old-king probes first (each frees ~206 GB for the next download), then raw / family probes / methods, then search
        far_vecs = {vec(n) for n in probes if n not in FAMILY and n not in OURS}

        def prio(q):
            if ms.key(q) in method_keys:
                return 0
            c = q.get("c", {})
            if q.get("mode") == "lin" and q.get("base") == A and len(c) == 1 and next(iter(c)) in far_vecs:
                return 1
            if not c or (len(c) == 1 and q.get("base") == A and q.get("mode") == "lin"):
                return 2
            return 3
        todo.sort(key=lambda q: -1 if ms.key(q) in injected else prio(q))
        methods = [q for q in todo if ms.key(q) in method_keys]
        rest = [q for q in todo if ms.key(q) not in method_keys]
        # previous kings' training steps (stepM40 = #40 − #39, …) fill the batch before blends of the latest kings
        def method_order(q):
            c = q.get("c") or {}
            steps = [k for k in c if str(k).startswith("step")]
            # ExPO away from a king's own step and soups of the reigning kings lead the pool: score them first
            if q.get("mode") == "stock" or (steps and any(c[s] < 0 for s in steps)) or len(c) >= 3:
                return (-1,)
            if steps and any(mg._famous_step(s) for s in steps):
                return (0,)
            if steps:
                return (1,)
            if mg.donors_in_point(q) & expand_donors:
                return (2,)
            mode = q.get("mode", "lin")
            if mode in ("ties", "dare", "della", "emr", "crumbs"):
                return (3,)
            if mode == "slerp":
                return (4,)
            return (5,)

        methods.sort(key=method_order)
        grads = [q for q in methods if any(str(k).startswith("step") for k in (q.get("c") or {}))]
        methods = grads + [q for q in methods if q not in grads]
        n_method = min(len(methods), args.batch if not rest else max(args.batch - 2, (args.batch * 3) // 4))
        todo = methods[:n_method] + rest[: max(0, args.batch - n_method)]
        pending = len([k for k in cands if k not in pool])
        log(f"round {rnd}: scoring {len(todo)} / {pending} pending ({sum(ms.key(q) in method_keys for q in todo)} method, "
            f"{len(methods)} method queued, focus={args.focus}, expand={len(expand_donors)} donors, "
            f"{len(probes)} probed, {len(pool)} in pool)")
        best_before = {c: pool[cur[c]]["mu"].get(c, -1) if cur.get(c) in valid else -1 for c in COMPS}
        add(todo + [{"base": A, "c": {}, "mode": "lin"}])
        valid = valid_keys()
        for comp, king in reigning.items():
            if king not in avail or best_before[comp] <= -1:
                continue  # no round-start reference (fresh / reset competition): every point would "beat" it
            for k, v in pool.items():
                p = v["point"]
                if p.get("base") != king or k not in valid:
                    continue
                mu = v["mu"].get(comp)
                if mu is None or mu < best_before[comp] + GAIN_KEEP:
                    continue
                promoted = mg.donors_in_point(p)
                new = promoted - expand_donors
                if new:
                    log(f"  expand donors for {comp.upper()} king {king}: {', '.join(sorted(new))} "
                        f"(mu {mu:+.6f} vs round-start best {best_before[comp]:+.6f})")
                expand_donors |= promoted
        base_mu = pool.get(anchor_key, {}).get("mu", {}) if anchor_key in valid else {}
        # probe gains: best of +- over the anchor, per competition
        keep = in_use()
        for n, pr in probes.items():
            if "gain" not in pr and pr["plus"] in valid and pr["minus"] in valid and base_mu:
                pr["gain"] = {c: round(max(pool[pr["plus"]]["mu"].get(c, -1), pool[pr["minus"]]["mu"].get(c, -1)) - base_mu.get(c, 0), 6) for c in COMPS}
                pr["best_sign"] = {c: (1 if pool[pr["plus"]]["mu"].get(c, -1) >= pool[pr["minus"]]["mu"].get(c, -1) else -1) for c in COMPS}
                log(f"  probe {n} (dose {pr['dose']}): gain " + "  ".join(f"{c.upper()} {pr['gain'][c]:+.6f}" for c in COMPS))
                if args.delete_old and n not in FAMILY and n not in OURS and n not in keep and max(pr["gain"].values()) < GAIN_KEEP:
                    path = avail.get(n)
                    if path and path.startswith("/data/teutonic/models/"):
                        to_cold(n, path)
                        pr["deleted"] = pr["archived"] = True
                        ms.VECS.pop(vec(n), None)
                        log(f"  {n}: no gain anywhere -> moving {path} to the cold volume (disk for the next download)")
        # move every competition to its best point so far (only points that are alive and scored as what they build)
        valid = valid_keys()
        for c in COMPS:
            ranked_c = sorted(((v["mu"][c], k) for k, v in pool.items() if c in v["mu"] and k in valid), reverse=True)
            if not ranked_c:
                continue
            best_mu, best_k = ranked_c[0]
            old = pool[cur[c]]["mu"].get(c, -1) if cur.get(c) in valid else -1
            if best_mu > old + EPS or cur.get(c) not in valid:
                cur[c] = best_k
                log(f"  {c.upper()} -> {best_mu:+.6f} (bar {COMPS[c][1]})  {best_k}")
            elif c in cur:
                steps[c] = {k: v / 2 for k, v in steps[c].items()}
        # a competition's best within 0.0002 of its bar (1 200 seqs) -> 4 800-seq screen; >= bar + 0.0002 there -> full
        # validator-size samples (A/B/C); every sample's LCB above the bar -> write the merge, upload it to HF, and
        # mark it SUBMIT-READY (search/public_ready.json). Submitting stays with the user.
        pend = pending_kings()
        for c in COMPS:
            if cur.get(c) not in valid:
                continue
            bar, mu = COMPS[c][1], pool[cur[c]]["mu"].get(c, -1)
            if mu < bar - 0.0002:
                continue
            if c in pend:
                say_once(f"pend-conf-{c}-{pend[c]}", f"  {c.upper()}: live king {pend[c][:12]} not ingested yet: no confirm runs vs the dethroned {COMPS[c][0]}")
                continue
            ck = f"{c}:{cur[c]}"
            label, path = recipe(pool[cur[c]]["point"])
            prev = confirm.get(ck) or {}
            retry = prev.get("status") == "scorer-failed" and prev.get("fails", 0) < 3
            if prev and prev.get("label") == label and not retry:
                continue
            log(f"  CONFIRM {c.upper()} {label} (+{mu:.6f} at {LIMIT} seqs, bar {bar}): 4 800-seq screen")
            rc = run43(path, c, "publicconf", "tag:A", 4800)
            scr = verdicts(label, c, 4800, "tag:A")
            failed = rc != 0 or not scr
            entry = {"label": label, "key": cur[c], "mu1200": mu, "screen": scr, "status": "screened", "king": COMPS[c][0]}
            if not failed:
                mu48 = max((v[0] for v in scr.values() if v[0] is not None), default=-1)
                log(f"  CONFIRM {c.upper()} {label}: 4 800 seqs mu {mu48:+.6f}")
                if mu48 >= bar + 0.0002:
                    log(f"  CONFIRM {c.upper()} {label}: full samples {FULL_SAMPLES[c]}")
                    rc = run43(path, c, f"{c}full", FULL_SAMPLES[c], 40000)
                    full = verdicts(label, c, 20000, FULL_SAMPLES[c])
                    entry["full"] = full
                    if rc != 0:
                        failed = True
                    else:
                        ok = bool(full) and all(v[2] for v in full.values()) and len(full) == len(FULL_SAMPLES[c].split(","))
                        entry["status"] = "PASS" if ok else "full-fail"
                        log(f"  CONFIRM {c.upper()} {label}: full " + "  ".join(f"{k[-6:]} mu {v[0]:+.6f} lcb {v[1]:+.6f} {'ok' if v[2] else 'no'}" for k, v in full.items())
                            + f" -> {entry['status']}")
                        if ok:
                            out = D / "submit" / f"{c}_{label}"
                            gpus = "cpu"
                            if stochastic(json.loads(path.read_text())):
                                gpus = "0,1,2,3,4,5,6,7"  # DARE / DELLA: the scored masks are the GPU RNG's
                            cmd = (f"cd {REPO} && source {D}/env.sh && nice -n 10 python scripts/merge/write_merge.py --recipe {path} --out {out} --gpus {gpus} --workers 32"
                                   f" && nice -n 19 python scripts/submit/hf_upload_ckpt.py --ckpt {out}")
                            ready = json.loads(READY.read_text()) if READY.exists() else {}
                            ready[c] = {"label": label, "recipe": str(path), "out": str(out), "full": full, "king": COMPS[c][0],
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                            if gpus != "cpu" and trainer_running():
                                entry["write"] = ready[c]["write"] = f"deferred (DARE/DELLA needs a GPU write, a trainer holds the GPUs): {cmd}"
                                log(f"  SUBMIT-READY {c.upper()} {label}: write deferred (DARE/DELLA needs the GPUs, a trainer holds them); "
                                    "command in search/public_ready.json; submission is the user's call")
                            else:
                                subprocess.Popen(["bash", "-c", cmd], stdout=open(D / "logs" / f"ready_{c}_{label}.log", "w"), stderr=subprocess.STDOUT)
                                log(f"  SUBMIT-READY {c.upper()} {label}: writing {out} (--gpus {gpus}, + HF upload); submission is the user's call")
                            READY.write_text(json.dumps(ready, indent=1))
            if failed:
                fails = prev.get("fails", 0) + 1 if prev.get("status") == "scorer-failed" and prev.get("label") == label else 1
                entry.update(status="scorer-failed", fails=fails, rc=rc)
                log(f"  CONFIRM {c.upper()} {label}: scorer failed (rc {rc}, {len(scr)} screen verdicts): not marked screened"
                    + (f", retry {fails}/3 next round" if fails < 3 else ", giving up after 3 tries"))
            confirm[ck] = entry
            save()
        # disk: old kings' directions are highly correlated (away from M37 / M38 / M39 all gain TEXT ~+0.00015), so only the
        # KEEP_OLD best old kings by gain stay on disk (plus any used by a competition's current point); the rest are deleted
        if args.delete_old:
            keep = in_use()
            olds = sorted(((max(pr["gain"].values()), n) for n, pr in probes.items()
                           if pr.get("gain") and not pr.get("deleted") and n not in FAMILY and n not in OURS), reverse=True)
            for g, n in olds[KEEP_OLD:]:
                path = avail.get(n)
                if n in keep or not path or not path.startswith("/data/teutonic/models/"):
                    continue
                to_cold(n, path)
                probes[n]["deleted"] = probes[n]["archived"] = True
                ms.VECS.pop(vec(n), None)
                log(f"  {n}: gain {g:+.6f} below the {KEEP_OLD} best old kings -> moving {path} to the cold volume")
        maybe_write()
        save()


if __name__ == "__main__":
    main()
