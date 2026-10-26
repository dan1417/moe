#!/usr/bin/env python3
"""Keep the latest public Teutonic kings on disk, in our merge basis (download.sh).

Every pass: read the four competitions' king chains from teutonic.ai, update configs/merge/public/public_models.json
(names: MAIN M<reign:02>, TEXT TX<reign>, CODE CD<reign>, MATH MA<reign>), pick the TOP most recently crowned kings plus
every reigning king plus T1 (the basis every hidden permutation is recovered against), and ingest the missing ones
(scripts/merge/ingest_king_fast.sh: 2 processes × 4 streams, permutation recovery, un-permute, register in models.json).
Then write /data/teutonic/search/public_kings.json: {competition: name} for every competition whose live king is ingested
(a dethroned king is never listed as reigning; a competition whose new king is not ingested yet is left out), and
/data/teutonic/search/public_kings_live.json: the live digest, name and readiness of each competition's king plus
"pending": {competition: digest} for kings not ingested yet. Readiness comes from scripts/box/kings.py (strict resolver).
With --loop it repeats every --poll seconds, so new kings are downloaded as they are crowned.

Failed ingests are recorded in /data/teutonic/search/ingest_failed.json (reason, attempts, time). A basis mismatch
(the hidden permutation cannot be recovered against T1: "cosine match is not a bijection", "column cosine too low"; the
MAIN #0-#26 lineage) is permanent: that king is skipped for good (--retry-failed clears those). Anything else (network,
incomplete download) is retried with backoff (15 min, 30 min, 1 h, 2 h, then every 6 h).

The eval sample outranks kings: while campaign43 is still building that sample, this loop downloads nothing.
One king at a time after that, 8 HTTPS connections total (not one connection per worker across every model).

  python scripts/data/download_kings.py --top 10 [--loop] [--poll 300]
  python scripts/data/download_kings.py --dry        # one pass: what would be downloaded / published; writes nothing
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/box"))
import kings as kr  # noqa: E402  (strict king resolver: live dashboards -> public_models.json name -> ingested path)

D = Path("/data/teutonic")
PUBLIC = REPO / "configs/merge/public/public_models.json"
MODELS = REPO / "configs/merge/k41/models.json"
KINGS = D / "search" / "public_kings.json"
LIVE = D / "search" / "public_kings_live.json"
FAILED = D / "search" / "ingest_failed.json"
COLD_REG = Path("/root/teutonic_cold/registry.json")
T1 = "95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0"
PREFIX = {"main": "M", "math": "MA", "code": "CD", "text": "TX"}
MODEL_GB = 221
# 2 processes × 4 range streams. Six models × 8 × 16 was ~790 connections and the sample never got the pipe.
DL_PROCS, DL_WORKERS = "2", "4"
SHARDS = D / "eval_cache" / "shards"
CAMPAIGN_LOG = D / "logs" / "pm2_campaign_public.log"
INGEST_LOGS = D / "logs" / "ingest_public"
# perm_from_embed.py: the checkpoint is not a permutation of T1's basis (another lineage); retrying cannot help
PERMANENT = ("cosine match is not a bijection", "column cosine too low", "may not share this hidden basis", "embed shape")
BACKOFF_S = (900, 1800, 3600, 7200, 21600)


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, obj, indent=None):
    """Atomic: readers (merge.sh, public_merge.py, the dashboard) never see a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=indent))
    tmp.replace(path)


def name_of(comp, reign):
    return f"M{int(reign):02d}" if comp == "main" else f"{PREFIX[comp]}{int(reign)}"


def chains(dry=False):
    """public_models.json rows from the live chains (existing names kept), the reigning king digest per competition,
    and the competitions whose dashboard could not be read this pass."""
    rows = read_json(PUBLIC, [])
    by_digest = {r["digest"]: r for r in rows}
    reigning, unreachable, changed = {}, [], False
    for comp in kr.COMPS:
        try:
            board = kr.board(comp)
        except kr.KingNotReady:
            unreachable.append(comp)
            continue
        king = board.get("king") or {}
        digest = king.get("king_digest") or king.get("model_digest")  # same field order as scripts/box/kings.py live()
        if digest:
            reigning[comp] = digest
        for k in board.get("king_chain") or []:
            d = k.get("model_digest")
            if not d:
                continue
            if d in by_digest:
                if k.get("ended_at") and by_digest[d].get("ended_at") != k["ended_at"]:
                    by_digest[d]["ended_at"], changed = k["ended_at"], True
                continue
            r = {"name": name_of(comp, k.get("reign_number") or 0), "digest": d, "comp": comp, "reign": k.get("reign_number"),
                 "repo": k.get("model_repo"), "coldkey": k.get("coldkey"), "uid": k.get("uid"), "crowned_at": k.get("crowned_at"),
                 "ended_at": k.get("ended_at")}
            rows.append(r)
            by_digest[d], changed = r, True
            log(f"new public king {r['name']}: {r['repo']} ({comp} #{r['reign']}, {d[:12]}, crowned {r['crowned_at']})")
    if changed and not dry:
        write_json(PUBLIC, rows, indent=1)
    return rows, reigning, unreachable


def on_disk(name, digest):
    m = read_json(MODELS, {})
    for path in ([m[name]] if name in m else []) + [p for p in m.values() if Path(p).name == digest or digest[:8] in Path(p).name]:
        if Path(path, "INGESTED").exists():
            return path
    return None


def archived(name):
    return name in read_json(COLD_REG, {})


# ------------------------------------------------------------------ failed ingests
def classify(logf):
    """(kind, reason) of a failed ingest from the tail of its log: 'permanent' for a basis mismatch, else 'transient'."""
    try:
        tail = Path(logf).read_text(errors="replace")[-6000:]
    except OSError:
        return "transient", "no ingest log"
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if any(sig in ln for sig in PERMANENT):
            return "permanent", ln[-300:]
    return "transient", (lines[-1][-300:] if lines else "empty ingest log")


def record_failure(fails, r, kind, reason, logf, source="ingest", when=None):
    when = time.time() if when is None else when
    f = fails.get(r["digest"]) or {"name": r["name"], "digest": r["digest"], "attempts": 0,
                                   "first": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when))}
    f.update(name=r["name"], kind=kind, reason=reason, log=str(logf), source=source, attempts=f["attempts"] + 1,
             last=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(when)))
    raw = D / "models" / r["digest"]
    if raw.exists():
        f["raw"] = str(raw)  # the downloaded, still permuted checkpoint (kept: deleting model dirs is the user's call)
    if kind == "transient":
        f["next_retry"] = when + BACKOFF_S[min(f["attempts"], len(BACKOFF_S)) - 1]
    else:
        f.pop("next_retry", None)
    fails[r["digest"]] = f
    return f


def seed_from_logs(rows, fails):
    """A king whose last ingest log already ends in a basis mismatch is recorded without another 2.5-min attempt."""
    seeded = []
    for r in rows:
        if r["digest"] in fails or on_disk(r["name"], r["digest"]):
            continue
        logf = INGEST_LOGS / f"{r['name']}.log"
        if not logf.exists():
            continue
        kind, reason = classify(logf)
        if kind == "permanent":
            record_failure(fails, r, kind, reason, logf, source="earlier log", when=logf.stat().st_mtime)
            seeded.append(r["name"])
    return seeded


def blocked(r, fails, now=None):
    """'failed' (permanent), 'backoff' (transient, not due yet) or None."""
    f = fails.get(r["digest"])
    if not f:
        return None
    if f.get("kind") == "permanent":
        return "failed"
    return "backoff" if (now or time.time()) < f.get("next_retry", 0) else None


# ------------------------------------------------------------------ the eval sample comes first
def _recent_scoring_in_log(within_s: float = 300.0) -> bool:
    """True while campaign43 is actively scoring (even across a quick restart with cached samples)."""
    try:
        data = CAMPAIGN_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    now = time.time()
    for ln in reversed(data[-50000:].splitlines()):
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
        if not m:
            continue
        age = now - dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        if age > within_s:
            break
        if "scoring item" in ln or "total scored" in ln:
            return True
    return False


def sample_download_active() -> bool:
    """campaign43 is up and has not reached scoring yet, so it is still fetching the eval sample."""
    if _recent_scoring_in_log():
        return False
    line = subprocess.run(["pgrep", "-f", "[s]cripts/merge/campaign43.py"], capture_output=True, text=True).stdout.strip()
    if not line:
        return False
    pid = line.split()[0]
    try:
        started = os.stat(f"/proc/{pid}").st_mtime
    except OSError:
        return False
    try:
        data = CAMPAIGN_LOG.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return True
    raw = data[-20000:]
    fresh = []
    for ln in raw.splitlines():
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
        if not m:
            continue
        if dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() >= started - 2:
            fresh.append(ln)
    text = "\n".join(fresh)
    if "scoring item" in text or "total scored" in text:
        return False
    return True


def wait_for_sample() -> None:
    announced = False
    while sample_download_active():
        if not announced:
            log("eval sample is still downloading; king downloads wait (merge and eval come first)")
            announced = True
        time.sleep(20)
    if announced:
        log("eval sample is ready; king downloads resume, one model, 8 connections")


def ingest(r, fails):
    """Download + un-permute + register one king; a failure is recorded (permanent basis mismatch or transient)."""
    wait_for_sample()
    logf = INGEST_LOGS / f"{r['name']}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    env = {**os.environ, "DL_PROCS": DL_PROCS, "DL_WORKERS": DL_WORKERS}
    with open(logf, "w") as fh:
        rc = subprocess.run(["bash", str(REPO / "scripts/merge/ingest_king_fast.sh"), r["digest"], r["name"]], cwd=REPO,
                            stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    mins = (time.time() - t0) / 60
    if rc == 0 and on_disk(r["name"], r["digest"]):
        fails.pop(r["digest"], None)
        write_json(FAILED, fails, indent=1)
        log(f"{r['name']} ({r['repo']}, {r['digest'][:12]}): ingested in {mins:.1f} min")
        return True
    kind, reason = classify(logf) if rc != 0 else ("transient", "ingest exited 0 but left no INGESTED marker")
    f = record_failure(fails, r, kind, reason, logf)
    write_json(FAILED, fails, indent=1)
    then = ("skipped for good (not in T1's hidden basis; --retry-failed to try again)" if kind == "permanent"
            else "retry after " + time.strftime("%H:%M", time.gmtime(f["next_retry"])))
    log(f"{r['name']} ({r['digest'][:12]}): FAILED rc={rc} after {mins:.1f} min, attempt {f['attempts']}, {kind}: {reason} -> {then}")
    return False


# ------------------------------------------------------------------ reigning kings
def publish(rows, reigning, unreachable, dry=False):
    """public_kings.json = {comp: name} of every competition whose LIVE king is ingested (kings.py resolve());
    public_kings_live.json = the live digest / name / readiness per competition + "pending": {comp: digest}.
    A competition whose dashboard was unreachable this pass keeps its previous entries."""
    by_digest = {r["digest"]: r for r in rows}
    old = read_json(KINGS, {})
    old_live = read_json(LIVE, {})
    flat = {k: v for k, v in old.items() if k not in kr.COMPS}  # keys other tools added stay
    live = {"updated": now_iso(), "kings": {}, "pending": {}}
    for comp in kr.COMPS:
        d = reigning.get(comp)
        if d is None:
            if comp in unreachable:
                if isinstance(old.get(comp), str):
                    flat[comp] = old[comp]
                if comp in (old_live.get("kings") or {}):
                    live["kings"][comp] = {**old_live["kings"][comp], "unreachable": now_iso()}
                if comp in (old_live.get("pending") or {}):
                    live["pending"][comp] = old_live["pending"][comp]
            continue
        row = by_digest.get(d) or {}
        entry = {"digest": d, "name": row.get("name"), "reign": row.get("reign"), "repo": row.get("repo")}
        try:
            entry["name"], entry["path"] = kr.resolve(d)
            entry["ingested"] = True
            flat[comp] = entry["name"]
        except kr.KingNotReady as exc:
            entry.update(ingested=False, why=str(exc))
            live["pending"][comp] = d
        live["kings"][comp] = entry
    if not dry:
        if flat != old:
            write_json(KINGS, flat)
        write_json(LIVE, live, indent=1)
    if flat != old or live["pending"] != (old_live.get("pending") or {}) or dry:
        pend = ", ".join(f"{c} {d[:12]} ({live['kings'][c].get('name') or '?'} not ingested)" for c, d in live["pending"].items())
        log(f"{'would publish' if dry else 'reigning kings'}: {json.dumps({c: v for c, v in flat.items() if c in kr.COMPS})}"
            + (f"; pending: {pend}" if pend else "") + ("" if dry else f" -> {KINGS}"))
    return flat, live


def fetch_rank(r, reigning_digests):
    """T1, then the reigning kings, then MAIN #40→#35 (the step chain that previously cut loss; each new download completes a
    gradient M_n − M_{n-1}), then the rest of #30–#45, then other competitions, then ancient MAIN."""
    if r["digest"] == T1:
        return (0, 0)
    if r["digest"] in reigning_digests:
        return (1, 0)
    m = re.fullmatch(r"(M|TX|CD|MA)(\d+)", r["name"])
    if m and m.group(1) == "M":
        n = int(m.group(2))
        if 35 <= n <= 40:
            return (2, -n)
        if 30 <= n <= 45:
            return (3, -n)
    if r.get("comp") != "main":
        return (4, -(r.get("reign") or 0))
    return (5, -(r.get("reign") or 0))


_last = {"summary": None, "t": 0.0}


def one_pass(top, reserve_gb, dry=False):
    rows, reigning, unreachable = chains(dry)
    by_digest = {r["digest"]: r for r in rows}
    latest = sorted(rows, key=lambda r: r.get("crowned_at") or "", reverse=True)[:top]
    want = {r["digest"]: r for r in latest}
    for d in list(reigning.values()) + [T1]:
        if d in by_digest:
            want[d] = by_digest[d]
    order = sorted(want.values(), key=lambda r: fetch_rank(r, set(reigning.values())))
    fails = read_json(FAILED, {})
    seeded = seed_from_logs(order, fails)
    if seeded and not dry:
        write_json(FAILED, fails, indent=1)
    state = {}
    for r in order:
        state[r["name"]] = ("disk" if on_disk(r["name"], r["digest"]) else "archived" if archived(r["name"])
                            else blocked(r, fails) or "todo")
    todo = [r for r in order if state[r["name"]] == "todo"]
    group = lambda s: [n for n, v in state.items() if v == s]  # noqa: E731
    failed, backoff = group("failed"), group("backoff")
    summary = (f"{len(want)} wanted (top {top} + reigning + T1): {len(group('disk'))} on disk, {len(group('archived'))} archived, "
               f"{len(failed)} failed for good" + (f" ({', '.join(failed[:6])}{' ...' if len(failed) > 6 else ''})" if failed else "")
               + f", {len(backoff)} in backoff" + (f" ({', '.join(backoff)})" if backoff else "")
               + f", {len(todo)} to download" + (f": {', '.join(r['name'] for r in todo[:8])}{' ...' if len(todo) > 8 else ''}" if todo else "")
               + (f"; dashboards unreachable: {', '.join(unreachable)}" if unreachable else "")
               + (f"; recorded from earlier logs as permanent failures: {', '.join(seeded)}" if seeded else ""))
    if dry or summary != _last["summary"] or time.time() - _last["t"] > 3600:
        log(("DRY " if dry else "pass: ") + summary)
        _last.update(summary=summary, t=time.time())
    if dry:
        for r in todo:
            log(f"DRY would download {r['name']} ({r['repo']}, {r['digest'][:12]}, {r['comp']} #{r['reign']})")
        publish(rows, reigning, unreachable, dry=True)
        return
    publish(rows, reigning, unreachable)
    if todo and todo[0]["digest"] == T1:  # the permutation reference first, alone
        ingest(todo.pop(0), fails)
        publish(rows, reigning, unreachable)
    batch = []
    for r in todo:
        free = shutil.disk_usage(D).free / 1e9
        if free - (len(batch) + 1) * 2 * MODEL_GB < reserve_gb:
            log(f"disk: {free:.0f} GB free, {reserve_gb} GB reserve -> {r['name']} and the rest wait for the next pass")
            break
        batch.append(r)
    # one model at a time no matter what --parallel says: the previous 6×8×16 stampede starved the eval sample
    for r in batch:
        if ingest(r, fails):
            publish(rows, reigning, unreachable)  # each reigning king is listed as soon as it is on disk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="keep the N most recently crowned public kings (plus reigning kings and T1)")
    ap.add_argument("--loop", action="store_true", help="keep watching for new kings")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--parallel", type=int, default=1, help="ignored; kings stay at one model so the eval sample keeps the network")
    ap.add_argument("--reserve-gb", type=float, default=400)
    ap.add_argument("--dry", action="store_true", help="one pass that downloads and writes nothing (prints the plan)")
    ap.add_argument("--retry-failed", action="store_true", help="forget permanent ingest failures (e.g. after a perm_from_embed fix)")
    args = ap.parse_args()
    if args.retry_failed and not args.dry:
        fails = read_json(FAILED, {})
        kept = {d: f for d, f in fails.items() if f.get("kind") != "permanent"}
        if len(kept) != len(fails):
            write_json(FAILED, kept, indent=1)
            log(f"--retry-failed: forgot {len(fails) - len(kept)} permanent failures")
    if args.dry:
        one_pass(args.top, args.reserve_gb, dry=True)
        return
    while True:
        wait_for_sample()
        one_pass(args.top, args.reserve_gb)
        if not args.loop:
            break
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
