#!/usr/bin/env python3
"""Ingest every public Teutonic king (configs/merge/public/public_models.json) into the k41 merge basis.

User 2026-10-12: "use all of the public models for model merging, previous kings too". 53 kings x 206 GB do not fit the
local disk, so this runs ingest_king41.sh (download -> hidden-perm recovery vs T1 -> un-permute -> register in models.json)
with N parallel slots and a disk budget: a slot starts only when free space covers a raw + un-permuted copy plus a
reserve for training checkpoints. The screen (public_screen.py) deletes old kings it has finished with, which frees room
for the next ones. Kings already on disk are registered under their public alias (M44, TX4, ...).

  python scripts/merge/ingest_public.py [--slots 4] [--reserve-gb 500] [--only M43,M42]
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = Path("/data/teutonic")
PUBLIC = REPO / "configs/merge/public/public_models.json"
MODELS = REPO / "configs/merge/k41/models.json"
LOGS = D / "logs/ingest_public"
# near family first (descendants of T1 / #40, all four competitions), then the older kings newest first, genesis last
# TX1 (= T1) first: every other king's hidden permutation is recovered against it (ingest_king41.sh uses $T1); then the
# reigning kings the campaign scores against, then the rest of the near family
TIER1 = ["TX1", "M45", "M44", "TX5", "TX4", "CD4", "MA1", "M43", "M42", "M41", "M40", "TX2", "TX3", "CD1", "CD2", "CD3"]
MODEL_GB = 221


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def models():
    return json.loads(MODELS.read_text())


def register(name, path):
    subprocess.run(["flock", str(D / "models_json.lock"), "python3", "-c",
                    "import json,sys; p=sys.argv[1]; m=json.load(open(p)); m[sys.argv[2]]=sys.argv[3]; json.dump(m, open(p,'w'), indent=1)",
                    str(MODELS), name, path], check=True)


def present(name, digest):
    """Path of this king on disk if fully ingested (INGESTED marker), under its alias or any earlier name, or its copy on
    the cold volume (/root/teutonic_cold, cold_archiver.py), else None."""
    try:
        cold = json.loads(Path("/root/teutonic_cold/registry.json").read_text())
    except (OSError, json.JSONDecodeError):
        cold = {}
    if name in cold:
        return cold[name]["path"]
    m = models()
    cands = [m[name]] if name in m else []
    cands += [path for path in m.values() if Path(path).name == digest or digest[:8] in Path(path).name]
    for path in cands:
        if Path(path, "INGESTED").exists():
            return path
    return None


def in_progress(digest):
    return bool(subprocess.run(["pgrep", "-f", f"ingest_king41.sh {digest}"], capture_output=True).stdout.strip())


def free_gb():
    return shutil.disk_usage(D).free / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--reserve-gb", type=float, default=500)
    ap.add_argument("--only", default="")
    ap.add_argument("--dl-workers", type=int, default=48)
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    rows = {r["name"]: r for r in json.loads(PUBLIC.read_text())}
    order = [n for n in TIER1 if n in rows] + sorted((n for n in rows if n.startswith("M") and n not in TIER1), reverse=True)
    order += [n for n in rows if n not in order]
    if args.only:
        order = [n for n in args.only.split(",") if n]
    running = {}
    for name in order:
        r = rows[name]
        have = present(name, r["digest"])
        if not have and in_progress(r["digest"]):
            log(f"{name} is being ingested by an earlier driver; skipped")
            continue
        if have:
            if models().get(name) != have and not have.startswith("/root/"):
                register(name, have)
            log(f"{name} already on disk: {have}")
            continue
        while True:
            for n, p in list(running.items()):
                if p.poll() is not None:
                    log(f"{n} finished rc={p.returncode}")
                    running.pop(n)
            # a slot needs room for the raw download plus the un-permuted copy
            # room for this download plus an un-permuted copy, and for the rest of every running download
            need = args.reserve_gb + 2 * MODEL_GB + len(running) * MODEL_GB
            if len(running) < args.slots and free_gb() > need:
                break
            time.sleep(30)
        log(f"start {name} ({r['repo']}, {r['comp']} #{r['reign']}, {r['digest'][:12]}), free {free_gb():.0f} GB")
        fh = open(LOGS / f"{name}.log", "w")
        running[name] = subprocess.Popen(["bash", str(REPO / "scripts/merge/ingest_king41.sh"), r["digest"], name],
                                         cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                                         env={**__import__("os").environ, "DL_WORKERS": str(args.dl_workers)})
        time.sleep(5)
    for n, p in running.items():
        p.wait()
        log(f"{n} finished rc={p.returncode}")
    log("all done")


if __name__ == "__main__":
    main()
