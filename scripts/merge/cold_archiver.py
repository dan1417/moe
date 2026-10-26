#!/usr/bin/env python3
"""Cold tier on the 13 TB attached volume (user 2026-10-12: "you can use it! from now, use it!").

/root is a gocryptfs view of a 13 TB XFS volume: ~0.7-1 GB/s, vs ~36 GB/s on the local NVMe /data. Too slow for merge loads
(every ingredient is re-read per candidate), big enough for every public king (54 x 206 GB). So models leave /data by MOVING
here instead of being deleted, and come back with --promote when a merge needs them.

  daemon   python scripts/merge/cold_archiver.py                  (pm2 cold_archiver; works through the queue)
  enqueue  append {"name": ..., "src": ...} to /data/teutonic/search/cold_queue.jsonl  (public_merge.py does this)
  promote  python scripts/merge/cold_archiver.py --promote M30    (copy back to /data/teutonic/models/<name>_cold, register)

Each move: copy every file with 8 parallel streams into <dst>.tmp, check file set and sizes, rename, record in
/root/teutonic_cold/registry.json, then remove the /data copy. Low CPU / IO priority (training and scoring come first).

Never moved (checked before the copy and again right before the /data copy is removed; a model that became busy during
the copy keeps its /data copy until it is free, registry "src_removed": false):
  * the reigning kings (search/public_kings.json names -> models.json, public_kings_live.json paths) and T1;
  * anything a running process has open or mapped, or names on its command line (a path, a models.json name such as
    campaign43's --king main=M45:0.003, or a recipe JSON whose terms are read): every recipe being loaded or scored;
  * models scored in the last LEASE_S seconds (eval_cache/results/*_duel.json; atime is useless on relatime /data):
    the next chunk or confirm run of that campaign may load them again;
  * the models of search/public.json's current best recipes.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path("/data/teutonic")
REPO = Path(__file__).resolve().parents[2]
COLD = Path("/root/teutonic_cold")
REG = COLD / "registry.json"
QUEUE = D / "search" / "cold_queue.jsonl"
MODELS_JSON = REPO / "configs/merge/k41/models.json"
KINGS = D / "search" / "public_kings.json"
KINGS_LIVE = D / "search" / "public_kings_live.json"
PUBLIC_STATE = D / "search" / "public.json"
RESULTS = D / "eval_cache" / "results"
LEASE_S = 3600


def log(msg):
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def registry():
    try:
        return json.loads(REG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_registry(reg):
    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=1))
    tmp.replace(REG)


def copy_tree(src: Path, dst: Path, workers: int = 8):
    files = [p for p in src.rglob("*") if p.is_file()]
    for p in files:
        (dst / p.relative_to(src)).parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda p: shutil.copyfile(p, dst / p.relative_to(src)), files))
    bad = [p for p in files if (dst / p.relative_to(src)).stat().st_size != p.stat().st_size]
    if bad:
        raise RuntimeError(f"size mismatch after copy: {[str(b) for b in bad[:3]]}")
    return sum(p.stat().st_size for p in files), len(files)


def _read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _norm(p) -> str:
    try:
        return str(Path(p).resolve())
    except (OSError, RuntimeError):
        return str(p)


def _recipe_models(path: str) -> list:
    """Model paths of a recipe JSON (terms[].model); [] for anything else."""
    try:
        if os.path.getsize(path) > 2 << 20:
            return []
        body = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return []
    terms = body.get("terms") if isinstance(body, dict) else None
    return [t["model"] for t in terms or [] if isinstance(t, dict) and isinstance(t.get("model"), str)]


def protected() -> dict:
    """{resolved path: why} of everything that must stay on /data right now (see the module docstring)."""
    out, models = {}, _read_json(MODELS_JSON, {})

    def add(path, why):
        if path:
            out.setdefault(_norm(path), why)

    for comp, name in _read_json(KINGS, {}).items():
        if isinstance(name, str) and name in models:
            add(models[name], f"reigning {comp} king {name}")
    for comp, k in (_read_json(KINGS_LIVE, {}).get("kings") or {}).items():
        if isinstance(k, dict):
            add(k.get("path"), f"live {comp} king {k.get('name')}")
            if k.get("digest"):
                add(D / "models" / k["digest"], f"live {comp} king {k.get('name')} (raw download)")
    add(models.get("T1"), "T1 (the basis every hidden permutation is recovered against)")
    for c, b in (_read_json(PUBLIC_STATE, {}).get("best") or {}).items():
        for m in _recipe_models(b.get("recipe", "")) if isinstance(b, dict) else []:
            add(m, f"public_merge's current best {c.upper()} recipe")
    me = os.getpid()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        if int(pid) == me:
            continue
        base = f"/proc/{pid}"
        try:
            argv = [a.decode(errors="replace") for a in Path(base, "cmdline").read_bytes().split(b"\0") if a]
        except OSError:
            continue
        what = f"pid {pid} ({' '.join(argv)[:80]})"
        for tok in re.split(r"[\s=,:'\"]+", " ".join(argv)):
            if tok.startswith("/"):
                add(tok, what)
                if tok.endswith(".json"):
                    for m in _recipe_models(tok):
                        add(m, f"recipe {Path(tok).name} of {what}")
            elif tok in models and not tok.isdigit():
                add(models[tok], f"{tok} named by {what}")
        try:
            for fd in os.listdir(f"{base}/fd"):
                try:
                    add(os.readlink(f"{base}/fd/{fd}"), f"open in {what}")
                except OSError:
                    pass
        except OSError:
            pass
        try:
            with open(f"{base}/maps") as fh:
                for line in fh:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith("/"):
                        add(parts[5].strip().removesuffix(" (deleted)"), f"mapped by {what}")
        except OSError:
            pass
    cutoff = time.time() - LEASE_S
    for f in RESULTS.glob("*_duel.json"):
        try:
            if f.stat().st_mtime < cutoff:
                continue
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for r in data.get("results", []):
            for side in ("challenger", "king"):
                for t in (r.get(side) or {}).get("terms") or []:
                    add(t.get("model"), f"scored {(time.time() - f.stat().st_mtime) / 60:.0f} min ago ({f.name})")
    return out


def busy(src, prot=None):
    """Why `src` (a model directory) must stay on /data now, or None."""
    prot = protected() if prot is None else prot
    s = _norm(src)
    for path, why in prot.items():
        if path == s or path.startswith(s + "/"):
            return why
    return None


def remove_src(name: str, reg: dict, prot=None) -> bool:
    """Remove the /data copy of an archived model unless it is in use; registry src_removed records it."""
    e = reg[name]
    why = busy(e["from"], prot)
    if why:
        return False
    shutil.rmtree(e["from"], ignore_errors=True)
    e["src_removed"] = not Path(e["from"]).exists()
    save_registry(reg)
    return e["src_removed"]


def move(name: str, src: Path, kind: str = "models"):
    dst = COLD / kind / name
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    nbytes, nfiles = copy_tree(src, tmp)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    reg = registry()
    reg[name] = {"path": str(dst), "from": str(src), "bytes": nbytes, "files": nfiles, "src_removed": False,
                 "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    save_registry(reg)
    rate = f"{nbytes / 1e9:.1f} GB, {nfiles} files -> {dst} in {(time.time() - t0) / 60:.1f} min ({nbytes / 1e9 / max(1, time.time() - t0):.2f} GB/s)"
    prot = protected()  # it may have become busy during the copy
    if remove_src(name, reg, prot):
        log(f"{name}: {rate}; {src} removed")
    else:
        log(f"{name}: {rate}; {src} KEPT while in use ({busy(src, prot)}), removed once it is free")


def promote(name: str):
    reg = registry()
    if name not in reg:
        raise SystemExit(f"{name} is not in the cold registry")
    dst = D / "models" / f"{name}_cold"
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    nbytes, _ = copy_tree(Path(reg[name]["path"]), tmp)
    tmp.rename(dst)
    (dst / "INGESTED").touch()
    subprocess.run(["flock", str(D / "models_json.lock"), "python3", "-c",
                    "import json,sys; p=sys.argv[1]; m=json.load(open(p)); m[sys.argv[2]]=sys.argv[3]; json.dump(m, open(p,'w'), indent=1)",
                    str(MODELS_JSON), name, str(dst)], check=True)
    log(f"{name}: promoted back to {dst} ({nbytes / 1e9:.1f} GB) and registered")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", default="")
    args = ap.parse_args()
    COLD.mkdir(parents=True, exist_ok=True)
    if args.promote:
        for n in args.promote.split(","):
            promote(n)
        return
    try:
        os.nice(10)
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False)
    except OSError:
        pass
    log(f"cold archiver: {COLD} ({shutil.disk_usage(COLD).free / 1e12:.1f} TB free), queue {QUEUE}")
    said = {}
    while True:
        reg = registry()
        jobs = []
        if QUEUE.exists():
            for line in QUEUE.read_text().splitlines():
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if j["name"] not in reg and Path(j["src"]).exists():
                    jobs.append(j)
        # archived while busy: the /data copy goes once nothing uses it ("src_removed" absent = removed by an older version)
        leftover = [n for n, e in reg.items() if e.get("src_removed") is False and Path(e["from"]).exists()]
        moved = False
        if jobs or leftover:
            prot = protected()
            for n in leftover:
                if remove_src(n, reg, prot):
                    log(f"{n}: {reg[n]['from']} removed (the cold copy {reg[n]['path']} was made earlier)")
            for j in jobs:
                why = busy(j["src"], prot)
                if why:
                    if said.get(j["name"]) != why or time.time() - said.get(j["name"] + "@", 0) > 3600:
                        log(f"{j['name']}: in use ({why}); not moved yet")
                        said[j["name"]], said[j["name"] + "@"] = why, time.time()
                    continue
                try:
                    move(j["name"], Path(j["src"]), j.get("kind", "models"))
                except Exception as exc:  # noqa: BLE001 - keep the daemon alive; the job stays queued and is retried
                    log(f"{j['name']}: move failed ({exc}); retry in 5 min")
                    time.sleep(300)
                moved = True
                break  # one model per pass; the queue and the protected set are re-read
        if not moved:
            time.sleep(30)


if __name__ == "__main__":
    main()
