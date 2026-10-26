#!/usr/bin/env python3
"""Chain LoRA runs until a checkpoint beats the king, then confirm and stop.

  chain_runs.py --start text_lora_d --king $T1 [--target 0.0026] [--max-runs 4] [--dry]

Every minute:
  * a scored checkpoint of the current run with screen mu >= --target (score_ckpts.py, first 2 400 sequences of the
    text sample) -> confirm: pause the trainer, duel the checkpoint vs the king on the FULL text samples tag:A and tag:B
    (30 000 sequences each). Confirmed = mu >= target and the validator's LCB > 0.002 on both -> stop training, write
    <run>/CONFIRMED.json, exit 0. Not confirmed -> resume training.
  * the run's `final` checkpoint scored -> next run (text_lora_e, _f, ...) from <run>/final on a fresh pool
    (text_cpt_v2, v3, ... built with eval/val/earlier-pool shards excluded), same recipe, its own scorer; dashboard follows.
  * stop (exit 3) if two consecutive runs gain < --min-gain each (start -> final on the screen): a plateau needs a human.
  * stop (exit 2) if the trainer dies without a final checkpoint; (exit 4) after --max-runs new runs.
Nothing is submitted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CKPT = Path("/data/teutonic/ckpt")
POOLS = Path("/data/teutonic/pools")
LOGS = Path("/data/teutonic/logs")
PY = str(REPO / ".venv-eval/bin/python")
RECIPE = ["--attn", "flex_attention", "--lr", "5e-5", "--warmup", "10", "--min-lr-frac", "0.5", "--steps", "200",
          "--eval-every", "20", "--save-every", "50"]
# global batch 1 024 sequences either way; bigger per-GPU batches amortize the base all-gathers, the last entry is the
# proven fallback (H100 80 GB: 8x16 then 4x32; H200 141 GB: 16x8, 8x16, 4x32). Set with --batches.
BATCHES = [("8", "16"), ("4", "32")]
MIN_TOK_S_K = 17.0  # batch 8 must beat batch 4's measured 16.7k tok/s at step 1, else it is thrashing -> fall back
EXTRA: list[str] = []  # extra trainer args, appended after RECIPE (later args win), e.g. a full-parameter recipe
NO_SCORER = False  # --no-scorer: no screen duels during training (each pauses the trainer); back up every checkpoint


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def scores(run: str) -> list[dict]:
    p = CKPT / run / "scores.json"
    return json.loads(p.read_text()) if p.exists() else []


def pm2_status(name: str) -> str | None:
    out = subprocess.run(["pm2", "jlist"], capture_output=True, text=True).stdout
    try:
        return {p["name"]: p["pm2_env"]["status"] for p in json.loads(out)}.get(name)
    except json.JSONDecodeError:
        return None


def pm2(*args: str) -> None:
    subprocess.run(["pm2", *args], capture_output=True)


def next_name(run: str) -> str:
    return run[:-1] + chr(ord(run[-1]) + 1)


_BUILDS: dict[int, subprocess.Popen] = {}


def ensure_pool(k: int, wait: bool = True) -> Path:
    pool = POOLS / f"text_cpt_v{k}"
    if (pool / "meta.json").exists():
        return pool
    if k in _BUILDS:  # a background build of this pool is already running: wait for it instead of starting another
        if wait and (_BUILDS.pop(k).wait() != 0 or not (pool / "meta.json").exists()):
            raise RuntimeError(f"pool build {pool.name} failed; see {LOGS}/pool_v{k}.log")
        return pool
    excludes = sorted(str(p) for p in POOLS.glob("text_cpt_v*") if (p / "meta.json").exists()) + [str(POOLS / "text_val_v1")]
    cmd = ["bash", str(REPO / "scripts/data/build_text_pool.sh"), pool.name, *excludes]
    with (LOGS / f"pool_v{k}.log").open("a") as h:
        proc = subprocess.Popen(cmd, stdout=h, stderr=subprocess.STDOUT)
        if not wait:
            _BUILDS[k] = proc
        else:
            if proc.wait() != 0 or not (pool / "meta.json").exists():
                raise RuntimeError(f"pool build {pool.name} failed; see {LOGS}/pool_v{k}.log")
    return pool


def duel(ckpt: Path, king: str, sample: str) -> dict:
    out = subprocess.run([PY, "scripts/eval/parity_eval.py", "--competition", "text", "duel", "--king", king,
                          "--challenger", str(ckpt), "--sample", sample], capture_output=True, text=True, cwd=REPO)
    with (LOGS / "chain_confirm.log").open("a") as h:
        h.write(out.stdout + out.stderr)
    m = re.search(r"^saved (.+\.json)$", out.stdout, re.M)
    if out.returncode != 0 or not m:
        raise RuntimeError(f"duel on {sample} failed ({out.returncode}); see {LOGS}/chain_confirm.log")
    v = json.loads(Path(m.group(1)).read_text())["results"][0]["verdict"]
    return {k: v.get(k) for k in ("mu_hat", "lcb", "accepted", "n", "se_iid", "se_shard_jackknife", "robust_lcb", "source_scores")}


def confirm(run: str, entry: dict, king: str, target: float) -> bool:
    out = CKPT / run
    ckpt = out / entry["ckpt"]
    log(f"{run}/{entry['ckpt']}: screen {entry['text_mu']:+.6f} >= {target} -> confirming on the full text samples")
    pm2("stop", f"score_{run}")
    training = pm2_status(run) == "online"
    if training:
        (out / "PAUSE").touch()
        t0 = time.time()
        while not (out / "PAUSED").exists() and pm2_status(run) == "online" and time.time() - t0 < 3600:
            time.sleep(10)
    try:
        result = {s: duel(ckpt, king, s) for s in ("tag:A", "tag:B")}
    except RuntimeError as exc:
        log(f"confirm failed to run: {exc}")
        result = None
    ok = bool(result) and all(r["mu_hat"] >= target and r["lcb"] > 0.002 for r in result.values())
    (out / f"confirm_{entry['ckpt']}.json").write_text(json.dumps({"entry": entry, "result": result, "confirmed": ok}, indent=1))
    if result:
        for s, r in result.items():
            log(f"  {s}: mu {r['mu_hat']:+.6f} lcb {r['lcb']:+.6f} accepted {r['accepted']} n {r['n']} splits {r['source_scores']}")
    if ok:
        pm2("delete", run)
        pm2("delete", f"score_{run}")
        (out / "PAUSE").unlink(missing_ok=True)
        (out / "CONFIRMED.json").write_text(json.dumps({"ckpt": str(ckpt), "screen": entry, "confirm": result}, indent=1))
        log(f"CONFIRMED: {ckpt} beats the king on both full samples; training stopped. Nothing submitted.")
        upload(run, entry, wait=True)
        return True
    (out / "PAUSE").unlink(missing_ok=True)
    pm2("restart", f"score_{run}")
    log("not confirmed; training resumes")
    return False


def first_step(run: str, trainer_log: Path, timeout: int = 1800) -> str:
    """Watch a fresh trainer until its step-1 line: 'ok', or why it failed (OOM, crash, thrashing, timeout)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        text = trainer_log.read_text(errors="replace") if trainer_log.exists() else ""
        if re.search(r"OutOfMemory|out of memory", text):
            return "out of memory"
        if re.search(r"Traceback|exitcode  :", text) or pm2_status(run) not in ("online", None):
            return "crashed"
        m = re.search(r"step 1/\d+: .*? ([\d.]+)k tok/s .*?mem ([\d.]+)/([\d.]+) GB", text)
        if m:
            tok_k, reserved = float(m.group(1)), float(m.group(3))
            if tok_k < MIN_TOK_S_K:
                return f"slow: {tok_k}k tok/s, {reserved} GB reserved"
            return "ok"
        time.sleep(15)
    return "timeout"


def best_entry(run: str) -> dict | None:
    """Best scored checkpoint of a run on the screen (the seeded step-0 entry belongs to the previous run)."""
    done = [s for s in scores(run) if s.get("status") == "done" and s.get("text_mu") is not None and s["ckpt"] != "step-0"]
    return max(done, key=lambda s: s["text_mu"], default=None)


BACKUP = CKPT / "backup_state.json"  # best checkpoint already on HF; read back after a restart of this driver
_backup: dict = {}  # the upload in flight: {"proc", "entry"}


def upload(run: str, entry: dict, wait: bool) -> subprocess.Popen | None:
    """Upload a checkpoint to the HF account (gated, random repo name; scripts/submit/hf_upload_ckpt.py).
    Returns the upload process (None if this checkpoint is already uploaded)."""
    ckpt = CKPT / run / entry["ckpt"]
    done = json.loads((CKPT / run / "uploads.json").read_text()) if (CKPT / run / "uploads.json").exists() else []
    if any(Path(u["ckpt"]) == ckpt.resolve() for u in done):
        return None
    inflight = _backup.get("proc")
    if inflight is not None and _backup["entry"]["ckpt_path"] == str(ckpt.resolve()):  # backup_best is on it already
        if wait:
            inflight.wait()
        return inflight
    logf = (LOGS / f"upload_{run}_{entry['ckpt']}.log").open("a")
    # lowest CPU / IO priority: hashing and reading 206 GB must not slow the trainer's data loading or its host threads
    proc = subprocess.Popen(["nice", "-n", "19", "ionice", "-c2", "-n7", PY, "scripts/submit/hf_upload_ckpt.py", "--ckpt", str(ckpt)],
                            cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    log(f"uploading {run}/{entry['ckpt']} (screen {entry['text_mu']:+.6f}) to Hugging Face; log {logf.name}")
    if wait:
        proc.wait()
        log(f"upload of {run}/{entry['ckpt']} finished with exit code {proc.returncode}")
    return proc


def backup_best(run: str) -> None:
    """Spot box (it can die any minute): send every new best scored checkpoint to HF as soon as it is scored, one
    upload at a time, instead of only the best of a run after the run ends. The floor is the run's step-0 score
    (its start checkpoint came from HF) and whatever was uploaded before."""
    proc = _backup.get("proc")
    if proc is not None:
        if proc.poll() is None:
            return
        if proc.returncode == 0:
            BACKUP.write_text(json.dumps(_backup["entry"], indent=1))
            log(f"backed up {_backup['entry']['run']}/{_backup['entry']['ckpt']} (screen {_backup['entry']['text_mu']:+.6f})")
        else:
            log(f"backup upload of {_backup['entry']['run']}/{_backup['entry']['ckpt']} failed ({proc.returncode}); retrying")
        _backup.clear()
        return
    best = best_entry(run)
    start = next((s for s in scores(run) if s["ckpt"] == "step-0" and s.get("text_mu") is not None), None)
    floor = max(json.loads(BACKUP.read_text())["text_mu"] if BACKUP.exists() else float("-inf"),
                start["text_mu"] if start else float("-inf"))
    if best is None or best["text_mu"] <= floor:
        return
    entry = dict(best, run=run, ckpt_path=str((CKPT / run / best["ckpt"]).resolve()))
    proc = upload(run, best, wait=False)
    if proc is None:  # already on HF (uploads.json)
        BACKUP.write_text(json.dumps(entry, indent=1))
    else:
        _backup.update(proc=proc, entry=entry)


def saved_ckpts(run: str) -> list[Path]:
    """Complete checkpoints of a run (the trainer writes SAVED after every shard, the index and the contract files)."""
    out = CKPT / run
    found = [p for p in out.glob("step-*") if p.is_dir() and not p.is_symlink() and (p / "SAVED").exists()]
    found.sort(key=lambda p: int(p.name.split("-")[1]))
    if (out / "final" / "SAVED").exists():
        found.append(out / "final")
    return found


def backup_latest(run: str) -> bool:
    """Spot box: keep the newest saved checkpoint on HF without waiting for a screen score, one upload at a time
    (a checkpoint saved while another uploads is skipped if a newer one exists by the time the upload ends).
    Returns True while an upload is running."""
    proc = _backup.get("proc")
    if proc is not None:
        if proc.poll() is None:
            return True
        name = _backup["entry"]["ckpt"]
        log(f"backup upload of {run}/{name} {'done' if proc.returncode == 0 else f'FAILED ({proc.returncode}); retrying'}")
        if proc.returncode == 0:
            BACKUP.write_text(json.dumps(_backup["entry"], indent=1))
        _backup.clear()
    ckpts = saved_ckpts(run)
    if not ckpts:
        return False
    newest = ckpts[-1]
    done = json.loads((CKPT / run / "uploads.json").read_text()) if (CKPT / run / "uploads.json").exists() else []
    if any(Path(u["ckpt"]) == newest.resolve() for u in done):
        return False
    entry = {"run": run, "ckpt": newest.name, "ckpt_path": str(newest.resolve()), "text_mu": None}
    proc = upload(run, dict(entry, text_mu=float("nan")), wait=False)
    if proc is not None:
        _backup.update(proc=proc, entry=entry)
        return True
    return False


def launch(run: str, init: Path, pool: Path, king: str, seed: dict | None = None) -> None:
    """Start <run> from checkpoint <init>. seed = the init checkpoint's screen entry (reused as step-0); without one,
    ckpt/<run>/step-0 links to <init> and the scorer duels it first, so every run has its own paired baseline."""
    out = CKPT / run
    out.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        (out / "scores.json").write_text(json.dumps([dict(seed, ckpt="step-0", note=f"= {init} (start of {run})")], indent=1))
    elif not (out / "step-0").exists():
        (out / "step-0").symlink_to(init.resolve())
    for i, (batch, accum) in enumerate(BATCHES):
        trainer_log = LOGS / f"pm2_{run}.out.log"
        trainer_log.unlink(missing_ok=True)
        subprocess.run(["bash", "scripts/train/pm2_cpt_lora.sh", run, "--init", str(init), "--pool", str(pool),
                        *RECIPE, *EXTRA, "--batch", batch, "--accum", accum], cwd=REPO, capture_output=True)
        if i == len(BATCHES) - 1:
            log(f"{run}: batch {batch} x accum {accum}")
            break
        verdict = first_step(run, trainer_log)
        if verdict == "ok":
            log(f"{run}: batch {batch} x accum {accum} accepted at step 1")
            break
        log(f"{run}: batch {batch} rejected ({verdict}); falling back to the next batch size")
        pm2("delete", run)
        time.sleep(20)
    pm2("delete", f"score_{run}")
    if NO_SCORER:
        log(f"{run}: no scorer during training (--no-scorer); every saved checkpoint is backed up to HF instead")
    else:
        subprocess.run(["pm2", "start", "bash", "--name", f"score_{run}", "--no-autorestart", "--time", "--merge-logs",
                    "--output", str(LOGS / f"pm2_score_{run}.out.log"), "--error", str(LOGS / f"pm2_score_{run}.err.log"), "--",
                    "-c", f"source /data/teutonic/env.sh && exec python scripts/train/score_ckpts.py --run {out} --king {king} "
                          f"--limit 2400 --no-mix --dashboard-args '--steps 200'"], cwd=REPO, capture_output=True)
    render = LOGS / "render_dashboard.sh"
    if render.exists():
        render.write_text(re.sub(r"^RUN=.*$", f"RUN={out}", render.read_text(), flags=re.M))
        pm2("restart", "dashboard_render")
    pm2("save")
    log(f"launched {run} from {init} on {pool.name} (pm2 {run}, scorer score_{run})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="current run name, e.g. text_lora_d")
    ap.add_argument("--king", required=True)
    ap.add_argument("--target", type=float, default=0.0026)
    ap.add_argument("--min-gain", type=float, default=0.0001)
    ap.add_argument("--max-runs", type=int, default=4)
    ap.add_argument("--first-pool", type=int, default=2, help="pool index for the first new run (text_cpt_v<k>)")
    ap.add_argument("--batches", default=None, help="per-GPU batch x accum to try in order, e.g. 16x8,8x16,4x32 (global 1 024)")
    ap.add_argument("--extra-args", default="", help="extra cpt_fsdp.py args (override the LoRA recipe), quoted string")
    ap.add_argument("--min-tok-s", type=float, default=None, help="step-1 throughput (k tok/s) a bigger batch must reach; 0 = any")
    ap.add_argument("--init-ckpt", type=Path, default=None,
                    help="fresh box: launch --start from this checkpoint on pool text_cpt_v<--first-pool> before chaining")
    ap.add_argument("--dry", action="store_true", help="print the decision for the current state and exit")
    ap.add_argument("--no-scorer", action="store_true",
                    help="no screen duels while training (they pause it): back up every saved checkpoint to HF, stop after the "
                         "final checkpoint is uploaded; score the checkpoints afterwards")
    args = ap.parse_args()
    global BATCHES, EXTRA, MIN_TOK_S_K, NO_SCORER
    NO_SCORER = args.no_scorer
    EXTRA = args.extra_args.split()
    if args.min_tok_s is not None:
        MIN_TOK_S_K = args.min_tok_s
    if args.batches:
        BATCHES = [tuple(b.split("x")) for b in args.batches.split(",")]
        assert len({int(b) * int(a) for b, a in BATCHES}) == 1, "every batch x accum must give the same global batch"
    run, k, launched, gains, tried = args.start, args.first_pool, 0, [], set()
    if args.init_ckpt and not args.dry and pm2_status(run) is None and not (CKPT / run / "train_log.jsonl").exists():
        launch(run, args.init_ckpt, ensure_pool(k), args.king)
        k += 1
        ensure_pool(k, wait=False)
    while True:
        if NO_SCORER and not args.dry:
            uploading = backup_latest(run)
            if pm2_status(run) not in ("online", None):
                if (CKPT / run / "final" / "SAVED").exists():
                    if not uploading and not backup_latest(run):
                        log(f"{run} finished and its final checkpoint is on HF; score the checkpoints, then start the next run")
                        sys.exit(0)
                else:
                    log(f"trainer {run} is {pm2_status(run)} without a final checkpoint; stopping the chain (resume from the "
                        f"newest step-N: {[p.name for p in saved_ckpts(run)][-1:]})")
                    sys.exit(2)
            time.sleep(60)
            continue
        if not args.dry:
            backup_best(run)
        done = [s for s in scores(run) if s.get("status") == "done" and s.get("text_mu") is not None]
        hits = [s for s in done if s["ckpt"] != "step-0" and s["text_mu"] >= args.target and s["ckpt"] not in tried]
        final = next((s for s in done if s["ckpt"] == "final"), None)
        if args.dry:
            best = max(done, key=lambda s: s["text_mu"], default=None)
            log(f"{run}: {len(done)} scored, best {best and best['ckpt']} {best and best['text_mu']:+.6f}, "
                f"hits {[h['ckpt'] for h in hits]}, final scored {final is not None}, trainer {pm2_status(run)}")
            return
        if hits:
            entry = max(hits, key=lambda s: s["text_mu"])
            tried.add(entry["ckpt"])
            if confirm(run, entry, args.king, args.target):
                return
            continue
        if final:
            start = next((s for s in done if s["ckpt"] == "step-0"), None)
            gain = final["text_mu"] - start["text_mu"] if start else float("nan")
            gains.append(gain)
            log(f"{run} finished: screen {start and start['text_mu']:+.6f} -> {final['text_mu']:+.6f} (gain {gain:+.6f})")
            best = best_entry(run)
            if len(gains) >= 2 and all(g < args.min_gain for g in gains[-2:]):
                log(f"PLATEAU: two runs in a row gained < {args.min_gain}; stopping the chain for a human decision")
                if best:
                    upload(run, best, wait=True)
                sys.exit(3)
            if launched >= args.max_runs:
                log(f"max runs ({args.max_runs}) reached; stopping")
                if best:
                    upload(run, best, wait=True)
                sys.exit(4)
            pm2("delete", f"score_{run}")
            pool = ensure_pool(k)
            nxt = next_name(run)
            launch(nxt, CKPT / run / "final", pool, args.king, seed=final)
            run, k, launched = nxt, k + 1, launched + 1
            ensure_pool(k, wait=False)  # build the following pool while this run trains
            continue
        if pm2_status(run) not in ("online", None) and not (CKPT / run / "final").exists():
            log(f"trainer {run} is {pm2_status(run)} without a final checkpoint; stopping the chain")
            sys.exit(2)
        time.sleep(60)


if __name__ == "__main__":
    main()
