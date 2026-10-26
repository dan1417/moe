#!/usr/bin/env python3
"""Back up a run's best exports to a private HF repo as bit-exact deltas (the pod can die any time).

Every export the trainer writes is matched with its val record in history.jsonl (EMA exports ->
``ema_vs_king``, raw -> ``vs_king``). When one beats every earlier export of the watched runs, it is
pinned on the RAM disk (``KEEP``, so ``export_keep`` pruning skips it), packed against the public start
body with ``scripts/data/ckpt_delta.py`` and uploaded to ``--repo``/<run>_<export>, and ``BEST.json`` in
the repo is updated. Older backups beyond ``--keep`` are deleted from the repo (private quota: 100 GB).

  python scripts/train/backup_watch.py --run line_c1 --repo touristAI/teutonic-cpt-backup
  (under pm2: pm2 start ... -- scripts/train/backup_watch.py ...)
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = Path("/workspace/teutonic/runs")
log = logging.getLogger("backup_watch")


def export_complete(path: Path) -> bool:
    last = path / "modeling_mimo_v2.py"  # the trainer copies the contract files last
    return (path / "model.safetensors.index.json").exists() and last.exists() and time.time() - last.stat().st_mtime > 20


def score_of(run: Path, export: Path) -> float | None:
    step = int(export.name[4:10])
    key = "ema_vs_king" if export.name.endswith("_ema") else "vs_king"
    history = run / "history.jsonl"
    if not history.exists():
        return None
    for line in history.read_text().splitlines():
        record = json.loads(line) if line.strip() else {}
        if record.get("step") == step and key in record:
            return float(record[key]["mu_hat"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="run name(s) under /workspace/teutonic/runs")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--base", default="/dev/shm/teutonic/start")
    parser.add_argument("--keep", type=int, default=3, help="backups kept in the repo")
    parser.add_argument("--min-score", type=float, default=0.0017, help="ignore exports below this vs_king")
    parser.add_argument("--poll", type=int, default=60)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    state_path = RUNS / f"backup_state_{args.repo.replace('/', '_')}.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"best": None, "uploaded": [], "seen": []}

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    log.info("watching %s -> %s (best so far: %s)", args.run, args.repo, state["best"])
    while True:
        for name in args.run:
            run = RUNS / name
            config = run / "config.yaml"
            if not config.exists():
                continue
            export_dir = Path(yaml.safe_load(config.read_text()).get("export_dir") or run / "exports")
            for export in sorted(export_dir.glob("step*_*")) if export_dir.exists() else []:
                if str(export) in state["seen"] or not export_complete(export):
                    continue
                score = score_of(run, export)
                if score is None:
                    continue  # val record not written yet (or no eval at this step): retry next poll
                state["seen"].append(str(export))
                best = state["best"]
                log.info("%s/%s: vs_king %+.6f (best %s)", name, export.name, score, f"{best['score']:+.6f}" if best else None)
                if score < args.min_score or (best and score <= best["score"]):
                    continue
                (export / "KEEP").touch()
                if best and Path(best["path"]).exists() and Path(best["path"]) != export:
                    (Path(best["path"]) / "KEEP").unlink(missing_ok=True)  # the old best may now be pruned
                folder = f"{name}_{export.name}"
                note = {"run": name, "export": export.name, "vs_king": score, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
                result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts/data/ckpt_delta.py"), "push", "--ckpt", str(export),
                                         "--base", args.base, "--repo", args.repo, "--name", folder, "--note", json.dumps(note)])
                if result.returncode != 0:
                    log.error("push of %s failed (exit %d); will not retry automatically", folder, result.returncode)
                    continue
                state["best"] = {**note, "path": str(export), "score": score, "folder": folder}
                state["uploaded"].append(folder)
                api.upload_file(path_or_fileobj=json.dumps(state["best"], indent=1).encode(), path_in_repo="BEST.json",
                                repo_id=args.repo, repo_type="model", commit_message=f"best: {folder} {score:+.6f}")
                while len(state["uploaded"]) > args.keep:
                    old = state["uploaded"].pop(0)
                    try:
                        api.delete_folder(path_in_repo=old, repo_id=args.repo, repo_type="model", commit_message=f"drop {old}")
                    except Exception as exc:  # noqa: BLE001 - keep watching even if the Hub hiccups
                        log.warning("could not delete %s: %s", old, exc)
                log.info("backed up %s (%+.6f) to %s", folder, score, args.repo)
                state_path.write_text(json.dumps(state, indent=1))
            state_path.write_text(json.dumps(state, indent=1))
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
