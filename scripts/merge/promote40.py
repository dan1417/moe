#!/usr/bin/env python3
"""Promote sweep40 winners: write the submission, verify it, upload it to a gated HF repo.

sweep40.py drops a record into /opt/teutonic/promote/<label>.json when a candidate reaches the
promotion threshold. This loop takes the best pending record that beats the last uploaded one by
--min-gain, then:
  1. scripts/merge/write_merge.py  (exact scored weights, contract files from the king)
  2. scripts/verify/verify_submission.py vs king #40
  3. uploads the directory to a new HF model repo (public + gated, manual approval, no README)
  4. records it in configs/merge/k40/uploads.json and commits + pushes that file.
The HF token is read from /root/.huggingface/upload_token (never written to the repo).

  nohup python scripts/merge/promote40.py > /opt/teutonic/logs/promote40.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import secrets
import shutil
import string
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PROMOTE = Path("/opt/teutonic/promote")
UPLOADS = REPO / "configs/merge/k40/uploads.json"
log = logging.getLogger("promote40")


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO, check=True)


def git_push(path: Path, message: str) -> None:
    rel = str(path.relative_to(REPO))
    subprocess.run(["git", "add", rel], cwd=REPO, check=False)
    subprocess.run(["git", "commit", "-q", "-m", message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                   cwd=REPO, check=False)
    for _ in range(3):
        if subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=REPO).returncode == 0:
            return
        subprocess.run(["git", "pull", "-q", "--rebase", "origin", "main"], cwd=REPO, check=False)


def upload(folder: Path, token: str, label: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    owner = api.whoami()["name"]
    name = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))
    repo_id = f"{owner}/{name}"
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=False)
    api.update_repo_settings(repo_id, gated="manual")
    log.info("created %s (public, gated=manual) for %s", repo_id, label)
    api.upload_large_folder(repo_id=repo_id, folder_path=str(folder), repo_type="model",
                            ignore_patterns=["README.md", "*.recipe.json", ".cache/**"], num_workers=16)
    files = set(api.list_repo_files(repo_id))
    local = {p.name for p in folder.iterdir() if p.is_file()}
    missing = local - files
    if missing:
        raise RuntimeError(f"upload incomplete, missing {sorted(missing)[:5]}")
    return repo_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", default="/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c")
    ap.add_argument("--submit-root", type=Path, default=Path("/root/teutonic_submit"))
    ap.add_argument("--token-file", type=Path, default=Path("/root/.huggingface/upload_token"))
    ap.add_argument("--min-gain", type=float, default=0.0002, help="must beat the last upload by this much")
    ap.add_argument("--gpus", default="cpu")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    token = args.token_file.read_text().strip()
    py = sys.executable
    while True:
        uploads = json.loads(UPLOADS.read_text()) if UPLOADS.exists() else []
        best_up = max((u["mu_hat"] for u in uploads), default=-1.0)
        pending = []
        for p in sorted(PROMOTE.glob("*.json")) if PROMOTE.exists() else []:
            rec = json.loads(p.read_text())
            pending.append((rec["result"]["mu_hat"], p, rec))
        pending.sort(key=lambda x: -x[0])
        todo = [x for x in pending if x[0] >= best_up + args.min_gain]
        for _, p, _ in pending:
            if not todo or p != todo[0][1]:
                p.rename(p.with_suffix(".skipped"))
        if todo:
            mu, path, rec = todo[0]
            label = rec["label"]
            recipe = REPO / "configs/merge/k40" / f"{label}.json"
            out = args.submit_root / label
            try:
                for old in args.submit_root.glob("*"):  # one submission on disk at a time (220 GB each)
                    if old.is_dir() and old != out:
                        shutil.rmtree(old)
                run([py, "scripts/merge/write_merge.py", "--recipe", str(recipe), "--out", str(out),
                     "--gpus", args.gpus, "--force"])
                run([py, "scripts/verify/verify_submission.py", "--model-dir", str(out), "--king-dir", args.king])
                repo_id = upload(out, token, label)
                uploads.append({"label": label, "repo": repo_id, "mu_hat": mu, "lcb": rec["result"]["lcb"],
                                "n": rec["result"]["n"], "sample": rec["result"]["sample"],
                                "recipe": str(recipe.relative_to(REPO)),
                                "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                UPLOADS.write_text(json.dumps(uploads, indent=1) + "\n")
                git_push(UPLOADS, f"k40: uploaded {label} (mu_hat {mu:+.5f}) to {repo_id}")
                log.info("UPLOADED %s -> %s", label, repo_id)
                path.rename(path.with_suffix(".done"))
            except Exception as exc:  # noqa: BLE001
                log.exception("promotion of %s failed: %s", label, exc)
                path.rename(path.with_suffix(".failed"))
        if args.once:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
