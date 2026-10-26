#!/usr/bin/env python3
"""Upload a validator-ready checkpoint to Hugging Face behind a manual-approval gate.

  hf_upload_ckpt.py --ckpt /data/teutonic/ckpt/text_lora_d/final [--repo yamalies/<name>] [--private] [--dry]

Token: /data/teutonic/hf_token (never printed). Free accounts cap private storage at ~100 GB, so by default the repo is
public with gated="manual" (no download without the owner's approval), like the account's other model repos, under a
random name. Order: create the empty repo -> set and verify the gate (abort and delete the repo if it does not stick)
-> upload the checkpoint files -> compare every file name and size with the local directory. The repo id is printed
and appended to <ckpt>/../uploads.json.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/teutonic/hf_home")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
TOKEN_FILE = Path("/data/teutonic/hf_token")
# upload target (user 02:00 JST Oct 12: "keep uploading gated repo models to the Hugging Face organization"): when this file
# exists, {"owner": "<org>", "token_file": "<org-scoped token file>"} sends backups to <org>/<random hex> instead of the account
TARGET_FILE = Path("/data/teutonic/hf_upload_target.json")


def local_files(ckpt: Path) -> dict[str, int]:
    return {p.name: p.stat().st_size for p in ckpt.iterdir() if p.is_file() and not p.name.startswith(".")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--repo", default=None, help="default: <account>/<random 12 hex>")
    ap.add_argument("--private", action="store_true", help="private repo (needs free private storage; else gated public)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="check token, checkpoint and name; create nothing")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    ckpt = args.ckpt.resolve()
    files = local_files(ckpt)
    shards = [f for f in files if f.endswith(".safetensors")]
    index = json.loads((ckpt / "model.safetensors.index.json").read_text())
    if len(set(index["weight_map"].values())) != len(shards) or "config.json" not in files:
        sys.exit(f"{ckpt} is not a complete checkpoint ({len(shards)} shards, index lists {len(set(index['weight_map'].values()))})")
    target = json.loads(TARGET_FILE.read_text()) if TARGET_FILE.exists() else {}
    api = HfApi(token=Path(target.get("token_file", TOKEN_FILE)).read_text().strip())
    account = target.get("owner") or api.whoami()["name"]
    repo = args.repo or f"{account}/{secrets.token_hex(6)}"
    total = sum(files.values())
    print(f"{ckpt}: {len(files)} files, {total / 1e9:.1f} GB -> {repo} ({'private' if args.private else 'public, gated=manual'})", flush=True)
    if args.dry:
        return

    api.create_repo(repo, repo_type="model", private=args.private, exist_ok=False)
    if not args.private:
        api.update_repo_settings(repo, gated="manual")
        if api.model_info(repo).gated != "manual":
            api.delete_repo(repo, repo_type="model")
            sys.exit(f"could not set gated=manual on {repo}; repo deleted, nothing uploaded")
    t0 = time.time()
    api.upload_large_folder(repo_id=repo, folder_path=str(ckpt), repo_type="model", num_workers=args.workers,
                            ignore_patterns=[".cache/**", "*.json.tmp"])
    shutil.rmtree(ckpt / ".cache", ignore_errors=True)  # upload_large_folder's resume metadata
    remote = {s.rfilename: s.size for s in (api.model_info(repo, files_metadata=True).siblings or [])}
    missing = sorted(f for f in files if f not in remote)
    wrong = sorted(f for f in files if f in remote and remote[f] != files[f])
    if missing or wrong:
        sys.exit(f"upload incomplete: {len(missing)} missing, {len(wrong)} size mismatches (e.g. {(missing + wrong)[:3]})")
    gated = api.model_info(repo).gated
    record = {"repo": repo, "ckpt": str(ckpt), "files": len(files), "bytes": total, "gated": gated, "private": args.private,
              "seconds": round(time.time() - t0), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    log = ckpt.parent / "uploads.json"
    entries = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(entries + [record], indent=1))
    print(f"UPLOADED {repo}: {len(files)} files, {total / 1e9:.1f} GB in {record['seconds']}s, gated={gated}", flush=True)


if __name__ == "__main__":
    main()
