#!/usr/bin/env python3
"""Upload the current best model split over several HF accounts (private repos, < 100 GB each).

Layout (repo per part, shards per part, sha256 of every file): configs/best/current_best.json.
Tokens are never stored in git: pass them via --tokens-file (one per line) or HF_TOKENS="tok1,tok2,...".
Each part is uploaded with whichever token belongs to the repo's owner (matched by whoami).

  python scripts/data/upload_best.py --tokens-file /root/.huggingface/best_split_tokens
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]


def load_tokens(path: str | None) -> list[str]:
    tokens = [t for t in os.environ.get("HF_TOKENS", "").split(",") if t.strip()]
    if path:
        tokens += Path(path).read_text().split()
    if not tokens:
        raise SystemExit("no tokens: pass --tokens-file or set HF_TOKENS")
    return tokens


def token_owners(tokens: list[str]) -> dict[str, str]:
    owners = {}
    for token in tokens:
        try:
            owners[HfApi(token=token).whoami()["name"]] = token
        except Exception as exc:  # an invalid token should not stop the others
            print(f"skipping a token: {type(exc).__name__}")
    return owners


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/best/current_best.json"))
    ap.add_argument("--tokens-file", default=None)
    ap.add_argument("--parts", default=None, help="comma-separated part indices (default: all)")
    ap.add_argument("--workers", type=int, default=6, help="upload workers per part")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    owners = token_owners(load_tokens(args.tokens_file))
    parts = list(enumerate(cfg["parts"]))
    if args.parts:
        wanted = {int(x) for x in args.parts.split(",")}
        parts = [(i, p) for i, p in parts if i in wanted]

    def upload(item) -> str:
        i, part = item
        owner = part["repo"].split("/")[0]
        if owner not in owners:
            return f"part {i} {part['repo']}: no token for account {owner}"
        api = HfApi(token=owners[owner])
        api.create_repo(part["repo"], repo_type="model", private=True, exist_ok=True)
        print(f"part {i}: uploading {len(part['shards'])} shards ({part['gb']} GB) to {part['repo']}", flush=True)
        api.upload_large_folder(repo_id=part["repo"], repo_type="model", folder_path=part["local_dir"],
                                num_workers=args.workers)
        remote = {s.rfilename for s in api.model_info(part["repo"]).siblings}
        missing = [f for f in part["shards"] + cfg["common_files"] if f not in remote]
        return f"part {i} {part['repo']}: " + ("OK" if not missing else f"MISSING {missing}")

    with ThreadPoolExecutor(len(parts)) as pool:
        for line in pool.map(upload, parts):
            print(line, flush=True)


if __name__ == "__main__":
    main()
