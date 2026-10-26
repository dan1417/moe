#!/usr/bin/env python3
"""Download the current best model (split over several private HF repos) into one directory and verify it.

Layout and sha256 of every file: configs/best/current_best.json. Tokens are never stored in git:
pass them via --tokens-file (one per line) or HF_TOKENS="tok1,tok2,...". For each part the token
belonging to the repo owner is used (matched by whoami).

  HF_TOKENS=hf_a,hf_b,hf_c,hf_d python scripts/data/download_best.py --out /root/best_model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = Path(__file__).resolve().parents[2]


def load_tokens(path: str | None) -> list[str]:
    tokens = [t for t in os.environ.get("HF_TOKENS", "").split(",") if t.strip()]
    if path:
        tokens += Path(path).read_text().split()
    if not tokens:
        raise SystemExit("no tokens: pass --tokens-file or set HF_TOKENS")
    return tokens


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/best/current_best.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens-file", default=None)
    ap.add_argument("--workers", type=int, default=8, help="parallel file downloads per part")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    owners = {}
    for token in load_tokens(args.tokens_file):
        try:
            owners[HfApi(token=token).whoami()["name"]] = token
        except Exception as exc:
            print(f"skipping a token: {type(exc).__name__}")

    def fetch(item) -> str:
        i, part = item
        owner = part["repo"].split("/")[0]
        if owner not in owners:
            raise SystemExit(f"part {i}: no token for account {owner}")
        snapshot_download(part["repo"], repo_type="model", local_dir=str(out), token=owners[owner],
                          allow_patterns=part["shards"] + (cfg["common_files"] if i == 0 else []),
                          max_workers=args.workers)
        return f"part {i} {part['repo']}: {len(part['shards'])} shards"

    with ThreadPoolExecutor(len(cfg["parts"])) as pool:
        for line in pool.map(fetch, list(enumerate(cfg["parts"]))):
            print(line, flush=True)

    if args.skip_verify:
        return
    names = sorted(cfg["files"])

    def check(name: str) -> str | None:
        path = out / name
        want = cfg["files"][name]
        if not path.exists() or path.stat().st_size != want["size"]:
            return f"{name}: missing or wrong size"
        return None if sha256(path) == want["sha256"] else f"{name}: sha256 mismatch"

    with ThreadPoolExecutor(16) as pool:
        bad = [b for b in pool.map(check, names) if b]
    print(f"verified {len(names) - len(bad)}/{len(names)} files" + (f"; FAILED: {bad}" if bad else " — ready"))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
