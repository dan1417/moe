#!/usr/bin/env python3
"""Materialize a scored recipe shard by shard and upload it to a new gated HF repo (no full local copy).

Each ~5 GB shard is built with src.parity.weights._fill (the code path the sweep scored), written to a small
staging dir, uploaded, checked against the LFS sha256 HF reports, and deleted before the next one is uploaded
(two shards in flight). Then the index and the six contract files (copied from the king, checked against
chain.toml like verify_submission.py) are uploaded. The repo is public with gated access (manual approval) and
has no README. The upload is recorded in configs/merge/k40/uploads.json and committed.

  python scripts/merge/stream_upload40.py --recipe configs/merge/k40/up_d36_050q.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import secrets
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "scripts/verify"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

log = logging.getLogger("stream_upload40")
UPLOADS = REPO / "configs/merge/k40/uploads.json"
SHARD_BYTES = 5 * 1024**3
MAX_UPLOAD_BYTES = 250_000_000_000


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--stage", type=Path, default=Path("/root/teutonic_stage"))
    ap.add_argument("--token-file", type=Path, default=Path("/root/.huggingface/upload_token"))
    ap.add_argument("--repo-id", default=None, help="resume into an existing repo")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from huggingface_hub import HfApi

    import verify_submission as vs
    from parity_eval import resolve_recipe
    from src.parity import upstream
    from src.parity.weights import _fill, _Reader

    torch.set_num_threads(args.threads)
    record = json.loads(Path(args.recipe).read_text())
    label = record.get("label", Path(args.recipe).stem)
    recipe = resolve_recipe(args.recipe)
    king = recipe.terms[0].path
    reader = _Reader(recipe)
    index = json.loads((king / "model.safetensors.index.json").read_text())["weight_map"]
    names = list(index)
    meta = {}
    for fn in sorted(set(index.values())):
        with safe_open(str(king / fn), framework="pt", device="cpu") as h:
            for n in h.keys():
                sl = h.get_slice(n)
                meta[n] = (tuple(sl.get_shape()), sl.get_dtype())
    assert all(d == "BF16" for _, d in meta.values()) and not any("mtp" in n.lower() for n in names)
    shards, cur, size = [], [], 0
    for n in names:
        nb = 2 * int(torch.tensor(meta[n][0]).prod().item()) if meta[n][0] else 2
        if cur and size + nb > SHARD_BYTES:
            shards.append(cur)
            cur, size = [], 0
        cur.append(n)
        size += nb
    if cur:
        shards.append(cur)
    total = len(shards)
    # contract files must match chain.toml (as verify_submission.py checks)
    expected, source = vs.contract_hashes()
    contract = list(upstream.chain_config.GENESIS_CONTRACT_FILES)
    for name in contract:
        got = sha256_file(king / name)
        if expected.get(name) and expected[name] != got:
            raise SystemExit(f"contract file {name} sha256 {got} != {source} {expected[name]}")
    token = args.token_file.read_text().strip()
    api = HfApi(token=token)
    if args.repo_id:
        repo_id = args.repo_id
    else:
        owner = api.whoami()["name"]
        repo_id = f"{owner}/" + "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(12))
        api.create_repo(repo_id, repo_type="model", private=False)
        api.update_repo_settings(repo_id, gated="manual")
    log.info("uploading %s (%d shards) to %s (public, gated=manual, no README)", label, total, repo_id)
    present = set(api.list_repo_files(repo_id))
    args.stage.mkdir(parents=True, exist_ok=True)
    weight_map, shas, written = {}, {}, [0]

    def build(i: int) -> tuple[str, Path]:
        fname = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
        tensors = {}
        for n in shards[i]:
            t = torch.empty(meta[n][0], dtype=torch.bfloat16)
            with torch.no_grad():
                _fill(t, n, reader, recipe)
            tensors[n] = t
        path = args.stage / fname
        save_file(tensors, str(path), metadata={"format": "pt"})
        written[0] += path.stat().st_size
        return fname, path

    def ship(fname: str, path: Path) -> None:
        sha = sha256_file(path)
        for attempt in range(5):
            try:
                api.upload_file(path_or_fileobj=str(path), path_in_repo=fname, repo_id=repo_id,
                                commit_message=f"{fname}")
                info = api.get_paths_info(repo_id, [fname])[0]
                remote = getattr(getattr(info, "lfs", None), "sha256", None)
                if remote != sha:
                    raise RuntimeError(f"{fname}: remote sha {remote} != local {sha}")
                break
            except Exception as exc:  # noqa: BLE001
                log.warning("upload %s attempt %d failed: %s", fname, attempt + 1, exc)
                if attempt == 4:
                    raise
                time.sleep(20 * (attempt + 1))
        shas[fname] = sha
        path.unlink()
        log.info("shipped %s (%s)", fname, sha[:12])

    started = time.time()
    with ThreadPoolExecutor(2) as builders, ThreadPoolExecutor(2) as shippers:
        pending = []
        for i in range(total):
            fname = f"model-{i + 1:05d}-of-{total:05d}.safetensors"
            for n in shards[i]:
                weight_map[n] = fname
            if fname in present:
                continue
            while len([p for p in pending if not p.done()]) >= 2:  # bound staging disk to ~4 shards
                time.sleep(2)
            fname_path = builders.submit(build, i)
            pending.append(shippers.submit(lambda f: ship(*f.result()), fname_path))
        for p in pending:
            p.result()
    total_size = sum(2 * int(torch.tensor(meta[n][0]).prod().item()) for n in names)
    if total_size > MAX_UPLOAD_BYTES:
        raise SystemExit("submission exceeds 250 GB")
    idx = args.stage / "model.safetensors.index.json"
    idx.write_text(json.dumps({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    api.upload_file(path_or_fileobj=str(idx), path_in_repo="model.safetensors.index.json", repo_id=repo_id,
                    commit_message="index")
    for name in contract:
        api.upload_file(path_or_fileobj=str(king / name), path_in_repo=name, repo_id=repo_id, commit_message=name)
    files = set(api.list_repo_files(repo_id))
    want = {f"model-{i + 1:05d}-of-{total:05d}.safetensors" for i in range(total)} | set(contract) | {"model.safetensors.index.json"}
    missing = want - files
    extra = {f for f in files if f not in want and f != ".gitattributes"}
    if missing or extra:
        raise SystemExit(f"repo check failed: missing {sorted(missing)[:5]} extra {sorted(extra)[:5]}")
    res = record.get("result", {})
    uploads = json.loads(UPLOADS.read_text()) if UPLOADS.exists() else []
    uploads.append({"label": label, "repo": repo_id, "mu_hat": res.get("mu_hat"), "lcb": res.get("lcb"), "n": res.get("n"),
                    "sample": res.get("sample"), "recipe": str(Path(args.recipe).resolve().relative_to(REPO)),
                    "shards": total, "total_size": total_size,
                    "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gated": "manual", "readme": False})
    UPLOADS.write_text(json.dumps(uploads, indent=1) + "\n")
    subprocess.run(["git", "add", str(UPLOADS.relative_to(REPO))], cwd=REPO, check=False)
    subprocess.run(["git", "commit", "-q", "-m", f"k40: uploaded {label} to {repo_id}\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                   cwd=REPO, check=False)
    subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=REPO, check=False)
    log.info("UPLOADED %s -> %s: %d shards, %.1f GB in %.0fs", label, repo_id, total, total_size / 1e9, time.time() - started)


if __name__ == "__main__":
    main()
