#!/usr/bin/env python3
"""Build a high-diversity CPT pool (or held-out val set) with HTTP range reads.

The validator's shard plan puts ~1,100 sequences of a sample on each of ~27 shards, and every
shard is a contiguous 6,144-sequence chunk of one sub-dataset, so shards differ in content
(the shard-level jackknife SE of a duel is ~4x its iid SE). A pool built from a few hundred
whole shards therefore trains on a few hundred topics; this builder instead reads one random
block of ``--block`` consecutive sequences from each of many randomly chosen shards:

  * sources in the live eval proportions (the validator's largest-remainder split);
  * within a source, shards drawn uniformly by capacity -- the category-stratified eval plan
    weights categories by shard count, so this is the eval distribution in expectation;
  * the validator's vocab filter; shards of local parity samples and --exclude-pool pools skipped.

The r2.dev dataset bucket rate-limits (~40-50 requests/s, >128 concurrent connections get
SSL resets), so a train pool uses blocks of 16-32 sequences and a val set blocks of 1.

  build_diverse_pool.py --out /workspace/teutonic/pools/val_div --n 8192 --block 1 --tag val_div
  build_diverse_pool.py --out /workspace/teutonic/pools/train_div --n 1000000 --block 32 --tag train_div \
      --exclude-pool /workspace/teutonic/pools/val_div
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cpt.data import eval_shard_names, pool_shard_names  # noqa: E402
from src.parity import upstream  # noqa: E402
from src.parity.sample import SEQ_LEN, VOCAB_SIZE, load_live_config  # noqa: E402

log = logging.getLogger("diverse_pool")
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "teutonic-cpt-pool/1.0"
    return _local.session


def fetch_block(task: dict, retries: int = 8) -> np.ndarray | None:
    """[block, SEQ_LEN] uint32 read with one Range request, or None after all retries fail."""
    first = task["header"] + task["start"] * SEQ_LEN * 4
    last = first + task["block"] * SEQ_LEN * 4 - 1
    for attempt in range(retries):
        try:
            response = _session().get(task["url"], headers={"Range": f"bytes={first}-{last}"}, timeout=60)
            if response.status_code == 206 and len(response.content) == last - first + 1:
                return np.frombuffer(response.content, dtype="<u4").reshape(task["block"], SEQ_LEN)
            log.debug("%s: status %s, %d bytes", task["name"], response.status_code, len(response.content))
        except requests.RequestException as exc:
            log.debug("%s: %s", task["name"], exc)
            _local.__dict__.pop("session", None)  # drop a connection the server reset
        time.sleep(min(30.0, 1.5 * 2**attempt))
    return None


def plan_blocks(live, *, n: int, block: int, rng: np.random.Generator, excluded: set[str],
                only_source: str | None = None) -> list[dict]:
    """Random (shard, block start) reads covering each source's share of ``n`` (+3% for filtering);
    ``only_source``: all ``n`` sequences from that one source."""
    proportions = [float(s.proportion) for s in live.settings.manifests]
    targets = upstream.configuration._source_targets(n, proportions)
    if only_source:
        targets = [n if s.name == only_source else 0 for s in live.settings.manifests]
    tasks: list[dict] = []
    for target, snapshot in zip(targets, live.settings.manifests):
        if target == 0:
            continue
        shards = [s for s in snapshot.manifest["shards"]
                  if int(s["n_tokens"]) // SEQ_LEN >= block and (s.get("url") or s["key"]).rsplit("/", 1)[-1] not in excluded]
        capacity = np.asarray([int(s["n_tokens"]) // SEQ_LEN for s in shards], dtype=np.float64)
        wanted = int(np.ceil(target * 1.03 / block))
        if wanted > len(shards):
            raise SystemExit(f"{snapshot.name}: {wanted} blocks wanted but only {len(shards)} eligible shards")
        picked = rng.choice(len(shards), size=wanted, replace=False, p=capacity / capacity.sum())
        for index in picked:
            shard = shards[int(index)]
            rows = int(shard["n_tokens"]) // SEQ_LEN
            tasks.append({
                "source": snapshot.name,
                "target": target,
                "name": (shard.get("url") or shard["key"]).rsplit("/", 1)[-1],
                "url": shard["url"] if shard.get("url") else upstream.configuration._shard_url(snapshot.manifest_url, shard["key"]),
                "header": int(shard["size_bytes"]) - int(shard["n_tokens"]) * 4,
                "start": int(rng.integers(0, rows - block + 1)),
                "block": block,
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", required=True, type=int, help="sequences (split by the live source proportions)")
    parser.add_argument("--block", type=int, default=32, help="consecutive sequences per range read")
    parser.add_argument("--tag", required=True, help="seed family; distinct per pool")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--only-source", help="draw every sequence from this source (e.g. txt360-v2)")
    parser.add_argument("--competition", default="main", help="eval config to mirror: main | math | code | text | mix")
    parser.add_argument("--exclude-pool", type=Path, action="append", default=[],
                        help="pool dir whose shards must not be reused (e.g. the val set when building a train pool)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    live = load_live_config(competition=args.competition)
    excluded = eval_shard_names() | pool_shard_names(args.exclude_pool)
    if not eval_shard_names():
        log.warning("no local parity samples found: build tag:A / tag:B first so their shards are excluded")
    seed = int.from_bytes(hashlib.blake2b(f"diverse:{args.tag}".encode(), digest_size=8).digest(), "little")
    tasks = plan_blocks(live, n=args.n, block=args.block, rng=np.random.default_rng(seed), excluded=excluded,
                        only_source=args.only_source)
    log.info("%d range reads of %d sequences (%d shards excluded), %d workers", len(tasks), args.block,
             len(excluded), args.workers)

    kept: dict[str, list[np.ndarray]] = {}
    used: list[dict] = []
    failed = dropped = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_block, task): task for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            task, rows = futures[future], future.result()
            if rows is None:
                failed += 1
            else:
                valid = rows[rows.max(axis=1) < VOCAB_SIZE]  # the validator drops out-of-vocab sequences
                dropped += len(rows) - len(valid)
                if len(valid):
                    kept.setdefault(task["source"], []).append(valid)
                    used.append({"name": task["name"], "start": task["start"], "rows": int(len(valid))})
            if done % 1000 == 0 or done == len(tasks):
                rate = done / max(time.time() - started, 1e-9)
                log.info("%d/%d reads (%.1f/s, eta %.0fs), %d failed, %d oov sequences dropped", done, len(tasks),
                         rate, (len(tasks) - done) / max(rate, 1e-9), failed, dropped)

    tokens, labels = [], []
    counts = {}
    targets = {t["source"]: t["target"] for t in tasks}
    for source, chunks in sorted(kept.items()):
        rows = np.concatenate(chunks)
        if len(rows) < targets[source]:
            log.warning("%s: %d sequences < target %d (failed reads)", source, len(rows), targets[source])
        rows = rows[: targets[source]]
        tokens.append(rows)
        labels.extend([source] * len(rows))
        counts[source] = int(len(rows))
    order = np.random.default_rng(seed + 1).permutation(len(labels))  # any prefix is a representative mix
    tokens = np.concatenate(tokens)[order]
    labels = np.asarray(labels)[order]
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "tokens.npy", tokens)
    np.save(args.out / "sources.npy", labels)
    (args.out / "meta.json").write_text(json.dumps({
        "tag": args.tag, "kind": "diverse_range_reads", "block": args.block, "n_sequences": int(len(labels)),
        "n_tokens": int(tokens.size), "source_counts": counts, "config_version": live.config_version,
        "shards": sorted({u["name"] for u in used}), "reads": used, "failed_reads": failed,
        "oov_dropped": dropped, "excluded_shards": len(excluded), "seconds": round(time.time() - started, 1),
    }, indent=1))
    log.info("pool %s: %d sequences (%.2fB tokens) from %d shards %s", args.out, len(labels), tokens.size / 1e9,
             len({u['name'] for u in used}), counts)


if __name__ == "__main__":
    main()
