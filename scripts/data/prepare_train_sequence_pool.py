#!/usr/bin/env python3
"""Precompute a deterministic per-run training “sequence pool”.

This pool is intended for fast training startup and deterministic `--resume` at
step boundaries by avoiding runtime random shard/sequence sampling.

It downloads exactly the shards it needs (with sha256 verification), samples the
required number of 2048-token sequences, and writes:
  - `pool_sequences.npy`: uint32 array of shape (N_total, 2048)
  - `pool_source_ids.npy`: uint16 array of shape (N_total,) mapping each row to a source
  - `pool_meta.json`: metadata used for validation by the training data loader
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.lib.format import open_memmap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_MANIFEST_URL, Manifest, SEQUENCE_LENGTH, ShardCache

log = logging.getLogger(__name__)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _round_with_largest_remainders(raw_counts: list[float], total: int) -> list[int]:
    """Convert float counts to integer counts summing exactly to `total`."""
    floors = [int(math.floor(x)) for x in raw_counts]
    current = sum(floors)
    remainder = total - current
    if remainder == 0:
        return floors

    fracs = [x - math.floor(x) for x in raw_counts]
    # Distribute remainder to largest fractional parts.
    order = sorted(range(len(raw_counts)), key=lambda i: fracs[i], reverse=True)
    if remainder > 0:
        for i in order[:remainder]:
            floors[i] += 1
    else:
        # remainder < 0: remove from smallest fractional parts
        for i in reversed(order[: -remainder]):
            floors[i] -= 1
    assert sum(floors) == total
    return floors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute deterministic training sequence pool")
    parser.add_argument("--config", default="configs/b300/train.yaml", help="Training config path")
    parser.add_argument("--pool-dir", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override selection seed")
    parser.add_argument("--world-size", type=int, default=None, help="Override world_size")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    training_cfg: dict[str, Any] = cfg["training"]
    data_cfg: dict[str, Any] = cfg.get("data", {})
    hardware_cfg: dict[str, Any] = cfg.get("hardware", {})

    micro_batch_size = int(training_cfg["micro_batch_size"])
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    total_steps = int(training_cfg["total_steps"])
    base_seed = int(training_cfg.get("base_seed", 42))
    seed = int(args.seed) if args.seed is not None else base_seed

    gpus_per_node = int(hardware_cfg.get("gpus_per_node", 8))
    num_nodes = int(hardware_cfg.get("num_nodes", 1))
    world_size = int(args.world_size) if args.world_size is not None else gpus_per_node * num_nodes

    data_pool_cfg = data_cfg.get("sequence_pool", {})
    pool_dir = Path(args.pool_dir or data_pool_cfg.get("pool_dir", "./sequence_pool"))
    pool_dir.mkdir(parents=True, exist_ok=True)

    selection_shards_range = data_pool_cfg.get("shards_per_source_range", [500, 2000])
    selection_seqs_range = data_pool_cfg.get("seqs_per_shard_range", [100, 2000])
    shards_min, shards_max = int(selection_shards_range[0]), int(selection_shards_range[1])
    seqs_min, seqs_max = int(selection_seqs_range[0]), int(selection_seqs_range[1])

    manifest_url = data_cfg.get("manifest_url") or DEFAULT_MANIFEST_URL
    manifest = Manifest(
        manifest_url=manifest_url,
        cache_ttl_sec=int(data_cfg.get("manifest_cache_ttl_sec", 60)),
    )
    sources = manifest.fetch(force=True)
    sources = sorted(sources, key=lambda s: s.name)

    sequences_per_step_per_rank = micro_batch_size * grad_accum_steps
    total_sequences_needed_global = sequences_per_step_per_rank * world_size * total_steps
    if total_sequences_needed_global <= 0:
        raise ValueError("computed total_sequences_needed_global must be > 0")

    raw_counts = [s.proportion * total_sequences_needed_global for s in sources]
    source_counts = _round_with_largest_remainders(raw_counts, total_sequences_needed_global)
    if sum(source_counts) != total_sequences_needed_global:
        raise RuntimeError("internal error: source_counts do not sum to total_sequences_needed_global")

    pool_sequences_path = pool_dir / "pool_sequences.npy"
    pool_source_ids_path = pool_dir / "pool_source_ids.npy"
    meta_path = pool_dir / "pool_meta.json"

    log.info(
        "Preparing sequence pool: world_size=%d total_steps=%d total_sequences=%d (seq_len=%d)",
        world_size,
        total_steps,
        total_sequences_needed_global,
        SEQUENCE_LENGTH,
    )
    log.info("Pool dir: %s", pool_dir)

    # Download shard cache to local disk for faster future runs.
    shard_cache = ShardCache(
        cache_dir=data_cfg.get("cache_dir", "./shard_cache"),
        max_gb=float(data_cfg.get("cache_max_gb", 500)),
        timeout=60,
        max_workers=int(data_cfg.get("sequence_pool_download_workers", 4)),
    )

    seq_mem = open_memmap(
        pool_sequences_path,
        mode="w+",
        dtype=np.uint32,
        shape=(total_sequences_needed_global, SEQUENCE_LENGTH),
    )
    src_mem = open_memmap(
        pool_source_ids_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_sequences_needed_global,),
    )

    rng = np.random.default_rng(seed)

    # Build a deterministic shuffled “source id” vector so training sees the
    # correct mixture at the sequence level (not as large per-source blocks).
    source_id_pool = np.empty(total_sequences_needed_global, dtype=np.uint16)
    write_pos = 0
    for source_idx in range(len(sources)):
        n_source = int(source_counts[source_idx])
        if n_source <= 0:
            continue
        source_id_pool[write_pos : write_pos + n_source] = np.uint16(source_idx)
        write_pos += n_source
    if write_pos != total_sequences_needed_global:
        raise RuntimeError(
            f"internal error: source_id_pool fill incomplete: wrote {write_pos}, expected {total_sequences_needed_global}"
        )
    rng.shuffle(source_id_pool)

    # Precompute per-source (selected_shard_local_idx, seq_index) requests.
    requests_shard_local: list[np.ndarray] = []
    requests_seq_index: list[np.ndarray] = []
    selected_shards_by_source: list[list[Any]] = []

    for source_idx, source in enumerate(sources):
        n_source = int(source_counts[source_idx])
        if n_source <= 0:
            requests_shard_local.append(np.empty((0,), dtype=np.int32))
            requests_seq_index.append(np.empty((0,), dtype=np.int32))
            selected_shards_by_source.append([])
            continue

        shards_by_seqmax_min = _ceil_div(n_source, seqs_max)
        shards_by_seqmin_max = n_source // seqs_min if seqs_min > 0 else shards_max
        shards_low = max(shards_min, shards_by_seqmax_min)
        shards_high = min(shards_max, shards_by_seqmin_max)
        if shards_high < shards_low:
            raise ValueError(
                f"Impossible selection ranges for source={source.name}: n_source={n_source}, "
                f"shards_range=[{shards_min},{shards_max}], seqs_range=[{seqs_min},{seqs_max}]"
            )

        selected_shard_count = int(rng.integers(shards_low, shards_high + 1))
        seqs_per_shard_target = int(_ceil_div(n_source, selected_shard_count))
        seqs_per_shard_target = max(seqs_min, min(seqs_max, seqs_per_shard_target))
        shard_indices = rng.choice(len(source.shards), size=selected_shard_count, replace=False)

        selected_shards: list[Any] = [source.shards[int(i)] for i in shard_indices]

        shard_local_idx_list: list[int] = []
        seq_index_list: list[int] = []

        remaining = n_source
        log.info(
            "[%d/%d] Source=%s n_sequences=%d selected_shards=%d seqs_per_shard_target=%d",
            source_idx + 1,
            len(sources),
            source.name,
            n_source,
            selected_shard_count,
            seqs_per_shard_target,
        )

        for shard_local_idx, shard in enumerate(selected_shards):
            if remaining <= 0:
                break
            array = shard_cache.load_shard(
                source.name,
                shard.key,
                shard.url,
                expected_sha256=shard.sha256,
            )
            n_in_shard = array.size // SEQUENCE_LENGTH
            if n_in_shard < 1:
                raise ValueError(f"Shard too small: source={source.name} shard_key={shard.key}")

            k = min(seqs_per_shard_target, remaining)
            if k >= n_in_shard:
                seq_indices = rng.choice(n_in_shard, size=k, replace=True)
            else:
                seq_indices = rng.choice(n_in_shard, size=k, replace=False)

            shard_local_idx_list.extend([shard_local_idx] * int(k))
            seq_index_list.extend([int(x) for x in seq_indices])
            remaining -= int(k)

        if len(seq_index_list) != n_source:
            raise RuntimeError(
                f"Sequence request count mismatch for source={source.name}: got {len(seq_index_list)} expected {n_source}"
            )

        requests_shard_local.append(np.asarray(shard_local_idx_list, dtype=np.int32))
        requests_seq_index.append(np.asarray(seq_index_list, dtype=np.int32))
        selected_shards_by_source.append(selected_shards)

    # Fill the final shuffled pool.
    per_source_cursor = [0 for _ in range(len(sources))]
    for global_i, source_idx in enumerate(source_id_pool):
        sid = int(source_idx)
        pos = per_source_cursor[sid]
        shard_local_idx = int(requests_shard_local[sid][pos])
        seq_index = int(requests_seq_index[sid][pos])
        per_source_cursor[sid] = pos + 1

        shard = selected_shards_by_source[sid][shard_local_idx]
        array = shard_cache.load_shard(
            sources[sid].name,
            shard.key,
            shard.url,
            expected_sha256=shard.sha256,
        )
        start = seq_index * SEQUENCE_LENGTH
        end = start + SEQUENCE_LENGTH
        seq_mem[global_i] = array[start:end]
        src_mem[global_i] = np.uint16(sid)

        if global_i % 10000 == 0 and global_i > 0:
            log.info("Pool fill progress: %d/%d", global_i, total_sequences_needed_global)

    # Flush memmaps.
    seq_mem.flush()
    src_mem.flush()

    meta_payload = {
        "pool_dir": str(pool_dir),
        "manifest_url": manifest.manifest_url,
        "manifest_config_version": manifest.config_version,
        "sequence_length": SEQUENCE_LENGTH,
        "world_size": world_size,
        "total_steps": total_steps,
        "micro_batch_size": micro_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "total_sequences_global": total_sequences_needed_global,
        "sequences_per_step_per_rank": sequences_per_step_per_rank,
        "seed": seed,
        "sources": [s.name for s in sources],
        "source_counts": {s.name: int(c) for s, c in zip(sources, source_counts)},
        "shards_per_source_range": [shards_min, shards_max],
        "seqs_per_shard_range": [seqs_min, seqs_max],
    }
    meta_path.write_text(json.dumps(meta_payload, indent=2))

    log.info("Sequence pool ready: %s", pool_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

