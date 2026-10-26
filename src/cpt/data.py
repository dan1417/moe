"""Training data drawn from the validator's own evaluation distribution.

Each *round* runs the validator's category-stratified shard planner
(``pretokenized_dataset_request``) with a fresh synthetic seed, then takes
``scale`` × the planned per-shard quota of random sequences from every planned shard.
Scaling the quotas uniformly keeps the source proportions (0.32/0.28/0.22/0.17/0.01)
and the per-category shares exactly as in a duel. Shards used by any local parity
eval sample are excluded so local validation stays uncontaminated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.parity import upstream
from src.parity.sample import SEQ_LEN, VOCAB_SIZE, LiveConfig, install_fetch_retry, synthetic_block_hash

log = logging.getLogger(__name__)
TRAIN_HOTKEY = "cpt-train"


def eval_shard_names(cache_root: Path | None = None) -> set[str]:
    """Shard file names used by every locally built parity sample."""
    root = (cache_root or upstream.DEFAULT_CACHE_ROOT) / "samples"
    names: set[str] = set()
    for meta in root.glob("*/meta.json"):
        for shard_list in json.loads(meta.read_text())["shard_names"].values():
            names.update(shard_list)
    return names


def pool_shard_names(pool_dirs: list[Path]) -> set[str]:
    """Shard file names already used by other pools (e.g. the held-out val pool)."""
    names: set[str] = set()
    for pool in pool_dirs:
        meta = Path(pool) / "meta.json"
        if meta.exists():
            names.update(json.loads(meta.read_text()).get("shards", []))
    return names


def plan_round(live: LiveConfig, tag: str, scale: float) -> list[dict]:
    """[{source, url, sha256, size_bytes, n_tokens, take}] for one training round."""
    request = upstream.configuration.pretokenized_dataset_request(
        live.settings, block_hash=synthetic_block_hash(tag), hotkey=TRAIN_HOTKEY, seq_len=SEQ_LEN
    )
    plan = []
    for source in request["sources"]:
        for shard in source["shards"]:
            capacity = int(shard["n_tokens"]) // SEQ_LEN
            plan.append({
                "source": source["name"],
                "url": shard["url"],
                "sha256": shard["sha256"],
                "size_bytes": int(shard["size_bytes"]),
                "n_tokens": int(shard["n_tokens"]),
                "take": min(capacity, int(round(int(shard["target_sequences"]) * scale))),
            })
    return plan


def _take_from_shard(entry: dict, seed: int) -> np.ndarray:
    """``take`` random in-vocab rows of one planned shard, read by HTTP range like the validator (>= 5118c32).

    Rows are visited in the same seeded permutation as the whole-shard reader this replaced (first ``take`` rows
    whose tokens are all < VOCAB_SIZE), so the same plan still yields the same pool -- without downloading the shard.
    Throttled reads (r2.dev 429) are retried by :func:`src.parity.sample.install_fetch_retry`.
    """
    install_fetch_retry()
    take = int(entry["take"])
    if take <= 0:
        return np.zeros((0, SEQ_LEN), dtype=np.uint32)
    ref = upstream.sources.ShardRef(
        source=entry["source"], url=entry["url"], sha256=entry["sha256"],
        size_bytes=entry["size_bytes"], n_tokens=entry["n_tokens"],
    )
    reader = upstream.sources.open_shard(ref)
    order = np.random.default_rng(seed).permutation(reader.sequence_count(SEQ_LEN))
    picked: list[np.ndarray] = []
    cursor = 0
    while len(picked) < take and cursor < len(order):
        batch = order[cursor : cursor + (take - len(picked)) * 11 // 10 + 8]  # a little spare for out-of-vocab rows
        cursor += len(batch)
        for row in reader.read_sequences(batch, SEQ_LEN):
            if int(row.max()) < VOCAB_SIZE:  # the validator drops out-of-vocab sequences too
                picked.append(np.asarray(row, dtype=np.uint32))
                if len(picked) == take:
                    break
    return np.stack(picked) if picked else np.zeros((0, SEQ_LEN), dtype=np.uint32)


def build_pool(
    live: LiveConfig,
    out_dir: Path,
    *,
    rounds: int,
    scale: float,
    tag: str = "pool",
    workers: int = 16,
    exclude_pools: list[Path] | None = None,
) -> Path:
    """Write tokens.npy (uint32 [N, 2048]), sources.npy and meta.json into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded = eval_shard_names() | pool_shard_names(exclude_pools or [])
    if not eval_shard_names():
        log.warning("no local parity samples found under %s: nothing excluded from the eval side", upstream.DEFAULT_CACHE_ROOT)
    chunks, labels, used = [], [], []
    started = time.time()
    for round_index in range(rounds):
        plan = [e for e in plan_round(live, f"{tag}:{round_index}", scale) if e["url"].rsplit("/", 1)[-1] not in excluded]
        seeds = [
            int.from_bytes(hashlib.blake2b(f"{tag}:{round_index}:{e['sha256']}".encode(), digest_size=4).digest(), "little")
            for e in plan
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            arrays = list(pool.map(_take_from_shard, plan, seeds))
        for entry, array in zip(plan, arrays):
            chunks.append(array)
            labels.extend([entry["source"]] * len(array))
            used.append(entry["url"].rsplit("/", 1)[-1])
        log.info("pool round %d/%d: %d sequences so far (%.0fs)", round_index + 1, rounds, len(labels), time.time() - started)
    # global shuffle so any prefix (e.g. val_n) is a representative mix
    order = np.random.default_rng(int.from_bytes(hashlib.blake2b(tag.encode(), digest_size=4).digest(), "little"))
    order = order.permutation(sum(len(c) for c in chunks))
    tokens = np.concatenate(chunks)[order]
    labels = [labels[i] for i in order]
    np.save(out_dir / "tokens.npy", tokens)
    np.save(out_dir / "sources.npy", np.asarray(labels))
    counts = {name: labels.count(name) for name in sorted(set(labels))}
    (out_dir / "meta.json").write_text(json.dumps({
        "tag": tag, "rounds": rounds, "scale": scale, "n_sequences": int(tokens.shape[0]),
        "n_tokens": int(tokens.size), "source_counts": counts, "config_version": live.config_version,
        "shards": used, "excluded_shards": len(excluded),
    }, indent=1))
    log.info("pool %s: %d sequences (%.2fB tokens) %s", out_dir, tokens.shape[0], tokens.size / 1e9, counts)
    return out_dir


@dataclass
class Batch:
    input_ids: torch.Tensor  # [B, 2048] int64 on device
    sources: np.ndarray  # [B] str


class PoolLoader:
    """Deterministic, resumable, rank-strided micro-batches over a token pool."""

    def __init__(self, pool_dir: Path, *, rank: int, world: int, micro_batch: int, seed: int, device):
        self.tokens = np.load(Path(pool_dir) / "tokens.npy", mmap_mode="r")
        self.sources = np.load(Path(pool_dir) / "sources.npy")
        self.rank, self.world, self.micro_batch = rank, world, micro_batch
        self.seed, self.device = seed, device
        self.position = 0  # global micro-batch counter (identical on all ranks)
        self.per_epoch = len(self.tokens) // (world * micro_batch)
        if self.per_epoch == 0:
            raise ValueError("pool smaller than one global micro-batch")

    def _order(self, epoch: int) -> np.ndarray:
        return np.random.default_rng(self.seed + epoch).permutation(len(self.tokens))

    def next(self) -> Batch:
        epoch, step = divmod(self.position, self.per_epoch)
        if not hasattr(self, "_epoch") or self._epoch != epoch:
            self._epoch, self._perm = epoch, self._order(epoch)
        start = (step * self.world + self.rank) * self.micro_batch
        index = np.sort(self._perm[start : start + self.micro_batch])
        self.position += 1
        rows = torch.from_numpy(np.asarray(self.tokens[index], dtype=np.int64))
        return Batch(rows.to(self.device, non_blocking=True), self.sources[index])

    def state_dict(self) -> dict:
        return {"position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        self.position, self.seed = int(state["position"]), int(state["seed"])


class KDPool:
    """Teacher top-k targets for pool sequences, as written by ``train.py --teacher-topk-out`` (one dir per rank)."""

    def __init__(self, kd_dir: Path, student_dir: Path | None = None):
        """``student_dir``: the frozen start model's top-k on the same sequences (ensemble-distillation target)."""
        self.shards = []
        self.student = student_dir is not None
        for rank_dir in sorted(Path(kd_dir).glob("rank*"), key=lambda p: int(p.name[4:])):
            tokens = np.load(rank_dir / "tokens.npy", mmap_mode="r")
            if len(tokens):
                shard = [tokens, np.load(rank_dir / "topk_idx.npy", mmap_mode="r"),
                         np.load(rank_dir / "topk_logp.npy", mmap_mode="r")]
                if student_dir is not None:
                    other = Path(student_dir) / rank_dir.name
                    if not np.array_equal(np.load(other / "pool_index.npy"), np.load(rank_dir / "pool_index.npy")):
                        raise ValueError(f"{other} labels different sequences than {rank_dir}")
                    shard += [np.load(other / "topk_idx.npy", mmap_mode="r"), np.load(other / "topk_logp.npy", mmap_mode="r")]
                self.shards.append(tuple(shard))
        if not self.shards:
            raise ValueError(f"no teacher top-k shards under {kd_dir}")
        self.offsets = np.cumsum([0] + [len(s[0]) for s in self.shards])
        self.n = int(self.offsets[-1])
        self.k = int(self.shards[0][1].shape[-1])

    def rows(self, index: np.ndarray) -> tuple[np.ndarray, ...]:
        """(tokens, teacher idx, teacher logp[, student idx, student logp]) for global row indices."""
        columns = [[] for _ in self.shards[0]]
        for i in index:
            shard = int(np.searchsorted(self.offsets, i, side="right") - 1)
            row = int(i - self.offsets[shard])
            for column, array in zip(columns, self.shards[shard]):
                column.append(np.asarray(array[row]))
        return tuple(np.stack(c) for c in columns)


@dataclass
class KDBatch(Batch):
    kd: tuple | None = None  # (idx [B, T-1, k] int64, logp [B, T-1, k] fp32, weight [B] fp32) on device


class MixedLoader:
    """Micro-batches of ``micro_batch - n_kd`` main-pool sequences (``kd_source`` excluded) + ``n_kd`` sequences
    with teacher targets (KD weight ``kd_lambda``); deterministic per (seed, rank), rank-strided like PoolLoader."""

    def __init__(self, pool_dir: Path, kd: KDPool, *, kd_source: str, kd_frac: float, kd_lambda: float,
                 rank: int, world: int, micro_batch: int, seed: int, device):
        self.tokens = np.load(Path(pool_dir) / "tokens.npy", mmap_mode="r")
        self.sources = np.load(Path(pool_dir) / "sources.npy")
        self.main_index = np.flatnonzero(self.sources != kd_source)
        self.kd, self.kd_source, self.kd_lambda = kd, kd_source, float(kd_lambda)
        self.n_kd = max(1, int(round(micro_batch * kd_frac)))
        self.n_main = micro_batch - self.n_kd
        self.rank, self.world, self.seed, self.device = rank, world, seed, device
        self.position = 0

    def _take(self, total: int, per: int, salt: int) -> np.ndarray:
        per_epoch = total // (self.world * per)
        epoch, step = divmod(self.position, per_epoch)
        perm = np.random.default_rng(self.seed * 1000 + salt + epoch).permutation(total)
        start = (step * self.world + self.rank) * per
        return perm[start : start + per]

    def next(self) -> KDBatch:
        main = np.sort(self.main_index[self._take(len(self.main_index), self.n_main, 1)]) if self.n_main else np.zeros(0, np.int64)
        kd_rows = self._take(self.kd.n, self.n_kd, 2)
        self.position += 1
        kd_tokens, *targets = self.kd.rows(kd_rows)
        tokens = np.concatenate([np.asarray(self.tokens[main]), kd_tokens]) if len(main) else kd_tokens
        n, t = len(tokens), tokens.shape[1] - 1
        dev = self.device
        padded = []
        for array in targets:  # (teacher idx, teacher logp[, student idx, student logp]); main rows get zeros
            full = np.zeros((n, t, array.shape[-1]), dtype=np.int64 if array.dtype.kind in "iu" else np.float32)
            full[len(main):] = array
            padded.append(torch.from_numpy(full).to(dev, non_blocking=True))
        weight = np.zeros(n, dtype=np.float32)
        weight[len(main):] = 1.0 if self.kd.student else self.kd_lambda  # mixture mode: row mask (weights in the head)
        sources = np.concatenate([self.sources[main], np.asarray([self.kd_source] * len(kd_rows))])
        kd = (*padded, torch.from_numpy(weight).to(dev, non_blocking=True))
        return KDBatch(torch.from_numpy(tokens.astype(np.int64)).to(dev, non_blocking=True), sources, kd)

    def state_dict(self) -> dict:
        return {"position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        self.position, self.seed = int(state["position"]), int(state["seed"])


class WeightedPoolLoader:
    """Like PoolLoader, but each micro-batch holds a fixed number of sequences per source (largest-remainder split of
    ``source_weights``), each source read from its own seeded permutation -- e.g. to spend more tokens on the source
    with the most learnable signal (txt360 gains most per king step) than its eval weight."""

    def __init__(self, pool_dir: Path, *, source_weights: dict, rank: int, world: int, micro_batch: int, seed: int, device,
                 source_pools: dict | None = None):
        """``source_pools``: {source: pool_dir} -- read that source from its own (e.g. fresh, larger) pool."""
        self.pools = {"": (np.load(Path(pool_dir) / "tokens.npy", mmap_mode="r"), np.load(Path(pool_dir) / "sources.npy"))}
        for source, extra in (source_pools or {}).items():
            self.pools[source] = (np.load(Path(extra) / "tokens.npy", mmap_mode="r"), np.load(Path(extra) / "sources.npy"))
        names = sorted(source_weights)
        weights = np.asarray([float(source_weights[n]) for n in names])
        raw = weights / weights.sum() * micro_batch
        counts = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - counts))[: micro_batch - counts.sum()]:
            counts[i] += 1
        self.plan = []
        for n, c in zip(names, counts):
            if c > 0:
                key = n if n in self.pools else ""
                self.plan.append((n, int(c), np.flatnonzero(self.pools[key][1] == n), key))
        missing = [n for n, _, idx, _ in self.plan if len(idx) == 0]
        if missing:
            raise ValueError(f"pool has no sequences of {missing}")
        self.rank, self.world, self.seed, self.device = rank, world, seed, device
        self.position = 0

    def describe(self) -> str:
        return ", ".join(f"{n} {c}" + (f" (own pool, {len(idx)} seqs)" if key else "") for n, c, idx, key in self.plan)

    def next(self) -> Batch:
        tokens, sources = [], []
        for salt, (_, count, index, key) in enumerate(self.plan):
            per_epoch = len(index) // (self.world * count)
            epoch, step = divmod(self.position, per_epoch)
            perm = np.random.default_rng(self.seed * 1000 + 17 * salt + epoch).permutation(len(index))
            start = (step * self.world + self.rank) * count
            rows = np.sort(index[perm[start : start + count]])
            pool_tokens, pool_sources = self.pools[key]
            tokens.append(np.asarray(pool_tokens[rows], dtype=np.int64))
            sources.append(pool_sources[rows])
        self.position += 1
        batch = torch.from_numpy(np.concatenate(tokens))
        return Batch(batch.to(self.device, non_blocking=True), np.concatenate(sources))

    def state_dict(self) -> dict:
        return {"position": self.position, "seed": self.seed}

    def load_state_dict(self, state: dict) -> None:
        self.position, self.seed = int(state["position"]), int(state["seed"])
