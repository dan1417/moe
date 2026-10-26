"""Multi-dataset weighted streaming data pipeline for Teutonic SN3."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

log = logging.getLogger(__name__)

DEFAULT_MANIFEST_URL = "https://teutonic.ai/datasets/manifest.json"
MANIFEST_CACHE_TTL_SEC = 60
MANIFEST_REFRESH_INTERVAL = 100
SEQUENCE_LENGTH = 2048
EVAL_SEED = 99999
EVAL_CACHE_DIRNAME = "eval_dataset_cache"
EVAL_CACHE_META = "meta.json"
DEFAULT_TIMEOUT = 60
DEFAULT_EVAL_PREFETCH_WORKERS = 32
USER_AGENT = "teutonic-miner/1.0"

def _stable_int_hash(*items: object) -> int:
    """Deterministic int hash across Python processes (unlike built-in `hash()`)."""
    h = hashlib.sha256()
    for item in items:
        h.update(str(item).encode("utf-8"))
        h.update(b"\x00")
    # 64-bit seed for NumPy RNG.
    return int.from_bytes(h.digest()[:8], byteorder="big", signed=False)

def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA256 of a file (used for shard integrity checks)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_fingerprint(manifest: "Manifest") -> str:
    payload = [
        {
            "name": source.name,
            "proportion": source.proportion,
            "shards": [(shard.key, shard.sha256, shard.size_bytes) for shard in source.shards],
        }
        for source in manifest.sources
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_rank0() -> bool:
    try:
        import torch.distributed as dist

        if dist.is_initialized():
            return dist.get_rank() == 0
    except (ImportError, RuntimeError):
        pass
    return True


def _rank0_log(level: int, msg: str, *args: Any) -> None:
    if _is_rank0():
        log.log(level, msg, *args)


@dataclass(frozen=True)
class ShardInfo:
    """Metadata for a single pre-tokenized shard."""

    key: str
    url: str
    sha256: str
    size_bytes: int
    n_tokens: int


@dataclass
class DatasetSource:
    """One dataset source from the Teutonic manifest."""

    name: str
    proportion: float
    manifest_url: str
    total_shards: int
    sequence_length: int
    dtype: str
    base_url: str
    shards: list[ShardInfo] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"DatasetSource(name={self.name!r}, proportion={self.proportion}, "
            f"total_shards={self.total_shards}, loaded_shards={len(self.shards)})"
        )


@dataclass
class CollatedBatch:
    """Batch returned by the training DataLoader collate function."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    datasets: list[str]

    def __repr__(self) -> str:
        return (
            f"CollatedBatch(shape={tuple(self.input_ids.shape)}, "
            f"n_datasets={len(self.datasets)})"
        )


class Manifest:
    """Fetches and caches the live Teutonic dataset manifest."""

    def __init__(
        self,
        manifest_url: str = DEFAULT_MANIFEST_URL,
        cache_ttl_sec: int = MANIFEST_CACHE_TTL_SEC,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.manifest_url = manifest_url
        self.cache_ttl_sec = cache_ttl_sec
        self.timeout = timeout
        self._sources: list[DatasetSource] = []
        self._delta_threshold: float = 0.1
        self._eval_n: int = 2000
        self._config_version: str = ""
        self._last_fetch: float = 0.0
        self._lock = threading.Lock()

    @property
    def delta_threshold(self) -> float:
        return self._delta_threshold

    @property
    def eval_n(self) -> int:
        return self._eval_n

    @property
    def config_version(self) -> str:
        return self._config_version

    @property
    def sources(self) -> list[DatasetSource]:
        return list(self._sources)

    def __repr__(self) -> str:
        return (
            f"Manifest(url={self.manifest_url!r}, config_version={self._config_version!r}, "
            f"sources={len(self._sources)}, eval_n={self._eval_n})"
        )

    def fetch(self, force: bool = False) -> list[DatasetSource]:
        """Return dataset sources, refreshing from the network when stale."""
        with self._lock:
            now = time.monotonic()
            if not force and self._sources and (now - self._last_fetch) < self.cache_ttl_sec:
                return list(self._sources)

            previous_version = self._config_version
            payload = self._http_get_json(self.manifest_url)
            self._delta_threshold = float(payload.get("delta_threshold", 0.1))
            self._eval_n = int(payload.get("eval_n", 2000))
            new_version = str(payload.get("config_version", ""))
            if previous_version and new_version and previous_version != new_version:
                _rank0_log(
                    logging.WARNING,
                    "Manifest config_version changed: %s -> %s",
                    previous_version,
                    new_version,
                )
            self._config_version = new_version

            sources: list[DatasetSource] = []
            for entry in payload.get("sources", []):
                manifest_url = str(entry["manifest_url"])
                base_url = manifest_url.rsplit("/", 1)[0]
                source = DatasetSource(
                    name=str(entry["name"]),
                    proportion=float(entry["proportion"]),
                    manifest_url=manifest_url,
                    total_shards=int(entry["total_shards"]),
                    sequence_length=int(entry.get("sequence_length", SEQUENCE_LENGTH)),
                    dtype=str(entry.get("dtype", "uint32")),
                    base_url=base_url,
                )
                source.shards = self._load_source_shards(source)
                sources.append(source)

            self._sources = sources
            self._last_fetch = now
            _rank0_log(
                logging.INFO,
                "Loaded manifest with %d sources (config_version=%s)",
                len(sources),
                self._config_version,
            )
            return list(self._sources)

    def _http_get_json(self, url: str) -> dict[str, Any]:
        response = requests.get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": USER_AGENT},
        )
        response.raise_for_status()
        return response.json()

    def _load_source_shards(self, source: DatasetSource) -> list[ShardInfo]:
        payload = self._http_get_json(source.manifest_url)
        manifest_shards = payload.get("shards", [])
        if len(manifest_shards) != source.total_shards:
            _rank0_log(
                logging.WARNING,
                "Shard count mismatch for %s: manifest=%d expected=%d",
                source.name,
                len(manifest_shards),
                source.total_shards,
            )
        shards: list[ShardInfo] = []
        for shard in manifest_shards:
            key = str(shard["key"])
            shards.append(
                ShardInfo(
                    key=key,
                    url=f"{source.base_url}/{key}",
                    sha256=str(shard["sha256"]),
                    size_bytes=int(shard.get("size_bytes", 0)),
                    n_tokens=int(shard.get("n_tokens", 0)),
                )
            )
        return shards


class ShardCache:
    """LRU disk cache for dataset shards with concurrent downloads."""

    def __init__(
        self,
        cache_dir: str | Path,
        max_gb: float = 500.0,
        timeout: float = DEFAULT_TIMEOUT,
        max_workers: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = int(max_gb * (1024**3))
        self.timeout = timeout
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._inflight: dict[str, Future[Path]] = {}
        self._inflight_targets: set[Path] = set()
        self._lru: OrderedDict[str, Path] = OrderedDict()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __repr__(self) -> str:
        return (
            f"ShardCache(cache_dir={self.cache_dir!r}, max_gb={self.max_bytes / 1024**3:.1f}, "
            f"entries={len(self._lru)})"
        )

    def is_cached(self, dataset_name: str, key: str) -> bool:
        """Return True if the shard is already on disk (current or legacy path)."""
        target = self._target_path(dataset_name, key)
        if target.exists():
            return True
        return self._legacy_target_path(dataset_name, key).exists()

    def prefetch_shards(self, dataset_name: str, shards: list[ShardInfo]) -> None:
        """Download many shards concurrently before sequential sequence sampling."""
        pending: list[tuple[ShardInfo, Path, str]] = []
        for shard in shards:
            cache_key = self._cache_key(dataset_name, shard.key)
            target = self._target_path(dataset_name, shard.key)
            if self.is_cached(dataset_name, shard.key):
                with self._lock:
                    if target.exists():
                        self._lru[cache_key] = target
                    else:
                        self._lru[cache_key] = self._legacy_target_path(dataset_name, shard.key)
                continue
            pending.append((shard, target, cache_key))

        if not pending:
            return

        _rank0_log(
            logging.INFO,
            "Prefetching %d/%d shards for %s (%d download workers)...",
            len(pending),
            len(shards),
            dataset_name,
            self.max_workers,
        )

        futures: dict[str, Future[Path]] = {}
        future_targets: dict[Future[Path], Path] = {}
        with self._lock:
            for shard, target, cache_key in pending:
                if self.is_cached(dataset_name, shard.key):
                    continue
                if cache_key in self._inflight:
                    futures[cache_key] = self._inflight[cache_key]
                    future_targets[self._inflight[cache_key]] = target
                    continue
                self._inflight_targets.add(target)
                future = self._executor.submit(
                    self._download_shard, dataset_name, shard.key, shard.url, shard.sha256, target
                )
                self._inflight[cache_key] = future
                futures[cache_key] = future
                future_targets[future] = target

        completed = 0
        total = len(futures)
        completed_paths: dict[str, Path] = {}
        for future in as_completed(futures.values()):
            path = future.result(timeout=self.timeout * 10)
            completed += 1
            for cache_key, inflight in futures.items():
                if inflight is future:
                    completed_paths[cache_key] = path
                    break
            if completed % 50 == 0 or completed == total:
                _rank0_log(
                    logging.INFO,
                    "Prefetch %s: %d/%d shards downloaded",
                    dataset_name,
                    completed,
                    total,
                )

        with self._lock:
            for cache_key, future in futures.items():
                self._inflight.pop(cache_key, None)
                path = completed_paths.get(cache_key)
                if path is not None:
                    self._lru[cache_key] = path
                    self._inflight_targets.discard(path)
                elif future in future_targets:
                    self._inflight_targets.discard(future_targets[future])
            self._evict_if_needed()

    def load_shard(
        self,
        dataset_name: str,
        key: str,
        url: str,
        *,
        expected_sha256: str | None = None,
    ) -> np.ndarray:
        """Return a memory-mapped uint32 shard array."""
        path = self._ensure_cached(dataset_name, key, url, expected_sha256=expected_sha256)
        with self._lock:
            self._lru.move_to_end(self._cache_key(dataset_name, key))
        return np.load(path, mmap_mode="r")

    def _cache_key(self, dataset_name: str, key: str) -> str:
        digest = hashlib.sha256(f"{dataset_name}:{key}".encode()).hexdigest()[:16]
        safe_name = key.replace("/", "__")
        return f"{dataset_name}__{digest}__{safe_name}"

    def _target_path(self, dataset_name: str, key: str) -> Path:
        cache_key = self._cache_key(dataset_name, key)
        # Shard keys already end in ".npy"; avoid "...npy.npy" cache filenames.
        filename = cache_key if cache_key.endswith(".npy") else f"{cache_key}.npy"
        return self.cache_dir / filename

    def _legacy_target_path(self, dataset_name: str, key: str) -> Path:
        """Pre-fix cache path that always appended .npy (produced *.npy.npy)."""
        return self.cache_dir / f"{self._cache_key(dataset_name, key)}.npy"

    @staticmethod
    def _verification_path(path: Path) -> Path:
        return path.with_name(f"{path.name}.verified.json")

    def _verify_cached(self, path: Path, expected_sha256: str | None) -> None:
        if expected_sha256 is None:
            return
        stat = path.stat()
        marker = self._verification_path(path)
        if marker.exists():
            try:
                payload = json.loads(marker.read_text())
                if (
                    payload.get("sha256") == expected_sha256
                    and int(payload.get("size", -1)) == stat.st_size
                    and int(payload.get("mtime_ns", -1)) == stat.st_mtime_ns
                ):
                    return
            except (OSError, ValueError, TypeError):
                pass
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != expected_sha256:
            path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            raise RuntimeError(
                f"Cached shard sha256 mismatch: path={path} "
                f"expected={expected_sha256} actual={actual_sha256}"
            )
        stat = path.stat()
        marker.write_text(
            json.dumps(
                {
                    "sha256": expected_sha256,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        )

    def _ensure_cached(
        self,
        dataset_name: str,
        key: str,
        url: str,
        *,
        expected_sha256: str | None = None,
    ) -> Path:
        cache_key = self._cache_key(dataset_name, key)
        target = self._target_path(dataset_name, key)
        if target.exists():
            self._verify_cached(target, expected_sha256)
            with self._lock:
                self._lru[cache_key] = target
            return target

        legacy = self._legacy_target_path(dataset_name, key)
        if legacy.exists() and legacy != target:
            try:
                os.replace(legacy, target)
                self._verify_cached(target, expected_sha256)
                with self._lock:
                    self._lru[cache_key] = target
                return target
            except OSError:
                self._verify_cached(legacy, expected_sha256)
                with self._lock:
                    self._lru[cache_key] = legacy
                return legacy

        with self._lock:
            if cache_key in self._inflight:
                future = self._inflight[cache_key]
            else:
                self._inflight_targets.add(target)
                future = self._executor.submit(
                    self._download_shard,
                    dataset_name,
                    key,
                    url,
                    expected_sha256,
                    target,
                )
                self._inflight[cache_key] = future

        try:
            path = future.result(timeout=self.timeout * 10)
        except Exception as exc:
            log.warning("Shard download failed for %s (%s): %s", dataset_name, key, exc)
            raise
        finally:
            with self._lock:
                self._inflight.pop(cache_key, None)
                self._inflight_targets.discard(target)

        with self._lock:
            self._lru[cache_key] = path
            self._evict_if_needed()
        return path

    def _download_shard(
        self,
        dataset_name: str,
        key: str,
        url: str,
        expected_sha256: str | None,
        target: Path,
    ) -> Path:
        if target.exists():
            self._verify_cached(target, expected_sha256)
            return target

        tmp_path = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            response = requests.get(
                url,
                timeout=self.timeout,
                stream=True,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            content_length = int(response.headers.get("Content-Length", "0") or 0)
            free_bytes = shutil.disk_usage(target.parent).free
            reserve = max(1024**3, content_length // 20)
            if content_length > 0 and free_bytes < content_length + reserve:
                raise OSError(
                    f"Insufficient disk space for {dataset_name}:{key}: "
                    f"need at least {content_length + reserve:,} bytes, free={free_bytes:,}"
                )
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if target.exists():
                return target
            if expected_sha256 is not None:
                actual_sha256 = _sha256_file(tmp_path)
                if actual_sha256 != expected_sha256:
                    tmp_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Shard sha256 mismatch for {dataset_name}:{key}: expected={expected_sha256} actual={actual_sha256}"
                    )
            os.replace(tmp_path, target)
            self._verify_cached(target, expected_sha256)
            return target
        finally:
            tmp_path.unlink(missing_ok=True)

    def _evict_if_needed(self) -> None:
        attempts = 0
        max_attempts = max(1, len(self._lru))
        while self._lru and self._current_size_bytes() > self.max_bytes and attempts < max_attempts:
            cache_key, path = self._lru.popitem(last=False)
            attempts += 1
            if path in self._inflight_targets or not path.exists():
                self._lru[cache_key] = path
                continue
            path.unlink(missing_ok=True)
            self._verification_path(path).unlink(missing_ok=True)
            attempts = 0
            max_attempts = max(1, len(self._lru))

    def _current_size_bytes(self) -> int:
        total = 0
        for path in self._lru.values():
            if path.exists():
                total += path.stat().st_size
        return total


class WeightedShardSampler:
    """Infinite weighted iterator over pre-tokenized shards."""

    def __init__(
        self,
        manifest: Manifest,
        shard_cache: ShardCache,
        rank: int,
        world_size: int,
        base_seed: int = 42,
        refresh_interval: int = MANIFEST_REFRESH_INTERVAL,
    ) -> None:
        self.manifest = manifest
        self.shard_cache = shard_cache
        self.rank = rank
        self.world_size = world_size
        self.base_seed = base_seed
        self.refresh_interval = refresh_interval
        self._yield_count = 0
        self._proportions: dict[str, float] = {}
        self._rank_shards: dict[str, list[ShardInfo]] = {}
        self._epoch = 0
        self._refresh_state(force=True)

    def __repr__(self) -> str:
        return (
            f"WeightedShardSampler(rank={self.rank}, world_size={self.world_size}, "
            f"epoch={self._epoch}, yields={self._yield_count})"
        )

    def __iter__(self) -> Iterator[tuple[str, np.ndarray]]:
        return self

    def __next__(self) -> tuple[str, np.ndarray]:
        if self._yield_count > 0 and self._yield_count % self.refresh_interval == 0:
            self._refresh_state(force=True)

        source_name = self._sample_source()
        shard = self._next_shard(source_name)
        array = self.shard_cache.load_shard(
            source_name,
            shard.key,
            shard.url,
            expected_sha256=shard.sha256,
        )
        sequence = self._sample_sequence(source_name, shard.key, array)
        self._yield_count += 1
        return source_name, sequence

    def _refresh_state(self, force: bool = False) -> None:
        sources = self.manifest.fetch(force=force)
        new_proportions = {source.name: source.proportion for source in sources}
        if self._proportions and new_proportions != self._proportions:
            _rank0_log(logging.WARNING, "Dataset proportions changed: %s -> %s", self._proportions, new_proportions)
        self._proportions = new_proportions
        self._rank_shards = {
            source.name: source.shards[self.rank :: self.world_size]
            for source in sources
        }

    def _sample_source(self) -> str:
        names = list(self._proportions.keys())
        weights = np.array([self._proportions[name] for name in names], dtype=np.float64)
        weights /= weights.sum()
        rng = np.random.default_rng(self.base_seed + self.rank + self._yield_count)
        index = int(rng.choice(len(names), p=weights))
        return names[index]

    def _next_shard(self, source_name: str) -> ShardInfo:
        shards = self._rank_shards.get(source_name, [])
        if not shards:
            raise RuntimeError(f"No shards assigned to rank {self.rank} for source {source_name}")
        rng = np.random.default_rng(self.base_seed + self.rank + self._epoch)
        shuffled = list(shards)
        rng.shuffle(shuffled)
        index = self._yield_count % len(shuffled)
        if index == 0 and self._yield_count > 0:
            self._epoch += 1
        return shuffled[index]

    def _sample_sequence(self, source_name: str, shard_key: str, array: np.ndarray) -> np.ndarray:
        if array.size < SEQUENCE_LENGTH:
            raise ValueError(f"Shard {shard_key} is smaller than one sequence ({array.size} tokens)")
        n_sequences = array.size // SEQUENCE_LENGTH
        flat = array[: n_sequences * SEQUENCE_LENGTH]
        sequences = flat.reshape(n_sequences, SEQUENCE_LENGTH)
        stable_seed = _stable_int_hash(
            self.base_seed,
            self.rank,
            self._yield_count,
            source_name,
            shard_key,
        )
        rng = np.random.default_rng(stable_seed)
        index = int(rng.integers(0, n_sequences))
        return np.asarray(sequences[index], dtype=np.uint32)


class TeutonicDataset(IterableDataset):
    """Iterable dataset wrapping the weighted shard sampler."""

    def __init__(
        self,
        sampler: WeightedShardSampler,
        rank: int,
        world_size: int,
    ) -> None:
        self.sampler = sampler
        self.rank = rank
        self.world_size = world_size

    def __repr__(self) -> str:
        return f"TeutonicDataset(rank={self.rank}, world_size={self.world_size})"

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            effective_rank = self.rank
            effective_world_size = self.world_size
        else:
            effective_rank = self.rank * worker_info.num_workers + worker_info.id
            effective_world_size = self.world_size * worker_info.num_workers

        local_sampler = WeightedShardSampler(
            manifest=self.sampler.manifest,
            shard_cache=self.sampler.shard_cache,
            rank=effective_rank,
            world_size=effective_world_size,
            base_seed=self.sampler.base_seed,
            refresh_interval=self.sampler.refresh_interval,
        )
        for dataset_name, sequence in local_sampler:
            input_ids = torch.from_numpy(sequence.astype(np.int64))
            yield {
                "input_ids": input_ids,
                "labels": input_ids.clone(),
                "dataset": dataset_name,
            }


class SequencePoolDataset(Dataset):
    """Map-style dataset backed by the precomputed sequence pool.

    The pool stores *global* sequence rows. Each rank reads a deterministic,
    strided subset so the overall ordering is stable and step-based resume
    aligns to checkpoint `meta.json.step`.
    """

    def __init__(
        self,
        *,
        pool_dir: Path,
        rank: int,
        world_size: int,
        start_step: int,
        total_steps: int,
        sequences_per_step_per_rank: int,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> None:
        self.pool_dir = pool_dir
        self.rank = rank
        self.world_size = world_size
        self.start_step = start_step
        self.total_steps = total_steps
        self.sequences_per_step_per_rank = sequences_per_step_per_rank
        self.sequence_length = sequence_length

        meta = json.loads((pool_dir / "pool_meta.json").read_text())
        required_meta = {
            "sources",
            "sequence_length",
            "world_size",
            "total_steps",
            "sequences_per_step_per_rank",
            "total_sequences_global",
        }
        missing_meta = sorted(required_meta - meta.keys())
        if missing_meta:
            raise ValueError(f"sequence_pool metadata missing required keys: {missing_meta}")
        self.source_names: list[str] = list(meta["sources"])
        if not self.source_names:
            raise ValueError("sequence_pool sources must not be empty")

        expected_world_size = int(meta["world_size"])
        if expected_world_size != world_size:
            raise ValueError(
                f"sequence_pool world_size mismatch: pool={expected_world_size} runtime={world_size}"
            )
        expected_total_steps = int(meta["total_steps"])
        if expected_total_steps != total_steps:
            raise ValueError(
                f"sequence_pool total_steps mismatch: pool={expected_total_steps} runtime={total_steps}"
            )
        if int(meta["sequence_length"]) != sequence_length:
            raise ValueError(
                f"sequence_pool sequence_length mismatch: pool={meta['sequence_length']} "
                f"runtime={sequence_length}"
            )
        if int(meta["sequences_per_step_per_rank"]) != sequences_per_step_per_rank:
            raise ValueError(
                "sequence_pool batch geometry mismatch: "
                f"pool={meta['sequences_per_step_per_rank']} "
                f"runtime={sequences_per_step_per_rank}"
            )

        sequences_path = pool_dir / "pool_sequences.npy"
        source_ids_path = pool_dir / "pool_source_ids.npy"
        self.sequences = np.load(sequences_path, mmap_mode="r")
        self.source_ids = np.load(source_ids_path, mmap_mode="r")

        if self.sequences.ndim != 2 or self.sequences.shape[1] != sequence_length:
            raise ValueError(
                f"Unexpected pool_sequences shape: got {self.sequences.shape}, expected (*,{sequence_length})"
            )

        if self.source_ids.ndim != 1 or self.source_ids.shape[0] != self.sequences.shape[0]:
            raise ValueError(
                "Unexpected pool_source_ids shape: "
                f"got {self.source_ids.shape}, expected ({self.sequences.shape[0]},)"
            )
        expected_rows = int(meta["total_sequences_global"])
        if self.sequences.shape[0] != expected_rows:
            raise ValueError(
                f"sequence_pool row count mismatch: metadata={expected_rows} "
                f"array={self.sequences.shape[0]}"
            )
        if self.source_ids.size and int(self.source_ids.max()) >= len(self.source_names):
            raise ValueError("sequence_pool source id exceeds sources metadata")

        self._start_seq_pos_per_rank = start_step * sequences_per_step_per_rank
        self._total_sequences_per_rank = total_steps * sequences_per_step_per_rank
        self._length = max(0, self._total_sequences_per_rank - self._start_seq_pos_per_rank)

    def __repr__(self) -> str:
        return (
            "SequencePoolDataset("
            f"pool_dir={str(self.pool_dir)!r}, rank={self.rank}, world_size={self.world_size}, "
            f"start_step={self.start_step}, length={self._length}"
            ")"
        )

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._length:
            raise IndexError(idx)
        global_pos = self.rank + (self._start_seq_pos_per_rank + idx) * self.world_size
        source_id = int(self.source_ids[global_pos])
        dataset_name = self.source_names[source_id]

        # Model expects int64 token ids.
        seq_u32: np.ndarray = self.sequences[global_pos]
        # Casting first avoids PyTorch warnings about non-writable NumPy memmaps.
        seq_i64: np.ndarray = seq_u32.astype(np.int64, copy=False)
        input_ids = torch.from_numpy(seq_i64)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "dataset": dataset_name,
        }


def _collate_batch(samples: list[dict[str, Any]]) -> CollatedBatch:
    input_ids = torch.stack([sample["input_ids"] for sample in samples], dim=0)
    labels = torch.stack([sample["labels"] for sample in samples], dim=0)
    datasets = [str(sample["dataset"]) for sample in samples]
    return CollatedBatch(input_ids=input_ids, labels=labels, datasets=datasets)


def build_train_loader(
    cfg: dict[str, Any],
    rank: int,
    world_size: int,
    *,
    start_step: int = 0,
    num_workers_override: int | None = None,
) -> DataLoader:
    """Build the distributed training DataLoader."""
    data_cfg = cfg.get("data", {})
    training_cfg = cfg.get("training", {})

    micro_batch_size = int(training_cfg.get("micro_batch_size", 1))
    grad_accum_steps = int(training_cfg.get("gradient_accumulation_steps", 1))
    total_steps = int(training_cfg.get("total_steps", 0))

    pool_cfg = data_cfg.get("sequence_pool", {})
    if bool(pool_cfg.get("enabled", False)):
        pool_dir = Path(pool_cfg.get("pool_dir", "./sequence_pool"))
        pool_meta_path = pool_dir / "pool_meta.json"
        if not pool_meta_path.exists():
            raise FileNotFoundError(
                f"sequence_pool is enabled but pool_meta.json is missing at {pool_meta_path}"
            )
        sequences_per_step_per_rank = micro_batch_size * grad_accum_steps
        dataset = SequencePoolDataset(
            pool_dir=pool_dir,
            rank=rank,
            world_size=world_size,
            start_step=start_step,
            total_steps=total_steps,
            sequences_per_step_per_rank=sequences_per_step_per_rank,
        )

        num_workers = (
            int(num_workers_override)
            if num_workers_override is not None
            else int(data_cfg.get("num_workers", 4))
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": micro_batch_size,
            "num_workers": num_workers,
            "pin_memory": bool(data_cfg.get("pin_memory", True)),
            "collate_fn": _collate_batch,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 8))
            loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        return DataLoader(dataset, **loader_kwargs)

    # Default runtime shard sampling (infinite iterator).
    manifest = Manifest(
        manifest_url=data_cfg.get("manifest_url", DEFAULT_MANIFEST_URL),
        cache_ttl_sec=int(data_cfg.get("manifest_cache_ttl_sec", MANIFEST_CACHE_TTL_SEC)),
    )
    manifest.fetch(force=True)
    shard_cache = ShardCache(
        cache_dir=data_cfg.get("cache_dir", "./shard_cache"),
        max_gb=float(data_cfg.get("cache_max_gb", 500)),
    )
    sampler = WeightedShardSampler(
        manifest=manifest,
        shard_cache=shard_cache,
        rank=rank,
        world_size=world_size,
        base_seed=int(training_cfg.get("base_seed", 42)),
        refresh_interval=int(data_cfg.get("manifest_refresh_interval", MANIFEST_REFRESH_INTERVAL)),
    )
    dataset = TeutonicDataset(sampler=sampler, rank=rank, world_size=world_size)
    num_workers = int(data_cfg.get("num_workers", 4))
    loader_kwargs: dict[str, Any] = {
        "batch_size": micro_batch_size,
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "collate_fn": _collate_batch,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 8))
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
    return DataLoader(dataset, **loader_kwargs)


class EvalDataset:
    """Pre-downloaded evaluation sequences stored on CPU."""

    def __init__(
        self,
        manifest: Manifest,
        shard_cache: ShardCache,
        seed: int = EVAL_SEED,
    ) -> None:
        self.manifest = manifest
        self.shard_cache = shard_cache
        self.seed = seed
        self._batches: dict[str, torch.Tensor] = {}

    def __repr__(self) -> str:
        sizes = {name: tensor.shape[0] for name, tensor in self._batches.items()}
        return f"EvalDataset(sources={list(self._batches.keys())}, sizes={sizes})"

    def build(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Pre-download deterministic evaluation sequences per manifest source."""
        del timeout  # reserved for future network-bound eval prefetch controls
        sources = self.manifest.fetch(force=True)
        eval_n = self.manifest.eval_n
        for source in sources:
            n_source = max(1, round(eval_n * source.proportion))
            _rank0_log(
                logging.INFO,
                "EvalDataset fetching %d sequences for %s (downloading shards, no per-shard log)...",
                n_source,
                source.name,
            )
            sequences = self._collect_sequences(source, n_source)
            self._batches[source.name] = torch.stack(sequences, dim=0)
            _rank0_log(
                logging.INFO,
                "EvalDataset built %d sequences for %s",
                n_source,
                source.name,
            )

    def get_batch(self, source_name: str) -> torch.Tensor:
        if source_name not in self._batches:
            raise KeyError(f"Unknown eval source: {source_name}")
        return self._batches[source_name]

    @property
    def source_names(self) -> list[str]:
        return list(self._batches.keys())

    def save_cache(self, cache_dir: str | Path) -> Path:
        """Persist assembled eval sequences for fast reload during offline eval."""
        out = Path(cache_dir) / EVAL_CACHE_DIRNAME
        out.mkdir(parents=True, exist_ok=True)
        for name, tensor in self._batches.items():
            torch.save(tensor, out / f"{name}.pt")
        meta = {
            "seed": self.seed,
            "eval_n": self.manifest.eval_n,
            "config_version": self.manifest.config_version,
            "manifest_fingerprint": _manifest_fingerprint(self.manifest),
            "sources": {name: list(tensor.shape) for name, tensor in self._batches.items()},
        }
        (out / EVAL_CACHE_META).write_text(json.dumps(meta, indent=2))
        _rank0_log(logging.INFO, "Saved eval dataset cache to %s", out)
        return out

    @classmethod
    def load_cache(
        cls,
        cache_dir: str | Path,
        manifest: Manifest,
        shard_cache: ShardCache,
    ) -> EvalDataset | None:
        """Load a previously saved eval dataset if manifest metadata still matches."""
        root = Path(cache_dir) / EVAL_CACHE_DIRNAME
        meta_path = root / EVAL_CACHE_META
        if not meta_path.exists():
            return None
        manifest.fetch(force=True)
        meta = json.loads(meta_path.read_text())
        if meta.get("config_version") != manifest.config_version:
            _rank0_log(
                logging.WARNING,
                "Eval cache stale (config_version %s != %s); rebuilding",
                meta.get("config_version"),
                manifest.config_version,
            )
            return None
        if meta.get("eval_n") != manifest.eval_n:
            _rank0_log(logging.WARNING, "Eval cache stale (eval_n mismatch); rebuilding")
            return None
        if int(meta.get("seed", -1)) != EVAL_SEED:
            _rank0_log(logging.WARNING, "Eval cache stale (seed mismatch); rebuilding")
            return None
        if meta.get("manifest_fingerprint") != _manifest_fingerprint(manifest):
            _rank0_log(logging.WARNING, "Eval cache stale (manifest fingerprint mismatch); rebuilding")
            return None

        dataset = cls(manifest=manifest, shard_cache=shard_cache, seed=int(meta.get("seed", EVAL_SEED)))
        for name, shape in meta.get("sources", {}).items():
            path = root / f"{name}.pt"
            if not path.exists():
                _rank0_log(logging.WARNING, "Eval cache missing %s; rebuilding", path)
                return None
            tensor = torch.load(path, map_location="cpu", weights_only=True)
            if list(tensor.shape) != shape:
                _rank0_log(logging.WARNING, "Eval cache shape mismatch for %s; rebuilding", name)
                return None
            dataset._batches[name] = tensor
        _rank0_log(logging.INFO, "Loaded eval dataset cache from %s (%s)", root, meta.get("sources"))
        return dataset

    def _collect_sequences(self, source: DatasetSource, n_sequences: int) -> list[torch.Tensor]:
        rng = np.random.default_rng(self.seed)
        shard_indices = rng.choice(len(source.shards), size=n_sequences, replace=True)

        unique_indices = np.unique(shard_indices)
        unique_shards = [source.shards[int(index)] for index in unique_indices]
        n_cached = sum(1 for shard in unique_shards if self.shard_cache.is_cached(source.name, shard.key))
        _rank0_log(
            logging.INFO,
            "EvalDataset %s: %d sequences from %d unique shards (%d cached)",
            source.name,
            n_sequences,
            len(unique_shards),
            n_cached,
        )
        self.shard_cache.prefetch_shards(source.name, unique_shards)

        sequences: list[torch.Tensor] = []
        for i, shard_index in enumerate(shard_indices):
            shard = source.shards[int(shard_index)]
            array = self.shard_cache.load_shard(
                source.name,
                shard.key,
                shard.url,
                expected_sha256=shard.sha256,
            )
            n_in_shard = max(1, array.size // SEQUENCE_LENGTH)
            seq_index = int(rng.integers(0, n_in_shard))
            start = seq_index * SEQUENCE_LENGTH
            chunk = array[start : start + SEQUENCE_LENGTH]
            sequences.append(torch.from_numpy(chunk.astype(np.int64)))
            if (i + 1) % 200 == 0 or i + 1 == n_sequences:
                _rank0_log(
                    logging.INFO,
                    "EvalDataset %s: sampled %d/%d sequences",
                    source.name,
                    i + 1,
                    n_sequences,
                )
        return sequences


def build_eval_dataset(
    cfg: dict[str, Any],
    *,
    save_cache: bool = False,
    prefetch_workers: int | None = None,
) -> EvalDataset:
    """Construct and populate an EvalDataset from config."""
    data_cfg = cfg.get("data", {})
    cache_dir = data_cfg.get("cache_dir", "./shard_cache")
    workers = int(
        prefetch_workers
        if prefetch_workers is not None
        else data_cfg.get("eval_prefetch_workers", DEFAULT_EVAL_PREFETCH_WORKERS)
    )
    manifest = Manifest(manifest_url=data_cfg.get("manifest_url", DEFAULT_MANIFEST_URL))
    shard_cache = ShardCache(
        cache_dir=cache_dir,
        max_gb=float(data_cfg.get("cache_max_gb", 500)),
        max_workers=workers,
    )
    cached = EvalDataset.load_cache(cache_dir, manifest=manifest, shard_cache=shard_cache)
    if cached is not None:
        return cached

    eval_dataset = EvalDataset(manifest=manifest, shard_cache=shard_cache)
    eval_dataset.build()
    if save_cache:
        eval_dataset.save_cache(cache_dir)
    return eval_dataset
