"""Validator-identical evaluation samples.

A sample is fully determined by the live evaluation config and the validator seed
material ``block_hash={block_hash}|hotkey={hotkey}``. Planning (category-stratified
shard selection) and sequence loading run through the validator's own functions, so
for a real duel's (block_hash, hotkey) this reproduces its exact 30k sequences; for a
synthetic block hash it draws a fresh sample from the same distribution.

Since validator 14a9bb2 (2026-10-09) every live config also draws ~6M tokens of complete
2049-8192-token documents (``long-documents-2k-8k-v1``), shuffled into the 30k windows.
Such samples are variable-length: ``tokens`` is then a :class:`Sequences` (flat tokens +
offsets) instead of an [n, 2048] array, and ``scored_tokens`` holds each sample's number of
scored targets under document masking (the verdict's weights).

Since validator 5118c32 (2026-10-12) windows and document fragments are read by HTTP range
(``sources.open_shard`` -> ``token_shards.NpyShard``) instead of downloading whole shards, so a
sample costs ~0.3 GB of reads instead of ~84 GB. A cached sample is only reused when it holds its
config's complete document budget (``documents_complete`` in meta.json): samples built while
documents were skipped are moved to ``samples/index_incomplete.json`` and rebuilt.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.parity import upstream

log = logging.getLogger(__name__)

LIVE_MANIFEST_URL = "https://teutonic.ai/datasets/manifest.json"
# Four competitions since validator f5bfc3f: MAIN (above) and the MATH / CODE / TEXT specialists, whose
# eval mixes draw 70% from their own split and 15% from each other split (split manifests carry explicit
# per-category weights). Until a specialist has a king, its challengers duel the MAIN king on that mix.
COMPETITIONS = ("main", "math", "code", "text")
SPLITS = ("math", "code", "text")


def competition_manifest_url(competition: str) -> str:
    if competition == "main":
        return LIVE_MANIFEST_URL
    if competition not in COMPETITIONS:
        raise ValueError(f"unknown competition {competition!r}")
    return f"https://teutonic.ai/competitions/{competition}/datasets/manifest.json"
DASHBOARD_URL = "https://teutonic.ai/dashboard.json"
USER_AGENT = "teutonic-parity-eval/1.0"
SEQ_LEN = 2048
VOCAB_SIZE = 152576
LOCAL_HOTKEY = "local-parity-eval"


def fetch_json(url: str, timeout: float = 300.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def synthetic_block_hash(tag: str) -> str:
    """Deterministic stand-in for a finalized block hash (``0x`` + 64 hex)."""
    return "0x" + hashlib.sha256(f"teutonic-parity-sample:{tag}".encode()).hexdigest()


@dataclass(frozen=True)
class LiveConfig:
    raw: dict
    settings: Any  # teutonic.evaluation.configuration.EvaluationSettings

    @property
    def config_version(self) -> str:
        return self.settings.config_version

    @property
    def n(self) -> int:
        return int(self.settings.n)

    @property
    def delta_threshold(self) -> float:
        return float(self.settings.delta_threshold)

    @property
    def shards_per_dataset(self) -> int:
        return int(self.settings.shards_per_dataset)


def _cached_dataset_manifest(source: dict, cache_dir: Path) -> dict:
    """Fetch a dataset manifest once per content hash; verify the validator digest."""
    configuration = upstream.configuration
    expected = str(source["manifest_sha256"])
    path = cache_dir / f"{source['name']}-{expected[:16]}.json"
    if path.exists():
        manifest = json.loads(path.read_text())
    else:
        log.info("fetching dataset manifest %s", source["manifest_url"])
        manifest = fetch_json(str(source["manifest_url"]), timeout=900.0)
    digest = hashlib.sha256(configuration.canonical_manifest_bytes(manifest)).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"dataset manifest {source['name']} digest {digest} != live config {expected}; "
            "the live config changed underneath us -- rerun with --refresh"
        )
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(manifest))
        tmp.replace(path)
    return manifest


def load_live_config(cache_root: Path | None = None, *, refresh: bool = False, competition: str = "main") -> LiveConfig:
    """Load the live eval config and rebuild the validator's EvaluationSettings.

    ``shards_per_dataset`` is not published in the manifest, but it is hashed into
    ``config_version``; it is recovered by matching that hash, which also proves the
    category rules and every other setting agree with the live validator.
    """
    configuration = upstream.configuration
    cache_root = cache_root or upstream.DEFAULT_CACHE_ROOT
    if competition == "mix":
        return mix_config(load_live_config(cache_root, refresh=refresh, competition="math"))
    live_path = cache_root / ("live_manifest.json" if competition == "main" else f"live_manifest_{competition}.json")
    if refresh or not live_path.exists() or time.time() - live_path.stat().st_mtime > 600:
        raw = fetch_json(competition_manifest_url(competition))
        live_path.parent.mkdir(parents=True, exist_ok=True)
        live_path.write_text(json.dumps(raw, indent=1))
    else:
        raw = json.loads(live_path.read_text())

    snapshots = []
    for source in raw["sources"]:
        manifest = _cached_dataset_manifest(source, cache_root / "manifests")
        snapshots.append(
            configuration.DatasetManifestSnapshot(
                name=str(source["name"]),
                manifest_url=str(source["manifest_url"]),
                manifest_sha256=str(source["manifest_sha256"]),
                proportion=float(source["proportion"]),
                manifest=manifest,
            )
        )
    snapshots = tuple(snapshots)

    # The long-document config is not published either; it is hashed into config_version too, so
    # the default one (6M tokens, the live index manifest) is confirmed by the same match.
    long_options = [None]
    try:  # 429 / 5xx retried with backoff (pool builds saturate r2.dev): a throttled read is not "no long-doc config"
        long_options.insert(0, _with_retry(lambda: upstream.long_documents.build_long_document_config(
            snapshots, index_manifest=index_manifest_pin(cache_root, refresh=refresh)), 8)())
    except Exception as exc:  # index unreachable or not covering these sources: window-only configs still match
        log.warning("long-document config unavailable (%s); trying window-only configs", exc)
    candidates = [int(upstream.chain_config._evaluation.get("shards_per_dataset", 4))]
    candidates += [value for value in range(1, 65) if value not in candidates]
    for shards_per_dataset in candidates:
        for long_documents in long_options:
            kwargs = dict(
                dataset_label=str(raw["dataset_label"]),
                n=int(raw["eval_n"]),
                delta_threshold=float(raw["delta_threshold"]),
                manifests=snapshots,
                shards_per_dataset=shards_per_dataset,
                long_documents=long_documents,
            )
            version = configuration.evaluation_config_version(**kwargs)
            if version == raw["config_version"]:
                settings = configuration.EvaluationSettings(config_version=version, **kwargs)
                return LiveConfig(raw=raw, settings=settings)
    raise RuntimeError(
        "could not reproduce live config_version from the published manifest; the "
        "validator's sampling policy or category rules changed -- update UPSTREAM_COMMIT"
    )


def index_manifest_pin(cache_root: Path, *, refresh: bool = False) -> dict:
    """The document-index manifest pinned by its exact bytes (re-fetched like the live config)."""
    index_manifest = upstream.module("teutonic.evaluation.index_manifest")
    path = cache_root / "doc_index_manifest_pin.json"
    if refresh or not path.exists() or time.time() - path.stat().st_mtime > 600:
        pin = index_manifest.fetch_index_manifest(index_manifest.DEFAULT_INDEX_MANIFEST_URL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(pin))
        return pin
    pin = json.loads(path.read_text())
    index_manifest.validate_index_manifest_pin(pin)
    return pin


def mix_config(base: LiveConfig, proportions: dict[str, float] | None = None, label: str = "mix-33-33-33") -> LiveConfig:
    """A synthetic eval config over the specialist split manifests with custom group proportions.

    The default (1/3 each) screening sample gives every split the same number of sequences, so one
    scoring pass yields per-split losses that can be re-weighted to any specialist mix (70/15/15).
    Planning and sampling inside each split are the validator's own (category-weighted) functions;
    only the split proportions and therefore ``config_version`` differ from a live competition.
    """
    configuration = upstream.configuration
    proportions = proportions or {name: 1.0 / len(SPLITS) for name in SPLITS}
    snapshots = tuple(
        configuration.DatasetManifestSnapshot(
            name=s.name, manifest_url=s.manifest_url, manifest_sha256=s.manifest_sha256,
            proportion=float(proportions[s.name]), manifest=s.manifest,
        )
        for s in base.settings.manifests
    )
    long_documents = base.settings.long_documents
    if long_documents is not None:  # same document budget, categories re-weighted to these proportions
        long_documents = upstream.long_documents.build_long_document_config(
            snapshots, token_budget=long_documents["token_budget"], index_manifest=long_documents["index_manifest"])
    kwargs = dict(dataset_label=label, n=base.n, delta_threshold=base.delta_threshold, manifests=snapshots,
                  shards_per_dataset=base.shards_per_dataset, long_documents=long_documents)
    version = configuration.evaluation_config_version(**kwargs)
    settings = configuration.EvaluationSettings(config_version=version, **kwargs)
    raw = dict(base.raw, config_version=version, dataset_label=label,
               sources=[dict(src, proportion=float(proportions[src["name"]])) for src in base.raw["sources"]])
    return LiveConfig(raw=raw, settings=settings)


def build_request(live: LiveConfig, *, block_hash: str, hotkey: str):
    """Validator EvalRequest for sampling (model fields are placeholders)."""
    configuration = upstream.configuration
    dataset = configuration.pretokenized_dataset_request(
        live.settings, block_hash=block_hash, hotkey=hotkey, seq_len=SEQ_LEN
    )
    long_documents = dataset.get("long_documents")
    return upstream.engine.EvalRequest(
        king_repo="king",
        challenger_repo="challenger",
        block_hash=block_hash,
        hotkey=hotkey,
        dataset_source="pretokenized_npy",
        dataset_sources=dataset["sources"],
        long_documents=long_documents,
        # the validator sets this from the models' max_position_embeddings (MiMo: far above 8192)
        max_document_tokens=upstream.long_documents.MAX_DOCUMENT_TOKENS if long_documents else 0,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        n=live.n,
    )


def planned_shard_names(live: LiveConfig, *, block_hash: str, hotkey: str) -> dict[str, list[str]]:
    """{source: [shard file names]} the validator would download for this seed."""
    request = build_request(live, block_hash=block_hash, hotkey=hotkey)
    return {
        source["name"]: [shard["url"].rsplit("/", 1)[-1] for shard in source["shards"]]
        for source in request.dataset_sources
    }


class Sequences:
    """Variable-length samples stored flat: ``seqs[i]`` is sample i's uint32 tokens.

    Stands in for the [n, 2048] array of window-only samples wherever one sample is read at a
    time (``shape[0]``, ``len``, integer indexing). Slices raise: there is no 2-D view.
    """

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat, self.offsets = flat, offsets
        self.shape = (len(offsets) - 1,)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, index):
        if not isinstance(index, (int, np.integer)):
            raise TypeError("variable-length sample: index one sample at a time (use a window-only sample for slices)")
        return self.flat[self.offsets[index]:self.offsets[index + 1]]

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


def scored_token_counts(tokens) -> np.ndarray:
    """Targets each sample contributes under document masking (validator ``scored_token_counts``):
    every position except the last whose token is not EOS -- a fragment's first token is never a target."""
    eos = upstream.masking.DOCUMENT_EOS_TOKEN_ID
    if isinstance(tokens, Sequences):
        not_eos = (np.asarray(tokens.flat) != eos).astype(np.int32)
        ends = tokens.offsets[1:] - 1  # each sample's last token is never an input of a target
        return (np.add.reduceat(not_eos, tokens.offsets[:-1]) - not_eos[ends]).astype(np.int64)
    return (np.asarray(tokens)[:, :-1] != eos).sum(axis=1).astype(np.int64)


@dataclass
class EvalSample:
    sample_id: str
    tokens: np.ndarray | Sequences  # validator order (post-shuffle); uint32 [n, seq_len] for window-only samples
    sources: np.ndarray  # str [n]
    provenance: np.ndarray  # int32 [n, 3]: shard_group_index, shard_index, shard_sequence_index (-1, -1 for documents)
    meta: dict
    scored_tokens: np.ndarray  # int64 [n]: scored targets per sample = the verdict's weights
    components: np.ndarray  # str [n]: "windows" | "long_documents"
    documents: dict[int, dict]  # sample index -> document provenance (dataset, category, length, length_bucket, ...)

    @property
    def n(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def has_documents(self) -> bool:
        return bool(self.documents)

    def lengths(self) -> np.ndarray:
        return self.tokens.lengths if isinstance(self.tokens, Sequences) else np.full(self.n, self.tokens.shape[1])

    def source_names(self) -> list[str]:
        return sorted(set(self.sources.tolist()))

    def shard_keys(self) -> np.ndarray:
        """Cluster id per sample ("group:shard"; a whole document is its own cluster), for resampling."""
        return np.asarray(
            [f"{g}:{s}" if s >= 0 else f"doc:{i}" for i, (g, s) in enumerate(self.provenance[:, :2].tolist())],
            dtype=object,
        )


def sample_dir(cache_root: Path, sample_id: str) -> Path:
    return cache_root / "samples" / sample_id


def load_sample(path: Path) -> EvalSample:
    meta = json.loads((path / "meta.json").read_text())
    tokens = np.load(path / "tokens.npy", mmap_mode="r")
    if (path / "offsets.npy").exists():
        tokens = Sequences(tokens, np.load(path / "offsets.npy"))
    n = int(tokens.shape[0])
    # samples built before the masked policy have no counts on disk; they follow from the tokens
    scored = np.load(path / "scored_tokens.npy") if (path / "scored_tokens.npy").exists() else scored_token_counts(tokens)
    documents_path = path / "documents.json"
    documents = {int(k): v for k, v in json.loads(documents_path.read_text()).items()} if documents_path.exists() else {}
    components = np.asarray(["long_documents" if i in documents else "windows" for i in range(n)])
    return EvalSample(
        sample_id=meta["sample_id"],
        tokens=tokens,
        sources=np.load(path / "sources.npy", allow_pickle=False),
        provenance=np.load(path / "provenance.npy"),
        meta=meta,
        scored_tokens=scored,
        components=components,
        documents=documents,
    )


def _with_retry(fn, attempts: int):
    """Retry ``fn`` on HTTP 429/5xx and connection errors. Stalls retry immediately; 429s wait at most a few seconds.

    ``RangeDownloadError`` is the validator's own range reader giving up after its 5 backed-off attempts (or on a
    Retry-After above 120 s): the bucket is throttling, so wait out a cooldown (15 s, 30 s, ... up to 2 min) and retry.
    """
    import random
    import urllib.error

    range_error = upstream.module("teutonic.evaluator.document_index").RangeDownloadError

    def wrapper(*args, **kwargs):
        for attempt in range(attempts):
            try:
                return fn(*args, **kwargs)
            except range_error as exc:
                if attempt == attempts - 1:
                    raise
                delay = min(120.0, 15.0 * (attempt + 1)) * (1 + 0.5 * random.random())
                log.warning("range read throttled (%s); cooling down %.0fs (retry %d/%d)", exc, delay, attempt + 1, attempts)
                time.sleep(delay)
                continue
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                    raise
                time.sleep(min(3.0, 0.4 * (attempt + 1)) * (1 + random.random()))
                continue
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == attempts - 1:
                    raise
                # a stalled shard must be retried at once; a 60s backoff is what dropped the sample to a few MB/s
                time.sleep(0.3 * (1 + random.random()))
                continue
        raise AssertionError("unreachable")

    wrapper._parity_retry = True
    return wrapper


# Seconds of silence before a range read is abandoned and retried. The validator waits 120 s, but r2.dev parks some
# throttled connections after the 206 header; at 120 s per stall a sample's ~5k document reads took >30 min.
RANGE_STALL_SECONDS = float(os.environ.get("PARITY_RANGE_STALL_SECONDS", "20"))
# Concurrent r2.dev connections per process (the validator's pool also allows 32).
RANGE_CONNECTIONS = int(os.environ.get("PARITY_RANGE_CONNECTIONS", "32"))


def legacy_shard_path(shard) -> Path:
    """Where validators <= 14a9bb2 stored a whole downloaded shard (``sources._cache_path``, SHA-256 verified)."""
    filename = Path(shard.url.split("?", 1)[0]).name or "shard.npy"
    root = Path(os.environ.get("TEUTONIC_PRETOKENIZED_CACHE_DIR", str(upstream.DEFAULT_CACHE_ROOT / "shards")))
    return root / f"{shard.sha256[:24]}-{filename}"


_verified_local: dict[tuple, bool] = {}
_verified_lock = threading.Lock()


def _local_verified(path: Path, shard) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False
    if stat.st_size != shard.size_bytes:
        return False
    key = (str(path), stat.st_mtime_ns, shard.sha256)
    with _verified_lock:
        if key in _verified_local:
            return _verified_local[key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    ok = digest.hexdigest() == shard.sha256
    with _verified_lock:
        _verified_local[key] = ok
    return ok


class _LocalReaderStats:
    downloaded_bytes = 0
    metadata_sha256_checked = True  # stronger: the whole file's SHA-256 was checked


class LocalNpyShard:
    """``token_shards.NpyShard``'s reader interface over a whole shard already on disk (SHA-256 verified).

    The ~100 GB of shards that older validators downloaded whole are read from disk instead of r2.dev; the tokens
    are byte-identical to the validator's range reads (same file, same logical row-major order).
    """

    def __init__(self, shard, path: Path):
        array = np.load(path, mmap_mode="r")
        if array.dtype != np.dtype("<u4") or array.ndim not in (1, 2) or array.size != shard.n_tokens:
            raise ValueError(f"cached shard {path} does not match its validator descriptor")
        self.shape, self.n_tokens = array.shape, int(array.size)
        self.flat = array.reshape(-1) if array.flags.c_contiguous else np.ascontiguousarray(array).reshape(-1)
        self.reader = _LocalReaderStats()

    def sequence_count(self, width):
        if width <= 0 or (len(self.shape) == 2 and self.shape[1] != width):
            raise ValueError(f"token shard sequence width differs from seq_len={width}")
        return self.n_tokens // width

    def read_tokens(self, spans):
        for start, length in spans:
            if start < 0 or length <= 0 or start + length > self.n_tokens:
                raise ValueError("token span is outside its shard")
        return [np.array(self.flat[start:start + length]) for start, length in spans]

    def read_sequences(self, indices, width):
        self.sequence_count(width)
        return self.read_tokens([(int(i) * width, width) for i in indices])


def _install_stall_timeout() -> None:
    """Swap the validator's shared range-read client for one that abandons a silent read after RANGE_STALL_SECONDS."""
    import httpx

    range_http = upstream.module("teutonic.evaluator.range_http")
    if getattr(range_http._client, "_parity_stall", False):
        return

    class StallTimeoutClient(httpx.Client):
        _parity_stall = True

        def stream(self, *args, timeout=None, **kwargs):  # noqa: ARG002 -- the caller's 120 s is what we replace
            return super().stream(*args, timeout=httpx.Timeout(RANGE_STALL_SECONDS, pool=120.0), **kwargs)

    old = range_http._client
    range_http._client = StallTimeoutClient(
        follow_redirects=True,
        limits=httpx.Limits(max_connections=RANGE_CONNECTIONS, max_keepalive_connections=RANGE_CONNECTIONS,
                            keepalive_expiry=60),
    )
    old.close()


def open_shard(shard):
    """``sources.open_shard`` with the legacy whole-shard cache first: a verified file on disk, else HTTP ranges."""
    path = legacy_shard_path(shard)
    if _local_verified(path, shard):
        return LocalNpyShard(shard, path)
    return _validator_open_shard(shard)


_validator_open_shard = None


def install_fetch_retry(attempts: int = 10) -> None:
    """Make the validator's data reads robust and cheap; the sampled tokens are unchanged.

    Since validator 5118c32 every byte of a sample -- document-index pages, window rows and document fragments -- is
    an HTTP range read through ``document_index.RangeObject._range`` (``PinnedObject`` for the index, ``NpyShard``'s
    reader for token shards), served by the public r2.dev bucket, which answers 429 under load (pool builds hit the
    same bucket) and sometimes stalls a response. This

    * wraps ``_range`` (5 attempts, 5-40 s backoff, then ``RangeDownloadError``) in a second, slower retry loop, so a
      throttled bucket delays a sample instead of failing it -- a retry repeats the same identity-checked read;
    * abandons a silent read after ``RANGE_STALL_SECONDS`` instead of 120 s;
    * serves shards that an older validator downloaded whole (``eval_cache/shards``, SHA-256 verified) from disk.
    """
    global _validator_open_shard
    # httpx logs every one of a sample's ~10k range requests at INFO
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger("httpx").level))
    index_module = upstream.module("teutonic.evaluator.document_index")
    for cls in (index_module.RangeObject, index_module.PinnedObject):
        current = cls.__dict__.get("_range")
        if current is not None and not getattr(current, "_parity_retry", False):
            cls._range = _with_retry(current, attempts)
    _install_stall_timeout()
    if upstream.sources.open_shard is not open_shard:
        _validator_open_shard = upstream.sources.open_shard
        upstream.sources.open_shard = open_shard


# The validator plans each category's documents to within about half a document (<= 4 096 tokens) of its quota
# (``document_index.sample_category``), i.e. the 6M budget to within ~1.6% over 24 categories; a sample below 95% of
# the budget, or short of its own planned total, had documents dropped while it was built.
DOCUMENT_COMPLETENESS = 0.95


def document_completeness(live: LiveConfig, long_meta: dict | None, document_tokens: int) -> dict:
    """``documents_complete`` + token totals of a sample's long-document component under ``live``'s config."""
    config = live.settings.long_documents
    if config is None:  # window-only config: no documents planned
        return {"documents_complete": document_tokens == 0, "document_tokens": int(document_tokens),
                "document_tokens_planned": 0, "document_token_budget": 0}
    budget = int(config["token_budget"])
    categories = (long_meta or {}).get("categories") or []
    planned = sum(int(c.get("sampled_tokens", 0)) for c in categories) if categories else None
    complete = document_tokens >= DOCUMENT_COMPLETENESS * budget and (planned is None or document_tokens == planned)
    return {"documents_complete": bool(complete), "document_tokens": int(document_tokens),
            "document_tokens_planned": planned, "document_token_budget": budget}


def sample_completeness(live: LiveConfig, sample: EvalSample) -> dict:
    """Completeness of a built sample (meta.json's record, else recomputed from its documents)."""
    if "documents_complete" in sample.meta:
        return {k: sample.meta.get(k) for k in
                ("documents_complete", "document_tokens", "document_tokens_planned", "document_token_budget")}
    index = sorted(sample.documents)
    tokens = int(sample.lengths()[index].sum()) if index else 0
    return document_completeness(live, sample.meta.get("long_documents"), tokens)


def _retire_incomplete(cache_root: Path, key: str, sample_id: str, reason: dict) -> None:
    """Record a cache key's incomplete sample in index_incomplete.json (its directory stays on disk)."""
    path = cache_root / "samples" / "index_incomplete.json"
    retired = json.loads(path.read_text()) if path.exists() else {}
    entries = retired.setdefault(key, [])
    if not any(e.get("sample_id") == sample_id for e in entries):
        entries.append({"sample_id": sample_id, **reason,
                        "retired_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(retired, indent=1, sort_keys=True))
    tmp.replace(path)


def _write_index(index_path: Path, index: dict) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, indent=1, sort_keys=True))
    tmp.replace(index_path)


def build_sample(
    live: LiveConfig,
    *,
    block_hash: str,
    hotkey: str,
    cache_root: Path | None = None,
    label: str = "",
) -> EvalSample:
    """Sample (or load cached) validator-identical sequences for one seed.

    One builder per seed at a time (``samples/.build-<key>.lock``): a second process asking for the same sample
    waits for the first and then reuses its result instead of sampling it again into the same directory.
    """
    import fcntl

    cache_root = cache_root or upstream.DEFAULT_CACHE_ROOT
    seed_material = f"block_hash={block_hash}|hotkey={hotkey}"
    key = hashlib.sha256(f"{live.config_version}|{seed_material}".encode()).hexdigest()[:16]
    lock_path = cache_root / "samples" / f".build-{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return _build_sample_locked(live, block_hash=block_hash, hotkey=hotkey, cache_root=cache_root,
                                        label=label, seed_material=seed_material, key=key)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _build_sample_locked(live: LiveConfig, *, block_hash: str, hotkey: str, cache_root: Path, label: str,
                         seed_material: str, key: str) -> EvalSample:
    index_path = cache_root / "samples" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    if key in index and (sample_dir(cache_root, index[key]) / "meta.json").exists():
        sample_id = index[key]
        cached = load_sample(sample_dir(cache_root, sample_id))
        status = sample_completeness(live, cached)
        if status["documents_complete"]:
            log.info("reusing cached eval sample %s", sample_id)
            return cached
        # built while documents were skipped: never reuse it for this seed (the directory stays, see index_incomplete.json)
        log.warning("cached eval sample %s is incomplete (%s document tokens of %s planned, budget %s); rebuilding",
                    sample_id, status["document_tokens"], status["document_tokens_planned"], status["document_token_budget"])
        _retire_incomplete(cache_root, key, sample_id, status)

    request = build_request(live, block_hash=block_hash, hotkey=hotkey)
    started = time.time()
    install_fetch_retry()
    # windows, then (live configs since 14a9bb2) complete documents shuffled in: the validator's own sampler
    sequences, meta = upstream.sources.sample_eval_sequences(
        request, on_phase=lambda event: log.info("sampler %s", event)
    )
    labels = np.asarray(meta.pop("_source_labels"))
    sample_provenance = meta.pop("_sample_provenance")
    provenance = np.asarray(
        [(p["shard_group_index"], p["shard_index"], p["shard_sequence_index"]) for p in sample_provenance],
        dtype=np.int32,
    )
    documents = {
        i: {k: v for k, v in p.items() if k not in ("shard_group_index", "shard_index", "shard_sequence_index")}
        for i, p in enumerate(sample_provenance) if p.get("component") == "long_documents"
    }
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)
    n_windows = len(sequences) - len(documents)
    if n_windows != live.n or (lengths[[i for i in range(len(sequences)) if i not in documents]] != SEQ_LEN).any():
        raise RuntimeError(f"sampled {n_windows} windows, expected {live.n} of {SEQ_LEN} tokens")
    completeness = document_completeness(live, meta.get("long_documents"), int(lengths[list(documents)].sum()) if documents else 0)
    if not completeness["documents_complete"]:
        raise RuntimeError(
            f"sample for {seed_material} has {completeness['document_tokens']} document tokens (planned "
            f"{completeness['document_tokens_planned']}, budget {completeness['document_token_budget']}): documents were "
            "dropped while sampling -- refusing to cache an incomplete sample")
    scored = np.asarray(upstream.masking.scored_token_counts(sequences), dtype=np.int64)
    sample_id = f"{live.config_version[:8]}-{meta['digest'][:16]}"
    final = sample_dir(cache_root, sample_id)
    # written to a private directory and renamed into place: a reader never maps a half-written tokens.npy
    out = final.parent / f".tmp-{sample_id}-{os.getpid()}"
    out.mkdir(parents=True, exist_ok=True)
    if documents:
        np.save(out / "tokens.npy", np.concatenate([np.asarray(s, dtype=np.uint32) for s in sequences]))
        np.save(out / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]))
        (out / "documents.json").write_text(json.dumps({str(i): d for i, d in documents.items()}))
    else:
        np.save(out / "tokens.npy", np.asarray(sequences, dtype=np.uint32))
    np.save(out / "scored_tokens.npy", scored)
    np.save(out / "sources.npy", labels)
    np.save(out / "provenance.npy", provenance)
    shard_names = {
        source["name"]: [ref.rsplit("/", 1)[-1] for ref in source["used_refs"]]
        for source in meta["sources"]
    }
    record = {
        "sample_id": sample_id,
        "label": label,
        "block_hash": block_hash,
        "hotkey": hotkey,
        "seed_material": seed_material,
        "dataset_seed": meta["dataset_seed"],
        "dataset_digest": meta["digest"],
        "config_version": live.config_version,
        "delta_threshold": live.delta_threshold,
        "n": len(sequences),
        "n_windows": n_windows,
        "n_documents": len(documents),
        "scored_tokens": int(scored.sum()),
        "document_scored_tokens": int(scored[list(documents)].sum()) if documents else 0,
        **completeness,
        "scoring_policy": upstream.masking.MASKED_POLICY_VERSION,
        "long_documents": {
            k: v for k, v in (meta.get("long_documents") or {}).items()
            if k in ("version", "target_tokens", "input_tokens", "n_documents", "context_limit", "categories")
        },
        "window_digest": meta.get("window_digest", meta["digest"]),
        "seq_len": SEQ_LEN,
        "shard_names": shard_names,
        "sources": [
            {k: v for k, v in source.items() if k not in ("used_shards", "used_refs")}
            for source in meta["sources"]
        ],
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sampling_seconds": round(time.time() - started, 1),
        "upstream_commit": upstream.UPSTREAM_COMMIT,
    }
    (out / "meta.json").write_text(json.dumps(record, indent=1))
    if (final / "meta.json").exists():  # the same sample (same digest) is already on disk: keep it, drop ours
        import shutil

        shutil.rmtree(out)
    else:
        if final.exists():  # leftovers of an interrupted build: set aside, never deleted
            final.rename(final.parent / f".partial-{sample_id}-{int(time.time())}")
        out.rename(final)
    out = final
    index = json.loads(index_path.read_text()) if index_path.exists() else {}  # re-read: other builds may have finished
    index[key] = sample_id
    _write_index(index_path, index)
    log.info("sample %s: %d windows + %d documents (%d scored tokens) in %.0fs",
             sample_id, n_windows, len(documents), int(scored.sum()), time.time() - started)
    return load_sample(out)
