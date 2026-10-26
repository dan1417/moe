#!/usr/bin/env python3
"""Download the current Teutonic king checkpoint from R2 with checksum verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

DASHBOARD_URL = "https://teutonic.ai/dashboard.json"
MODEL_BASE_URL = "https://pub-0821d4e196224864af220294345fd141.r2.dev/models/sha256"
DEFAULT_TIMEOUT = 120
DEFAULT_WORKERS = 16
USER_AGENT = "teutonic-miner/1.0"
CHUNK_SIZE = 16 * 1024 * 1024
RANGE_CHUNK = 256 * 1024 * 1024
# A metered R2 socket trickles instead of stalling. Cut it and open a new one; 8 MB keeps the retry small.
PIECE = 8 * 1024 * 1024
MIN_BPS = 4 * 1024 * 1024


def _fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


def _sha256_file(path: Path, timeout: float = DEFAULT_TIMEOUT) -> str:
    del timeout  # file I/O is local; timeout kept for API consistency
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _format_gbps(num_bytes: int, elapsed_sec: float) -> float:
    if elapsed_sec <= 0:
        return 0.0
    return (num_bytes / (1024**3)) / elapsed_sec


def get_king_digest(dashboard_url: str = DASHBOARD_URL, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Return king metadata from the live dashboard."""
    dashboard = _fetch_json(dashboard_url, timeout=timeout)
    king = dashboard.get("king")
    if not king:
        raise RuntimeError("dashboard.json does not contain a current king")
    digest = king.get("king_digest") or king.get("model_digest")
    if not digest:
        raise RuntimeError("Could not find king digest in dashboard.json")
    return {
        "digest": str(digest),
        "model_name": king.get("model_repo") or king.get("model_name", "unknown"),
        "hotkey": king.get("hotkey"),
        "crowned_at": king.get("crowned_at"),
        "avg_king_loss": king.get("avg_king_loss"),
        "model_reference": king.get("model_reference"),
    }


def fetch_model_manifest(digest: str, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    manifest_url = f"{MODEL_BASE_URL}/{digest}/manifest.json"
    return _fetch_json(manifest_url, timeout=timeout)


def download_file(
    digest: str,
    file_info: dict[str, Any],
    output_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
    verify_only: bool = False,
    session: requests.Session | None = None,
) -> None:
    rel_path = str(file_info["path"])
    expected_sha = str(file_info["sha256"])
    expected_size = int(file_info.get("size", 0))
    target = output_dir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        actual_sha = _sha256_file(target)
        if actual_sha == expected_sha:
            log.info("Already present: %s (sha256 ok)", rel_path)
            return
        if verify_only:
            raise RuntimeError(f"Checksum mismatch for existing file {rel_path}")
        log.warning("Checksum mismatch for %s — re-downloading", rel_path)

    if verify_only:
        raise FileNotFoundError(f"Missing file during verify-only: {rel_path}")

    url = f"{MODEL_BASE_URL}/{digest}/{rel_path}"
    start = time.monotonic()
    http = session or requests
    response = http.get(url, timeout=timeout, stream=True, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    downloaded = 0
    with open(tmp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            if chunk:
                handle.write(chunk)
                downloaded += len(chunk)
    tmp_path.replace(target)
    elapsed = time.monotonic() - start
    size_bytes = downloaded if downloaded > 0 else expected_size
    speed = _format_gbps(size_bytes, elapsed)
    log.info("Downloaded %s (%.2f GB/s, %.2f GB)", rel_path, speed, size_bytes / (1024**3))

    actual_sha = _sha256_file(target)
    if actual_sha != expected_sha:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch after download for {rel_path}")


def _read_piece(session: requests.Session, url: str, start: int, end: int) -> bytes:
    """One range. A socket under 4 MB/s is closed so the caller can open a fresh one."""
    expected = end - start + 1
    response = session.get(
        url,
        timeout=(10, 8),
        headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"},
        stream=True,
    )
    try:
        response.raise_for_status()
        buf = bytearray()
        window_at = time.monotonic()
        window_n = 0
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            buf += chunk
            window_n += len(chunk)
            now = time.monotonic()
            if now - window_at < 2.0:
                continue
            rate = window_n / (now - window_at)
            if rate < MIN_BPS:
                raise TimeoutError(f"slow stream {rate / 1e6:.1f} MB/s")
            window_at = now
            window_n = 0
        if len(buf) != expected:
            raise RuntimeError(f"short range {len(buf)} != {expected}")
        return bytes(buf)
    finally:
        response.close()


def _download_files_parallel(
    digest: str,
    files: list[dict[str, Any]],
    output_dir: Path,
    timeout: float,
    workers: int,
) -> None:
    """Download manifest files as RANGE_CHUNK byte ranges spread over `workers` streams.

    R2 serves ~20-50 MB/s per stream, and some kings ship 28 GB shards, so a
    file-per-worker scheme leaves most streams idle at the tail of a download.
    A stream under 4 MB/s is dropped; eight streams is the rate this host actually holds.
    """
    del timeout  # each piece uses a short read timeout so a trickle cannot pin a worker for 120s
    todo: list[dict[str, Any]] = []
    for file_info in files:
        target = output_dir / str(file_info["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size == int(file_info.get("size", -1)):
            if _sha256_file(target) == str(file_info["sha256"]):
                log.info("Already present: %s (sha256 ok)", file_info["path"])
                continue
        todo.append(file_info)

    chunks: list[tuple[dict[str, Any], int, int]] = []
    remaining: dict[str, int] = {}
    for file_info in todo:
        size = int(file_info["size"])
        tmp = output_dir / (str(file_info["path"]) + ".tmp")
        done = _completed_ranges(tmp, size)
        with open(tmp, "r+b" if tmp.exists() else "wb") as handle:
            handle.truncate(size)
        spans = [
            (start, min(size, start + RANGE_CHUNK) - 1)
            for start in range(0, size, RANGE_CHUNK)
            if start not in done
        ]
        if done:
            log.info("Resuming %s: %d/%d ranges already present", file_info["path"], len(done), len(done) + len(spans))
        remaining[str(file_info["path"])] = len(spans)
        chunks.extend((file_info, start, end) for start, end in spans)
        if not spans:
            _finalize(file_info, output_dir)
    lock = threading.Lock()
    started = time.monotonic()
    total = sum(int(f["size"]) for f in todo)
    done_bytes = [0]

    def fetch(chunk: tuple[dict[str, Any], int, int]) -> None:
        file_info, start, end = chunk
        rel_path = str(file_info["path"])
        url = f"{MODEL_BASE_URL}/{digest}/{rel_path}"
        session = requests.Session()
        pos = start
        try:
            while pos <= end:
                piece_end = min(end, pos + PIECE - 1)
                body = None
                last_exc: Exception | None = None
                for attempt in range(8):
                    try:
                        body = _read_piece(session, url, pos, piece_end)
                        break
                    except (requests.RequestException, TimeoutError, RuntimeError, OSError) as exc:
                        last_exc = exc
                        session.close()
                        session = requests.Session()
                        if attempt == 7:
                            raise
                        log.warning("range %s[%d-%d] retry %d: %s", rel_path, pos, piece_end, attempt + 1, exc)
                        time.sleep(0.2)
                if body is None:
                    raise last_exc or RuntimeError(f"empty range {rel_path}[{pos}-{piece_end}]")
                fd = os.open(output_dir / (rel_path + ".tmp"), os.O_WRONLY)
                try:
                    os.pwrite(fd, body, pos)
                finally:
                    os.close(fd)
                with lock:
                    done_bytes[0] += len(body)
                pos = piece_end + 1
        finally:
            session.close()
        with lock:
            with open(output_dir / (rel_path + ".tmp.ranges"), "a") as progress:
                progress.write(f"{start}\n")
            remaining[rel_path] -= 1
            finished = remaining[rel_path] == 0
        if finished:
            _finalize(file_info, output_dir)
            elapsed = time.monotonic() - started
            log.info(
                "Downloaded %s (%.2f GB); total %.1f/%.1f GB at %.0f MB/s",
                rel_path, int(file_info["size"]) / 1e9, done_bytes[0] / 1e9, total / 1e9,
                done_bytes[0] / 1e6 / max(elapsed, 1e-9),
            )

    log.info("Downloading %d files (%.1f GB) as %d ranges over %d streams", len(todo), total / 1e9, len(chunks), workers)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for future in as_completed([pool.submit(fetch, chunk) for chunk in chunks]):
            future.result()  # propagate errors


def _completed_ranges(tmp: Path, size: int) -> set[int]:
    """Range starts already on disk: logged ranges, or a legacy sequential prefix."""
    progress = tmp.with_name(tmp.name + ".ranges")
    if progress.exists() and tmp.exists():
        return {int(line) for line in progress.read_text().split()}
    if tmp.exists() and tmp.stat().st_size < size:
        # written front-to-back by the single-stream downloader: whole ranges only
        prefix = tmp.stat().st_size // RANGE_CHUNK * RANGE_CHUNK
        return set(range(0, prefix, RANGE_CHUNK))
    return set()


def _finalize(file_info: dict[str, Any], output_dir: Path) -> None:
    rel_path = str(file_info["path"])
    tmp = output_dir / (rel_path + ".tmp")
    if _sha256_file(tmp) != str(file_info["sha256"]):
        tmp.unlink(missing_ok=True)
        tmp.with_name(tmp.name + ".ranges").unlink(missing_ok=True)
        raise RuntimeError(f"Checksum mismatch after download for {rel_path}")
    tmp.replace(output_dir / rel_path)
    tmp.with_name(tmp.name + ".ranges").unlink(missing_ok=True)


def verify_manifest_files(
    digest: str,
    manifest: dict[str, Any],
    output_dir: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    files = manifest.get("files", [])
    expected_count = len(files)
    present = 0
    for file_info in files:
        rel_path = str(file_info["path"])
        target = output_dir / rel_path
        if not target.exists():
            raise FileNotFoundError(f"Missing file: {rel_path}")
        actual_sha = _sha256_file(target, timeout=timeout)
        if actual_sha != str(file_info["sha256"]):
            raise RuntimeError(f"Checksum mismatch for {rel_path}")
        present += 1
    if present != expected_count:
        raise RuntimeError(f"File count mismatch: expected {expected_count}, found {present}")
    log.info("Verified %d/%d files for digest %s", present, expected_count, digest)


def get_king_chain(dashboard_url: str = DASHBOARD_URL, timeout: float = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    """Return the full king_chain from the dashboard (current + previous reigns)."""
    dashboard = _fetch_json(dashboard_url, timeout=timeout)
    chain = dashboard.get("king_chain")
    if not isinstance(chain, list):
        raise RuntimeError("dashboard.json does not contain king_chain")
    return chain


def get_king_by_reign(
    reign: int,
    dashboard_url: str = DASHBOARD_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Look up a king entry by reign number from king_chain."""
    for entry in get_king_chain(dashboard_url=dashboard_url, timeout=timeout):
        if int(entry.get("reign_number", -1)) == reign:
            digest = entry.get("model_digest") or entry.get("king_digest")
            if not digest:
                raise RuntimeError(f"Reign {reign} has no model digest in king_chain")
            return {
                "digest": str(digest),
                "model_name": entry.get("model_repo") or entry.get("model_name", "unknown"),
                "hotkey": entry.get("hotkey"),
                "crowned_at": entry.get("crowned_at"),
                "ended_at": entry.get("ended_at"),
                "reign_number": reign,
                "model_reference": entry.get("model_reference"),
            }
    raise RuntimeError(f"No king found for reign {reign} in king_chain")


def list_kings(dashboard_url: str = DASHBOARD_URL, timeout: float = DEFAULT_TIMEOUT) -> None:
    """Print all known kings from king_chain."""
    chain = get_king_chain(dashboard_url=dashboard_url, timeout=timeout)
    current = get_king_digest(dashboard_url=dashboard_url, timeout=timeout)
    current_digest = current["digest"]
    print(f"{'Reign':>5}  {'Digest':<12}  Model")
    print("-" * 72)
    for entry in chain:
        digest = str(entry.get("model_digest") or entry.get("king_digest") or "")
        reign = entry.get("reign_number", "?")
        model = entry.get("model_repo") or entry.get("model_name") or "unknown"
        marker = " *" if digest == current_digest else ""
        print(f"{reign:>5}  {digest[:12]}...  {model}{marker}")
    print()
    print("* = current king")


def download_model_by_digest(
    digest: str,
    output_dir: Path,
    king_meta: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    verify_only: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Download or verify any model checkpoint by SHA256 digest."""
    manifest = fetch_model_manifest(digest, timeout=timeout)
    files = manifest.get("files", [])
    if not files:
        raise RuntimeError(f"Manifest for {digest} contains no files")

    output_dir.mkdir(parents=True, exist_ok=True)
    if verify_only:
        verify_manifest_files(digest, manifest, output_dir, timeout=timeout)
        log.info("Verify-only passed for digest %s", digest)
        return output_dir

    start = time.monotonic()
    _download_files_parallel(digest, files, output_dir, timeout=timeout, workers=workers)
    # every file was sha256-verified as it completed; confirm nothing is missing
    for file_info in files:
        target = output_dir / str(file_info["path"])
        if not target.exists() or target.stat().st_size != int(file_info.get("size", -1)):
            raise RuntimeError(f"Missing or truncated after download: {file_info['path']}")
    elapsed = time.monotonic() - start
    total_gb = sum(int(f.get("size", 0)) for f in files) / (1024**3)
    log.info(
        "Download complete in %.1f min (%.2f GB total, %.2f GB/s effective, %d workers)",
        elapsed / 60,
        total_gb,
        total_gb / elapsed if elapsed > 0 else 0.0,
        workers,
    )

    info = {
        "digest": digest,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest_protocol_version": manifest.get("protocol_version"),
        "file_count": len(files),
    }
    if king_meta:
        info.update({k: v for k, v in king_meta.items() if k != "digest"})
    info_path = output_dir / "king_info.json"
    info_path.write_text(json.dumps(info, indent=2))
    log.info("Model checkpoint ready at %s (%d files, digest=%s)", output_dir, len(files), digest)
    return output_dir


def download_king(
    output_dir: Path,
    dashboard_url: str = DASHBOARD_URL,
    timeout: float = DEFAULT_TIMEOUT,
    verify_only: bool = False,
    digest: str | None = None,
    reign: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> Path:
    """Download or verify a king checkpoint (current, by digest, or by reign number)."""
    if digest is not None and reign is not None:
        raise ValueError("Provide only one of digest or reign, not both")
    if digest is not None:
        king_meta = {"digest": digest}
    elif reign is not None:
        king_meta = get_king_by_reign(reign, dashboard_url=dashboard_url, timeout=timeout)
        digest = king_meta["digest"]
    else:
        king_meta = get_king_digest(dashboard_url=dashboard_url, timeout=timeout)
        digest = king_meta["digest"]

    return download_model_by_digest(
        digest=digest,
        output_dir=output_dir,
        king_meta=king_meta,
        timeout=timeout,
        verify_only=verify_only,
        workers=workers,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Teutonic king checkpoints from R2.")
    parser.add_argument("--output", type=Path, default=Path("./king"), help="Output directory")
    parser.add_argument("--dashboard-url", default=DASHBOARD_URL, help="Dashboard JSON URL")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument(
        "--digest",
        help="Download a specific model by SHA256 digest (any past or current king)",
    )
    parser.add_argument(
        "--reign",
        type=int,
        help="Download a king by reign number from dashboard king_chain",
    )
    parser.add_argument(
        "--list-kings",
        action="store_true",
        help="List all kings in king_chain and exit",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel shard downloads (default: {DEFAULT_WORKERS}; try 8-16 on fast links)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify existing files against manifest without downloading",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        if args.list_kings:
            list_kings(dashboard_url=args.dashboard_url, timeout=args.timeout)
            return 0
        download_king(
            output_dir=args.output,
            dashboard_url=args.dashboard_url,
            timeout=args.timeout,
            verify_only=args.verify_only,
            digest=args.digest,
            reign=args.reign,
            workers=args.workers,
        )
    except (requests.RequestException, OSError, RuntimeError, FileNotFoundError) as exc:
        log.error("King download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
