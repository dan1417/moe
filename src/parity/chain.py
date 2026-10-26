"""Minimal finney JSON-RPC helpers to recover a past duel's sampling seed.

The validator seeds each duel's sample with the hash of the block that finalized the
miner's ``r2ready:v1`` commitment. Scanning blocks for that commitment (signed by the
duel's hotkey) recovers the exact seed, so a past duel can be replayed locally on the
very same 30k sequences and compared against the published verdict.
"""

from __future__ import annotations

import itertools
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

log = logging.getLogger(__name__)

RPC_URLS = (
    "https://entrypoint-finney.opentensor.ai:443",
    "https://archive.chain.opentensor.ai:443",
    "https://lite.chain.opentensor.ai:443",
)
BLOCK_SECONDS = 12.0
READY_PREFIX = b"r2ready:v1"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_calls = itertools.count()


def rpc(method: str, params: list, *, retries: int = 8, timeout: float = 30.0):
    """JSON-RPC over HTTPS, rotating public endpoints and backing off on 429s."""
    body = json.dumps({"id": 1, "jsonrpc": "2.0", "method": method, "params": params}).encode()
    for attempt in range(retries):
        url = RPC_URLS[(next(_calls) + attempt) % len(RPC_URLS)]
        try:
            request = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read())
            if "error" in payload:
                raise RuntimeError(f"{method} failed: {payload['error']}")
            return payload["result"]
        except urllib.error.HTTPError as exc:
            if attempt == retries - 1:
                raise
            time.sleep((5.0 if exc.code == 429 else 1.5) * (attempt + 1))
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise AssertionError("unreachable")


def ss58_public_key(address: str) -> bytes:
    """32-byte public key of an SS58 address (1-byte network prefix, e.g. 42)."""
    number = 0
    for char in address:
        number = number * 58 + _B58.index(char)
    raw = number.to_bytes((number.bit_length() + 7) // 8, "big")
    raw = b"\x00" * (len(address) - len(address.lstrip("1"))) + raw
    if len(raw) == 35:
        return raw[1:33]
    if len(raw) == 36:
        return raw[2:34]
    raise ValueError(f"unexpected SS58 payload length {len(raw)} for {address}")


def block_hash(number: int) -> str:
    return rpc("chain_getBlockHash", [number])


def block_extrinsics(hash_: str) -> list[str]:
    return rpc("chain_getBlock", [hash_])["block"]["extrinsics"]


def block_timestamp(hash_: str) -> datetime:
    """Decode the Timestamp.set inherent (first extrinsic) of a block."""
    extrinsic = bytes.fromhex(block_extrinsics(hash_)[0][2:])
    # compact(len) | version(0x04) | pallet | call | compact<u64 millis>
    offset = 1 if extrinsic[0] & 0b11 == 0 else 2 if extrinsic[0] & 0b11 == 1 else 4
    offset += 3
    mode = extrinsic[offset] & 0b11
    if mode != 3:
        raise ValueError("unexpected timestamp encoding")
    length = (extrinsic[offset] >> 2) + 4
    millis = int.from_bytes(extrinsic[offset + 1 : offset + 1 + length], "little")
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)


def estimate_block(at: datetime, *, ref_block: int, ref_time: datetime) -> int:
    return ref_block - int(round((ref_time - at).total_seconds() / BLOCK_SECONDS))


def find_ready_block(
    hotkey: str, *, start: int, end: int, workers: int = 6
) -> tuple[int, str] | None:
    """Scan [start, end] (newest first) for the hotkey's r2ready commitment."""
    key_hex = ss58_public_key(hotkey).hex()
    marker = READY_PREFIX.hex()

    def probe(number: int) -> tuple[int, str] | None:
        hash_ = block_hash(number)
        for extrinsic in block_extrinsics(hash_):
            if marker in extrinsic and key_hex in extrinsic:
                return number, hash_
        return None

    numbers = list(range(end, start - 1, -1))
    chunk = max(workers * 4, 64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index in range(0, len(numbers), chunk):
            batch = numbers[index : index + chunk]
            hits = [hit for hit in pool.map(probe, batch) if hit is not None]
            if hits:
                return max(hits)
            log.info("scanned blocks %d..%d", batch[-1], batch[0])
    return None
