#!/usr/bin/env python3
"""Pre-submission gate: mirror every validator check that can reject a checkpoint.

Checks (all must pass):
  1. the six contract files exist and match chain.toml SHA-256 (live main, pinned fallback)
  2. no manifest.json, no symlinks, total size <= 250 GB
  3. tensor names == king's, dtype/shape identical, all BF16, no "mtp" keys
  4. every safetensors file referenced by the index exists and parses
  5. weights differ from the king (validator's safetensors digest)
  6. config lock keys equal the king's (validate_config_lock)

Usage: verify_submission.py --model-dir <dir> --king-dir <king dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from safetensors import safe_open  # noqa: E402

from src.parity import upstream  # noqa: E402

MAX_UPLOAD_BYTES = 250_000_000_000
CHAIN_TOML_URL = "https://raw.githubusercontent.com/unarbos/teutonic/main/chain.toml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def contract_hashes() -> tuple[dict[str, str], str]:
    try:
        with urllib.request.urlopen(CHAIN_TOML_URL, timeout=30) as response:
            doc = tomllib.loads(response.read().decode())
        return dict(doc["seed"]["contract_files"]), "live chain.toml"
    except Exception:  # offline: fall back to the pinned validator package
        return dict(upstream.chain_config.GENESIS_CONTRACT_FILES), "pinned chain.toml"


def headers(model_dir: Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    out = {}
    for filename in sorted(set(index["weight_map"].values())):
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"index references missing file {filename}")
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                sliced = handle.get_slice(name)
                out[name] = (sliced.get_dtype(), tuple(sliced.get_shape()))
    missing = set(index["weight_map"]) - set(out)
    if missing:
        raise RuntimeError(f"index lists tensors absent from files: {sorted(missing)[:4]}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--king-dir", required=True, type=Path)
    args = parser.parse_args()
    model_dir, king_dir = args.model_dir, args.king_dir
    failures: list[str] = []

    def check(ok: bool, message: str) -> None:
        print(("PASS " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

    expected, source = contract_hashes()
    for name, digest in sorted(expected.items()):
        path = model_dir / name
        check(path.is_file() and sha256_file(path) == digest, f"contract file {name} matches {source}")

    files = [p for p in model_dir.rglob("*")]
    check(not any(p.is_symlink() for p in files), "no symlinks")
    check(not (model_dir / "manifest.json").exists(), "no reserved manifest.json")
    total = sum(p.stat().st_size for p in files if p.is_file())
    check(total <= MAX_UPLOAD_BYTES, f"total size {total / 1e9:.1f} GB <= 250 GB")

    ours, king = headers(model_dir), headers(king_dir)
    check(set(ours) == set(king), f"tensor names match king ({len(ours)} vs {len(king)})")
    mismatched = [n for n in set(ours) & set(king) if ours[n] != king[n]]
    check(not mismatched, f"dtype/shape match king{'' if not mismatched else ': ' + mismatched[0]}")
    check(all(dtype == "BF16" for dtype, _ in ours.values()), "all tensors BF16")
    check(not any("mtp" in n.lower() for n in ours), "no MTP tensors")

    engine = upstream.engine
    ours_digest = engine.snapshot_safetensors_digest(str(model_dir))
    king_digest = engine.snapshot_safetensors_digest(str(king_dir))
    check(ours_digest != king_digest, f"weights differ from king (digest {ours_digest[:16]})")

    ours_cfg = json.loads((model_dir / "config.json").read_text())
    king_cfg = json.loads((king_dir / "config.json").read_text())
    lock = upstream.policy.validate_config_lock(
        king_cfg, ours_cfg, extra_lock_keys=tuple(upstream.chain_config.EXTRA_LOCK_KEYS)
    )
    check(lock is None, f"config lock keys match king{'' if lock is None else ': ' + lock}")

    print(f"\n{'READY' if not failures else 'NOT READY'}: {len(failures)} failure(s); safetensors digest {ours_digest}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
