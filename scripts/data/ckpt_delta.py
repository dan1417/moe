#!/usr/bin/env python3
"""Compact, bit-exact checkpoint backups as deltas against a public base checkpoint.

A trainer export differs from its start body only in the trained tensors, and small updates move
most bf16 values by 0 or 1 ulp. So each changed tensor is stored as the wrapped int16 difference of
its bf16 bit patterns (``ckpt_bits - base_bits mod 2**16``, mostly 0/+-1), zstd-compressed; unchanged
tensors are stored as references. ``restore`` rebuilds the exact safetensors files (same tensors, same
metadata -> byte-identical, SHA-256 checked per file), i.e. the same validator digest.

  ckpt_delta.py pack    --ckpt <export dir> --base <base dir> --out <delta dir>
  ckpt_delta.py restore --delta <delta dir> --base <base dir> --out <ckpt dir>
  ckpt_delta.py push    --ckpt <export dir> --base <base dir> --repo <user/repo> --name <folder> [--note '{json}']

Restore on another machine:
  hf download touristAI/teutonic-II-110B-merge-k30u --local-dir base      # public start body
  hf download <private repo> --include '<folder>/*' --local-dir backup     # needs HF_TOKEN
  python scripts/data/ckpt_delta.py restore --delta backup/<folder> --base base --out ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import zstandard
from safetensors import safe_open
from safetensors.torch import save_file

log = logging.getLogger("ckpt_delta")
CONTRACT_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja.txt",
                  "configuration_mimo_v2.py", "modeling_mimo_v2.py")
BASE_REPO = "touristAI/teutonic-II-110B-merge-k30u"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Reader:
    """Tensor access by name across a sharded safetensors checkpoint."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self._handles: dict[str, object] = {}

    def get(self, name: str) -> torch.Tensor:
        filename = self.weight_map[name]
        if filename not in self._handles:
            self._handles[filename] = safe_open(str(self.root / filename), framework="pt", device="cpu")
        return self._handles[filename].get_tensor(name)


def _wrap16(values: torch.Tensor) -> torch.Tensor:
    return (torch.remainder(values + 32768, 65536) - 32768).to(torch.int16)


def pack(ckpt: Path, base: Path, out: Path, *, workers: int = 16, level: int = 3) -> dict:
    ckpt_reader, base_reader = Reader(ckpt), Reader(base)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(set(ckpt_reader.weight_map.values()))
    manifest = {"format": "bf16-int16-delta-zstd/1", "base_repo": BASE_REPO, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "ckpt_index": ckpt_reader.index, "files": {}, "tensors": {}}
    started = time.time()

    def pack_file(position: int, filename: str) -> tuple[str, dict, dict]:
        compressor = zstandard.ZstdCompressor(level=level)
        names = [n for n, f in ckpt_reader.weight_map.items() if f == filename]
        blob_name = f"delta-{position:05d}.bin"
        entries, offset, changed = {}, 0, 0
        with open(out / blob_name, "wb") as blob:
            for name in sorted(names):
                mine = ckpt_reader.get(name)
                other = base_reader.get(name)
                if mine.dtype != torch.bfloat16 or other.shape != mine.shape or other.dtype != mine.dtype:
                    raise SystemExit(f"{name}: unsupported dtype/shape {mine.dtype} {tuple(mine.shape)} vs base {other.dtype} {tuple(other.shape)}")
                a, b = mine.view(torch.int16), other.view(torch.int16)
                if torch.equal(a, b):
                    entries[name] = {"mode": "same", "shape": list(mine.shape)}
                    continue
                delta = _wrap16(a.to(torch.int32) - b.to(torch.int32))
                data = compressor.compress(delta.numpy().tobytes())
                blob.write(data)
                entries[name] = {"mode": "delta16", "shape": list(mine.shape), "blob": blob_name, "offset": offset,
                                 "length": len(data), "changed": int((delta != 0).sum())}
                offset += len(data)
                changed += 1
        file_info = {"sha256": sha256_file(ckpt / filename), "size": (ckpt / filename).stat().st_size,
                     "blob": blob_name, "blob_bytes": offset, "changed_tensors": changed, "tensors": len(names)}
        return filename, file_info, entries

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for filename, file_info, entries in pool.map(lambda p: pack_file(*p), enumerate(files)):
            manifest["files"][filename] = file_info
            manifest["tensors"].update(entries)
            log.info("%s: %d/%d tensors changed -> %.2f GB", filename, file_info["changed_tensors"], file_info["tensors"],
                     file_info["blob_bytes"] / 1e9)
    for name in CONTRACT_FILES:
        shutil.copyfile(ckpt / name, out / name)
    changed = sum(e.get("changed", 0) for e in manifest["tensors"].values())
    total = sum(int(torch.Size(e["shape"]).numel()) for e in manifest["tensors"].values())
    manifest["summary"] = {"elements": total, "changed_elements": changed,
                           "delta_bytes": sum(f["blob_bytes"] for f in manifest["files"].values()),
                           "ckpt_bytes": sum(f["size"] for f in manifest["files"].values()),
                           "seconds": round(time.time() - started, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("packed %s: %.2f GB of deltas for %.1f GB checkpoint (%.2f%% of elements changed) in %.0fs",
             ckpt, manifest["summary"]["delta_bytes"] / 1e9, manifest["summary"]["ckpt_bytes"] / 1e9,
             100.0 * changed / max(total, 1), time.time() - started)
    return manifest


def restore(delta: Path, base: Path, out: Path, *, workers: int = 16) -> None:
    manifest = json.loads((delta / "manifest.json").read_text())
    base_reader = Reader(base)
    out.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[str]] = {}
    for name, filename in manifest["ckpt_index"]["weight_map"].items():
        by_file.setdefault(filename, []).append(name)

    def restore_file(filename: str) -> str:
        decompressor = zstandard.ZstdDecompressor()
        tensors = {}
        blob_path = delta / manifest["files"][filename]["blob"]
        with open(blob_path, "rb") as blob:
            for name in sorted(by_file[filename]):
                entry = manifest["tensors"][name]
                other = base_reader.get(name)
                if entry["mode"] == "same":
                    tensors[name] = other.clone()
                    continue
                blob.seek(entry["offset"])
                raw = decompressor.decompress(blob.read(entry["length"]))
                d = torch.frombuffer(bytearray(raw), dtype=torch.int16).view(other.shape)
                bits = _wrap16(other.view(torch.int16).to(torch.int32) + d.to(torch.int32))
                tensors[name] = bits.view(torch.bfloat16)
        save_file(tensors, str(out / filename), metadata={"format": "pt"})
        digest = sha256_file(out / filename)
        if digest != manifest["files"][filename]["sha256"]:
            raise SystemExit(f"{filename}: restored sha256 {digest} != original {manifest['files'][filename]['sha256']}")
        return filename

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for done, filename in enumerate(pool.map(restore_file, sorted(by_file)), 1):
            log.info("restored %s (%d/%d, sha256 ok)", filename, done, len(by_file))
    (out / "model.safetensors.index.json").write_text(json.dumps(manifest["ckpt_index"], indent=2))
    for name in CONTRACT_FILES:
        shutil.copyfile(delta / name, out / name)
    log.info("restored %s: every shard byte-identical to the original export", out)


def push(ckpt: Path, base: Path, repo: str, name: str, note: dict, staging_root: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    staging = staging_root / name
    shutil.rmtree(staging, ignore_errors=True)
    manifest = pack(ckpt, base, staging)
    (staging / "NOTE.json").write_text(json.dumps({**note, "source_dir": str(ckpt), "summary": manifest["summary"]}, indent=1))
    started = time.time()
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(staging), path_in_repo=name,
                      commit_message=f"backup {name}: {json.dumps(note)[:200]}")
    log.info("uploaded %s -> %s/%s (%.2f GB) in %.0fs", staging, repo, name, manifest["summary"]["delta_bytes"] / 1e9,
             time.time() - started)
    shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack"); p.add_argument("--ckpt", type=Path, required=True); p.add_argument("--base", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("restore"); r.add_argument("--delta", type=Path, required=True); r.add_argument("--base", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    u = sub.add_parser("push"); u.add_argument("--ckpt", type=Path, required=True); u.add_argument("--base", type=Path, required=True)
    u.add_argument("--repo", required=True); u.add_argument("--name", required=True); u.add_argument("--note", default="{}")
    u.add_argument("--staging", type=Path, default=Path("/workspace/teutonic/delta_staging"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    torch.set_num_threads(4)
    if args.command == "pack":
        pack(args.ckpt, args.base, args.out)
    elif args.command == "restore":
        restore(args.delta, args.base, args.out)
    else:
        push(args.ckpt, args.base, args.repo, args.name, json.loads(args.note), args.staging)


if __name__ == "__main__":
    sys.exit(main())
