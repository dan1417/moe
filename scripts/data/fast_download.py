#!/usr/bin/env python3
"""Download a model from R2 by digest with N processes (download_king.py's verified, resumable range downloader per file
subset). One download_king process tops out at ~120 MB/s (Python-bound; the NIC is 400 Gb/s), so a single urgent model
(a new king) is split over --procs processes by file size.

  python scripts/data/fast_download.py --digest <sha256> --output <dir> [--procs 8] [--workers 16]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_king as dk  # noqa: E402


def run(digest, group, out, workers):
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dk._download_files_parallel(digest, group, Path(out), timeout=dk.DEFAULT_TIMEOUT, workers=workers)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digest", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    manifest = dk.fetch_model_manifest(args.digest)
    files = sorted(manifest["files"], key=lambda f: -int(f.get("size", 0)))
    groups = [[] for _ in range(args.procs)]
    load = [0] * args.procs
    for f in files:  # largest first onto the lightest group
        i = load.index(min(load))
        groups[i].append(f)
        load[i] += int(f.get("size", 0))
    args.output.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    procs = [mp.Process(target=run, args=(args.digest, g, str(args.output), args.workers)) for g in groups if g]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [f["path"] for f in files if not (args.output / f["path"]).exists()
           or (args.output / f["path"]).stat().st_size != int(f.get("size", -1))]
    if bad or any(p.exitcode for p in procs):
        raise SystemExit(f"download incomplete: {bad[:5]} exitcodes {[p.exitcode for p in procs]}")
    total = sum(int(f.get("size", 0)) for f in files)
    (args.output / "king_info.json").write_text(json.dumps({"digest": args.digest, "file_count": len(files),
                                                            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    print(f"downloaded {len(files)} files, {total / 1e9:.1f} GB in {(time.time() - t0) / 60:.1f} min "
          f"({total / 1e9 / max(1, time.time() - t0):.2f} GB/s, {len(procs)} processes)", flush=True)


if __name__ == "__main__":
    main()
