#!/usr/bin/env python3
"""Build a CPT token pool (or held-out val set) from the live eval distribution.

Each round = one validator shard plan (27 shards over 24 categories) with quotas
scaled by --scale; ~30k*scale sequences (~61M*scale tokens) per round. Shards used
by local parity samples are excluded.

  build_train_pool.py --out /data/pools/line_a --rounds 10 --scale 2 --tag line_a
  build_train_pool.py --out /data/pools/val --rounds 1 --scale 0.1 --tag val
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cpt.data import build_pool  # noqa: E402
from src.parity.sample import load_live_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tag", default="pool", help="seed family; use distinct tags per line and for val")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--exclude-pool", type=Path, action="append", default=[],
                        help="pool dir whose shards must not be reused (pass the val pool when building train pools)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    build_pool(load_live_config(), args.out, rounds=args.rounds, scale=args.scale, tag=args.tag, workers=args.workers,
               exclude_pools=args.exclude_pool)


if __name__ == "__main__":
    main()
