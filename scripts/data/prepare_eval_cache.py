#!/usr/bin/env python3
"""Prefetch evaluation sequences used by offline paired-bootstrap eval."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from src.data import DEFAULT_EVAL_PREFETCH_WORKERS, build_eval_dataset

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch eval shard cache for offline eval.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pro6000/train.yaml"),
        help="Training config (uses data.* and manifest_url)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_EVAL_PREFETCH_WORKERS,
        help=f"Parallel shard download workers (default {DEFAULT_EVAL_PREFETCH_WORKERS})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with open(args.config) as handle:
        cfg = yaml.safe_load(handle)

    log.info(
        "Building eval dataset from %s (seed=99999, %d download workers)...",
        args.config,
        args.workers,
    )
    eval_dataset = build_eval_dataset(cfg, save_cache=True, prefetch_workers=args.workers)
    sizes = {name: tensor.shape[0] for name, tensor in eval_dataset._batches.items()}
    total = sum(sizes.values())
    log.info("Eval cache ready: %d sequences across %d sources (%s)", total, len(sizes), sizes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
