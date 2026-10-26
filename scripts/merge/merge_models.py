#!/usr/bin/env python3
"""Merge two or more MoE model checkpoints with MoE-aware strategies."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from src.submit.merge import (
    MergeConfig,
    MergeStrategy,
    load_model_state,
    merge_model_paths,
    summarize_groups,
)

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two MiMo-V2 / DeepSeek-style MoE checkpoints. "
            "Use torchrun + --distributed for fast 8-GPU shard merge on Pro 6000."
        ),
    )
    parser.add_argument(
        "--model-a",
        type=Path,
        required=True,
        help="Base model directory or checkpoint (backbone source for aoe/experts_only)",
    )
    parser.add_argument(
        "--model-b",
        type=Path,
        help="Donor model directory or checkpoint to blend in",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=Path,
        help="Three or more models for SMA merge (overrides --model-a/--model-b)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./merged_model"),
        help="Output directory for merged safetensors / model.pt",
    )
    parser.add_argument(
        "--strategy",
        choices=[item.value for item in MergeStrategy],
        default=MergeStrategy.AOE.value,
        help="linear | sma | aoe | experts_only",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help=(
            "Donor weight for blended groups: (1-a)*A + a*B. "
            "a in (0,1) interpolates; a<0 extrapolates away from B; a>1 past B. "
            "Allowed range [-2, 2]."
        ),
    )
    parser.add_argument(
        "--group-alpha",
        action="append",
        default=[],
        metavar="GROUP=VALUE",
        help="Override alpha per group: routed_expert, shared_expert, router, attention, embed_lm, other",
    )
    parser.add_argument(
        "--contract-dir",
        type=Path,
        default=None,
        help="Contract files copied into output (default: model-a)",
    )
    parser.add_argument(
        "--genesis-dir",
        type=Path,
        default=Path("./genesis"),
        help="Genesis directory for --export-hf verification",
    )
    parser.add_argument(
        "--export-hf",
        action="store_true",
        help="Also export validator-ready safetensors (rank 0, single-process only)",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Parallel shard merge across GPUs (requires torchrun; model-a must be safetensors)",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=int(os.environ.get("GPUS_PER_NODE", "8")),
        help="GPUs per node for distributed merge topology",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for single-process streaming merge: cuda or cpu",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print tensor group summary for model-a and exit",
    )
    parser.add_argument(
        "--non-strict-keys",
        action="store_true",
        help="Merge only shared keys when checkpoints differ slightly",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def _parse_group_alphas(values: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid --group-alpha {item!r}; expected GROUP=VALUE")
        group, raw = item.split("=", 1)
        parsed[group.strip()] = float(raw.strip())
    return parsed


def _setup_distributed(gpus_per_node: int) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.inspect:
        state = load_model_state(args.model_a)
        summary = summarize_groups(state)
        print(json.dumps({"model": str(args.model_a), "n_tensors": len(state), "groups": summary}, indent=2))
        return 0

    if args.models:
        model_paths = args.models
    else:
        if args.model_b is None:
            log.error("Provide --model-b or --models")
            return 2
        model_paths = [args.model_a, args.model_b]

    if not -2.0 <= args.alpha <= 2.0:
        log.error("--alpha must be between -2 and 2 (got %s)", args.alpha)
        return 2
    if args.alpha < 0.0:
        log.warning(
            "alpha=%s < 0: extrapolating away from model-b past model-a (task-arithmetic style)",
            args.alpha,
        )
    elif args.alpha > 1.0:
        log.warning(
            "alpha=%s > 1: extrapolating past model-b away from model-a",
            args.alpha,
        )

    if args.distributed:
        _setup_distributed(args.gpus_per_node)
        if args.export_hf and dist.get_rank() != 0:
            args.export_hf = False

    config = MergeConfig(
        strategy=MergeStrategy(args.strategy),
        alpha=args.alpha,
        group_alphas=_parse_group_alphas(args.group_alpha),
        strict_keys=not args.non_strict_keys,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    try:
        out, report = merge_model_paths(
            model_paths=model_paths,
            output_dir=args.output,
            config=config,
            contract_dir=args.contract_dir or args.model_a,
            genesis_dir=args.genesis_dir,
            export_hf=args.export_hf and not args.distributed,
            device=device,
            distributed=args.distributed,
            gpus_per_node=args.gpus_per_node,
        )
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        log.error("Merge failed: %s", exc)
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()
        return 1

    is_rank0 = not args.distributed or not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        print(report)
        print(f"Merged model saved under {out}")
        if (out / "merge_report.json").exists():
            print(f"Report: {out / 'merge_report.json'}")

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
