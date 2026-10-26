"""Offline paired-bootstrap evaluation mirroring the Teutonic validator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from src.data import EvalDataset, Manifest
from src.dist.process_groups import ProcessGroups

log = logging.getLogger(__name__)

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_ALPHA = 0.001
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_SAFETY_MARGIN = 0.05


def paired_bootstrap_lcb(
    diffs: np.ndarray,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Compute paired-bootstrap mean and one-sided lower confidence bound."""
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        raise ValueError("diffs must contain at least one value")
    mu_hat = float(values.mean())
    rng = np.random.default_rng(seed=seed)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        sample = rng.choice(values, size=values.size, replace=True)
        boot_means[index] = sample.mean()
    lcb = float(2.0 * mu_hat - np.quantile(boot_means, 1.0 - alpha))
    return mu_hat, lcb


def compute_per_sequence_ce(
    model: nn.Module,
    token_seqs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-sequence cross-entropy in nats/token without mean reduction."""
    model.eval()
    all_losses: list[torch.Tensor] = []
    num_sequences = token_seqs.shape[0]
    progress = tqdm(range(0, num_sequences, batch_size), desc="offline_eval", leave=False)
    with torch.inference_mode():
        for start in progress:
            batch = token_seqs[start : start + batch_size].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=batch, use_cache=False).logits
            shift_logits = logits[:, :-1, :].float()
            shift_labels = batch[:, 1:]
            token_ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view(batch.size(0), -1)
            seq_ce = token_ce.mean(dim=1)
            all_losses.append(seq_ce.detach().cpu())
    return torch.cat(all_losses, dim=0)


def _broadcast_batch(batch: torch.Tensor, pg: ProcessGroups) -> torch.Tensor:
    """Broadcast a batch tensor from rank 0 to all ranks."""
    if not dist.is_initialized():
        return batch
    if pg.is_rank0:
        shape = torch.tensor(list(batch.shape), dtype=torch.long, device=batch.device)
    else:
        shape = torch.empty(batch.dim(), dtype=torch.long, device=batch.device)
    dist.broadcast(shape, src=0)
    if not pg.is_rank0:
        batch = torch.empty(tuple(shape.tolist()), dtype=batch.dtype, device=batch.device)
    dist.broadcast(batch, src=0)
    return batch


def compute_per_sequence_ce_distributed(
    model: nn.Module,
    token_seqs: torch.Tensor,
    batch_size: int,
    device: torch.device,
    pg: ProcessGroups,
    *,
    progress_desc: str = "offline_eval",
) -> torch.Tensor:
    """FSDP-safe per-sequence CE where all ranks participate in each forward."""
    model.eval()
    if dist.is_initialized():
        count = torch.tensor([token_seqs.shape[0] if pg.is_rank0 else 0], device=device, dtype=torch.long)
        dist.broadcast(count, src=0)
        num_sequences = int(count.item())
    else:
        num_sequences = token_seqs.shape[0]

    all_losses: list[torch.Tensor] = []
    progress = (
        tqdm(range(0, num_sequences, batch_size), desc=progress_desc, leave=False)
        if pg.is_rank0
        else range(0, num_sequences, batch_size)
    )
    with torch.inference_mode():
        for start in progress:
            if pg.is_rank0:
                batch = token_seqs[start : start + batch_size].to(device, non_blocking=True)
            else:
                batch = torch.empty((1, 1), dtype=torch.long, device=device)
            batch = _broadcast_batch(batch, pg)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=batch, use_cache=False).logits
            if pg.is_rank0:
                shift_logits = logits[:, :-1, :].float()
                shift_labels = batch[:, 1:]
                token_ce = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                ).view(batch.size(0), -1)
                all_losses.append(token_ce.mean(dim=1).detach().cpu())
    if not pg.is_rank0:
        return torch.empty(0)
    return torch.cat(all_losses, dim=0)


def score_source_distributed(
    model: nn.Module,
    eval_dataset: EvalDataset,
    source_name: str,
    n_sequences: int,
    eval_batch_size: int,
    device: torch.device,
    pg: ProcessGroups,
    *,
    progress_desc: str,
) -> torch.Tensor:
    """Score one manifest source; returns empty tensor on non-rank0."""
    batch = eval_dataset.get_batch(source_name)[:n_sequences] if pg.is_rank0 else torch.empty(0)
    return compute_per_sequence_ce_distributed(
        model,
        batch,
        eval_batch_size,
        device,
        pg,
        progress_desc=progress_desc,
    )


def build_paired_eval_result(
    king_ces: dict[str, torch.Tensor],
    challenger_ces: dict[str, torch.Tensor],
    sources: list[Any],
    eval_n: int,
    delta: float,
) -> EvalResult:
    """Build validator-style paired result from per-sequence CE tensors."""
    per_source: dict[str, dict[str, float | int]] = {}
    all_diffs: list[np.ndarray] = []
    for source in sources:
        n_source = max(1, round(eval_n * source.proportion))
        k_ce = king_ces[source.name]
        c_ce = challenger_ces[source.name]
        diffs = k_ce - c_ce
        all_diffs.append(diffs.numpy())
        per_source[source.name] = {
            "n": n_source,
            "king": float(k_ce.mean().item()),
            "challenger": float(c_ce.mean().item()),
            "mu_hat": float(diffs.mean().item()),
            "challenger_wins_pct": float((diffs > 0).float().mean().item() * 100.0),
        }

    concatenated = np.concatenate(all_diffs)
    mu_hat, lcb = paired_bootstrap_lcb(concatenated)
    return EvalResult(
        mu_hat=mu_hat,
        lcb=lcb,
        accepted=lcb > delta,
        delta=delta,
        n_sequences=int(concatenated.size),
        per_source=per_source,
    )


def log_paired_source_row(
    source_name: str,
    n: int,
    king_mean: float,
    challenger_mean: float,
    mu_hat: float,
    *,
    challenger_wins_pct: float | None = None,
) -> None:
    """Log one validator-style king vs challenger row (rank 0 only)."""
    delta_ce = king_mean - challenger_mean
    winner = "challenger" if mu_hat > 0 else "king" if mu_hat < 0 else "tie"
    extra = f" | challenger_wins={challenger_wins_pct:.1f}%" if challenger_wins_pct is not None else ""
    log.info(
        "Paired eval %s (n=%d): king=%.4f challenger=%.4f delta=%+.4f mu_hat=%+.4f -> %s%s",
        source_name,
        n,
        king_mean,
        challenger_mean,
        delta_ce,
        mu_hat,
        winner,
        extra,
    )


def log_paired_eval_summary(result: EvalResult, *, king_label: str, challenger_label: str) -> None:
    """Log full paired dual-eval table after both models are scored."""
    log.info(
        "Paired dual eval complete: king=%s vs challenger=%s | mu_hat=%.6f lcb=%.6f accepted=%s",
        king_label,
        challenger_label,
        result.mu_hat,
        result.lcb,
        result.accepted,
    )
    log.info(
        "Positive mu_hat means challenger beats king on the same sequences (lower CE is better)."
    )
    for source, stats in sorted(result.per_source.items()):
        log_paired_source_row(
            source,
            int(stats["n"]),
            float(stats["king"]),
            float(stats["challenger"]),
            float(stats["mu_hat"]),
            challenger_wins_pct=float(stats.get("challenger_wins_pct", 0.0)),
        )


@dataclass
class EvalResult:
    """Result of an offline paired-bootstrap evaluation."""

    mu_hat: float
    lcb: float
    accepted: bool
    delta: float
    n_sequences: int
    per_source: dict[str, dict[str, float | int]] = field(default_factory=dict)

    def is_safe_to_submit(self, safety_margin: float = DEFAULT_SAFETY_MARGIN) -> bool:
        return self.lcb > self.delta + safety_margin

    def __str__(self) -> str:
        status = "✅" if self.accepted else "❌"
        safe = "✅" if self.is_safe_to_submit() else "❌"
        lines = [
            f"Paired Dual Eval {status} | mu_hat={self.mu_hat:.6f} | lcb={self.lcb:.6f} | "
            f"delta={self.delta:.3f} | n={self.n_sequences}",
            "mu_hat = mean(king_ce - challenger_ce); positive => challenger wins on same sequences",
            f"Safe to submit (margin={DEFAULT_SAFETY_MARGIN}): {safe}",
            "Per-source breakdown (identical sequences for king and challenger):",
            f"{'source':<24} {'n':>6} {'king':>10} {'challenger':>12} {'mu_hat':>10} {'chall_w%':>8}",
        ]
        for source, stats in sorted(self.per_source.items()):
            chall_w = float(stats.get("challenger_wins_pct", 0.0))
            lines.append(
                f"{source:<24} {int(stats['n']):>6} "
                f"{float(stats['king']):>10.4f} {float(stats['challenger']):>12.4f} "
                f"{float(stats['mu_hat']):>10.4f} {chall_w:>7.1f}%"
            )
        return "\n".join(lines)


class OfflineEvaluator:
    """Runs validator-style paired bootstrap evaluation offline."""

    def __init__(
        self,
        king_model: nn.Module,
        eval_dataset: EvalDataset,
        manifest: Manifest,
        device: torch.device,
        eval_batch_size: int,
    ) -> None:
        self.king_model = king_model
        self.eval_dataset = eval_dataset
        self.manifest = manifest
        self.device = device
        self.eval_batch_size = eval_batch_size

    def __repr__(self) -> str:
        return (
            f"OfflineEvaluator(device={self.device}, eval_batch_size={self.eval_batch_size}, "
            f"sources={self.eval_dataset.source_names})"
        )

    def evaluate(self, challenger_model: nn.Module) -> EvalResult:
        sources = self.manifest.fetch(force=True)
        eval_n = self.manifest.eval_n
        delta = self.manifest.delta_threshold
        per_source: dict[str, dict[str, float | int]] = {}
        all_diffs: list[np.ndarray] = []

        for source in sources:
            n_source = max(1, round(eval_n * source.proportion))
            batch = self.eval_dataset.get_batch(source.name)[:n_source]
            king_ce = compute_per_sequence_ce(
                self.king_model,
                batch,
                self.eval_batch_size,
                self.device,
            )
            challenger_ce = compute_per_sequence_ce(
                challenger_model,
                batch,
                self.eval_batch_size,
                self.device,
            )
            diffs = king_ce - challenger_ce
            all_diffs.append(diffs.numpy())
            per_source[source.name] = {
                "n": n_source,
                "king": float(king_ce.mean().item()),
                "challenger": float(challenger_ce.mean().item()),
                "mu_hat": float(diffs.mean().item()),
            }

        concatenated = np.concatenate(all_diffs)
        mu_hat, lcb = paired_bootstrap_lcb(concatenated)
        accepted = lcb > delta
        result = EvalResult(
            mu_hat=mu_hat,
            lcb=lcb,
            accepted=accepted,
            delta=delta,
            n_sequences=int(concatenated.size),
            per_source=per_source,
        )
        log.info("Offline evaluation complete: mu_hat=%.6f lcb=%.6f accepted=%s", mu_hat, lcb, accepted)
        return result

    def evaluate_distributed(self, challenger_model: nn.Module, pg: ProcessGroups) -> EvalResult:
        """Run paired-bootstrap evaluation with FSDP-safe distributed forwards."""
        sources = self.manifest.fetch(force=True)
        eval_n = self.manifest.eval_n
        delta = self.manifest.delta_threshold
        per_source: dict[str, dict[str, float | int]] = {}
        all_diffs: list[np.ndarray] = []

        for source in sources:
            n_source = max(1, round(eval_n * source.proportion))
            batch = self.eval_dataset.get_batch(source.name)[:n_source] if pg.is_rank0 else torch.empty(0)
            king_ce = compute_per_sequence_ce_distributed(
                self.king_model,
                batch,
                self.eval_batch_size,
                self.device,
                pg,
            )
            challenger_ce = compute_per_sequence_ce_distributed(
                challenger_model,
                batch,
                self.eval_batch_size,
                self.device,
                pg,
            )
            if pg.is_rank0:
                diffs = king_ce - challenger_ce
                all_diffs.append(diffs.numpy())
                per_source[source.name] = {
                    "n": n_source,
                    "king": float(king_ce.mean().item()),
                    "challenger": float(challenger_ce.mean().item()),
                    "mu_hat": float(diffs.mean().item()),
                }

        if not pg.is_rank0:
            return EvalResult(mu_hat=0.0, lcb=0.0, accepted=False, delta=delta, n_sequences=0)

        concatenated = np.concatenate(all_diffs)
        mu_hat, lcb = paired_bootstrap_lcb(concatenated)
        accepted = lcb > delta
        result = EvalResult(
            mu_hat=mu_hat,
            lcb=lcb,
            accepted=accepted,
            delta=delta,
            n_sequences=int(concatenated.size),
            per_source=per_source,
        )
        log.info("Offline evaluation complete: mu_hat=%.6f lcb=%.6f accepted=%s", mu_hat, lcb, accepted)
        return result
