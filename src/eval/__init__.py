"""Offline paired-bootstrap evaluation (validator-style)."""

from src.eval.offline import (
    DEFAULT_SAFETY_MARGIN,
    build_paired_eval_result,
    log_paired_eval_summary,
    log_paired_source_row,
    score_source_distributed,
)

__all__ = [
    "DEFAULT_SAFETY_MARGIN",
    "build_paired_eval_result",
    "log_paired_eval_summary",
    "log_paired_source_row",
    "score_source_distributed",
]
