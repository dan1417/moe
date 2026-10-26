"""Per-(sample, model, scorer) cache of per-sequence losses."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from src.parity import upstream

MOE_KERNEL = "grouped_mm_v1"


def scorer_signature(attn_implementation: str, lm_head_chunk: int, deterministic: bool = True) -> str:
    # deterministic (the validator's mode since 14a9bb2) and default kernels differ at rounding level:
    # king and challenger losses must come from the same mode
    mode = "" if deterministic else "|nondet"
    # the scoring-code commit ("14a9bb2cd8f2"), not the installed pin: moving the pin to 5118c32 (range reads) left
    # every loss-producing file byte-identical, and the loss caches of kings/candidates on unchanged samples must survive
    return f"{attn_implementation}|chunk{lm_head_chunk}|{MOE_KERNEL}|{upstream.SCORING_CODE_COMMIT[:12]}{mode}"


class LossCache:
    def __init__(self, root: Path | None = None):
        self.root = (root or upstream.DEFAULT_CACHE_ROOT) / "losses"

    def _paths(self, sample_id: str, model_id: str, signature: str) -> tuple[Path, Path]:
        key = hashlib.sha256(f"{model_id}|{signature}".encode()).hexdigest()[:24]
        base = self.root / sample_id / key
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def load(self, sample_id: str, model_id: str, signature: str, n: int) -> np.ndarray:
        data, _ = self._paths(sample_id, model_id, signature)
        if data.exists():
            return np.load(data)
        return np.full(n, np.nan, dtype=np.float64)

    def save(self, sample_id: str, model_id: str, signature: str, losses: np.ndarray, info: dict) -> None:
        data, meta = self._paths(sample_id, model_id, signature)
        data.parent.mkdir(parents=True, exist_ok=True)
        tmp = data.with_suffix(".tmp.npy")
        np.save(tmp, losses)
        tmp.replace(data)
        record = {
            "model_id": model_id,
            "signature": signature,
            "sample_id": sample_id,
            "scored": int(np.isfinite(losses).sum()),
            "mean_loss_scored": float(np.nanmean(losses)) if np.isfinite(losses).any() else None,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            **info,
        }
        meta.write_text(json.dumps(record, indent=1))

    def entries(self, sample_id: str) -> list[dict]:
        folder = self.root / sample_id
        if not folder.exists():
            return []
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
