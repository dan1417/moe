"""Per-token cross-entropy (validator numerics) for token-level diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.parity import upstream

SEQ_POS = 2047  # targets per 2048-token sequence


@torch.no_grad()
def per_token_ce(model, token_batches: list[list[int]], chunk_size: int) -> list[list[float]]:
    """[batch][2047] CE per target position, computed exactly like the validator's scorer
    (bf16 logits in lm_head chunks) but without the per-sequence mean."""
    engine = upstream.engine
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=engine.model_input_device(model))
    hidden = model.model(input_ids, use_cache=False).last_hidden_state
    head = engine.lm_head_device(model)
    hidden, labels = hidden.to(head), input_ids.to(head)
    n_pos = labels.shape[1] - 1
    pieces = []
    for start in range(0, n_pos, chunk_size):
        end = min(start + chunk_size, n_pos)
        logits = model.lm_head(hidden[:, start:end, :])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, start + 1 : end + 1].reshape(-1), reduction="none")
        pieces.append(loss.view(len(token_batches), -1).float())
        del logits
    return torch.cat(pieces, dim=1).cpu().tolist()
