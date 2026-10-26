"""Logit-temperature diagnostics: is the model over- or under-confident on the eval mix?

Scaling ``lm_head`` (or the final norm) by ``c`` scales every logit by ``c`` -- a
weights-only change the locked architecture allows. For per-token CE(c) with logits
``c·z``: dCE/dc|_{c=1} = E_p[z] - z_y and d²CE/dc² = Var_p[z]. One pass gives the
Newton estimate c* = 1 - E[g]/E[v] and the predicted gain E[g]²/(2·E[v]).
"""

from __future__ import annotations

import numpy as np
import torch

from src.parity import upstream

STAT_WIDTH = 4  # ce_sum, grad_sum, curvature_sum, n_tokens


@torch.no_grad()
def calibration_stats(model, token_batches: list[list[int]], chunk_size: int) -> list[list[float]]:
    engine = upstream.engine
    chunk_size = min(chunk_size, 512)
    input_ids = torch.tensor(token_batches, dtype=torch.long, device=engine.model_input_device(model))
    hidden = model.model(input_ids, use_cache=False).last_hidden_state
    head = engine.lm_head_device(model)
    hidden, labels = hidden.to(head), input_ids.to(head)
    n_pos = labels.shape[1] - 1
    out = torch.zeros(len(token_batches), STAT_WIDTH, dtype=torch.float64, device=head)
    for start in range(0, n_pos, chunk_size):
        end = min(start + chunk_size, n_pos)
        logits = model.lm_head(hidden[:, start:end]).float()
        target = labels[:, start + 1 : end + 1]
        log_norm = torch.logsumexp(logits, dim=-1)
        probs = torch.exp(logits - log_norm.unsqueeze(-1))
        mean_z = (probs * logits).sum(-1)
        mean_z2 = (probs * logits.square()).sum(-1)
        del probs
        z_y = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        out[:, 0] += (log_norm - z_y).sum(1).double()
        out[:, 1] += (mean_z - z_y).sum(1).double()
        out[:, 2] += (mean_z2 - mean_z.square()).sum(1).double()
        del logits
    out[:, 3] = n_pos
    return out.cpu().tolist()


def summarize(stats: np.ndarray, sources: np.ndarray) -> dict:
    """Optimal logit scale and predicted CE gain, overall and per source."""
    def one(rows: np.ndarray) -> dict:
        tokens = rows[:, 3].sum()
        ce, grad, curv = (rows[:, i].sum() / tokens for i in range(3))
        scale = 1.0 - grad / curv
        return {"n": int(len(rows)), "ce": round(ce, 6), "grad": round(grad, 6), "curvature": round(curv, 4),
                "optimal_scale": round(scale, 5), "predicted_gain": round(grad * grad / (2 * curv), 7)}

    done = np.isfinite(stats[:, 0])
    rows, labels = stats[done], sources[done]
    result = {"all": one(rows)}
    for name in sorted(set(labels.tolist())):
        result[name] = one(rows[labels == name])
    return result
