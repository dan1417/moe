"""Fit only ``lm_head`` on the eval mixture, on top of a frozen (merged) body.

With the body frozen the objective is convex in the head (softmax regression on
fixed features), so this cannot damage representations and converges predictably.
The frozen forward runs through :class:`src.parity.pipeline.PipelineScorer`; the last
GPU keeps an fp32 master copy + AdamW moments (~6 GB), accumulates dCE/dW, steps
every ``accum_seqs`` sequences and writes the bf16 head back in place, so later
batches -- and the exported weights -- use exactly the bf16 head that is scored.
"""

from __future__ import annotations

import logging
import threading

import torch

from src.parity import upstream

log = logging.getLogger(__name__)


class HeadTuner:
    def __init__(
        self,
        model,
        *,
        lr: float,
        accum_seqs: int,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        grad_clip: float = 1.0,
        chunk: int = 512,
    ):
        self.model = model
        self.weight = model.lm_head.weight  # bf16, on the last GPU
        self.master = self.weight.detach().float().clone()
        self.initial = self.weight.detach().clone()
        self.exp_avg = torch.zeros_like(self.master)
        self.exp_avg_sq = torch.zeros_like(self.master)
        self.grad = torch.zeros_like(self.master)
        self.lr, self.accum_seqs, self.betas, self.eps = lr, accum_seqs, betas, eps
        self.grad_clip, self.chunk = grad_clip, chunk
        self.pending_seqs = self.pending_tokens = 0
        self.steps = 0
        self.training = True
        self.lock = threading.Lock()
        self.history: list[dict] = []

    @torch.no_grad()
    def __call__(self, model, token_batches: list[list[int]], lm_head_chunk: int) -> list[float]:
        """PipelineScorer ``fn``: per-sequence CE with the current head + gradient accumulation."""
        engine = upstream.engine
        input_ids = torch.tensor(token_batches, dtype=torch.long, device=engine.model_input_device(model))
        hidden = model.model(input_ids, use_cache=False).last_hidden_state
        head_device = self.weight.device
        hidden, labels = hidden.to(head_device), input_ids.to(head_device)
        batch, n_pos = labels.shape[0], labels.shape[1] - 1
        totals = torch.zeros(batch, dtype=torch.float64, device=head_device)
        with self.lock:  # the head update must not interleave with another batch's forward
            for start in range(0, n_pos, self.chunk):
                end = min(start + self.chunk, n_pos)
                feats = hidden[:, start:end].reshape(-1, hidden.shape[-1])
                target = labels[:, start + 1 : end + 1].reshape(-1)
                logits = torch.nn.functional.linear(feats, self.weight).float()
                log_probs = torch.log_softmax(logits, dim=-1)
                del logits
                token_ce = -log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
                totals += token_ce.view(batch, -1).sum(1).double()
                probs = log_probs.exp_()
                probs[torch.arange(len(target), device=head_device), target] -= 1.0  # dCE/dlogits
                if self.training:
                    self.grad.add_(torch.mm(probs.to(torch.bfloat16).t(), feats, out_dtype=torch.float32))
                del probs, log_probs
            if self.training:
                self.pending_seqs += batch
                self.pending_tokens += batch * n_pos
                if self.pending_seqs >= self.accum_seqs:
                    self._step()
        return (totals / n_pos).float().cpu().tolist()

    def _step(self) -> None:
        grad = self.grad.div_(self.pending_tokens)
        norm = float(grad.norm())
        if self.grad_clip and norm > self.grad_clip:
            grad.mul_(self.grad_clip / norm)
        self.steps += 1
        beta1, beta2 = self.betas
        self.exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias1 = 1 - beta1**self.steps
        bias2 = 1 - beta2**self.steps
        denom = (self.exp_avg_sq / bias2).sqrt_().add_(self.eps)
        self.master.addcdiv_(self.exp_avg, denom, value=-self.lr / bias1)
        self.weight.copy_(self.master)
        self.history.append({"step": self.steps, "grad_norm": round(norm, 5), "tokens": self.pending_tokens})
        self.grad.zero_()
        self.pending_seqs = self.pending_tokens = 0

    def relative_change(self) -> float:
        return float((self.weight.float() - self.initial.float()).norm() / self.initial.float().norm())
