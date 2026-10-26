"""Duel verdicts: the validator's exact math plus submission-risk diagnostics.

Since validator 14a9bb2 (policy ``document-masked-token-bootstrap-v1``) every average is
token-weighted: a sample's weight is its number of scored targets after document masking
(``EvalSample.scored_tokens``), the bootstrap resamples whole windows/documents and recomputes
the weighted ratio. Samples with whole documents are always scored in full (no early stopping).
"""

from __future__ import annotations

import math
import re

import numpy as np

from src.parity import upstream
from src.parity.sample import EvalSample

Z_ALPHA = 3.090232306167813  # one-sided alpha = 0.001, the validator's alpha
# Early-stopping values the live validator uses are private (PostgreSQL); these match
# its dataclass defaults and the public history (first stop at 20% of n, 100-step
# checks). Only weak challengers ever trigger it.
ASSUMED_EARLY_STOP = dict(enabled=True, min_fraction=0.2, advantage_quantile=0.95, margin=0.0, check_interval=100)


def category_of_shard(name: str) -> str:
    match = re.match(r"([a-z0-9-]+?)__([a-zA-Z0-9_-]+?)--", name)
    return f"{match.group(1)}/{match.group(2)}" if match else "*"


def simulate_early_stop(king: np.ndarray, challenger: np.ndarray, weights: np.ndarray, delta: float,
                        policy: dict) -> dict | None:
    """Replay the validator's (token-weighted) futility checks over the sample order."""
    early = upstream.early_stopping
    rule = early.EarlyStoppingPolicy(**policy)
    total = len(king)
    check = max(1, math.ceil(total * rule.min_fraction))
    while check < total:
        decision = early.challenger_futility_decision(
            king[:check].tolist(), challenger[:check].tolist(),
            total_sequences=total, delta_threshold=delta, policy=rule,
            scored_tokens=weights[:check].tolist(), total_scored_tokens=int(weights.sum()),
        )
        if decision is not None:
            return {"stopped_at": check, **{k: round(v, 6) for k, v in decision.items()}}
        check += rule.check_interval
    return None


def weighted_se(diff: np.ndarray, weights: np.ndarray) -> float:
    """SE of the token-weighted mean sum(w d) / sum(w) over iid samples (ratio estimator)."""
    n = len(diff)
    mu = np.average(diff, weights=weights)
    return float(math.sqrt(n / max(n - 1, 1) * np.sum(weights**2 * (diff - mu) ** 2)) / weights.sum())


def stratified_shard_jackknife(diff: np.ndarray, weights: np.ndarray, sample: EvalSample) -> float:
    """Delete-one-cluster jackknife SE of the token-weighted mean; strata = sources (fixed token
    shares), clusters = window shards, each whole document its own cluster."""
    sources = sample.sources
    shards = sample.shard_keys()
    total = weights.sum()
    variance = 0.0
    for name in sorted(set(sources.tolist())):
        in_source = sources == name
        clusters, inverse = np.unique(shards[in_source], return_inverse=True)
        m = len(clusters)
        if m < 2:
            continue
        w, wd = weights[in_source], weights[in_source] * diff[in_source]
        cw, cwd = np.bincount(inverse, w, m), np.bincount(inverse, wd, m)
        estimates = (wd.sum() - cwd) / (w.sum() - cw)
        variance += (w.sum() / total) ** 2 * (m - 1) / m * np.sum((estimates - estimates.mean()) ** 2)
    return float(math.sqrt(variance))


def duel(
    king: np.ndarray,
    challenger: np.ndarray,
    sample: EvalSample,
    *,
    delta: float,
    n_bootstrap: int = 10_000,
    alpha: float = 0.001,
    bootstrap_seed: int = 0,
    early_stop: dict | None = None,
) -> dict:
    """Validator verdict on (a prefix of) the sample, plus risk diagnostics."""
    scored = ~(np.isnan(king) | np.isnan(challenger))
    n = int(scored.sum())
    if n == 0:
        raise ValueError("no sequence has both losses")
    full = n == len(king)
    king_s, chall_s = king[scored], challenger[scored]
    weights = sample.scored_tokens[scored].astype(np.float64)
    counts = sample.scored_tokens[scored].tolist()
    engine = upstream.engine
    verdict = upstream.policy.paired_bootstrap_verdict(
        king_s.tolist(), chall_s.tolist(),
        bootstrap_seed=bootstrap_seed, n_bootstrap=n_bootstrap, alpha=alpha, delta_threshold=delta,
        scored_tokens=counts,
    )
    verdict.pop("timestamp", None)
    verdict["n_scored_tokens"] = int(weights.sum())
    labels = sample.sources[scored].tolist()
    verdict["source_scores"] = engine._compute_source_scores(king_s.tolist(), chall_s.tolist(), labels, counts)

    indices = np.flatnonzero(scored)
    if sample.has_documents:
        components = sample.components[scored].tolist()
        verdict["component_scores"] = engine._compute_source_scores(king_s.tolist(), chall_s.tolist(), components, counts)
        provenance = [{"component": "long_documents", **sample.documents[i]} if i in sample.documents
                      else {"component": "windows"} for i in indices.tolist()]
        verdict["long_document_scores"] = engine.long_document_scores(king_s.tolist(), chall_s.tolist(), provenance, counts)

    diff = king_s - chall_s
    shard_names = sample.meta["shard_names"]
    source_order = [s["name"] for s in sample.meta["sources"]]
    categories = np.asarray([
        f"doc:{sample.documents[i]['dataset']}/{sample.documents[i]['category']}" if i in sample.documents
        else category_of_shard(shard_names[source_order[g]][s])
        for i, (g, s) in zip(indices.tolist(), sample.provenance[scored][:, :2].tolist())
    ])
    verdict["category_scores"] = {
        cat: {"n": int((categories == cat).sum()),
              "mu_hat": round(float(np.average(diff[categories == cat], weights=weights[categories == cat])), 6)}
        for cat in sorted(set(categories.tolist()))
    }
    se_iid = weighted_se(diff, weights)
    verdict["se_iid"] = round(se_iid, 7)
    verdict["win_rate"] = round(float((diff > 0).mean()), 4)
    if full:
        se_shard = stratified_shard_jackknife(king - challenger, sample.scored_tokens.astype(np.float64), sample)
        verdict["se_shard_jackknife"] = round(se_shard, 7)
        verdict["robust_lcb"] = round(float(np.average(diff, weights=weights)) - Z_ALPHA * max(se_shard, se_iid), 6)
        # the validator never stops a mixed window/document sample early
        verdict["early_stop"] = None if sample.has_documents else simulate_early_stop(
            king, challenger, sample.scored_tokens, delta, early_stop or ASSUMED_EARLY_STOP)
    verdict["complete"] = full
    return verdict
