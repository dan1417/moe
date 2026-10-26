#!/usr/bin/env python3
"""Check document-masked parity scoring (validator 14a9bb2) through the real pipeline.

  python scripts/verify/verify_doc_mask_eval.py --model <dir> --sample id:<sample_id> [--windows 8] [--attn flash_attention_4]

For --windows windows of the sample that contain at least two EOS-terminated fragments:
1. the window's masked loss (PipelineScorer -> validator compute_per_sequence_loss) equals the scored-token-weighted
   mean of its fragments' losses when every fragment is scored as a sample of its own (isolation + position reset);
   the fragments are batched together with the sample's whole documents, so packed mixed-length rows are covered too;
2. the documents' losses are finite and their scored-token counts are length - 1.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.parity import upstream  # noqa: E402
from src.parity.sample import Sequences, load_sample, sample_dir, scored_token_counts  # noqa: E402

EOS = upstream.masking.DOCUMENT_EOS_TOKEN_ID


def fragments(row: np.ndarray) -> list[np.ndarray]:
    cuts = np.flatnonzero(row[:-1] == EOS) + 1
    return [f for f in np.split(row, cuts)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--documents", type=int, default=4)
    ap.add_argument("--attn", default="flash_attention_4")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all)")
    ap.add_argument("--deterministic", type=int, default=1, help="the validator's execution mode (parity_eval default)")
    args = ap.parse_args()
    import os
    import time

    import torch

    if args.deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)

    from src.parity.pipeline import PipelineScorer
    from src.parity.weights import Recipe, free_model, load_model

    sample = load_sample(sample_dir(upstream.DEFAULT_CACHE_ROOT, args.sample.removeprefix("id:")))
    windows = [i for i in range(sample.n) if i not in sample.documents
               and sum(len(f) >= 2 for f in fragments(np.asarray(sample.tokens[i]))) >= 2][: args.windows]
    docs = sorted(sample.documents)[: args.documents]
    rows, owner = [], []  # owner: (window index, None) for windows, (window index, k) for its fragments, (doc, -1)
    for i in windows:
        rows.append(np.asarray(sample.tokens[i]))
        owner.append((i, None))
    for i in docs:
        rows.append(np.asarray(sample.tokens[i]))
        owner.append((i, -1))
    for i in windows:
        for k, frag in enumerate(fragments(np.asarray(sample.tokens[i]))):
            if (frag[:-1] != EOS).sum() >= 1:  # a fragment with no target cannot be scored alone
                rows.append(frag)
                owner.append((i, k))
    copies = []  # decisive leak test: a repeated chunk is near-free to predict only if attention crosses the EOS
    for i in windows[:4]:
        row = np.asarray(sample.tokens[i])
        chunk = row[row != EOS][:1023]
        rows.append(np.concatenate([chunk, [EOS], chunk, [EOS]]))
        owner.append((i, "copy"))
        rows.append(np.concatenate([chunk, [EOS]]))
        owner.append((i, "single"))
        copies.append(i)
    flat = np.concatenate(rows).astype(np.uint32)
    seqs = Sequences(flat, np.concatenate([[0], np.cumsum([len(r) for r in rows])]))
    counts = scored_token_counts(seqs)
    print(f"{len(windows)} windows ({sum(o[1] not in (None, -1) for o in owner)} fragments), {len(docs)} documents "
          f"(lengths {[len(sample.tokens[i]) for i in docs]})")

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    loaded = load_model(Recipe.single(args.model), gpus, attn_implementation=args.attn)
    if args.attn == "flash_attention_4":
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
        lazy_import_flash_attention(args.attn)
        from src.parity.pipeline import serialize_fa4_host_calls
        serialize_fa4_host_calls()
    scorer = PipelineScorer(loaded)
    try:
        t0 = time.time()
        losses = scorer.score(seqs, list(range(len(rows))), batch_size=args.batch_size, log_every=0)
        elapsed = time.time() - t0
        print(f"scored {len(rows)} samples ({int(seqs.lengths.sum())} tokens) in {elapsed:.1f}s: "
              f"{len(rows) / elapsed:.2f} samples/s, {seqs.lengths.sum() / elapsed / 1e3:.1f}k tok/s "
              f"(deterministic={args.deterministic})")
    finally:
        scorer.close()
        free_model(loaded)
    worst = 0.0
    for i in windows:
        whole = losses[owner.index((i, None))]
        parts = [(losses[j], counts[j]) for j, o in enumerate(owner) if o[0] == i and isinstance(o[1], int) and o[1] >= 0]
        combined = sum(loss * c for loss, c in parts) / sum(c for _, c in parts)
        worst = max(worst, abs(whole - combined))
        n_targets = counts[owner.index((i, None))]
        print(f"window {i}: masked {whole:.6f} | fragments alone {combined:.6f} | diff {whole - combined:+.2e} | "
              f"{len(parts)} fragments, {n_targets} targets (= {sum(c for _, c in parts)})")
        assert n_targets == sum(c for _, c in parts), "target counts differ"
    for i in docs:
        j = owner.index((i, -1))
        assert np.isfinite(losses[j]) and counts[j] == len(sample.tokens[i]) - 1
        print(f"document {i} ({sample.documents[i]['dataset']}/{sample.documents[i]['category']}, "
              f"{len(sample.tokens[i])} tokens): loss {losses[j]:.6f}")
    diffs = [losses[owner.index((i, None))] - sum(losses[j] * counts[j] for j, o in enumerate(owner) if o[0] == i and isinstance(o[1], int) and o[1] >= 0)
             / sum(counts[j] for j, o in enumerate(owner) if o[0] == i and isinstance(o[1], int) and o[1] >= 0) for i in windows]
    # bf16 + MoE rounding: a sample scored in another batch moves by ~1e-3 (sd); leakage would bias the window LOWER
    print(f"window - fragments: mean {np.mean(diffs):+.2e}, sd {np.std(diffs):.2e}, max |.| {worst:.2e}")
    for i in copies:
        both, single = losses[owner.index((i, "copy"))], losses[owner.index((i, "single"))]
        print(f"copy test {i}: [X EOS X EOS] {both:.4f} vs [X EOS] {single:.4f} (a leak would roughly halve it)")
        assert abs(both - single) < 0.05 * single, "the repeated chunk is predicted from across the EOS: masking not applied"
    assert abs(np.mean(diffs)) < 2e-3, "windows systematically differ from their fragments"
    print("PASS")


if __name__ == "__main__":
    main()
