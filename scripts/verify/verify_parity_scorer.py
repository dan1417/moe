#!/usr/bin/env python3
"""Check the parity scorer against the validator's own code paths on real weights.

1. pipelined (threaded, hand-over-hand stages) vs sequential scoring: must match
2. src.parity.moe.grouped_moe vs the validator's grouped_mimo_moe: must match up to
   index_add_ atomic-order noise (the validator itself is nondeterministic there)
3. eager vs fa4sim attention: reports the per-sequence loss shift between backends

Usage: verify_parity_scorer.py --model /opt/teutonic/models/<digest> --sample id:<sample_id> [-n 16]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.parity import upstream  # noqa: E402
from src.parity.moe import grouped_moe  # noqa: E402
from src.parity.pipeline import PipelineScorer  # noqa: E402
from src.parity.sample import load_sample, sample_dir  # noqa: E402
from src.parity.weights import Recipe, load_model  # noqa: E402


def sequential(model, tokens, indices, chunk=1024):
    out = []
    for index in indices:
        seq = np.asarray(tokens[index], dtype=np.int64).tolist()
        out.append(upstream.engine.compute_per_sequence_loss(model, [seq], chunk, reset_peak_memory=False)[0])
    return np.asarray(out)


def set_moe(model, fn):
    for module in model.modules():
        if module.__class__.__name__ == "MiMoV2MoE":
            module.moe = types.MethodType(fn, module)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample", required=True, help="id:<sample_id>")
    parser.add_argument("-n", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    sample = load_sample(sample_dir(upstream.DEFAULT_CACHE_ROOT, args.sample.removeprefix("id:")))
    indices = list(range(args.n))
    loaded = load_model(Recipe.single(args.model), list(range(torch.cuda.device_count())), attn_implementation="eager")
    model = loaded.model
    for gpu in sorted(set(loaded.layer_devices)):
        print(f"cuda:{gpu} allocated {torch.cuda.memory_allocated(gpu) / 2**30:.2f} GiB")

    scorer = PipelineScorer(loaded)
    t = time.time()
    piped = scorer.score(sample.tokens, indices, log_every=0)[indices]
    print(f"pipelined eager: {time.time() - t:.1f}s for {len(indices)} seqs; mean {piped.mean():.6f}")
    scorer.close()

    # hooks for device moves are needed for sequential runs too
    scorer = PipelineScorer(loaded)
    scorer.locks = [type("NoLock", (), {"acquire": lambda self: None, "release": lambda self: None})() for _ in scorer.locks]
    t = time.time()
    seq_grouped = sequential(model, sample.tokens, indices)
    print(f"sequential eager: {time.time() - t:.1f}s; max|pipelined - sequential| = {np.abs(piped - seq_grouped).max():.3e}")

    set_moe(model, upstream.engine.grouped_mimo_moe)
    seq_validator = sequential(model, sample.tokens, indices)
    print(f"validator grouped_mimo_moe: max|ours - validator| = {np.abs(seq_grouped - seq_validator).max():.3e}")
    set_moe(model, grouped_moe)
    again = sequential(model, sample.tokens, indices)
    print(f"ours run-to-run: max diff = {np.abs(seq_grouped - again).max():.3e}")

    model.config._attn_implementation = "fa4sim"
    from src.parity.attention import register_fa4sim

    register_fa4sim()
    fa = sequential(model, sample.tokens, indices)
    diff = fa - seq_grouped
    print(f"fa4sim - eager per-seq loss: mean {diff.mean():+.3e} sd {diff.std():.3e} max|.| {np.abs(diff).max():.3e}")
    print("eager :", np.round(seq_grouped[:8], 6))
    print("fa4sim:", np.round(fa[:8], 6))
    print("sources:", sample.sources[indices][:8])


if __name__ == "__main__":
    main()
