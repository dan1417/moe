#!/usr/bin/env python3
"""sweep41 plugin: per-split routed-expert usage on the screening sample (MoE-aware merge planning).

Queue item:  {"kind": "plugin", "module": "routing41", "label": "route_K41", "model": "<recipe or ckpt>",
              "limit": 1500}
Materializes `model` (default: king #41) in the server's resident skeleton, scores the first `limit`
sequences of the mix sample with forward hooks on every MoE gate, and saves
  /data/teutonic/eval_cache/routing/<label>.npy  int64 [layers, splits, experts] token counts
  /data/teutonic/eval_cache/routing/<label>.json {layers, splits, n, model}
Per-sequence attribution: each worker thread scores its own batch end to end, so a thread-local list of
the batch's split ids maps the gate's [batch, seq, top_k] output rows to splits.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import torch

OUT = Path("/data/teutonic/eval_cache/routing")


def run(server, path: Path, raw: dict) -> None:
    from src.parity import upstream

    label = raw.get("label", path.stem)
    spec = raw.get("model") or json.loads((server.args.queue.parent / "kings.json").read_text())["main"]
    recipe = server.pe.resolve_recipe(spec) if str(spec).endswith(".json") else server.Recipe.single(spec)
    server.materialize(recipe)
    sample = server.mix
    n = min(int(raw.get("limit", 1500)), sample.n)
    if raw.get("by", "split") == "category":  # group by the shard's category (txt360-qa, planning, ...) instead of split
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from catdiag41 import shard_categories
        cats_of = shard_categories()
        names = sample.meta["shard_names"]
        order = [s["name"] for s in sample.meta["sources"]]
        group_name = [cats_of.get(names[order[g]][s], "?") for g, s in sample.provenance[:, :2].tolist()]
    else:
        group_name = [str(x) for x in sample.sources.tolist()]
    splits = sorted(set(group_name[:n]))
    split_of = {tuple(np.asarray(sample.tokens[i][:64]).tolist()): splits.index(group_name[i]) for i in range(n)}
    local = threading.local()
    counts, hooks = {}, []
    for li, layer in enumerate(server.model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None or not hasattr(gate, "e_score_correction_bias"):
            continue
        dev = gate.e_score_correction_bias.device
        n_exp = gate.e_score_correction_bias.shape[0]
        counts[li] = torch.zeros(len(splits), n_exp, dtype=torch.int64, device=dev)

        def hook(module, inputs, output, li=li, n_exp=n_exp):
            groups = getattr(local, "groups", None)
            if groups is None:
                return
            idx = output[0].reshape(len(groups), -1)
            for row, g in enumerate(groups):
                counts[li][g] += torch.bincount(idx[row], minlength=n_exp)

        hooks.append(gate.register_forward_hook(hook))
    engine = upstream.engine

    def fn(model, batches, chunk):
        local.groups = [split_of[tuple(b[:64])] for b in batches]
        try:
            return engine.compute_per_sequence_loss(model, batches, chunk, reset_peak_memory=False)
        finally:
            local.groups = None

    try:
        server.scorer.score(sample.tokens, list(range(n)), batch_size=server.args.batch_size, fn=fn, log_every=0)
    finally:
        for h in hooks:
            h.remove()
    layers = sorted(counts)
    table = np.stack([counts[i].cpu().numpy() for i in layers])
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{label}.npy", table)
    (OUT / f"{label}.json").write_text(json.dumps({"layers": layers, "splits": splits, "n": n,
                                                   "model": recipe.describe(), "sample": sample.sample_id}, indent=1))
    share = table.sum(axis=(0, 2)) / table.sum()
    print(f"routing {label}: token share per split {dict(zip(splits, np.round(share, 4).tolist()))}")
