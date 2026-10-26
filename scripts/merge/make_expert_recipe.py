#!/usr/bin/env python3
"""Routing-aware (MoE-specific) transplant recipe.

Lineage B's update (donor − base) helps reasoning/code/math but hurts web text (txt360).
Both lineages share the same frozen router, so expert e of layer l means the same thing
in both. Using the king's measured routing (parity_eval.py routing), each expert gets a
web affinity = (its share of txt360 tokens) / (global txt360 share); the donor update is
applied to web-light experts only, with separate strengths for the rest of the model.

  make_expert_recipe.py --routing eval_cache/routing/<stem>.npy --out configs/merge/expertB_a0.8.json \
      --max-affinity 0.8 --beta-experts 0.35 --beta-rest 0.25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--routing", required=True, type=Path, help=".npy from parity_eval.py routing")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--king", default="reign:29")
    parser.add_argument("--donor", default="reign:26")
    parser.add_argument("--base", default="reign:22")
    parser.add_argument("--web-source", default="txt360-v2")
    parser.add_argument("--max-affinity", type=float, default=0.8, help="experts at or below get beta-experts")
    parser.add_argument("--beta-experts", type=float, default=0.35)
    parser.add_argument("--beta-web-experts", type=float, default=0.0, help="for experts above max-affinity")
    parser.add_argument("--beta-rest", type=float, default=0.25, help="attention, shared experts, norms, head")
    args = parser.parse_args()

    table = np.load(args.routing)  # [layers, sources, experts]
    meta = json.loads(args.routing.with_suffix(".json").read_text())
    web = meta["sources"].index(args.web_source)
    per_expert = table.sum(axis=1)
    global_share = table[:, web].sum() / table.sum()
    affinity = table[:, web] / np.maximum(per_expert, 1) / global_share
    tensor_coefs = {}
    selected = 0
    for position, layer in enumerate(meta["layers"]):
        for expert in range(table.shape[2]):
            light = affinity[position, expert] <= args.max_affinity
            beta = args.beta_experts if light else args.beta_web_experts
            selected += int(light)
            for proj in ("gate_proj", "up_proj", "down_proj"):
                tensor_coefs[f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight"] = [1.0, beta, -beta]
    total = len(meta["layers"]) * table.shape[2]
    recipe = {
        "label": args.out.stem,
        "description": (
            f"king + beta·(donor−base): experts with {args.web_source} affinity ≤ {args.max_affinity} get "
            f"{args.beta_experts} ({selected}/{total} experts), others {args.beta_web_experts}, non-experts {args.beta_rest}"
        ),
        "terms": [
            {"model": args.king, "coef": 1.0},
            {"model": args.donor, "coef": args.beta_rest},
            {"model": args.base, "coef": -args.beta_rest},
        ],
        "tensor_coefs": tensor_coefs,
        "routing": str(args.routing),
    }
    args.out.write_text(json.dumps(recipe))
    print(f"{args.out}: {selected}/{total} experts web-light (global {args.web_source} share {global_share:.3f}); "
          f"affinity quantiles {np.round(np.quantile(affinity, [0.1, 0.25, 0.5, 0.75, 0.9]), 2).tolist()}")


if __name__ == "__main__":
    main()
