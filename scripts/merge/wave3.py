#!/usr/bin/env python3
"""Combine the winning per-module / per-depth directions from wave 2 into wave-3 recipes.

Reads eval_cache/results, compares each u30w2_<group><mult> variant against the base
u30_dstack_0.17_0.13 on sample A, keeps directions that win by > 1.5 sigma, and queues:
the full combination, a half-step version, and the best single direction pushed further.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

D26, D25 = "/opt/teutonic/deltas/d26_22", "/opt/teutonic/deltas/d25_23"
A, B = 0.17, 0.13
GROUPS = {"experts": r"\.mlp\.experts\.", "attn": r"\.self_attn\.", "head": r"^lm_head\.|\.shared_experts\.",
          "upper": r"^model\.layers\.(2[7-9]|3[0-9]|4[0-4])\.", "lower": r"^model\.layers\.([0-9]|1[0-9]|2[0-6])\."}


def main() -> None:
    res = {}
    for path in glob.glob("eval_cache/results/*_duel.json"):
        for r in json.load(open(path))["results"]:
            v = r["verdict"]
            if r["sample"].endswith("16160cb531a3faaa") and v["n_sequences"] == 4000:
                res[r["challenger"]["label"]] = (v["mu_hat"], v["se_iid"])
    base = res.get("u30_dstack_0.17_0.13")
    if base is None:
        raise SystemExit("base result missing")
    wins = []
    for group in ("experts", "attn", "head"):
        for mult in (1.3, 0.7):
            v = res.get(f"u30w2_{group}{mult}")
            if v and v[0] - base[0] > 1.5 * max(v[1], base[1]):
                wins.append((group, mult, v[0] - base[0]))
    for name, mult_map in (("upper1.3", {"upper": 1.3, "lower": 0.7}), ("lower1.3", {"lower": 1.3, "upper": 0.7})):
        v = res.get(f"u30w2_{name}")
        if v and v[0] - base[0] > 1.5 * max(v[1], base[1]):
            for g, m in mult_map.items():
                wins.append((g, m, v[0] - base[0]))
    print("base", base, "winning directions:", wins)
    if not wins:
        Path("/opt/teutonic/logs/wave3_queued").write_text("no winning directions\n")
        return
    def recipe(name, desc, mults):
        overrides = [{"pattern": GROUPS[g], "coefs": [1.0, round(A * m, 4), round(B * m, 4)]} for g, m in mults.items()]
        json.dump({"label": name, "description": desc, "terms": [{"model": "reign:30", "coef": 1.0}, {"model": D26, "coef": A}, {"model": D25, "coef": B}],
                   "overrides": overrides}, open(f"configs/merge/{name}.json", "w"), indent=1)
        return f"--attn fa4sim --limit 4000 duel --king reign:30 --challenger configs/merge/{name}.json --sample tag:A"
    full = {g: m for g, m, _ in wins}
    half = {g: 1 + (m - 1) / 2 for g, m in full.items()}
    best_g, best_m, _ = max(wins, key=lambda w: w[2])
    pushed = {best_g: 1 + (best_m - 1) * 2}
    lines = [recipe("u30w3_combo", f"all winning directions {full}", full),
             recipe("u30w3_combo_half", f"half-step {half}", half),
             recipe("u30w3_push", f"best direction pushed further {pushed}", pushed)]
    with open("eval_cache/queue.txt", "a") as f:
        f.write("# --- wave 3b: combined directions ---\n" + "\n".join(lines) + "\n")
    Path("/opt/teutonic/logs/wave3_queued").write_text(json.dumps(wins) + "\n")
    print("queued wave 3b")


if __name__ == "__main__":
    main()
