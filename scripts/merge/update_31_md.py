#!/usr/bin/env python3
"""Regenerate MoE/31.md results table from /opt/teutonic/logs/scores.jsonl + recipe JSONs."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

REPO = Path("/root/workspace/MoE")
SCORES = Path("/opt/teutonic/logs/scores.jsonl")
RECIPES = REPO / "configs/merge/k31"
OUT = REPO / "31.md"

HEADER = """# 31.md — merge campaign against king #31

## Current king: #31 (`teutonic-II-110B-A7B-5ac02e46`)
- Crowned 2026-09-25 23:27 UTC by **mu_hat +0.003403** / LCB +0.003298 over #30 (same coldkey 5Hmim6 as #26/#29/#30).
- Duelsource gains vs #30: code +0.0031, math +0.0017, pretrain +0.0036, sft +0.0031, **txt360 +0.0056**.
- **#31 is a new hidden-dimension permutation** (relL2 vs #30/#29 ≈ 1.41; embedding columns match 1.0000 but order ≠ #30's shuffle).
  We score on **#31u** = #31 mapped into the k30u / #29 basis (`scripts/merge/perm_from_embed.py` → `unpermute_checkpoint.py`).
- Bar to crown: LCB > **0.003** on a fresh 30k draw → want **mu_hat ≥ ~0.0036** here before burning a hotkey.

## Lessons carried from MERGE.md (#30 campaign)
- Only steps **outside** the king's ancestry help; extrapolating the king's own update hurts.
- Best public-king merge vs #30 was `#30u + 0.17·D26 + 0.13·D25` with experts×1.15 / attn×0.7 / head×0.7 → **+0.0018** (half the bar).
- TIES / DARE / SLERP / sign-agree / orthogonal rarely beat plain task arithmetic.
- Module scaling (experts↑, attn/head↓) was the only useful structure.
- txt360 is the usual conflict; gains on code/math/pretrain often pay for web damage.

## Ingredients (this campaign)
| name | what | path |
|---|---|---|
| **#31u** | current king, unpermuted | `/opt/teutonic/hf/king31u` |
| **k30u** | best merge body vs #30 (`touristAI/teutonic-II-110B-merge-k30u`) | `/opt/teutonic/hf/k30u` |
| **CPT** | line_c1 EMA@150 (`touristAI/teutonic-cpt-backup`); trainer +0.0024 vs #30 | `/opt/teutonic/hf/cpt-ema150` |
| **D_CPT** | CPT − k30u (the training step we transplant onto #31u) | (implicit in recipes) |
| **#30u** | king #30, unpermuted into the same basis (`k30_hidden_perm.json`) | `/opt/teutonic/hf/king30u` |
| **Dm** | k30u − #30u = the public-king merge step `0.17·D26 + 0.13·D25` (experts×1.15, attn/head×0.7) | sweep vector |
| **S31** | #31u − #30u = king #31's own training step | sweep vector |

Geometry (`scripts/merge/delta_geometry.py`, |Δ|/|W| and cosines):
- **S31** touches only layers 16–47 + lm_head (layers 0–15, embeddings, routers bit-identical to #30); |S31|/|W| = 0.016.
- **Dm** is nearly orthogonal to S31 (cos **0.17**; attn 0.24, experts 0.18, lm_head −0.05) → #31 did **not** absorb the king merge.
- **D_CPT** is tiny (|D_CPT|/|W| = 0.00034, 50× smaller than S31) → the CPT transplants score ≈ 0.

Recipes live in `configs/merge/k31/<name>.json`. Scorer: `parity_eval.py` FA4, sample `tag:A`.
Screens: first **800** sequences (batch 8, 1×8-GPU pipeline). Positive **mu_hat** = beats #31.
`sweep31` rows are rebuilt in GPU from RAM-held vectors (`scripts/merge/sweep31.py`); their JSON holds the exact
shippable recipe under `ship`.

## Conclusion (campaign stopped 2026-09-26 13:20 UTC)
- **Best: `L1_coef_g060` = +0.00185** (LCB +0.0014, 800 seqs of tag:A): `#31u + Dm` with per-tensor / per-expert
  coefficients for Dm and S31 learned from one 512-sequence gradient probe on a separate sample (tag:G).
  Plain public-king merge `K_dm100` = +0.00166; CPT-step transplants ≤ +0.0002; extrapolating #31's step ≤ 0.
- The bar is ≈ +0.0036, so merging gets about half of it. The only public step outside #31's ancestry is the
  D26/D25 pair (Dm); its gains on code/math/pretrain/sft are paid back on txt360 as its scale grows, and learned
  coefficients added +0.0002 per step before the noisy gradient's trust region ran out. Beating #31 needs
  new training signal (CPT from this merge), not more merge arithmetic.
- The best candidate was written with `write_merge.py` from its exact recipe (`ship` block of
  `configs/merge/k31/L1_coef_g060.json`); storage/download split over four private HF repos: see
  [CURRENT_BEST.md](CURRENT_BEST.md).

## Campaign log (king-merge phase, 2026-09-26)
- **Dm scale curve** (`#31u + c·Dm`, 800 seqs): c=0.5 **+0.00125**, 0.75 **+0.00162**, 1.0 **+0.00166**, 1.25 **+0.00145**,
  1.5 **+0.00114**. code/math/pretrain/sft keep rising to c=1.5 (+0.0028/+0.0018/+0.0033/+0.0034) but txt360 falls
  linearly (+0.0001 → −0.0045): txt360 damage is what caps the public-king merge at ≈ +0.0017 (half the bar).
- `#31u + Dm − 0.1·S31` +0.00162 (no gain from giving back #31's step).
- **Learned merge coefficients** (AdaMerging-style, first order): the sweep server back-propagates the eval objective
  (mean per-sequence CE) on a *separate* sample (tag:G) at a merge point and records g_t = <dL/dW_t, vec_t> for every
  tensor and every routed expert (`GRAD_*` items, `scripts/merge/grad_report31.py`); `scripts/merge/grad_step31.py`
  turns them into per-tensor coefficients (`L*_…` rows, method `learned-coef`), screened on tag:A.
  Probe 1 (512 seqs at c=1): dL/dc(Dm) = +1.9e-4 (c=1 slightly past the optimum), dL/dc(S31) = −6.9e-4;
  per-tensor split-half corr 0.27 (Dm) / 0.07 (S31) → noisy; a 2,048-seq probe follows.

## Results vs #31u (live; regenerated from scores.jsonl)
<!-- K31_RESULTS -->
{table}
<!-- /K31_RESULTS -->

## How to read
- **mu_hat** = mean(king CE − candidate CE). Positive = better than the king.
- Crown needs ≈ **+0.0036**. Screens use 800 seqs (~±2e-4 noise); promote winners to full 30k × 2 samples before submit.
- Recipe JSON path: `configs/merge/k31/<candidate>.json`.
- **Batch matters:** MoE kernel rounding depends on batch size. The first campaign31 rows compared batch-8 challengers
  with a batch-2 king cache; after sweep31 re-scored the king at batch 8, `scripts/merge/rescore31.py` re-derived those
  rows from cached losses (original numbers kept as `summary_orig` in scores.jsonl). All rows now use batch 8 on both sides.

## Running
```bash
source /opt/teutonic/env.sh
# campaign auto-appends scores.jsonl and rewrites this file after each duel
python scripts/merge/campaign31.py
python scripts/merge/update_31_md.py   # refresh table only
# fast sweep server (RAM-held task vectors; drop recipe JSONs into configs/merge/k31/queue/)
python scripts/merge/sweep31.py --vec Dm=/opt/teutonic/hf/k30u:/opt/teutonic/hf/king30u \
    --vec S31=/opt/teutonic/hf/king31u:/opt/teutonic/hf/king30u
```

Live logs: `/opt/teutonic/logs/campaign31.log`, `/opt/teutonic/logs/sweep31.log` · JSONL: `/opt/teutonic/logs/scores.jsonl`
"""


def parse_summary(summary: str) -> dict:
    out = {}
    if not summary or summary == "FAIL":
        return out
    for key in ("n_sequences", "mu_hat", "lcb", "avg_king_loss", "avg_challenger_loss"):
        m = re.search(rf"{key}=([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", summary)
        if m:
            out[key] = float(m.group(1)) if key != "n_sequences" else int(float(m.group(1)))
    return out


def recipe_blurb(name: str) -> tuple[str, str]:
    path = RECIPES / f"{name}.json"
    if not path.exists():
        return "?", name
    raw = json.loads(path.read_text())
    method = raw.get("method") or raw.get("mode", "linear")
    what = raw.get("description") or raw.get("label") or name
    return method, what


def fmt(x, digits=5):
    if x is None:
        return "—"
    return f"{x:+.{digits}f}"


def build_table() -> str:
    if not SCORES.exists():
        return "_No scores yet._\n"
    rows = []
    for line in SCORES.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        name = o["name"]
        summary = parse_summary(o.get("summary") or "")
        src = o.get("sources") or {}
        method, what = recipe_blurb(name)
        mu = summary.get("mu_hat")
        rows.append({
            "name": name,
            "method": method,
            "what": what,
            "n": summary.get("n_sequences") or "?",
            "mu": mu,
            "lcb": summary.get("lcb"),
            "code": src.get("code-reasoning"),
            "math": src.get("math-reasoning"),
            "pretrain": src.get("pretrain-behaviors"),
            "sft": src.get("sft-reasoning"),
            "txt360": src.get("txt360-v2"),
            "secs": o.get("seconds"),
            "fail": (o.get("summary") == "FAIL"),
        })
    rows.sort(key=lambda r: (r["fail"], -(r["mu"] if r["mu"] is not None else -99)))
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines = [
        f"_Regenerated {stamp} from `{SCORES}`; positive mu_hat beats #31; need ≈ +0.0036._",
        "",
        "| # | candidate | method | what it does | seqs | **mu_hat** | LCB | code | math | pretrain | sft | txt360 | secs |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(rows):
        if r["fail"]:
            lines.append(
                f"| {i} | `{r['name']}` | {r['method']} | {r['what']} | — | FAIL | — | — | — | — | — | — | {r['secs'] or '—'} |"
            )
            continue
        mu_s = f"**{fmt(r['mu'])}**" if i == 0 and r["mu"] is not None else fmt(r["mu"])
        lines.append(
            f"| {i} | `{r['name']}` | {r['method']} | {r['what']} | {r['n']} | {mu_s} | "
            f"{fmt(r['lcb'])} | {fmt(r['code'], 4)} | {fmt(r['math'], 4)} | {fmt(r['pretrain'], 4)} | "
            f"{fmt(r['sft'], 4)} | {fmt(r['txt360'], 4)} | {r['secs'] or '—'} |"
        )
    if rows and rows[0]["mu"] is not None:
        best = rows[0]
        gap = 0.0036 - best["mu"]
        lines += [
            "",
            f"**Best so far:** `{best['name']}` = {fmt(best['mu'])} "
            f"(gap to bar ≈ {gap:+.4f}; recipe `configs/merge/k31/{best['name']}.json`).",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(HEADER.format(table=build_table()))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
