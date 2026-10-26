#!/usr/bin/env python3
"""Regenerate the results table in 40.md from configs/merge/k40/scores.jsonl (latest row per name)."""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCORES = REPO / "configs/merge/k40/scores.jsonl"
DOC = REPO / "40.md"
BEGIN, END = "<!-- K40_RESULTS -->", "<!-- /K40_RESULTS -->"
SRC = ["code-reasoning", "math-reasoning", "pretrain-behaviors", "sft-reasoning", "txt360-v2"]


def table() -> str:
    rows: dict[str, dict] = {}
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["name"], r.get("sample"), r.get("n"))
                rows[key] = r
    best: dict[tuple, dict] = {}
    for (name, sample, n), r in rows.items():
        k = (name, sample)
        if k not in best or (r.get("n") or 0) >= (best[k].get("n") or 0):
            best[k] = r
    ordered = sorted(best.values(), key=lambda r: -(r.get("mu_hat") or -1))
    out = [f"_Regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from `configs/merge/k40/scores.jsonl`; "
           "positive mu_hat beats #40; crown needs ≈ +0.0036, upload threshold +0.001._", "",
           "| # | candidate | method | what it does | sample | seqs | **mu_hat** | LCB | code | math | pretrain | sft | txt360 |",
           "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, r in enumerate(ordered):
        s = r.get("sources", {})
        out.append(f"| {i} | `{r['name']}` | {r.get('method', '')} | {r.get('description', '')} | "
                   f"{(r.get('sample') or '')[-6:]} | {r.get('n')} | **{r['mu_hat']:+.5f}** | {r['lcb']:+.5f} | "
                   + " | ".join(f"{s[k]:+.4f}" if k in s else "–" for k in SRC) + " |")
    return "\n".join(out)


def main() -> None:
    text = DOC.read_text() if DOC.exists() else f"# 40.md — merge campaign against king #40\n\n{BEGIN}\n{END}\n"
    if BEGIN not in text:
        text += f"\n{BEGIN}\n{END}\n"
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(f"{head}{BEGIN}\n{table()}\n{END}{tail}")


if __name__ == "__main__":
    main()
