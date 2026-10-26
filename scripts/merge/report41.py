#!/usr/bin/env python3
"""Regenerate the result tables and the full recipe book in 41.md from configs/merge/k41/.

Every scored recipe appears twice: in the results table (per-competition mu / LCB) and in the recipe
book (the exact combination, readable: models by short name from configs/merge/k41/models.json, vector
notation when the item was built by queue41.py), so 41.md alone is enough to rebuild any candidate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "configs/merge/k41"
SCORES = DIR / "scores.jsonl"
DOC = REPO / "41.md"
MODELS = DIR / "models.json"
TAGS = {
    "results": ("<!-- K41_RESULTS -->", "<!-- /K41_RESULTS -->"),
    "recipes": ("<!-- K41_RECIPES -->", "<!-- /K41_RECIPES -->"),
}
BAR = {"main": 0.003, "math": 0.002, "code": 0.002, "text": 0.002}


def short_names() -> dict[str, str]:
    if not MODELS.exists():
        return {}
    return {v: k for k, v in json.loads(MODELS.read_text()).items()}


def latest_rows() -> list[dict]:
    rows: dict[tuple, dict] = {}
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["name"], r.get("sample"), r.get("n"))] = r
    best: dict[str, dict] = {}
    for (name, _sample, n), r in rows.items():
        if name not in best or (n or 0) >= (best[name].get("n") or 0):
            best[name] = r
    return list(best.values())


def fmt(x, nd=5) -> str:
    return "–" if x is None else f"{x:+.{nd}f}"


def results_table(rows: list[dict]) -> str:
    out = [f"_Regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} from `configs/merge/k41/scores.jsonl`. "
           "MATH and MAIN are vs the current MAIN king (#41 until 22:26 UTC, then #42 = CK3 — see the kings column), CODE vs our CODE king from 20:20 UTC (vs #41 before), TEXT vs T1. "
           "`lcb30k` = mu − 3.09·SE with the screen's variances at the validator's 30k counts "
           "(21000 own split + 4500 + 4500): if mu is real, a duel passes when lcb30k > bar (0.002 / MAIN 0.003)._", ""]
    for comp in ("math", "code", "text", "main"):
        def key(r, comp=comp):
            if comp == "main":
                return (r.get("main") or {}).get("mu", -9)
            return r["comps"][comp]["mu"]
        top = sorted((r for r in rows if key(r) > -9), key=lambda r: -key(r))[:8]
        out.append(f"**Top {comp.upper()}** (bar {BAR[comp]}): " + ", ".join(
            f"`{r['name']}` {key(r):+.5f}" for r in top) if top else f"**Top {comp.upper()}**: –")
        out.append("")
    out += ["| # | candidate | method | what it does | n | **MATH** | lcb30k | **CODE** | lcb30k | **TEXT** | lcb30k | "
            "MAIN (n) | math vs#41 | code vs#41 | text vs#41 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ordered = sorted(rows, key=lambda r: -max(r["comps"][c]["mu"] for c in ("math", "code", "text")))
    for i, r in enumerate(ordered):
        c = r["comps"]
        m = r.get("main") or {}
        k41 = r.get("kings", {}).get("math")
        g = (r.get("groups_vs") or {}).get(k41, {})
        out.append(
            f"| {i} | `{r['name']}` | {r.get('method', '')} | {r.get('description', '')} | {r['n']} | "
            f"**{fmt(c['math']['mu'])}** | {fmt(c['math'].get('lcb30k'))} | **{fmt(c['code']['mu'])}** | "
            f"{fmt(c['code'].get('lcb30k'))} | **{fmt(c['text']['mu'])}** | {fmt(c['text'].get('lcb30k'))} | "
            f"{fmt(m.get('mu'))}{' (%d)' % m['n'] if m else ''} | {fmt(g.get('math'), 4)} | "
            f"{fmt(g.get('code'), 4)} | {fmt(g.get('text'), 4)} |")
    return "\n".join(out)


def recipe_text(rec: dict, names: dict[str, str]) -> str:
    nm = lambda p: names.get(p, p)  # noqa: E731
    lines = []
    if rec.get("vec_spec"):
        lines.append(f"- vectors: `{rec['vec_spec']}`")
    mode = rec.get("mode", "linear")
    terms = rec.get("terms", [])
    lines.append(f"- mode `{mode}`; terms: " + ", ".join(f"{t['coef']:+g}·{nm(t['model'])}" for t in terms))
    for o in rec.get("overrides", []):
        lines.append(f"  - override `{o['pattern']}`: " + ", ".join(
            f"{c:+g}·{nm(t['model'])}" for c, t in zip(o["coefs"], terms) if c))
    if rec.get("mode_args"):
        lines.append(f"  - mode_args `{json.dumps(rec['mode_args'])}`")
    if rec.get("tensor_coefs"):
        lines.append(f"  - per-tensor coefficients for {len(rec['tensor_coefs'])} tensors (exact values in the JSON)")
    for t in rec.get("transforms", []):
        lines.append(f"  - transform `{t['pattern']}` {t['op']} {json.dumps(t.get('args', {}))}")
    if rec.get("replacements"):
        lines.append(f"  - replacements {json.dumps(rec['replacements'])}")
    return "\n".join(lines)


def recipe_book(rows: list[dict]) -> str:
    names = short_names()
    out = ["Exact recipes (scored = shipped). Models: " + ", ".join(f"`{k}` = `{v}`" for v, k in sorted(
        names.items(), key=lambda kv: kv[1])) + ". Rebuild: `python scripts/merge/write_merge.py --recipe "
           "configs/merge/k41/<label>.json --out <dir>`.", ""]
    for r in sorted(rows, key=lambda r: r.get("time", "")):
        path = DIR / f"{r['name']}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        c = r["comps"]
        out.append(f"### `{r['name']}` — {r.get('method', '')}")
        out.append(f"{r.get('description', '')}  ")
        out.append(f"MATH {fmt(c['math']['mu'])} · CODE {fmt(c['code']['mu'])} · TEXT {fmt(c['text']['mu'])}"
                   + (f" · MAIN {fmt(r['main']['mu'])}" if r.get("main") else "") + f" (n={r['n']}, {r.get('time', '')})")
        out.append(recipe_text(rec, names))
        out.append("")
    return "\n".join(out)


def splice(text: str, which: str, body: str) -> str:
    begin, end = TAGS[which]
    if begin not in text:
        text += f"\n{begin}\n{end}\n"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    return f"{head}{begin}\n{body}\n{end}{tail}"


LEDGER = REPO / "42.md"
LEDGER_TAGS = ("<!-- K42_LEADERBOARDS -->", "<!-- /K42_LEADERBOARDS -->")
SAFE_MARGIN = 0.0006  # the validator's draw ran 0.0004 below our local estimate on 2026-10-07


def leaderboards(rows: list[dict]) -> str:
    """Per competition: every candidate scored against the competition's CURRENT king, best first."""
    kings = json.loads((DIR / "kings.json").read_text())
    cur = {c: Path(p).name[:8] for c, p in kings.items()}
    names = {Path(v).name[:8]: k for k, v in json.loads(MODELS.read_text()).items()} if MODELS.exists() else {}
    out = [f"_Regenerated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. Each table: candidates scored against that "
           "competition's current king (older rows measured against a previous king are left out). mu = mean per-sequence "
           f"CE gain on the competition's mix (2400 screen seqs; MAIN 800); verdict needs mu ≥ bar + {SAFE_MARGIN} to be "
           "worth a hotkey; 'close' = above the bar but inside the margin._", ""]
    for comp, bar in (("code", 0.002), ("math", 0.002), ("text", 0.002), ("main", 0.003)):
        king = cur.get(comp, "")
        label = names.get(king, king)
        table = []
        for r in rows:
            if comp == "main":
                m = r.get("main")
                if not m or r.get("kings", {}).get("main") != king:
                    continue
                mu, lcb, n = m["mu"], m.get("lcb"), m.get("n")
            else:
                if r.get("kings", {}).get(comp) != king:
                    continue
                c = r["comps"][comp]
                mu, lcb, n = c["mu"], c.get("lcb30k"), r["n"]
            table.append((mu, lcb, n, r))
        table.sort(key=lambda x: -x[0])
        out.append(f"### {comp.upper()} — king `{label}` ({king}…), bar LCB > {bar}, safe target mu ≥ {bar + SAFE_MARGIN:.4f} — {len(table)} candidates")
        out.append("")
        out.append("| rank | candidate | what it is | mu | lcb(30k) | gap to safe target | verdict |")
        out.append("|---:|---|---|---:|---:|---:|---|")
        for i, (mu, lcb, n, r) in enumerate(table[:25], 1):
            gap = mu - (bar + SAFE_MARGIN)
            verdict = "**BEATS (safe)**" if gap >= 0 else ("close" if mu >= bar else ("worse than king" if mu < 0 else "no"))
            desc = (r.get("description") or "")[:90]
            out.append(f"| {i} | `{r['name']}` | {desc} | {mu:+.5f} | {fmt(lcb)} | {gap:+.5f} | {verdict} |")
        if len(table) > 25:
            out.append(f"| … | {len(table) - 25} more in 41.md | | | | | |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    rows = latest_rows()
    text = DOC.read_text() if DOC.exists() else "# 41.md\n"
    text = splice(text, "results", results_table(rows))
    text = splice(text, "recipes", recipe_book(rows))
    DOC.write_text(text)
    if LEDGER.exists():
        ledger = LEDGER.read_text()
        begin, end = LEDGER_TAGS
        if begin not in ledger:
            ledger = ledger.replace("## King lineage", f"## Leaderboards — what is good and what is worse (auto-updated after every scored recipe)\n{begin}\n{end}\n\n## King lineage", 1)
        head, rest = ledger.split(begin, 1)
        _, tail = rest.split(end, 1)
        LEDGER.write_text(f"{head}{begin}\n{leaderboards(rows)}\n{end}{tail}")


if __name__ == "__main__":
    main()
