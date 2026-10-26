#!/usr/bin/env python3
"""The reigning king of every competition, resolved strictly to a model on this box.

    python scripts/box/kings.py                  # table: competition, live digest, our name, path, ready?
    python scripts/box/kings.py --args main,math # campaign43 arguments: --king main=M45:0.003 --king math=MA2:0.002
    python scripts/box/kings.py --json

live digest (teutonic.ai dashboards) -> name (configs/merge/public/public_models.json, written by download_kings.py)
-> path (configs/merge/k41/models.json) that must hold an ingested model in our basis (INGESTED marker, or SAVED for our
own checkpoints). Anything that does not resolve raises KingNotReady: a screen against a guessed or stale king is
worse than no screen (king_args.py used to fall back to K43 / to the MAIN king and the caller's `|| true` hid it).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLIC = REPO / "configs/merge/public/public_models.json"
MODELS = REPO / "configs/merge/k41/models.json"
COMPS = ("main", "math", "code", "text")
BARS = {"main": 0.003, "math": 0.002, "code": 0.002, "text": 0.002}
URLS = {"main": "https://teutonic.ai/dashboard.json",
        **{c: f"https://teutonic.ai/competitions/{c}/dashboard.json" for c in ("math", "code", "text")}}
OURS = {"d4e670ff65ef": "T007"}  # our CODE crown t007 (a permutation of m1/step-150), models.json T007


class KingNotReady(RuntimeError):
    pass


def board(comp: str, tries: int = 4) -> dict:
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(URLS[comp], headers={"User-Agent": "teutonic-miner/1.0"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001 - network: retry, then raise
            last = exc
            time.sleep(5 * (attempt + 1))
    raise KingNotReady(f"{comp}: dashboard unreachable ({last})")


def live(comp: str) -> dict:
    """The live king of `comp` straight from its dashboard (digest, repo, reign, crowned_at, bar)."""
    d = board(comp)
    k = d.get("king") or {}
    digest = k.get("king_digest") or k.get("model_digest")
    if not digest:
        raise KingNotReady(f"{comp}: dashboard has no king")
    bar = next((float(c.get("delta_threshold")) for c in _competitions() if c.get("competition") == comp
                and c.get("delta_threshold") is not None), BARS[comp])
    return {"comp": comp, "digest": digest, "repo": k.get("model_repo"), "reign": k.get("reign_number"),
            "crowned_at": k.get("crowned_at"), "hotkey": k.get("hotkey"), "bar": bar,
            "queue": len(d.get("queue") or []), "evaluating": bool(d.get("current_eval"))}


_COMP_CACHE: list = []


def _competitions() -> list:
    if not _COMP_CACHE:
        try:
            req = urllib.request.Request("https://teutonic.ai/competitions.json", headers={"User-Agent": "teutonic-miner/1.0"})
            _COMP_CACHE.extend(json.loads(urllib.request.urlopen(req, timeout=30).read()).get("competitions") or [])
        except Exception:  # noqa: BLE001 - the published defaults stand in
            pass
    return _COMP_CACHE


def _ready(path: str) -> bool:
    p = Path(path)
    return (p / "model.safetensors.index.json").exists() and ((p / "INGESTED").exists() or (p / "SAVED").exists())


def resolve(digest: str) -> tuple[str, str]:
    """(name, path) of the model with this digest, ingested in our basis; KingNotReady otherwise."""
    models = json.loads(MODELS.read_text()) if MODELS.exists() else {}
    names = [OURS[d] for d in OURS if digest.startswith(d)]
    rows = json.loads(PUBLIC.read_text()) if PUBLIC.exists() else []
    names += [r["name"] for r in rows if r.get("digest") == digest]
    for name in names:
        path = models.get(name)
        if path and _ready(path):
            return name, path
    # registered under another name whose directory is the digest itself
    for name, path in models.items():
        if Path(path).name == digest and _ready(path):
            return name, path
    known = ", ".join(f"{n} -> {models.get(n, 'not in models.json')}" for n in names) or "not in public_models.json"
    raise KingNotReady(f"king {digest[:12]} is not ingested yet ({known}); download_kings.py fetches and un-permutes it")


def reigning(comps=COMPS) -> dict:
    out = {}
    for c in comps:
        k = live(c)
        k["name"], k["path"] = resolve(k["digest"])
        out[c] = k
    return out


def king_args(comps) -> str:
    return " ".join(f"--king {c}={k['name']}:{k['bar']}" for c, k in reigning(comps).items())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--args", default=None, help="comma list: print campaign43 --king arguments (exit 2 if any is not ready)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.args:
        try:
            print(king_args([c for c in a.args.split(",") if c]))
        except KingNotReady as exc:
            print(f"KING NOT READY: {exc}", file=sys.stderr)
            sys.exit(2)
        return
    rows = {}
    for c in COMPS:
        try:
            k = live(c)
            try:
                k["name"], k["path"] = resolve(k["digest"])
                k["ready"] = True
            except KingNotReady as exc:
                k["ready"], k["why"] = False, str(exc)
        except KingNotReady as exc:
            k = {"comp": c, "ready": False, "why": str(exc)}
        rows[c] = k
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    for c, k in rows.items():
        print(f"{c:5s} {str(k.get('digest', ''))[:12]:12s} #{k.get('reign')} {k.get('name', '-'):6s} bar {k.get('bar')} "
              f"{'READY ' + k['path'] if k.get('ready') else 'NOT READY: ' + k.get('why', '')}")


if __name__ == "__main__":
    main()
