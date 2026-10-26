#!/usr/bin/env python3
"""Multi-competition merge sweep server (8 x 48 GB GPUs, model layer-sharded across all of them).

Since validator f5bfc3f there are four competitions: MAIN (vs the MAIN king, bar 0.003) and the MATH /
CODE / TEXT specialists (bar 0.002) whose eval mix is 70% own split + 15% each other split; a specialist
without a king duels the MAIN king on its mix. One scoring pass on the equal-thirds screening sample
(``--competition mix``) gives per-split losses, so every candidate is scored once and re-weighted:

    mu_c = sum_g w_c[g] * mean_g(king_c - candidate)      SE_c^2 = sum_g w_c[g]^2 * var_g / n_g

with the king of each competition read from ``configs/merge/k41/kings.json`` (re-read per item, so a new
specialist king only needs a new entry). Optionally the first ``main_limit`` sequences of a MAIN sample
are scored too (MAIN is a different source mix).

The model skeleton is allocated once; each queued recipe is materialized with ``src.parity.weights._fill``
(the code path ``write_merge.py`` uses, so what is scored is what is shipped), re-reading only tensors
whose per-tensor combination changed since the previous candidate.

Queue item (JSON in --queue, processed by (priority, mtime)) = any recipe ``parity_eval.resolve_recipe``
understands + bookkeeping keys: label, description, method, limit, main_limit, priority.
Every scored recipe -> configs/merge/k41/<label>.json (recipe + result), configs/merge/k41/scores.jsonl,
and the tables in 41.md (scripts/merge/report41.py); committed locally (no push).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger("sweep41")
RECIPES = REPO / "configs/merge/k41"
SCORES = RECIPES / "scores.jsonl"
KINGS = RECIPES / "kings.json"
GIT_LOCK = threading.Lock()
Z = 3.090232306167813  # one-sided alpha 0.001
# specialist eval mixes (live manifests: <split>-70-15-15) and the 30k draw's per-split counts
MIX = {
    "math": {"math": 0.70, "code": 0.15, "text": 0.15},
    "code": {"math": 0.15, "code": 0.70, "text": 0.15},
    "text": {"math": 0.15, "code": 0.15, "text": 0.70},
}
BAR = {"main": 0.003, "math": 0.002, "code": 0.002, "text": 0.002}


def git_commit(paths: list[Path], message: str) -> None:
    """Commit recipe files locally in the background; never fatal (no push: no credentials here)."""

    def run() -> None:
        with GIT_LOCK:
            try:
                rel = [str(p.relative_to(REPO)) for p in paths if p.exists()]
                subprocess.run(["git", "add", *rel], cwd=REPO, check=True, capture_output=True)
                if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO).returncode == 0:
                    return
                subprocess.run(["git", "commit", "-q", "-m",
                                message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                               cwd=REPO, check=True, capture_output=True)
            except Exception as exc:  # noqa: BLE001
                log.warning("git commit failed: %s", exc)

    threading.Thread(target=run, daemon=True).start()


def tensor_signature(recipe, name: str) -> str:
    """What decides tensor `name` under this recipe (equal signature -> identical bf16 tensor)."""
    ops = [t for t in recipe.transforms if t.matches(name)]
    tail = [t.describe() for t in ops]
    if recipe.mode != "linear":
        return json.dumps(["mode", recipe.mode, recipe.mode_args, [[str(t.path), t.coef] for t in recipe.terms],
                           recipe.coefs_for(name), tail], sort_keys=True, default=str)
    coefs = recipe.coefs_for(name)
    active = [[str(recipe.terms[i].path), c] for i, c in enumerate(coefs) if c != 0.0]
    sig = {"lin": active, "ops": tail}
    if name in recipe.replacements:
        sig["rep"] = recipe.replacements[name]
    if name == "lm_head.weight" and recipe.head_rows_from_king:
        sig["rows"] = len(recipe.head_rows_from_king)
    return json.dumps(sig, sort_keys=True)


def group_stats(king: np.ndarray, chall: np.ndarray, groups: np.ndarray) -> dict:
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        d = king[m] - chall[m]
        out[g] = {"n": int(m.sum()), "mu": float(d.mean()), "var": float(d.var(ddof=1)) if m.sum() > 1 else 0.0}
    return out


def comp_stats(stats: dict, weights: dict, full_counts: dict | None = None) -> dict:
    mu = sum(w * stats[g]["mu"] for g, w in weights.items())
    se = math.sqrt(sum(w * w * stats[g]["var"] / max(stats[g]["n"], 1) for g, w in weights.items()))
    res = {"mu": round(mu, 6), "se": round(se, 7), "lcb": round(mu - Z * se, 6)}
    if full_counts:  # what the validator's 30k draw would give if mu is the truth
        se30 = math.sqrt(sum(w * w * stats[g]["var"] / full_counts[g] for g, w in weights.items()))
        res["lcb30k"] = round(mu - Z * se30, 6)
    return res


class Server:
    def __init__(self, args):
        import parity_eval as pe
        from src.parity.cache import LossCache, scorer_signature
        from src.parity.moe import grouped_moe
        from src.parity.pipeline import PipelineScorer
        from src.parity.sample import load_live_config
        from src.parity.verdict import duel
        from src.parity.weights import LoadedModel, Recipe, _allocate, build_skeleton, plan_layer_devices

        self.args, self.pe, self.duel, self.Recipe = args, pe, duel, Recipe
        self.mix = pe.resolve_sample(args.sample, load_live_config(competition="mix"))
        self.main_live = load_live_config(competition="main")
        self.main = pe.resolve_sample(args.main_sample, self.main_live) if args.main_sample else None
        self.cache = LossCache()
        self.signature = scorer_signature(args.attn, args.lm_head_chunk)
        base = Path(args.base)
        model = build_skeleton(base, args.attn)
        self.layer_devices = plan_layer_devices(model, list(range(torch.cuda.device_count())),
                                                reserve_first_gib=args.reserve_gib, reserve_last_gib=args.reserve_gib + 8)
        self.placement = _allocate(model, self.layer_devices)
        for module in model.modules():
            if module.__class__.__name__ == "MiMoV2MoE":
                module.moe = grouped_moe.__get__(module)
        model.eval()
        model.config.use_cache = False
        model.config._attn_implementation = args.attn
        model.hf_device_map = {
            **{f"model.layers.{i}": d for i, d in enumerate(self.layer_devices)},
            "model.embed_tokens": self.layer_devices[0], "model.norm": self.layer_devices[-1],
            "lm_head": self.layer_devices[-1],
        }
        self.model = model
        self.loaded_sig: dict[str, str] = {}
        self.scorer = PipelineScorer(LoadedModel(model, Recipe.single(base), self.layer_devices, args.attn, 0.0),
                                     lm_head_chunk=args.lm_head_chunk)
        log.info("mix sample %s n=%d; main sample %s; %d tensors on GPUs %s", self.mix.sample_id, self.mix.n,
                 self.main.sample_id if self.main else None, len(self.placement), sorted(set(self.layer_devices)))

    # ------------------------------------------------------------ build / score
    def materialize(self, recipe) -> tuple[float, int]:
        from src.parity.weights import _Reader, _check_keys, _fill

        started = time.time()
        names = sorted(self.placement)
        sigs = {n: tensor_signature(recipe, n) for n in names}
        todo = [n for n in names if self.loaded_sig.get(n) != sigs[n]]
        if not todo:
            return 0.0, 0
        reader = _Reader(recipe)
        _check_keys(self.model, reader)
        for n in todo:
            self.loaded_sig.pop(n, None)  # a failed fill must not leave a stale signature
        done = [0]
        lock = threading.Lock()

        def work(name: str) -> None:
            with torch.no_grad():
                _fill(self.placement[name], name, reader, recipe)
            self.loaded_sig[name] = sigs[name]
            with lock:
                done[0] += 1
                if done[0] % 8000 == 0:
                    log.info("  filled %d/%d tensors (%.0fs)", done[0], len(todo), time.time() - started)

        with ThreadPoolExecutor(max_workers=self.args.io_threads) as pool:
            list(pool.map(work, todo))
        for gpu in sorted(set(self.layer_devices)):
            torch.cuda.synchronize(gpu)
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        return time.time() - started, len(todo)

    def losses(self, sample, recipe, limit: int) -> tuple[np.ndarray, float, float, int]:
        model_id = recipe.identity()
        cached = self.cache.load(sample.sample_id, model_id, self.signature, sample.n)
        todo = [i for i in range(min(limit, sample.n)) if not np.isfinite(cached[i])]
        if not todo:
            return cached, 0.0, 0.0, 0
        build, refilled = self.materialize(recipe)
        started = time.time()
        got = self.scorer.score(sample.tokens, todo, batch_size=self.args.batch_size, log_every=0)
        cached[todo] = got[todo]
        self.cache.save(sample.sample_id, model_id, self.signature, cached,
                        {"batch": self.args.batch_size, "recipe": recipe.describe()})
        return cached, build, time.time() - started, refilled

    def kings(self) -> dict[str, str]:
        return json.loads(KINGS.read_text())

    # ------------------------------------------------------------ items
    def score_item(self, path: Path, raw: dict) -> None:
        raw.setdefault("label", path.stem)
        name = raw["label"]
        limit = int(raw.get("limit", self.args.limit))
        main_limit = int(raw.get("main_limit", self.args.main_limit)) if self.main else 0
        recipe = self.pe.resolve_recipe(str(path))
        model_id = recipe.identity()
        kings = self.kings()
        # references first (cached after the first item), then the candidate: the candidate stays loaded
        refs = {}
        for key in sorted(set(kings.values())):
            refs[key] = self.losses(self.mix, self.Recipe.single(key), limit)[0]
        chall, build, secs, refilled = self.losses(self.mix, recipe, limit)
        n = min(limit, self.mix.n)
        groups = self.mix.sources[:n]
        per_ref = {key: group_stats(ref[:n], chall[:n], groups) for key, ref in refs.items()}
        counts30 = {g: int(round(30000 * w)) for g, w in MIX["math"].items()}
        comps = {}
        for comp, weights in MIX.items():
            counts = {g: int(round(30000 * w)) for g, w in weights.items()}
            comps[comp] = comp_stats(per_ref[kings[comp]], weights, counts)
        del counts30
        main = None
        msecs = 0.0
        if main_limit:
            mking = self.losses(self.main, self.Recipe.single(kings["main"]), main_limit)[0]
            mchall, mb, msecs, mr = self.losses(self.main, recipe, main_limit)
            build, refilled = build + mb, refilled + mr
            m = min(main_limit, self.main.n)
            mask = np.zeros(self.main.n, dtype=bool)
            mask[:m] = True
            v = self.duel(np.where(mask, mking, np.nan), np.where(mask, mchall, np.nan), self.main,
                          delta=self.main_live.delta_threshold)
            main = {"n": v["n_sequences"], "mu": round(v["mu_hat"], 6), "lcb": round(v["lcb"], 6),
                    "se": v.get("se_iid"), "sources": {k: round(s["mu_hat"], 6) for k, s in v["source_scores"].items()}}
        groups_vs = {Path(key).name[:8]: {g: round(s["mu"], 6) for g, s in st.items()} for key, st in per_ref.items()}
        result = {
            "n": n, "comps": comps, "main": main, "groups_vs": groups_vs,
            "group_n": {g: s["n"] for g, s in next(iter(per_ref.values())).items()},
            "kings": {c: Path(k).name[:8] for c, k in kings.items()},
            "sample": self.mix.sample_id, "main_sample": self.main.sample_id if (self.main and main_limit) else None,
            "model_id": model_id, "batch": self.args.batch_size, "attn": self.args.attn,
            "build_s": round(build, 1), "score_s": round(secs + msecs, 1), "refilled_tensors": refilled,
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        record = {k: val for k, val in raw.items() if k not in ("priority",)}
        record["result"] = result
        out = RECIPES / f"{name}.json"
        out.write_text(json.dumps(record, indent=1) + "\n")
        with SCORES.open("a") as handle:
            handle.write(json.dumps({"name": name, "description": raw.get("description", ""),
                                     "method": raw.get("method", recipe.mode), **result}) + "\n")
        log.info("RESULT %-30s MATH %+.5f CODE %+.5f TEXT %+.5f%s | build %.0fs (%d tensors) score %.0fs", name,
                 comps["math"]["mu"], comps["code"]["mu"], comps["text"]["mu"],
                 f" MAIN {main['mu']:+.5f}" if main else "", build, refilled, secs + msecs)
        subprocess.run([sys.executable, str(REPO / "scripts/merge/report41.py")], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        git_commit([out, SCORES, REPO / "41.md"],
                   f"k41 sweep: {name} MATH {comps['math']['mu']:+.5f} CODE {comps['code']['mu']:+.5f} "
                   f"TEXT {comps['text']['mu']:+.5f}" + (f" MAIN {main['mu']:+.5f}" if main else ""))

    def handle(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        kind = raw.get("kind", "score")
        if kind == "score":
            self.score_item(path, raw)
        elif kind == "plugin":  # hot-reloaded extension: scripts/merge/<module>.py:run(server, path, raw)
            import importlib
            module = importlib.import_module(raw["module"])
            importlib.reload(module).run(self, path, raw)
        else:
            raise ValueError(f"unknown kind {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data/teutonic/models/1f1efb68b4a03d4b54304562597b34a71aa9b58d916757dfc83821df0b6d61c4",
                    help="checkpoint whose skeleton/config is used (king #41)")
    ap.add_argument("--queue", type=Path, default=RECIPES / "queue")
    ap.add_argument("--sample", default="tag:A", help="mix (equal-thirds) screening sample")
    ap.add_argument("--main-sample", default="tag:A")
    ap.add_argument("--limit", type=int, default=2400)
    ap.add_argument("--main-limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--attn", default="flash_attention_4")
    ap.add_argument("--lm-head-chunk", type=int, default=1024)
    ap.add_argument("--io-threads", type=int, default=32)
    ap.add_argument("--reserve-gib", type=float, default=4.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sys.path.insert(0, str(REPO / "scripts/merge"))
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue / "done").mkdir(exist_ok=True)
    server = Server(args)

    def next_item() -> Path | None:
        files = list(args.queue.glob("*.json"))
        if not files:
            return None

        def key(p: Path):
            try:
                return (float(json.loads(p.read_text()).get("priority", 100)), p.stat().st_mtime)
            except (json.JSONDecodeError, OSError):
                return (1e9, 0)
        return min(files, key=key)

    log.info("sweep41 ready; watching %s", args.queue)
    while True:
        if (args.queue / "STOP").exists():
            log.info("STOP file found; exiting")
            break
        path = next_item()
        if path is None:
            time.sleep(3)
            continue
        try:
            server.handle(path)
        except Exception as exc:  # keep serving the queue
            log.exception("FAIL %s: %s", path.name, exc)
            if path.exists():
                shutil.move(str(path), args.queue / "done" / (path.name + ".failed"))
            server.loaded_sig.clear()  # a failed fill may leave tensors half-written
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if path.exists():
            shutil.move(str(path), args.queue / "done" / path.name)


if __name__ == "__main__":
    main()
