#!/usr/bin/env python3
"""Single-GPU merge sweep server against king #40 (one B300 holds the whole 220 GB model).

The model skeleton is allocated once on the GPU. Each queued recipe is materialized with
``src.parity.weights._fill`` -- the code path ``write_merge.py`` uses to write a submission, so
what is scored is what is shipped -- but only tensors whose per-tensor combination changed since
the previous candidate are re-read (a sweep over attention / head coefficients re-reads ~9 GB
instead of 220 GB per source). Then the first ``limit`` sequences of the sample are scored and the
paired verdict vs the king (cached losses, same batch size) is recorded.

Queue item (drop JSON into --queue; processed by (priority, mtime)): any recipe that
``parity_eval.resolve_recipe`` understands, plus bookkeeping keys:
  {"label": "s36_050", "description": "#40 + 0.5*(#36 - #35)", "method": "task-arith",
   "terms": [{"model": "/opt/teutonic/models/<k40>", "coef": 1.0},
             {"model": "/opt/teutonic/hf/r36_b40", "coef": 0.5},
             {"model": "/opt/teutonic/hf/r35_b40", "coef": -0.5}],
   "overrides": [{"pattern": "\\.mlp\\.experts\\.", "coefs": [1.0, 0.6, -0.6]}],
   "limit": 800, "priority": 100}

Every scored recipe is written to configs/merge/k40/<label>.json (with its result) and appended to
configs/merge/k40/scores.jsonl, then committed and pushed (so a lost GPU box loses nothing).
Candidates with mu_hat >= --promote-at are handed to scripts/merge/promote40.py (write, verify,
upload to a gated HF repo) through /opt/teutonic/promote/.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
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

log = logging.getLogger("sweep40")
RECIPES = REPO / "configs/merge/k40"
SCORES = RECIPES / "scores.jsonl"
PROMOTE = Path("/opt/teutonic/promote")
GIT_LOCK = threading.Lock()


def git_push(paths: list[Path], message: str) -> None:
    """Commit and push recipe files in the background; never fatal."""

    def run() -> None:
        with GIT_LOCK:
            try:
                rel = [str(p.relative_to(REPO)) for p in paths]
                subprocess.run(["git", "add", *rel], cwd=REPO, check=True, capture_output=True)
                staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=REPO)
                if staged.returncode == 0:
                    return
                subprocess.run(["git", "commit", "-q", "-m", message + "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"],
                               cwd=REPO, check=True, capture_output=True)
                for _ in range(3):
                    r = subprocess.run(["git", "push", "-q", "origin", "HEAD"], cwd=REPO, capture_output=True, text=True)
                    if r.returncode == 0:
                        return
                    subprocess.run(["git", "pull", "-q", "--rebase", "origin", "main"], cwd=REPO, capture_output=True)
                log.warning("git push failed: %s", r.stderr[-300:])
            except Exception as exc:  # noqa: BLE001
                log.warning("git commit/push failed: %s", exc)

    threading.Thread(target=run, daemon=True).start()


def tensor_signature(recipe, name: str) -> str:
    """What decides tensor `name` under this recipe (equal signature -> identical bf16 tensor)."""
    if recipe.mode != "linear":
        return json.dumps(["mode", recipe.mode, recipe.mode_args, [[str(t.path), t.coef] for t in recipe.terms],
                           recipe.coefs_for(name)], sort_keys=True, default=str)
    coefs = recipe.coefs_for(name)
    active = [[str(recipe.terms[i].path), c] for i, c in enumerate(coefs) if c != 0.0]
    sig = {"lin": active}
    if name in recipe.replacements:
        sig["rep"] = recipe.replacements[name]
    if name == "lm_head.weight" and recipe.head_rows_from_king:
        sig["rows"] = len(recipe.head_rows_from_king)
    return json.dumps(sig, sort_keys=True)


class Server:
    def __init__(self, args):
        import parity_eval as pe
        from src.parity.cache import LossCache, scorer_signature
        from src.parity.moe import grouped_moe
        from src.parity.pipeline import PipelineScorer
        from src.parity.sample import load_live_config
        from src.parity.verdict import duel
        from src.parity.weights import LoadedModel, Recipe, _allocate, build_skeleton, model_identity, plan_layer_devices

        self.args, self.pe, self.duel, self.Recipe = args, pe, duel, Recipe
        self.live = load_live_config()
        self.sample = pe.resolve_sample(args.sample, self.live)
        self.cache = LossCache()
        self.signature = scorer_signature(args.attn, args.lm_head_chunk)
        self.king = Recipe.single(args.king)
        self.king_id = model_identity(args.king)
        model = build_skeleton(args.king, args.attn)
        self.layer_devices = plan_layer_devices(model, list(range(torch.cuda.device_count())))
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
        if args.attn == "flash_attention_4":
            from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
            lazy_import_flash_attention(args.attn)
        self.scorer = PipelineScorer(LoadedModel(model, self.king, self.layer_devices, args.attn, 0.0),
                                     lm_head_chunk=args.lm_head_chunk)
        log.info("sample %s n=%d; king %s; %d tensors on GPUs %s", self.sample.sample_id, self.sample.n,
                 self.king_id[:16], len(self.placement), sorted(set(self.layer_devices)))

    # ------------------------------------------------------------ build
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

    def losses(self, recipe, model_id: str, limit: int) -> tuple[np.ndarray, float, float, int]:
        sample = self.sample
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

    # ------------------------------------------------------------ items
    def score_item(self, path: Path, raw: dict) -> None:
        raw.setdefault("label", path.stem)
        name = raw["label"]
        limit = int(raw.get("limit", self.args.limit))
        recipe = self.pe.resolve_recipe(str(path))
        model_id = recipe.identity()
        king, _, ks, _ = self.losses(self.king, self.king_id, limit)
        if ks:
            log.info("king scored up to %d (%.0fs)", limit, ks)
        chall, build, secs, refilled = self.losses(recipe, model_id, limit)
        n = min(limit, self.sample.n)
        mask = np.zeros(self.sample.n, dtype=bool)
        mask[:n] = True
        v = self.duel(np.where(mask, king, np.nan), np.where(mask, chall, np.nan), self.sample,
                      delta=self.live.delta_threshold)
        sources = {k: round(s["mu_hat"], 6) for k, s in v["source_scores"].items()}
        result = {"n": v["n_sequences"], "mu_hat": v["mu_hat"], "lcb": v["lcb"], "robust_lcb": v.get("robust_lcb"),
                  "win_rate": v.get("win_rate"), "avg_king_loss": v["avg_king_loss"],
                  "avg_challenger_loss": v["avg_challenger_loss"], "sources": sources,
                  "sample": self.sample.sample_id, "model_id": model_id, "batch": self.args.batch_size,
                  "attn": self.args.attn, "build_s": round(build, 1), "score_s": round(secs, 1),
                  "refilled_tensors": refilled, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        record = {k: val for k, val in raw.items() if k not in ("priority",)}
        record["result"] = result
        out = RECIPES / f"{name}.json"
        out.write_text(json.dumps(record, indent=1) + "\n")
        with SCORES.open("a") as handle:
            handle.write(json.dumps({"name": name, "description": raw.get("description", ""),
                                     "method": raw.get("method", recipe.mode), **result}) + "\n")
        log.info("RESULT %-28s mu_hat %+.5f lcb %+.5f n %d | %s | build %.0fs (%d tensors) score %.0fs",
                 name, v["mu_hat"], v["lcb"], v["n_sequences"],
                 " ".join(f"{k.split('-')[0]} {s:+.4f}" for k, s in sorted(sources.items())), build, refilled, secs)
        subprocess.run([sys.executable, str(REPO / "scripts/merge/report40.py")], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        git_push([out, SCORES, REPO / "40.md"], f"k40 sweep: {name} mu_hat {v['mu_hat']:+.5f} (n={v['n_sequences']})")
        if v["mu_hat"] >= self.args.promote_at and v["n_sequences"] >= 800:
            PROMOTE.mkdir(parents=True, exist_ok=True)
            (PROMOTE / f"{name}.json").write_text(json.dumps(record, indent=1))
            log.info("PROMOTE %s (mu_hat %+.5f >= %.4f)", name, v["mu_hat"], self.args.promote_at)

    def handle(self, path: Path) -> None:
        raw = json.loads(path.read_text())
        kind = raw.get("kind", "score")
        if kind == "score":
            self.score_item(path, raw)
        elif kind == "grad":
            import sweep40_grad
            import importlib
            importlib.reload(sweep40_grad).grad_item(self, path, raw)
        else:
            raise ValueError(f"unknown kind {kind}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--king", type=Path, default=Path("/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c"))
    ap.add_argument("--queue", type=Path, default=RECIPES / "queue")
    ap.add_argument("--sample", default="tag:A")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--attn", default="flash_attention_4")
    ap.add_argument("--lm-head-chunk", type=int, default=1024)
    ap.add_argument("--io-threads", type=int, default=32)
    ap.add_argument("--promote-at", type=float, default=0.001)
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

    log.info("sweep40 ready; watching %s", args.queue)
    while True:
        if (args.queue / "STOP").exists() or (args.queue / "PAUSE").exists():
            log.info("STOP/PAUSE file found; exiting (frees the GPU; run_sweep40.sh restarts after PAUSE is removed)")
            break
        path = next_item()
        if path is None:
            time.sleep(5)
            continue
        try:
            server.handle(path)
        except Exception as exc:  # keep serving the queue
            log.exception("FAIL %s: %s", path.name, exc)
            if path.exists():
                shutil.move(str(path), args.queue / "done" / (path.name + ".failed"))
            gc.collect()
            torch.cuda.empty_cache()
            continue
        if path.exists():  # may have been withdrawn while it ran
            shutil.move(str(path), args.queue / "done" / path.name)


if __name__ == "__main__":
    main()
