#!/usr/bin/env python3
"""Validator-parity evaluation for Teutonic SN3 (see docs/PARITY_EVAL.md).

Samples:  tag:<name>              fresh validator-distribution sample (synthetic seed)
          seed:<block_hash>:<hotkey>  explicit validator seed material
          duel:<challenge_id>     exact sample of a past duel (seed recovered on-chain)
          id:<sample_id>          previously built sample
Models:   <dir> | reign:<n> | digest:<sha256> | c1:<spec>,c2:<spec> | recipe.json

Examples:
  parity_eval.py config --drift
  parity_eval.py replay --challenge-id latest
  parity_eval.py duel --king reign:29 --challenger merged.json --sample tag:a --sample tag:b
  parity_eval.py score --model reign:29 --sample tag:a --limit 3000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.parity import chain, upstream  # noqa: E402
from src.parity.cache import LossCache, scorer_signature  # noqa: E402
from src.parity.sample import (  # noqa: E402
    DASHBOARD_URL,
    LOCAL_HOTKEY,
    EvalSample,
    build_sample,
    fetch_json,
    load_live_config,
    load_sample,
    planned_shard_names,
    sample_dir,
    synthetic_block_hash,
)
from src.parity.verdict import duel as duel_verdict  # noqa: E402

log = logging.getLogger("parity_eval")
# Colon-separated search path for downloaded checkpoints named by digest.
MODELS_DIRS = [
    Path(p) for p in os.environ.get("TEUTONIC_MODELS_DIRS", f"{REPO_ROOT / 'models'}").split(":") if p
]
RESULTS_DIR = upstream.DEFAULT_CACHE_ROOT / "results"


# --------------------------------------------------------------------- resolving
_dashboard_cache: dict = {}
COMPETITION = "main"  # set from --competition in main()


def dashboard(competition: str = "main") -> dict:
    """MAIN dashboard (reign:<n> lookups) or a specialist competition's dashboard (its duels / kings)."""
    key = "main" if competition in ("main", "mix") else competition
    if key not in _dashboard_cache:
        url = DASHBOARD_URL if key == "main" else f"https://teutonic.ai/competitions/{key}/dashboard.json"
        _dashboard_cache[key] = fetch_json(url)
    return _dashboard_cache[key]


def _digest_dir(digest: str) -> str:
    for base in MODELS_DIRS:
        index = base / digest / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if all((base / digest / shard).exists() for shard in shards):
                return str(base / digest)
    raise SystemExit(f"model {digest} not downloaded under {':'.join(map(str, MODELS_DIRS))}")


def resolve_model_path(spec: str) -> str:
    if spec.startswith("reign:"):
        reign = int(spec.split(":", 1)[1])
        for entry in dashboard()["king_chain"]:
            if int(entry["reign_number"]) == reign:
                return _digest_dir(entry["model_digest"])
        raise SystemExit(f"reign {reign} not in dashboard king_chain")
    if spec.startswith("digest:"):
        return _digest_dir(spec.split(":", 1)[1])
    return spec


def resolve_recipe(spec: str):
    from src.parity.weights import Recipe, Term, Transform

    if spec.endswith(".json"):
        raw = json.loads(Path(spec).read_text())
        return Recipe(
            terms=[Term(Path(resolve_model_path(t["model"])), float(t["coef"])) for t in raw["terms"]],
            overrides=[(o["pattern"], [float(c) for c in o["coefs"]]) for o in raw.get("overrides", [])],
            label=raw.get("label", Path(spec).stem),
            replacements=dict(raw.get("replacements", {})),
            mode=raw.get("mode", "linear"),
            mode_args=dict(raw.get("mode_args", {})),
            tensor_coefs={k: [float(c) for c in v] for k, v in raw.get("tensor_coefs", {}).items()},
            head_rows_from_king=[int(i) for i in (json.loads(Path(raw["head_rows_file"]).read_text()) if raw.get("head_rows_file") else raw.get("head_rows_from_king", []))],
            transforms=[Transform(t["pattern"], t["op"], dict(t.get("args", {}))) for t in raw.get("transforms", [])],
        )
    if "," in spec or (":" in spec and not spec.startswith(("reign:", "digest:"))):
        parts = []
        for part in spec.split(","):
            coef, model = part.split(":", 1)
            parts.append(f"{coef}:{resolve_model_path(model)}")
        return Recipe.parse(",".join(parts))
    return Recipe.single(resolve_model_path(spec))


def find_duel(challenge_id: str) -> dict:
    history = sorted((h for h in dashboard(COMPETITION)["history"] if h.get("timestamp")), key=lambda x: x["timestamp"])
    if challenge_id == "latest":
        return next(h for h in reversed(history) if h.get("accepted"))
    for entry in history:
        if entry.get("challenge_id") == challenge_id:
            return entry
    raise SystemExit(f"challenge {challenge_id} not in dashboard history")


def recover_duel_seed(entry: dict, live, *, window: int = 4000) -> tuple[str, str]:
    """Find the ready block hash whose shard plan reproduces the duel's shards_used."""
    board = dashboard(COMPETITION)["chain"]
    ref_time = datetime.fromisoformat(board["observed_at"].replace("Z", "+00:00"))
    finished = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
    started = finished - timedelta(seconds=float(entry.get("wall_time_s") or 0))
    end = chain.estimate_block(started, ref_block=int(board["last_finalized_block"]), ref_time=ref_time) + 25
    hit = chain.find_ready_block(entry["hotkey"], start=end - window, end=end)
    if hit is None:
        raise SystemExit("ready commitment not found; widen --window")
    block_number, block_hash = hit
    # window shards only: since 14a9bb2 the whole-document shards are listed too ("component": "long_documents")
    want = {s["source"]: s["names"] for s in entry.get("shards_used") or [] if s.get("component") is None}
    got = planned_shard_names(live, block_hash=block_hash, hotkey=entry["hotkey"])
    if want and got != want:
        raise SystemExit(
            f"block {block_number} seed does not reproduce shards_used (config changed since this duel?)"
        )
    log.info("duel %s seeded by block %d %s (shards verified)", entry["challenge_id"], block_number, block_hash)
    return block_hash, entry["hotkey"]


def resolve_sample(spec: str, live) -> EvalSample:
    if spec.startswith("id:"):
        return load_sample(sample_dir(upstream.DEFAULT_CACHE_ROOT, spec[3:]))
    if spec.startswith("tag:"):
        tag = spec[4:]
        return build_sample(live, block_hash=synthetic_block_hash(tag), hotkey=LOCAL_HOTKEY, label=spec)
    if spec.startswith("seed:"):
        _, block_hash, hotkey = spec.split(":", 2)
        return build_sample(live, block_hash=block_hash, hotkey=hotkey, label=spec)
    if spec.startswith("duel:"):
        entry = find_duel(spec[5:])
        block_hash, hotkey = recover_duel_seed(entry, live)
        return build_sample(live, block_hash=block_hash, hotkey=hotkey, label=f"duel:{entry['challenge_id']}")
    raise SystemExit(f"unknown sample spec {spec!r}")


# ----------------------------------------------------------------------- scoring
class Scorer:
    """Loads one model at a time and fills the loss cache for requested samples."""

    def __init__(self, args):
        self.args = args
        self.cache = LossCache()
        self.signature = scorer_signature(args.attn, args.lm_head_chunk, bool(args.deterministic))
        self.gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else None

    def losses(self, recipe, samples: list[EvalSample], limit: int | None) -> list[np.ndarray]:
        model_id = recipe.identity()
        wanted = []
        results = []
        for sample in samples:
            cached = self.cache.load(sample.sample_id, model_id, self.signature, sample.n)
            upto = sample.n if limit is None else min(limit, sample.n)
            missing = [i for i in range(upto) if not np.isfinite(cached[i])]
            results.append(cached)
            if missing:
                wanted.append((sample, cached, missing))
        if not wanted:
            return results
        import torch
        from concurrent.futures import ThreadPoolExecutor

        from src.parity.pipeline import PipelineScorer
        from src.parity.weights import clone_model, load_model

        gpus = self.gpus or list(range(torch.cuda.device_count()))
        replicas = max(1, int(getattr(self.args, "replicas", 1) or 1))
        if len(gpus) % replicas:
            raise SystemExit(f"--replicas {replicas} must divide the {len(gpus)} GPUs")
        per = len(gpus) // replicas
        groups = [gpus[r * per:(r + 1) * per] for r in range(replicas)]
        if replicas > 1 and getattr(self.args, "replica_procs", 1):
            # one process per replica: no GIL contention between replicas (threads left the GPUs 60-85 % idle)
            from src.parity.mpscore import score_replicas

            def save(sample, cached, elapsed, recipe=recipe, model_id=model_id):
                self.cache.save(sample.sample_id, model_id, self.signature, cached,
                                {"recipe": recipe.describe(), "score_seconds": round(elapsed, 1), "replicas": replicas,
                                 "replica_procs": True})

            for sample, _, missing in wanted:
                log.info("scoring %s on %s: %d sequences on %d replica process(es)", recipe.label or model_id,
                         sample.sample_id, len(missing), replicas)
            score_replicas(recipe, groups, wanted, attn=self.args.attn, lm_head_chunk=self.args.lm_head_chunk,
                           batch_size=self.args.batch_size, deterministic=bool(self.args.deterministic), on_save=save)
            return results
        # One full model copy per group (a 220 GB king fits one 275 GB B300): the first is read from
        # disk (and merged, for recipes); the others are cloned device-to-device over NVLink. The copies
        # score disjoint interleaved slices, so throughput scales with the number of replicas.
        first = load_model(recipe, groups[0], attn_implementation=self.args.attn)
        with ThreadPoolExecutor(max_workers=max(1, replicas - 1)) as pool:
            loaded = [first] + list(pool.map(lambda g: clone_model(first, g), groups[1:]))
        scorers = [PipelineScorer(l, lm_head_chunk=self.args.lm_head_chunk) for l in loaded]
        if self.args.attn == "flash_attention_4":
            # transformers imports the FA kernels lazily on the first attention call and publishes
            # `_loaded_implementation` before the function globals: concurrent first calls from several
            # replica threads can see `_flash_fn is None`. Import once here, before any thread starts.
            from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

            lazy_import_flash_attention(self.args.attn)
            from src.parity.pipeline import serialize_fa4_host_calls

            serialize_fa4_host_calls()  # concurrent first calls JIT-compile FA4 kernels at once: segfault
        try:
            for sample, cached, missing in wanted:
                log.info("scoring %s on %s: %d sequences on %d replica(s)", recipe.label or model_id,
                         sample.sample_id, len(missing), replicas)
                started = time.time()
                last_save = [time.time()]
                save_lock = threading.Lock()
                count = [0]

                def checkpoint(index, loss, sample=sample, cached=cached):
                    cached[index] = loss
                    with save_lock:
                        count[0] += 1
                        if count[0] % 1000 == 0:
                            rate = count[0] / max(time.time() - started, 1e-9)
                            log.info("total scored %d/%d (%.1f seq/s, eta %.0fs)", count[0], len(missing), rate,
                                     (len(missing) - count[0]) / max(rate, 1e-9))
                    if time.time() - last_save[0] > 120 and save_lock.acquire(blocking=False):
                        try:
                            last_save[0] = time.time()
                            self.cache.save(sample.sample_id, model_id, self.signature, cached.copy(), {"recipe": recipe.describe()})
                        finally:
                            save_lock.release()

                slices = [missing[r::replicas] for r in range(replicas)]
                with ThreadPoolExecutor(max_workers=replicas) as pool:
                    list(pool.map(
                        lambda pair: pair[0].score(sample.tokens, pair[1], batch_size=self.args.batch_size,
                                                   on_result=checkpoint, log_every=0 if replicas > 1 else 1000),
                        [(s, sl) for s, sl in zip(scorers, slices) if sl],
                    ))
                elapsed = time.time() - started
                log.info("scored %d sequences in %.0fs (%.1f seq/s)", len(missing), elapsed, len(missing) / max(elapsed, 1e-9))
                self.cache.save(
                    sample.sample_id, model_id, self.signature, cached,
                    {"recipe": recipe.describe(), "score_seconds": round(elapsed, 1),
                     "load_seconds": round(max(l.load_seconds for l in loaded), 1), "replicas": replicas},
                )
        finally:
            # Drop every reference (scorers hold the models too) before collecting: the models contain
            # reference cycles, so their GPU memory is only returned after a full gc pass.
            devices = sorted({d for l in loaded for d in l.layer_devices})
            for scorer in scorers:
                scorer.close()
                scorer.model = None
                scorer.static.clear()
            for l in loaded:
                l.model = None
            scorers.clear()
            loaded.clear()
            first = None
            import gc

            gc.collect()
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
        return results


def print_verdict(title: str, verdict: dict, published: dict | None = None) -> None:
    print(f"\n=== {title}")
    keys = ("n_sequences", "n_scored_tokens", "avg_king_loss", "avg_challenger_loss", "mu_hat", "lcb", "delta", "accepted")
    print("  " + "  ".join(f"{k}={verdict.get(k)}" for k in keys))
    extra = ("se_iid", "se_shard_jackknife", "robust_lcb", "win_rate", "early_stop", "complete")
    print("  " + "  ".join(f"{k}={verdict.get(k)}" for k in extra if k in verdict))
    if published:
        print("  published: " + "  ".join(f"{k}={published.get(k)}" for k in keys if k in published))
    print(f"  {'source':<20}{'n':>6}{'king':>11}{'chall':>11}{'mu_hat':>11}" + ("   published mu_hat" if published else ""))
    pub_sources = {s["source"]: s for s in (published or {}).get("source_scores", [])}
    for name, row in sorted(verdict["source_scores"].items()):
        pub = pub_sources.get(name)
        tail = f"   {pub['mu_hat']:+.6f} (king {pub['avg_king_loss']:.6f})" if pub else ""
        print(f"  {name:<20}{row['n_sequences']:>6}{row['avg_king_loss']:>11.6f}{row['avg_challenger_loss']:>11.6f}{row['mu_hat']:>+11.6f}{tail}")
    # token-weighted split: 2048-token windows vs whole 2049-8192-token documents (and documents by length)
    for group in ("component_scores", ("long_document_scores", "by_bucket")):
        rows = verdict.get(group) if isinstance(group, str) else (verdict.get(group[0]) or {}).get(group[1])
        for name, row in sorted((rows or {}).items()):
            print(f"  {name:<20}{row['n_sequences']:>6}{row['avg_king_loss']:>11.6f}{row['avg_challenger_loss']:>11.6f}"
                  f"{row['mu_hat']:>+11.6f}   {row['n_scored_tokens']} tokens")


def save_result(kind: str, payload: dict) -> Path:
    """Write a result file that never replaces another one (campaigns report many cached recipes per second; with
    second-resolution names all but the last of them were lost)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now)) + f"{now % 1:.6f}"[1:] + "Z"
    body = json.dumps(payload, indent=1, default=str)
    for attempt in range(1000):
        path = RESULTS_DIR / f"{stamp}{'' if attempt == 0 else f'-{attempt}'}_{kind}.json"
        try:
            with open(path, "x") as handle:
                handle.write(body)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"no free result file name for {stamp}")


# ---------------------------------------------------------------------- commands
def cmd_config(args) -> None:
    live = load_live_config(refresh=True, competition=args.competition)
    print(json.dumps({
        "config_version": live.config_version, "n": live.n, "delta_threshold": live.delta_threshold,
        "shards_per_dataset": live.shards_per_dataset,
        "sources": [{k: s[k] for k in ("name", "proportion", "manifest_sha256")} for s in live.raw["sources"]],
        "upstream_commit": upstream.UPSTREAM_COMMIT,
    }, indent=1))
    if args.drift:
        drift = upstream.upstream_drift()
        print("upstream scoring files changed since pin:" if drift else "upstream scoring files: unchanged", drift or "")


def cmd_sample(args) -> None:
    live = load_live_config(competition=args.competition)
    for spec in args.sample:
        sample = resolve_sample(spec, live)
        print(sample.sample_id, sample.n, sample.meta["label"], sample.meta["block_hash"])


def cmd_score(args) -> None:
    live = load_live_config(competition=args.competition)
    samples = [resolve_sample(spec, live) for spec in args.sample]
    scorer = Scorer(args)
    for spec in args.model:
        recipe = resolve_recipe(spec)
        for sample, losses in zip(samples, scorer.losses(recipe, samples, args.limit)):
            done = np.isfinite(losses)
            print(f"{recipe.label or spec} on {sample.sample_id}: scored {done.sum()} mean {np.nanmean(losses):.6f}")


def cmd_duel(args) -> None:
    live = load_live_config(competition=args.competition)
    samples = [resolve_sample(spec, live) for spec in args.sample]
    scorer = Scorer(args)
    king = resolve_recipe(args.king)
    king_losses = scorer.losses(king, samples, args.limit)
    summary = []
    for spec in args.challenger:
        recipe = resolve_recipe(spec)
        chall_losses = scorer.losses(recipe, samples, args.limit)
        for sample, k, c in zip(samples, king_losses, chall_losses):
            upto = sample.n if args.limit is None else min(args.limit, sample.n)
            mask = np.zeros(sample.n, dtype=bool)
            mask[:upto] = True
            verdict = duel_verdict(np.where(mask, k, np.nan), np.where(mask, c, np.nan), sample, delta=live.delta_threshold)
            print_verdict(f"{recipe.label or spec} vs king {king.label} on {sample.sample_id}", verdict)
            summary.append({"challenger": recipe.describe(), "king": king.describe(), "sample": sample.sample_id, "verdict": verdict})
    if len(samples) > 1:
        for spec in args.challenger:
            label = resolve_recipe(spec).label or spec
            mus = [s["verdict"]["mu_hat"] for s in summary if (s["challenger"]["label"] or spec) == label]
            print(f"\n{label}: mu_hat per sample {mus} mean {np.mean(mus):+.6f} sd {np.std(mus, ddof=1) if len(mus) > 1 else 0:.6f}")
    print("saved", save_result("duel", {"delta": live.delta_threshold, "attn": args.attn, "results": summary}))


def cmd_replay(args) -> None:
    live = load_live_config(competition=args.competition)
    entry = find_duel(args.challenge_id)
    if entry.get("dataset_version") and entry["dataset_version"] != live.config_version:
        raise SystemExit("that duel ran under a different eval config; replay needs the same manifests")
    block_hash, hotkey = recover_duel_seed(entry, live)
    sample = build_sample(live, block_hash=block_hash, hotkey=hotkey, label=f"duel:{entry['challenge_id']}")
    if args.king:
        king = resolve_recipe(args.king)
    else:
        reigns = [r for board in (dashboard(), dashboard(COMPETITION)) for r in board["king_chain"]
                  if r.get("hotkey") == entry["baseline_hotkey"]]
        if not reigns:
            raise SystemExit("the duel's king is no longer in king_chain; pass --king")
        king = resolve_recipe(f"digest:{reigns[-1]['model_digest']}")
    if args.challenger:
        challenger = resolve_recipe(args.challenger)
    elif entry.get("challenger_digest"):
        challenger = resolve_recipe(f"digest:{entry['challenger_digest']}")
    else:
        raise SystemExit("challenger weights are private for non-winning duels; pass --challenger")
    scorer = Scorer(args)
    (k,) = scorer.losses(king, [sample], args.limit)
    (c,) = scorer.losses(challenger, [sample], args.limit)
    verdict = duel_verdict(k, c, sample, delta=float(entry.get("delta") or live.delta_threshold))
    print_verdict(f"replay {entry['challenge_id']} ({args.attn})", verdict, published=entry)
    print("saved", save_result("replay", {"entry": {k2: entry.get(k2) for k2 in ("challenge_id", "mu_hat", "lcb", "avg_king_loss", "avg_challenger_loss", "source_scores")}, "attn": args.attn, "verdict": verdict, "king": king.describe(), "challenger": challenger.describe()}))


def cmd_leaderboard(args) -> None:
    """Rank every cached model on a sample against the king (common scored prefix)."""
    live = load_live_config(competition=args.competition)
    cache = LossCache()
    signature = scorer_signature(args.attn, args.lm_head_chunk, bool(args.deterministic))
    king = resolve_recipe(args.king)
    for spec in args.sample:
        sample = resolve_sample(spec, live)
        king_losses = cache.load(sample.sample_id, king.identity(), signature, sample.n)
        rows = []
        for entry in cache.entries(sample.sample_id):
            if entry["model_id"] == king.identity() or entry["signature"] != signature:
                continue
            chall = cache.load(sample.sample_id, entry["model_id"], signature, sample.n)
            both = np.isfinite(king_losses) & np.isfinite(chall)
            upto = int(np.argmin(both)) if not both.all() else sample.n  # common prefix length
            if upto < 500:
                continue
            mask = np.zeros(sample.n, dtype=bool)
            mask[:upto] = True
            verdict = duel_verdict(np.where(mask, king_losses, np.nan), np.where(mask, chall, np.nan),
                                   sample, delta=live.delta_threshold)
            label = (entry.get("recipe") or {}).get("label") or entry["model_id"][:24]
            rows.append((verdict["mu_hat"], label, upto, verdict))
        print(f"\n=== {sample.sample_id} ({sample.meta.get('label')}) vs king {king.label}")
        print(f"  {'candidate':<34}{'n':>7}{'mu_hat':>11}{'lcb':>11}  worst source")
        for mu, label, upto, verdict in sorted(rows, key=lambda r: -r[0]):
            worst = min(verdict["source_scores"].items(), key=lambda kv: kv[1]["mu_hat"])
            print(f"  {label[:34]:<34}{upto:>7}{mu:>+11.6f}{verdict['lcb']:>+11.6f}  {worst[0]} {worst[1]['mu_hat']:+.5f}")


def cmd_routing(args) -> None:
    """Per-layer, per-source expert routing counts (MoE-aware merge planning)."""
    import torch

    from src.parity.pipeline import PipelineScorer
    from src.parity.weights import free_model, load_model

    live = load_live_config(competition=args.competition)
    sample = resolve_sample(args.sample[0], live)
    recipe = resolve_recipe(args.model)
    upto = sample.n if args.limit is None else min(args.limit, sample.n)
    names = sample.source_names()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    loaded = load_model(recipe, gpus, attn_implementation=args.attn)
    model = loaded.model
    local = threading.local()
    counts = {}
    hooks = []
    for layer_index, layer in enumerate(model.model.layers):
        gate = getattr(layer.mlp, "gate", None)
        if gate is None:
            continue
        device = next(gate.parameters()).device
        counts[layer_index] = torch.zeros(len(names), gate.n_routed_experts, dtype=torch.int64, device=device)

        def hook(module, inputs, output, layer_index=layer_index):
            source = getattr(local, "source", None)
            if source is not None:
                counts[layer_index][source] += torch.bincount(output[0].flatten(), minlength=module.n_routed_experts)

        hooks.append(gate.register_forward_hook(hook))
    engine = upstream.engine
    order = list(range(upto))
    # workers may reach the model in a different order than they picked batches, so
    # identify each (batch_size=1) sequence by its leading tokens
    source_of = {tuple(np.asarray(sample.tokens[i][:64]).tolist()): names.index(str(sample.sources[i])) for i in order}
    scorer = PipelineScorer(loaded, lm_head_chunk=args.lm_head_chunk)

    def by_sequence(model_, batches, chunk):
        local.source = source_of[tuple(batches[0][:64])]
        try:
            return engine.compute_per_sequence_loss(model_, batches, chunk, reset_peak_memory=False)
        finally:
            local.source = None

    try:
        scorer.score(sample.tokens, order, batch_size=1, fn=by_sequence)
    finally:
        for hook_handle in hooks:
            hook_handle.remove()
        scorer.close()
    table = np.stack([counts[i].cpu().numpy() for i in sorted(counts)])  # [layers, sources, experts]
    layers = sorted(counts)
    free_model(loaded)
    out_dir = upstream.DEFAULT_CACHE_ROOT / "routing"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{recipe.label or 'model'}_{sample.sample_id}_{upto}".replace("/", "_")
    np.save(out_dir / f"{stem}.npy", table)
    (out_dir / f"{stem}.json").write_text(json.dumps({"layers": layers, "sources": names, "n": upto,
                                                      "model": recipe.describe(), "sample": sample.sample_id}, indent=1))
    share = table.sum(axis=(0, 2)) / table.sum()
    print("token share per source:", dict(zip(names, np.round(share, 4).tolist())))
    print("saved", out_dir / f"{stem}.npy")


def cmd_tokendiag(args) -> None:
    """Where does a challenger lose/gain vs the king? Per target token id, position and source."""
    import torch

    from src.parity.pipeline import PipelineScorer
    from src.parity.tokens import SEQ_POS, per_token_ce
    from src.parity.weights import free_model, load_model

    live = load_live_config(competition=args.competition)
    sample = resolve_sample(args.sample[0], live)
    upto = sample.n if args.limit is None else min(args.limit, sample.n)
    idx = list(range(upto))
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    per_model = {}
    for role, spec in (("king", args.king), ("challenger", args.challenger)):
        recipe = resolve_recipe(spec)
        loaded = load_model(recipe, gpus, attn_implementation=args.attn)
        scorer = PipelineScorer(loaded, lm_head_chunk=args.lm_head_chunk)
        try:
            per_model[role] = scorer.score(sample.tokens, idx, batch_size=args.batch_size, fn=per_token_ce, width=SEQ_POS)[:upto]
        finally:
            scorer.close()
            scorer.model = None  # the scorer must not keep the model alive across loads
            del scorer
            free_model(loaded)
            del loaded
            import gc; gc.collect(); torch.cuda.empty_cache()
    diff = per_model["king"] - per_model["challenger"]  # >0 where the challenger is better
    targets = np.asarray(sample.tokens[:upto], dtype=np.int64)[:, 1:]
    sources = sample.sources[:upto]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(resolve_recipe(args.king).terms[0].path), trust_remote_code=True)
    report = {"n_sequences": upto, "mu_hat": float(diff.mean()), "by_source": {}}
    for name in sorted(set(sources.tolist())):
        m = sources == name
        d, t = diff[m], targets[m]
        sums = np.bincount(t.ravel(), weights=d.ravel(), minlength=152576)
        counts = np.bincount(t.ravel(), minlength=152576)
        order = np.argsort(sums)  # most negative first = largest loss increase
        neg_total = float(sums[sums < 0].sum())
        top = [(int(i), float(sums[i]), int(counts[i]), tok.decode([int(i)])) for i in order[:40]]
        pos_buckets = {f"{a}-{b}": float(d[:, a:b].mean()) for a, b in ((0, 128), (128, 512), (512, 1024), (1024, 2047))}
        cum = np.cumsum(np.sort(sums[sums < 0]))
        k50 = int(np.searchsorted(cum, 0.5 * neg_total)) + 1 if neg_total < 0 else 0
        report["by_source"][name] = {"n": int(m.sum()), "mu_hat": float(d.mean()), "loss_increase_total": neg_total,
                                     "tokens_explaining_50pct_of_increase": k50, "worst_tokens": top, "by_position": pos_buckets}
        print(f"== {name}: n={int(m.sum())} mu_hat={d.mean():+.5f}  50% of the loss increase comes from {k50} token ids  by position {pos_buckets}")
        for i, sm, c, txt in top[:12]:
            print(f"     id {i:>6} {txt!r:<24} total {sm:+.3f} over {c} occurrences (mean {sm / max(c, 1):+.4f})")
    out = RESULTS_DIR / f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_tokendiag.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"king": args.king, "challenger": args.challenger, "sample": sample.sample_id, **report}, indent=1))
    np.save(out.with_suffix(".diff.npy"), diff.astype(np.float32))
    print("saved", out)


def cmd_calibrate(args) -> None:
    """Optimal logit (lm_head) scale for a model on the first --limit sequences of a sample."""
    import torch

    from src.parity.calibration import STAT_WIDTH, calibration_stats, summarize
    from src.parity.pipeline import PipelineScorer
    from src.parity.weights import free_model, load_model

    live = load_live_config(competition=args.competition)
    sample = resolve_sample(args.sample[0], live)
    recipe = resolve_recipe(args.model)
    upto = sample.n if args.limit is None else min(args.limit, sample.n)
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    loaded = load_model(recipe, gpus, attn_implementation=args.attn)
    scorer = PipelineScorer(loaded, lm_head_chunk=args.lm_head_chunk)
    try:
        stats = scorer.score(sample.tokens, list(range(upto)), batch_size=args.batch_size,
                             fn=calibration_stats, width=STAT_WIDTH)
    finally:
        scorer.close()
        free_model(loaded)
    report = summarize(stats[:upto], sample.sources[:upto])
    for name, row in report.items():
        print(f"{name:<20} {row}")
    print("saved", save_result("calibrate", {"model": recipe.describe(), "sample": sample.sample_id, "n": upto, "report": report}))


def cmd_headtune(args) -> None:
    """Fit lm_head on a training pool over a frozen body; keep the best head on held-out data."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    from src.parity.headtune import HeadTuner
    from src.parity.pipeline import PipelineScorer
    from src.parity.weights import free_model, load_model

    live = load_live_config(competition=args.competition)
    val = resolve_sample(args.val, live)
    body = resolve_recipe(args.body)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = LossCache()
    signature = scorer_signature(args.attn, args.lm_head_chunk, bool(args.deterministic))
    val_idx = list(range(min(args.val_limit, val.n)))
    baseline = cache.load(val.sample_id, body.identity(), signature, val.n)

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else list(range(torch.cuda.device_count()))
    loaded = load_model(body, gpus, attn_implementation=args.attn)
    scorer = PipelineScorer(loaded, lm_head_chunk=args.lm_head_chunk)
    tuner = HeadTuner(loaded.model, lr=args.lr, accum_seqs=args.accum)
    pool = np.load(Path(args.pool) / "tokens.npy", mmap_mode="r")
    records = []
    best = None
    try:
        missing = [i for i in val_idx if not np.isfinite(baseline[i])]
        if missing:
            baseline[missing] = scorer.score(val.tokens, missing, batch_size=args.batch_size, log_every=0)[missing]
            cache.save(val.sample_id, body.identity(), signature, baseline, {"recipe": body.describe()})
        base_mean = float(baseline[val_idx].mean())
        for start in range(0, min(args.train_n, len(pool)), args.segment):
            idx = list(range(start, min(start + args.segment, args.train_n, len(pool))))
            tuner.training = True
            online = scorer.score(pool, idx, batch_size=args.batch_size, fn=tuner, log_every=0)[idx]
            tuner.training = False
            current = scorer.score(val.tokens, val_idx, batch_size=args.batch_size, log_every=0)[val_idx]
            diff = baseline[val_idx] - current
            record = {
                "seen": idx[-1] + 1, "steps": tuner.steps, "online_train_ce": round(float(online.mean()), 6),
                "val_ce": round(float(current.mean()), 6), "val_mu_hat": round(float(diff.mean()), 6),
                "val_se": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 7),
                "head_rel_change": round(tuner.relative_change(), 5),
            }
            records.append(record)
            log.info("headtune %s", json.dumps(record))
            if best is None or record["val_mu_hat"] > best["val_mu_hat"]:
                best = record
                save_file({"lm_head.weight": loaded.model.lm_head.weight.detach().cpu().contiguous()},
                          str(out / "best_head.safetensors"), metadata={"format": "pt"})
    finally:
        scorer.close()
        free_model(loaded)
    recipe_json = json.loads(Path(args.body).read_text()) if args.body.endswith(".json") else {
        "terms": [{"model": str(t.path), "coef": t.coef} for t in body.terms]}
    recipe_json["label"] = f"{body.label}+head:{out.name}"
    recipe_json["replacements"] = {"lm_head.weight": str(out / "best_head.safetensors")}
    (out / "recipe.json").write_text(json.dumps(recipe_json, indent=1))
    (out / "history.json").write_text(json.dumps({"args": vars(args), "baseline_val_ce": base_mean,
                                                   "records": records, "best": best}, indent=1))
    print(json.dumps({"baseline_val_ce": base_mean, "best": best}, indent=1))
    print("recipe with tuned head:", out / "recipe.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attn", default="flash_attention_4", choices=["flash_attention_4", "eager", "fa4sim"],
                        help="flash_attention_4 = the validator's backend (sm_90/sm_100/sm_103; document-masked via "
                             "explicit fragment boundaries); eager = the validator's masked fallback (small tests: "
                             "W x W masks); fa4sim cannot score the document-masked policy (validator >= 14a9bb2)")
    parser.add_argument("--competition", default="main", choices=["main", "math", "code", "text", "mix"],
                        help="eval config: MAIN, a specialist mix (70/15/15), or 'mix' = equal math/code/text "
                             "screening sample (re-weight per split to any specialist mix)")
    parser.add_argument("--lm-head-chunk", type=int, default=1024)
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N sequences")
    parser.add_argument("--batch-size", type=int, default=4, help="sequences per pipeline step")
    parser.add_argument("--replicas", type=int, default=1,
                        help="independent model copies (each on len(gpus)/N GPUs) scoring disjoint slices; "
                             "use 8 on 8xB300 where a whole 110B model fits one GPU")
    parser.add_argument("--replica-procs", type=int, default=1,
                        help="with --replicas > 1: one process per replica (1, default) or threads of this process (0)")
    parser.add_argument("--deterministic", type=int, default=1,
                        help="1 = the validator's execution mode since 14a9bb2 (strict deterministic algorithms, "
                             "CUBLAS_WORKSPACE_CONFIG=:4096:8, deterministic cuDNN); 0 = default kernels")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("config")
    p.add_argument("--drift", action="store_true")
    p = sub.add_parser("sample")
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("score")
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("duel")
    p.add_argument("--king", required=True)
    p.add_argument("--challenger", action="append", required=True)
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("replay")
    p.add_argument("--challenge-id", default="latest")
    p.add_argument("--king")
    p.add_argument("--challenger")
    p = sub.add_parser("calibrate")
    p.add_argument("--model", required=True)
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("leaderboard")
    p.add_argument("--king", default="reign:29")
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("routing")
    p.add_argument("--model", required=True)
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("tokendiag")
    p.add_argument("--king", required=True)
    p.add_argument("--challenger", required=True)
    p.add_argument("--sample", action="append", required=True)
    p = sub.add_parser("headtune")
    p.add_argument("--body", required=True, help="frozen body recipe/model")
    p.add_argument("--pool", required=True, help="training pool dir (build_train_pool.py)")
    p.add_argument("--val", default="tag:H", help="held-out sample (not used for merge screening)")
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--train-n", type=int, default=48000)
    p.add_argument("--segment", type=int, default=8000, help="sequences between validations")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--accum", type=int, default=64, help="sequences per optimizer step")
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    global COMPETITION
    COMPETITION = args.competition
    if args.deterministic:
        import torch

        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # read when cuBLAS initializes: before any CUDA work
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    commands = {"config": cmd_config, "sample": cmd_sample, "score": cmd_score, "duel": cmd_duel,
                "replay": cmd_replay, "calibrate": cmd_calibrate, "headtune": cmd_headtune,
                "leaderboard": cmd_leaderboard, "routing": cmd_routing, "tokendiag": cmd_tokendiag}
    commands[args.command](args)


if __name__ == "__main__":
    main()
