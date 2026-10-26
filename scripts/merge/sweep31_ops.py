"""Hot-reloaded operations of the sweep31 server (reloaded before every queue item).

Queue items (JSON in configs/merge/k31/queue/):
  score (default)  {"label", "description", "method", "vecs", "overrides", "transforms", "ties",
                    "tensor_vecs": {name: {vec: coef}}, "limit", "priority"}
  grad             {"type": "grad", "label", "at": <score recipe>, "sample": "tag:G", "n": 512,
                    "batch": 2, "flush_every": 64, "vecs": ["Dm", "S31"]}
                   -> per-tensor (per expert) <dL/dW, vec> at W = recipe, saved to
                      /opt/teutonic/logs/grad31/<label>.json (mean per-sequence CE, same objective as mu_hat)

Weights: W = bf16( base + sum_v coef_v(name) * T_v(vec_v) [+ lambda * TIES(...)] ), fp32 on the GPU.
`tensor_vecs` (exact tensor name -> {vec: coef}) takes precedence over `overrides` (regex) and `vecs`.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

CHUNK_BYTES = 512 << 20
GRAD_DIR = Path("/opt/teutonic/logs/grad31")


# ------------------------------------------------------------------ recipes
def coef_fn(recipe: dict, vec_names: list[str]):
    default = {v: float(recipe.get("vecs", {}).get(v, 0.0)) for v in vec_names}
    compiled = []
    for o in recipe.get("overrides", []):
        merged = dict(default)
        merged.update({v: float(c) for v, c in o["vecs"].items()})
        compiled.append((re.compile(o["pattern"]), merged))
    exact = {}
    for name, table in (recipe.get("tensor_vecs") or {}).items():
        merged = dict(default)
        merged.update({v: float(c) for v, c in table.items()})
        exact[name] = merged
    used = set(recipe.get("vecs", {})) | {v for o in recipe.get("overrides", []) for v in o["vecs"]}
    used |= set(recipe.get("transforms", {})) | set(recipe.get("ties", {}).get("vecs", {}))
    used |= {v for t in (recipe.get("tensor_vecs") or {}).values() for v in t}
    unknown = used - set(vec_names)
    if unknown:
        raise ValueError(f"unknown vectors {sorted(unknown)}")

    def coefs(name: str) -> dict[str, float]:
        if name in exact:
            return exact[name]
        for pattern, table in compiled:
            if pattern.search(name):
                return table
        return default

    return coefs


def is_linear(recipe: dict) -> bool:
    return not recipe.get("transforms") and not recipe.get("ties")


def ship_recipe(recipe: dict, base: Path, vec_specs: dict) -> dict | None:
    """Equivalent src/parity/weights.py linear recipe (exact fp32 arithmetic on the checkpoints)."""
    if not is_linear(recipe):
        return None
    paths = [base] + sorted({p for pair in vec_specs.values() for p in pair} - {base}, key=str)

    def coefs_for(table: dict[str, float]) -> list[float]:
        acc = {p: 0.0 for p in paths}
        acc[base] += 1.0
        for v, c in table.items():
            a, b = vec_specs[v]
            acc[a] += c
            acc[b] -= c
        return [round(acc[p], 10) for p in paths]

    default = {v: float(recipe.get("vecs", {}).get(v, 0.0)) for v in vec_specs}

    def merged(table):
        m = dict(default)
        m.update({v: float(c) for v, c in table.items()})
        return m

    out = {
        "label": recipe["label"],
        "description": recipe.get("description", ""),
        "terms": [{"model": str(p), "coef": c} for p, c in zip(paths, coefs_for(default))],
        "overrides": [{"pattern": o["pattern"], "coefs": coefs_for(merged(o["vecs"]))} for o in recipe.get("overrides", [])],
    }
    if recipe.get("tensor_vecs"):
        out["tensor_coefs"] = {n: coefs_for(merged(t)) for n, t in recipe["tensor_vecs"].items()}
    return out


def sweep_identity(recipe: dict, vec_specs, base_id: str) -> str:
    material = {
        "base": base_id,
        "vecs": {v: [str(a), str(b)] for v, (a, b) in vec_specs.items()},
        "default": recipe.get("vecs", {}),
        "overrides": recipe.get("overrides", []),
        "transforms": recipe.get("transforms", {}),
        "ties": recipe.get("ties", {}),
    }
    if recipe.get("tensor_vecs"):
        material["tensor_vecs"] = hashlib.sha256(json.dumps(recipe["tensor_vecs"], sort_keys=True).encode()).hexdigest()
    if (not any(recipe.get("vecs", {}).values()) and not recipe.get("overrides")
            and not recipe.get("ties") and not recipe.get("tensor_vecs")):
        return base_id  # plain base == the king itself (exact copy)
    return "sweep:" + hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ per-tensor ops (rows = original tensors)
def _gen(device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed & 0x7FFFFFFFFFFFFFFF)
    return g


def _seed(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def _row_threshold(x: torch.Tensor, keep: float) -> torch.Tensor:
    """Per-row |x| value above which a fraction `keep` of entries lie (sampled for huge rows)."""
    flat = x.abs().reshape(x.shape[0], -1)
    n = flat.shape[1]
    if n > (1 << 22):
        idx = torch.randint(0, n, (1 << 22,), device=x.device, generator=_gen(x.device, 1234))
        flat = flat[:, idx]
        n = flat.shape[1]
    k = min(n, max(1, int(round((1.0 - keep) * n))))
    return flat.kthvalue(k, dim=1).values.view(-1, *([1] * (x.dim() - 1)))


def apply_ops(x: torch.Tensor, ops: list[dict], refs: dict, key: str, start: int) -> torch.Tensor:
    zero = torch.zeros((), device=x.device)
    for op in ops:
        kind = op["op"]
        if kind == "dare":
            p = float(op["p"])
            mask = torch.rand(x.shape, device=x.device, generator=_gen(x.device, _seed(key, start, op.get("seed", 0))))
            x = torch.where(mask >= p, x / (1.0 - p), zero)
        elif kind == "trim":
            thr = _row_threshold(x, float(op["density"]))
            x = torch.where(x.abs() >= thr, x, zero)
        elif kind == "drop_top":
            thr = _row_threshold(x, float(op["frac"]))
            x = torch.where(x.abs() < thr, x, zero)
        elif kind == "clip":
            thr = _row_threshold(x, 1.0 - float(op["q"]))
            x = torch.maximum(torch.minimum(x, thr), -thr)
        elif kind == "sign":
            ref = refs.get(op["ref"])
            if ref is None:
                if op.get("zero_ref", "keep") == "drop":
                    x = torch.zeros_like(x)
                continue
            agree = torch.sign(x) == torch.sign(ref)
            if op.get("keep", "agree") == "disagree":
                agree = ~agree & (ref != 0)
            else:
                agree = agree | ((ref == 0) & (op.get("zero_ref", "keep") == "keep"))
            x = torch.where(agree, x, zero)
        elif kind == "orth":
            ref = refs.get(op["ref"])
            if ref is None:
                continue
            dims = tuple(range(1, x.dim()))
            num = (x * ref).sum(dim=dims, keepdim=True)
            den = (ref * ref).sum(dim=dims, keepdim=True).clamp_min(1e-30)
            x = x - float(op.get("beta", 1.0)) * (num / den) * ref
        else:
            raise ValueError(f"unknown op {kind}")
    return x


def ties_merge(chunks: dict, weights: dict, density: float, dare, key: str, start: int) -> torch.Tensor:
    trimmed = {}
    for v, x in chunks.items():
        if dare:
            x = apply_ops(x, [{"op": "dare", "p": dare, "seed": 7}], {}, key + v, start)
        elif density < 1.0:
            x = apply_ops(x, [{"op": "trim", "density": density}], {}, key, start)
        trimmed[v] = x
    # weights only steer the sign election; kept values are unscaled (disjoint mean)
    elected = torch.sign(sum(x * weights[v] for v, x in trimmed.items()))
    num = torch.zeros_like(elected)
    cnt = torch.zeros_like(elected)
    for x in trimmed.values():
        agree = (torch.sign(x) == elected) & (x != 0)
        num += torch.where(agree, x, torch.zeros((), device=x.device))
        cnt += agree.float()
    return num / cnt.clamp_min(1.0)


# ------------------------------------------------------------------ rebuild
def rebuild(ctx, recipe: dict) -> float:
    started = time.time()
    store = ctx.store
    coefs = coef_fn(recipe, ctx.vec_names)
    transforms = recipe.get("transforms", {})
    ties = recipe.get("ties") or None
    refs_needed = {op["ref"] for ops in transforms.values() for op in ops if "ref" in op}

    def fill(u) -> None:
        dev = u.target.device
        tables = [coefs(n) for n in u.names]
        per_vec = {v: [t[v] for t in tables] for v in ctx.vec_names}
        active = [v for v in ctx.vec_names if store.vecs[v].nonzero(u) and any(per_vec[v])]
        tie_vecs = [v for v in ties.get("vecs", {}) if store.vecs[v].nonzero(u)] if ties else []
        base = ctx.base_of(u)
        with torch.no_grad(), torch.cuda.device(dev):
            if not active and not tie_vecs:
                u.target.copy_(base)
                return
            rows = len(u.names)
            step = max(1, CHUNK_BYTES // (u.per_numel * 4))
            needed = set(active) | set(tie_vecs) | {r for r in refs_needed if store.vecs[r].nonzero(u)}
            for s in range(0, rows, step):
                e = min(rows, s + step)
                acc = base[s:e].to(dev, non_blocking=True).float()
                chunks = {v: store.vecs[v].unit(u)[s:e].to(dev, non_blocking=True).float() for v in needed}
                for v in active:
                    x = chunks[v]
                    if v in transforms:
                        x = apply_ops(x.clone(), transforms[v], chunks, u.key, s)
                    c = per_vec[v][s:e]
                    if len(set(c)) == 1:
                        acc.add_(x, alpha=c[0])
                    else:
                        acc.add_(x * torch.tensor(c, device=dev).view(-1, *([1] * (x.dim() - 1))))
                if tie_vecs:
                    w = {v: float(ties["vecs"][v]) for v in tie_vecs}
                    merged = ties_merge({v: chunks[v] for v in tie_vecs}, w, float(ties.get("density", 1.0)),
                                        ties.get("dare"), u.key, s)
                    acc.add_(merged, alpha=float(ties.get("lambda", 1.0)))
                u.target[s:e].copy_(acc)

    with ThreadPoolExecutor(ctx.args.fill_threads) as pool:
        list(pool.map(fill, ctx.order))
    for gpu in sorted(set(ctx.layer_devices)):
        torch.cuda.synchronize(gpu)
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()
    return time.time() - started


# ------------------------------------------------------------------ scoring
def losses_for(ctx, model_id: str, limit: int, recipe: dict):
    cache, sample = ctx.cache, ctx.sample
    cached = cache.load(sample.sample_id, model_id, ctx.signature, sample.n)
    todo = [i for i in range(min(limit, sample.n)) if not np.isfinite(cached[i])]
    if not todo:
        return cached, 0.0, 0.0
    build = rebuild(ctx, recipe)
    started = time.time()
    got = ctx.scorer.score(sample.tokens, todo, batch_size=ctx.args.batch_size, log_every=0)
    cached[todo] = got[todo]
    cache.save(sample.sample_id, model_id, ctx.signature, cached,
               {"batch": ctx.args.batch_size,
                "recipe": {"sweep": {k: v for k, v in recipe.items() if k != "tensor_vecs"},
                           "vec_specs": {k: [str(a), str(b)] for k, (a, b) in ctx.vec_specs.items()},
                           "base": str(ctx.args.base)}})
    return cached, build, time.time() - started


def score_item(ctx, path: Path, recipe: dict) -> None:
    log = ctx.log
    recipe.setdefault("label", path.stem)
    name = recipe["label"]
    limit = int(recipe.get("limit", ctx.args.limit))
    model_id = sweep_identity(recipe, ctx.vec_specs, ctx.base_id)
    king_losses, _, ks = losses_for(ctx, ctx.base_id, limit, {"label": "king31u", "vecs": {}})
    if ks:
        log.info("king scored up to %d (%.0fs)", limit, ks)
    chall, build, secs = losses_for(ctx, model_id, limit, recipe)
    sample = ctx.sample
    mask = np.zeros(sample.n, dtype=bool)
    mask[:min(limit, sample.n)] = True
    verdict = ctx.duel_verdict(np.where(mask, king_losses, np.nan), np.where(mask, chall, np.nan), sample,
                               delta=ctx.live.delta_threshold)
    summary = (f"n_sequences={verdict['n_sequences']}  avg_king_loss={verdict['avg_king_loss']}  "
               f"avg_challenger_loss={verdict['avg_challenger_loss']}  mu_hat={verdict['mu_hat']}  "
               f"lcb={verdict['lcb']}  delta={verdict.get('delta')}  accepted={verdict.get('accepted')}")
    sources = {k: v["mu_hat"] for k, v in verdict["source_scores"].items()}
    record = dict(recipe)
    record["method"] = recipe.get("method", "task-arith")
    record["ship"] = ship_recipe(recipe, ctx.args.base, ctx.vec_specs)
    record["vec_specs"] = {k: [str(a), str(b)] for k, (a, b) in ctx.vec_specs.items()}
    (ctx.recipes / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    with ctx.scores.open("a") as handle:
        handle.write(json.dumps({
            "name": name, "seconds": round(build + secs, 1), "summary": summary, "sources": sources,
            "recipe": str(ctx.recipes / f"{name}.json"), "attn": ctx.args.attn, "limit": limit,
            "batch": ctx.args.batch_size, "engine": "sweep31", "win_rate": verdict.get("win_rate"),
            "categories": verdict.get("category_scores"),
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }) + "\n")
    log.info("SCORE %s rebuild=%.0fs score=%.0fs %s | %s", name, build, secs, summary,
             " ".join(f"{k.split('-')[0]}={v:+.5f}" for k, v in sorted(sources.items())))


# ------------------------------------------------------------------ gradient probe
def _seq_ce(model, ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Mean over sequences of per-sequence mean next-token CE (the eval objective), differentiable."""
    hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state
    head = model.lm_head
    hidden = hidden.to(head.weight.device)
    labels = ids[:, 1:].to(head.weight.device)
    total = torch.zeros(ids.shape[0], device=head.weight.device, dtype=torch.float32)
    t = labels.shape[1]
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        logits = head(hidden[:, s:e]).float()
        ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, s:e].reshape(-1),
                                               reduction="none").view(ids.shape[0], -1)
        total = total + ce.sum(dim=1)
    return (total / t).mean()


def grad_item(ctx, path: Path, spec: dict) -> None:
    log = ctx.log
    label = spec.get("label", path.stem)
    GRAD_DIR.mkdir(parents=True, exist_ok=True)
    at = dict(spec["at"])
    at.setdefault("label", label + "_at")
    vecs = [v for v in spec.get("vecs", ctx.vec_names)]
    n = int(spec.get("n", 512))
    batch = int(spec.get("batch", 2))
    flush_every = int(spec.get("flush_every", 64))
    offset = int(spec.get("offset", 0))  # first sequence of the gradient sample (disjoint probes)
    grad_sample = ctx.pe.resolve_sample(spec.get("sample", "tag:G"), ctx.live)
    t0 = time.time()
    build = rebuild(ctx, at)
    # grads need the GPU room the resident base copy occupies: park the base in pinned host memory
    ctx.park_base()
    params = []
    for u in ctx.units:
        if not any(ctx.store.vecs[v].nonzero(u) for v in vecs):
            continue
        # expert stacks are used directly by grouped_moe; single tensors live in the module's nn.Parameter
        leaf = u.leaf if len(u.names) > 1 else ctx.model.get_parameter(u.names[0])
        leaf.requires_grad_(True)
        params.append((u, leaf))
    model = ctx.model
    old_attn = model.config._attn_implementation
    model.config._attn_implementation = spec.get("attn", "eager")
    # activation checkpointing per decoder layer (the pipeline pre-hooks run before the wrapper, so the
    # recompute sees inputs already on the layer's GPU)
    from torch.utils.checkpoint import checkpoint
    wrapped = []
    for layer in model.model.layers:
        inner = layer.forward
        layer.forward = (lambda f: (lambda *a, **k: checkpoint(f, *a, use_reentrant=False, **k)))(inner)
        wrapped.append((layer, inner))
    dots = {v: {} for v in vecs}
    chunks_out = []  # per-flush dots (split-half noise estimate)
    vnorm = {v: {} for v in vecs}
    gnorm = {}
    losses = []

    def flush() -> None:
        part = {v: {} for v in vecs}
        chunks_out.append(part)
        with torch.no_grad():
            for u, leaf in params:
                g = leaf.grad
                if g is None:
                    continue
                g = g.view(len(u.names), -1)
                for v in vecs:
                    if not ctx.store.vecs[v].nonzero(u):
                        continue
                    host = ctx.store.vecs[v].unit(u).view(len(u.names), -1)
                    step = max(1, CHUNK_BYTES // (u.per_numel * 4))
                    for s in range(0, len(u.names), step):
                        e = min(len(u.names), s + step)
                        hv = host[s:e].to(g.device, non_blocking=True).float()
                        d = (g[s:e].float() * hv).sum(dim=1).cpu().tolist()
                        if len(chunks_out) == 1:
                            for i, val in zip(range(s, e), hv.pow(2).sum(dim=1).cpu().tolist()):
                                vnorm[v][u.names[i]] = val
                        for i, val in zip(range(s, e), d):
                            dots[v][u.names[i]] = dots[v].get(u.names[i], 0.0) + val
                            part[v][u.names[i]] = val
                gg = g.float().pow(2).sum(dim=1).cpu().tolist()
                for i, val in enumerate(gg):
                    gnorm[u.names[i]] = gnorm.get(u.names[i], 0.0) + val
                leaf.grad = None

    try:
        steps = 0
        for s in range(0, n, batch):
            ids = torch.as_tensor(np.asarray(grad_sample.tokens[offset + s:offset + min(n, s + batch)], dtype=np.int64),
                                  device=f"cuda:{ctx.layer_devices[0]}")
            loss = _seq_ce(model, ids) * (ids.shape[0] / n)
            loss.backward()
            losses.append(float(loss) * n / ids.shape[0])
            steps += 1
            if steps % flush_every == 0:
                flush()
                log.info("grad %s: %d/%d seqs, mean CE %.5f (%.0fs)", label, min(n, s + batch), n,
                         float(np.mean(losses)), time.time() - t0)
        flush()
    finally:
        for _, leaf in params:
            leaf.grad = None
            leaf.requires_grad_(False)
        model.config._attn_implementation = old_attn
        for layer, inner in wrapped:
            layer.forward = inner
        for gpu in sorted(set(ctx.layer_devices)):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        ctx.unpark_base()
    out = {"label": label, "at": at, "sample": spec.get("sample", "tag:G"), "n": n, "offset": offset, "mean_ce": float(np.mean(losses)),
           "vecs": vecs, "dots": dots, "dots_chunks": chunks_out, "vnorm2": vnorm, "gnorm2": gnorm, "seconds": round(time.time() - t0, 1), "rebuild": round(build, 1)}
    (GRAD_DIR / f"{label}.json").write_text(json.dumps(out))
    tot = {v: sum(d.values()) for v, d in dots.items()}
    log.info("GRAD %s done in %.0fs: mean CE %.5f; total <g,vec> %s -> %s", label, time.time() - t0,
             out["mean_ce"], {v: f"{x:+.3e}" for v, x in tot.items()}, GRAD_DIR / f"{label}.json")


# ------------------------------------------------------------------ dispatch
def handle(ctx, path: Path) -> None:
    item = json.loads(path.read_text())
    kind = item.get("type", "score")
    if kind == "score":
        score_item(ctx, path, item)
    elif kind == "grad":
        grad_item(ctx, path, item)
    elif kind == "addvec":
        addvec_item(ctx, path, item)
    else:
        raise ValueError(f"unknown item type {kind}")


# ------------------------------------------------------------------ add a task vector to the running server
def addvec_item(ctx, path: Path, spec: dict) -> None:
    """{"type": "addvec", "name": "Dc", "a": dir, "b": dir}: vector a - b into pinned host memory."""
    import threading
    main = sys.modules["__main__"]
    name, a, b = spec["name"], Path(spec["a"]), Path(spec["b"])
    if name in ctx.store.vecs:
        ctx.log.info("vector %s already loaded", name)
        return
    started = time.time()
    ra, rb = main.Reader(a), main.Reader(b)
    arena = main.HostArena(name)
    jobs = [(u, i) for u in ctx.units for i in range(len(u.names))]
    lock = threading.Lock()

    def work(job) -> None:
        u, i = job
        tname = u.names[i]
        size = u.per_numel * 2
        ta, tb = bytearray(size), bytearray(size)
        ra.into(tname, memoryview(ta))
        rb.into(tname, memoryview(tb))
        diff = torch.frombuffer(ta, dtype=torch.bfloat16).float() - torch.frombuffer(tb, dtype=torch.bfloat16).float()
        if bool(diff.any()):
            arena.slot(u)[i].view(-1).copy_(diff.to(torch.bfloat16))
            with lock:
                arena.written[u.key].add(i)

    with ThreadPoolExecutor(64) as pool:
        list(pool.map(work, jobs))
    gib = arena.finish()
    ctx.store.vecs[name] = arena
    ctx.vec_specs[name] = (a, b)
    if name not in ctx.vec_names:
        ctx.vec_names.append(name)
    ctx.log.info("ADDVEC %s = %s - %s: %d/%d units nonzero, %.0f GiB pinned (%.0fs)", name, a, b,
                 len(arena.slots), len(ctx.units), gib, time.time() - started)
