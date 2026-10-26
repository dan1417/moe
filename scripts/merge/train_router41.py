#!/usr/bin/env python3
"""sweep41 plugin: fine-tune only the MoE routers (and optionally the RMSNorm gains) of a merged model.

Why: every weight merge of one lineage lives in the span of the ingredients' differences, and the per-category
diagnostics show the TEXT limit is a trade-off *inside the experts* (the #40->#41 step lifts txt360-qa but hurts
web/planning/general in the same experts). The one lever a static merge cannot pull is per-token: which experts a
token uses and with what weight. That is the router: 45 x [256 x 3072] = 35M parameters (0.03 % of the model),
trained with the merged experts frozen on a diverse pool that mirrors the competition mix.

Mechanics (same machinery as grad41): the recipe is materialized in the server's resident layer-sharded model,
every decoder layer runs under torch.utils.checkpoint, attention is eager (fa4sim has no backward), the routers'
bf16 weights get fp32 master copies updated by AdamW, and the model never leaves eval mode (MiMo-V2 noaux_tc
routing raises in train mode; the gradient reaches the router through the normalized sigmoid top-k weights).

Queue item:
  {"kind": "plugin", "module": "train_router41", "label": "rt_best_a", "recipe": "ck3_d41_025_pmom015",
   "pool": "/data/teutonic/pools/text_train", "steps": 600, "batch": 2, "lr": 1e-5, "val_n": 96,
   "eval_every": 100, "params": "router", "attn": "eager", "priority": -60}
params: "router" (mlp.gate.weight), "router|norm" (+ every *norm.weight), or any regex over tensor names.
Output: /data/teutonic/vec/routers/<label>.safetensors (best-by-val snapshot, bf16),
        a queue item <label>.json = base recipe + replacements (scored like any candidate; the recipe JSON and the
        41.md / 42.md tables then carry it), and configs/merge/k41/routers/<label>.json (training log).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DIR = REPO / "configs/merge/k41"
OUT = Path("/data/teutonic/vec/routers")
PARAM_SETS = {"router": r"\.mlp\.gate\.weight$", "norm": r"norm\.weight$",
              "shared": r"\.mlp\.shared_experts\.(gate|up|down)_proj\.weight$",  # the dense expert every token uses (415M)
              "dense": r"^model\.layers\.0\.mlp\.(gate|up|down)_proj\.weight$",  # layer-0 dense MLP (151M)
              "attn_o": r"\.self_attn\.o_proj\.weight$"}


def _param_regex(spec: str) -> re.Pattern:
    parts = [PARAM_SETS.get(p, p) for p in spec.split("|")]
    return re.compile("|".join(f"(?:{p})" for p in parts))


def run(server, path: Path, raw: dict) -> None:
    import sweep41
    from safetensors.torch import save_file

    log = sweep41.log
    label = raw["label"]
    base_path = DIR / f"{raw['recipe']}.json"
    recipe = server.pe.resolve_recipe(str(base_path))
    server.materialize(recipe)
    model = server.model
    pool = Path(raw.get("pool", "/data/teutonic/pools/text_train"))
    tokens = np.load(pool / "tokens.npy", mmap_mode="r")
    steps, batch, lr = int(raw.get("steps", 600)), int(raw.get("batch", 2)), float(raw.get("lr", 1e-5))
    val_n, eval_every = int(raw.get("val_n", 96)), int(raw.get("eval_every", 100))
    clip = float(raw.get("clip", 1.0))
    accum = int(raw.get("accum", 1))  # micro-batches of `batch` seqs per optimizer step
    rng = np.random.default_rng(int(raw.get("seed", 0)))
    val_idx = np.arange(val_n)
    train_idx = rng.permutation(np.arange(val_n, tokens.shape[0]))
    need = steps * batch * accum
    if need > len(train_idx):
        train_idx = np.concatenate([rng.permutation(train_idx) for _ in range((need + len(train_idx) - 1) // len(train_idx))])
    train_idx = train_idx[:need]

    pat = _param_regex(raw.get("params", "router"))
    params = {n: p for n, p in model.named_parameters() if pat.search(n)}
    if not params:
        raise RuntimeError(f"no parameters match {pat.pattern}")
    already = [n for n, p in model.named_parameters() if p.requires_grad]
    if already:
        raise RuntimeError(f"{len(already)} parameters already require grad (e.g. {already[0]})")
    n_train = sum(p.numel() for p in params.values())
    masters = {n: p.detach().float().clone().requires_grad_(True) for n, p in params.items()}
    opt = torch.optim.AdamW(list(masters.values()), lr=lr, betas=(0.9, 0.98), weight_decay=0.0)
    warmup = max(1, int(raw.get("warmup", 20)))
    decay = str(raw.get("decay", "none"))  # none | cosine (to 10 % of lr at the last step)
    import math
    def lr_mult(s: int) -> float:
        w = min(1.0, (s + 1) / warmup)
        if decay == "cosine" and s >= warmup:
            return w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
        return w
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    attn_before = model.config._attn_implementation
    model.config._attn_implementation = raw.get("attn", "eager")
    import torch.utils.checkpoint as cp
    orig_forward = {}
    for layer in model.model.layers:
        orig_forward[layer] = layer.forward

        def make(f):
            def wrapped(*args, **kwargs):
                return cp.checkpoint(f, *args, use_reentrant=False, **kwargs)
            return wrapped
        layer.forward = make(layer.forward)
    head_dev = torch.device(f"cuda:{server.layer_devices[-1]}")
    in_dev = f"cuda:{server.layer_devices[0]}"

    def batch_loss(idx: np.ndarray) -> torch.Tensor:
        ids = torch.tensor(np.asarray(tokens[np.sort(idx)], dtype=np.int64), device=in_dev)
        hidden = model.model(ids, use_cache=False).last_hidden_state.to(head_dev)
        labels = ids.to(head_dev)[:, 1:]
        n_pos = hidden.shape[1] - 1
        total = None
        for s in range(0, n_pos, 1024):
            e = min(s + 1024, n_pos)
            logits = model.lm_head(hidden[:, s:e, :])
            part = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels[:, s:e].reshape(-1), reduction="sum")
            total = part if total is None else total + part
            del logits
        return total / (n_pos * ids.shape[0])

    def evaluate() -> float:
        with torch.no_grad():
            vals = [float(batch_loss(val_idx[i:i + batch])) for i in range(0, val_n, batch)]
        return float(np.mean(vals))

    log.info("train_router41 %s: base %s, %d tensors / %.1fM params (%s), %d steps x %d seqs, lr %g, val %d, attn %s",
             label, raw["recipe"], len(params), n_train / 1e6, pat.pattern, steps, batch, lr, val_n,
             model.config._attn_implementation)
    history, best = [], None
    t0 = time.time()
    try:
        for p in params.values():
            p.requires_grad_(True)
        val0 = evaluate()
        best = (val0, 0, {n: m.detach().cpu().clone() for n, m in masters.items()})
        history.append({"step": 0, "val": val0})
        log.info("  step 0: val %.5f (%.0fs)", val0, time.time() - t0)
        run_loss, run_n = 0.0, 0
        for step in range(1, steps + 1):
            for a in range(accum):
                lo = ((step - 1) * accum + a) * batch
                loss = batch_loss(train_idx[lo: lo + batch]) / accum
                loss.backward()
                run_loss += float(loss.detach()) * accum
                del loss
            run_n += accum
            gn2 = 0.0
            for n, p in params.items():
                g = p.grad
                masters[n].grad = None if g is None else g.float()
                p.grad = None
                if g is not None:
                    gn2 += float(g.float().pow(2).sum())
            gnorm = gn2 ** 0.5
            if clip > 0 and gnorm > clip:
                for m in masters.values():
                    if m.grad is not None:
                        m.grad.mul_(clip / (gnorm + 1e-12))
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for n, p in params.items():
                    p.data.copy_(masters[n].detach().to(p.dtype))
            if step % 10 == 0 or step == 1:
                log.info("  step %d/%d: train %.5f (last %d), |g| %.4f, %.1fs/step", step, steps, run_loss / run_n, run_n,
                         gnorm, (time.time() - t0) / step)
            if step % eval_every == 0 or step == steps:
                val = evaluate()
                history.append({"step": step, "train": run_loss / max(run_n, 1), "val": val})
                log.info("  step %d: train %.5f val %.5f (best %.5f @%d, %.0fs)", step, run_loss / max(run_n, 1), val,
                         best[0], best[1], time.time() - t0)
                run_loss, run_n = 0.0, 0
                if val < best[0]:
                    best = (val, step, {n: m.detach().cpu().clone() for n, m in masters.items()})
    finally:
        for p in params.values():
            p.requires_grad_(False)
            p.grad = None
        model.config._attn_implementation = attn_before
        for layer, f in orig_forward.items():
            layer.forward = f
        for n in params:  # the resident tensors now hold trained values: force a refill before the next candidate
            server.loaded_sig.pop(n, None)
        for gpu in sorted(set(server.layer_devices)):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
    if best is None:
        return
    if best[1] == 0:
        log.info("ROUTERFT %s: no step improved the held-out CE (val0 %.5f; history %s) - nothing to score", label, val0,
                 [(h['step'], round(h['val'], 5)) for h in history])
        (DIR / "routers").mkdir(exist_ok=True)
        (DIR / "routers" / f"{label}.json").write_text(json.dumps({"label": label, "item": raw, "history": history,
                                                                  "best_step": 0, "val0": val0, "seconds": round(time.time() - t0)}, indent=1))
        sweep41.git_commit([DIR / "routers" / f"{label}.json"], f"k41 router-ft: {label} (no gain)")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{label}.safetensors"
    save_file({n: t.to(torch.bfloat16).contiguous() for n, t in best[2].items()}, str(out_path),
              metadata={"base": raw["recipe"], "best_step": str(best[1]), "val": f"{best[0]:.6f}"})
    base = json.loads(base_path.read_text())
    item = {k: v for k, v in base.items() if k not in ("priority", "time", "model_id", "result", "kings", "groups_vs", "sample", "n")}
    item.update({"label": label, "method": "router-ft",
                 "description": f"ROUTER FINE-TUNE of {raw['recipe']}: {pat.pattern} ({n_train / 1e6:.1f}M params) trained "
                                f"{best[1]} steps x {batch * accum} seqs on {pool.name} (lr {lr:g}); val CE {val0:.5f} -> {best[0]:.5f}",
                 "replacements": {**base.get("replacements", {}), **{n: str(out_path) for n in params}},
                 "priority": float(raw.get("score_priority", -60)), "main_limit": raw.get("main_limit", 800)})
    (server.args.queue / f"{label}__score.json").write_text(json.dumps(item) + "\n")  # not <label>.json: the server moves the plugin item of that name to done/ after run()
    (DIR / "routers").mkdir(exist_ok=True)
    (DIR / "routers" / f"{label}.json").write_text(json.dumps({"label": label, "item": raw, "history": history,
                                                              "best_step": best[1], "best_val": best[0], "val0": val0,
                                                              "seconds": round(time.time() - t0), "out": str(out_path)}, indent=1))
    log.info("ROUTERFT %s: val CE %.5f -> %.5f at step %d (%.0fs); queued %s for scoring", label, val0, best[0], best[1],
             time.time() - t0, label)
    sweep41.git_commit([DIR / "routers" / f"{label}.json"], f"k41 router-ft: {label}")
