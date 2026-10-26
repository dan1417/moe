#!/usr/bin/env python3
"""sweep41 plugin: first-order gradient probe for learned per-tensor / per-expert merge coefficients (AdaMerging).

At the merge point W (a scored recipe, materialized in the server's resident layer-sharded model) compute, for every
probed tensor t and every task vector V (an int8/bf16 delta checkpoint),

    g_t = < dL/dW_t , V_t >      L = mean per-sequence CE over `n` sequences of a held-out sample (not the screen)

so that changing tensor t's coefficient of V by delta_t moves mu_hat by about -g_t * delta_t. The full gradient never
exists: each weight gets a post-accumulate hook that dots its gradient against the staged V slice and frees it.
Routed experts are handled on their stacked [E, out, in] buffers (that is what the grouped MoE forward uses), with the
dot taken per expert. V slices are staged on the layer's GPU as int8 + fp32 row scales; only `layers_per_pass` layers
per GPU fit next to the activations, so the probe makes several passes over the same sequences.

Queue item:
  {"kind": "plugin", "module": "grad41", "label": "G_pe015_d36", "recipe": "t1_d36050_pe015",
   "vecs": {"D36": "/data/teutonic/vec/d36_35q"}, "sample": "tag:B", "n": 384, "chunk": 64,
   "layers_per_pass": 3, "only": "\\.mlp\\.experts\\.", "attn": "fa4sim", "priority": -20}
Output: configs/merge/k41/probes/<label>.json.gz with {"at", "vecs", "n", "chunk", "dots": {v: {tensor: g}},
         "dots_chunks": [{v: {tensor: mean g over the chunk}}, ...], "vnorm2": {v: {tensor: |V_t|^2}}, "loss"}.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
PROBES = REPO / "configs/merge/k41/probes"
EXPERT = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
EXPERT_CHUNK = 32  # experts per dequantize/dot step (bounds the fp32 temporaries)


def _layer_of(name: str) -> int:
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1


def _read_int8(handle, sl) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Raw int8 rows (+ fp32 scales) of one tensor, or the bf16 tensor when the delta is not quantized."""
    if not getattr(sl, "int8", False):
        return sl[:], None
    buf = bytearray(sl.end - sl.begin)
    view, done = memoryview(buf), 0
    while done < len(buf):
        got = os.preadv(handle.fd, [view[done:]], handle.data_start + sl.begin + done)
        if got <= 0:
            raise IOError("short read")
        done += got
    return torch.frombuffer(buf, dtype=torch.int8).view(*sl.shape), handle.scales(sl.name)


class VecStore:
    """One task vector: per-tensor slices (dense weights) and stacked per-(layer, proj) expert slices on the GPUs."""

    def __init__(self, path: Path):
        from src.parity.weights import Recipe, _Reader

        self.reader = _Reader(Recipe.single(path))
        self.names = set(self.reader.maps[0])
        self.dense: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {}
        self.experts: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor | None]] = {}  # (prefix, proj) -> [E,..]

    def stage(self, names: list[str], device_of) -> int:
        groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
        total = 0
        for name in names:
            if name not in self.names:
                continue
            m = EXPERT.match(name)
            if m:
                groups.setdefault((m.group(1), m.group(4)), []).append((int(m.group(3)), name))
                continue
            q, scale = _read_int8(self.reader.handle(0, name), self.reader.handle(0, name).get_slice(name))
            dev = device_of(name)
            self.dense[name] = (q.to(dev), None if scale is None else scale.to(dev))
            total += q.numel() * q.element_size()
        for key, members in groups.items():
            members.sort()
            qs, scales = [], []
            for _, name in members:
                q, scale = _read_int8(self.reader.handle(0, name), self.reader.handle(0, name).get_slice(name))
                qs.append(q)
                scales.append(scale)
            dev = device_of(members[0][1])
            q = torch.stack(qs).to(dev)
            sc = None if scales[0] is None else torch.stack(scales).to(dev)
            self.experts[key] = (q, sc)
            total += q.numel() * q.element_size()
        return total

    def dense_vec(self, name: str) -> torch.Tensor | None:
        item = self.dense.get(name)
        if item is None:
            return None
        q, scale = item
        return q.float() if scale is None else q.float().mul_(scale.view(-1, *([1] * (q.dim() - 1))))

    def expert_dots(self, key: tuple[str, str], grad: torch.Tensor) -> torch.Tensor | None:
        """per-expert <grad[e], V[e]> for a stacked gradient [E, out, in]."""
        item = self.experts.get(key)
        if item is None:
            return None
        q, scale = item
        out = torch.empty(q.shape[0], dtype=torch.float64, device=grad.device)
        for s in range(0, q.shape[0], EXPERT_CHUNK):
            e = min(s + EXPERT_CHUNK, q.shape[0])
            v = q[s:e].float()
            if scale is not None:
                v.mul_(scale[s:e].unsqueeze(-1))
            out[s:e] = (grad[s:e].float() * v).sum(dim=(1, 2)).double()
        return out

    def expert_norm2(self, key: tuple[str, str]) -> torch.Tensor | None:
        item = self.experts.get(key)
        if item is None:
            return None
        q, scale = item
        out = torch.empty(q.shape[0], dtype=torch.float64, device=q.device)
        for s in range(0, q.shape[0], EXPERT_CHUNK):
            e = min(s + EXPERT_CHUNK, q.shape[0])
            v = q[s:e].float()
            if scale is not None:
                v.mul_(scale[s:e].unsqueeze(-1))
            out[s:e] = v.pow(2).sum(dim=(1, 2)).double()
        return out

    def clear(self) -> None:
        self.dense.clear()
        self.experts.clear()


def run(server, path: Path, raw: dict) -> None:
    import sweep41
    from src.parity.sample import load_live_config

    log = sweep41.log
    label = raw["label"]
    recipe = server.pe.resolve_recipe(str(REPO / "configs/merge/k41" / f"{raw['recipe']}.json"))
    server.materialize(recipe)
    model = server.model
    live = load_live_config(competition=raw.get("competition", "mix"))
    sample = server.pe.resolve_sample(raw.get("sample", "tag:B"), live)
    n, chunk = int(raw.get("n", 384)), int(raw.get("chunk", 64))
    only = re.compile(raw["only"]) if raw.get("only") else None
    vecs = {k: VecStore(Path(v)) for k, v in raw["vecs"].items()}
    all_names = sorted(server.placement)
    probed = [nm for nm in all_names if (only is None or only.search(nm)) and any(nm in v.names for v in vecs.values())]
    layers = sorted({_layer_of(nm) for nm in probed})
    per_gpu: dict[int, list[int]] = {}
    for li in layers:
        dev = server.layer_devices[li] if li >= 0 else server.layer_devices[-1]
        per_gpu.setdefault(dev, []).append(li)
    lpp = int(raw.get("layers_per_pass", 3))
    passes = max((len(v) + lpp - 1) // lpp for v in per_gpu.values())
    pass_layers = [sorted(li for ls in per_gpu.values() for li in ls[k * lpp:(k + 1) * lpp]) for k in range(passes)]
    # stacked expert buffers (what the grouped MoE forward actually uses) and plain parameters
    stacked: dict[tuple[str, str], torch.Tensor] = {}
    for mod_name, module in model.named_modules():
        if getattr(module, "_parity_stacked", None):
            for proj, buf in module._parity_stacked.items():
                stacked[(f"{mod_name}.experts", proj)] = buf
    params = dict(model.named_parameters())
    attn_before = model.config._attn_implementation
    if raw.get("attn"):
        model.config._attn_implementation = raw["attn"]
    # Gradient checkpointing per decoder layer: a 45-layer forward under autograd keeps ~45 GB of activations across
    # the GPUs; checkpointing keeps only each layer's input and recomputes the layer during backward.
    import torch.utils.checkpoint as cp
    orig_forward = {}
    for layer in model.model.layers:
        orig_forward[layer] = layer.forward

        def make(f):
            def wrapped(*args, **kwargs):
                return cp.checkpoint(f, *args, use_reentrant=False, **kwargs)
            return wrapped
        layer.forward = make(layer.forward)
    log.info("grad41 %s: point %s, %d probed tensors, %d layers, %d passes x %d seqs, attn %s", label, raw["recipe"],
             len(probed), len(layers), passes, n, model.config._attn_implementation)

    dots = {v: {} for v in vecs}
    vnorm2 = {v: {} for v in vecs}
    chunks: list[dict] = [{v: {} for v in vecs} for _ in range((n + chunk - 1) // chunk)]
    losses = np.full(n, np.nan)
    head_dev = torch.device(f"cuda:{server.layer_devices[-1]}")
    in_dev = f"cuda:{server.layer_devices[0]}"
    t_start = time.time()
    try:
        for k, lset in enumerate(pass_layers):
            names = [nm for nm in probed if _layer_of(nm) in lset]
            dense_names = [nm for nm in names if not EXPERT.match(nm)]
            expert_keys = sorted({(m.group(1), m.group(4)) for nm in names for m in [EXPERT.match(nm)] if m})
            staged = sum(v.stage(names, lambda nm: server.placement[nm].device) for v in vecs.values())
            for v, store in vecs.items():
                for nm in dense_names:
                    vt = store.dense_vec(nm)
                    if vt is not None:
                        vnorm2[v][nm] = float(vt.pow(2).sum())
                for key in expert_keys:
                    nrm = store.expert_norm2(key)
                    if nrm is not None:
                        for e, val in enumerate(nrm.tolist()):
                            vnorm2[v][f"{key[0]}.{e}.{key[1]}.weight"] = val
            log.info("  pass %d/%d: %d dense + %d expert groups, %.1f GB staged", k + 1, passes, len(dense_names),
                     len(expert_keys), staged / 1e9)
            acc = {v: {} for v in vecs}
            cur = {v: {} for v in vecs}
            handles, leaves = [], []

            def add(v, nm, val):
                acc[v][nm] = acc[v].get(nm, 0.0) + val
                cur[v][nm] = cur[v].get(nm, 0.0) + val

            for nm in dense_names:
                p = params[nm]
                p.requires_grad_(True)
                leaves.append(p)

                def dhook(param, nm=nm):
                    g = param.grad
                    if g is not None:
                        gf = g.float()
                        for v, store in vecs.items():
                            vt = store.dense_vec(nm)
                            if vt is not None:
                                add(v, nm, float((gf * vt).sum()))
                    param.grad = None

                handles.append(p.register_post_accumulate_grad_hook(dhook))
            for key in expert_keys:
                buf = stacked[key]
                buf.requires_grad_(True)
                leaves.append(buf)

                def ehook(t, key=key):
                    g = t.grad
                    if g is not None:
                        for v, store in vecs.items():
                            d = store.expert_dots(key, g)
                            if d is not None:
                                for e, val in enumerate(d.tolist()):
                                    add(v, f"{key[0]}.{e}.{key[1]}.weight", val)
                    t.grad = None

                handles.append(buf.register_post_accumulate_grad_hook(ehook))
            try:
                for i in range(n):
                    ids = torch.tensor(np.asarray(sample.tokens[i], dtype=np.int64)[None], device=in_dev)
                    hidden = model.model(ids, use_cache=False).last_hidden_state.to(head_dev)
                    labels = ids.to(head_dev)[:, 1:]
                    n_pos = hidden.shape[1] - 1
                    loss = None
                    for s in range(0, n_pos, 1024):
                        e = min(s + 1024, n_pos)
                        logits = model.lm_head(hidden[:, s:e, :])
                        part = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels[:, s:e].reshape(-1), reduction="sum")
                        loss = part if loss is None else loss + part
                        del logits
                    loss = loss / n_pos
                    (loss / n).backward()  # accumulates the mean-CE gradient over the n probe sequences
                    if k == 0:
                        losses[i] = float(loss)
                    del hidden, loss
                    if (i + 1) % chunk == 0 or i + 1 == n:
                        ci = i // chunk
                        for v in vecs:
                            for nm, val in cur[v].items():
                                chunks[ci][v][nm] = val * n / chunk  # mean per-seq gradient over this chunk
                            cur[v] = {}
                    if (i + 1) % 32 == 0:
                        log.info("    pass %d seq %d/%d (%.0fs)", k + 1, i + 1, n, time.time() - t_start)
            finally:
                for h in handles:
                    h.remove()
                for t in leaves:
                    t.requires_grad_(False)
                    t.grad = None
                for v in vecs.values():
                    v.clear()
                for gpu in sorted(set(server.layer_devices)):
                    with torch.cuda.device(gpu):
                        torch.cuda.empty_cache()
            for v in vecs:
                dots[v].update(acc[v])
    finally:
        model.config._attn_implementation = attn_before
        for layer, f in orig_forward.items():
            layer.forward = f
        for gpu in sorted(set(server.layer_devices)):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
    PROBES.mkdir(parents=True, exist_ok=True)
    out = {"label": label, "at": raw["recipe"], "vecs": list(vecs), "n": n, "chunk": chunk, "sample": sample.sample_id,
           "attn": raw.get("attn") or attn_before, "loss": float(np.nanmean(losses)), "dots": dots, "dots_chunks": chunks,
           "vnorm2": vnorm2, "seconds": round(time.time() - t_start), "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with gzip.open(PROBES / f"{label}.json.gz", "wt") as f:
        json.dump(out, f)
    for v in vecs:
        tot = sum(dots[v].values())
        top = sorted(dots[v].items(), key=lambda kv: -abs(kv[1]))[:3]
        log.info("GRAD %s %s: mean CE %.5f, dL/dc(%s) = %+.6f over %d tensors (%.0fs); top |g|: %s", label, v, out["loss"], v,
                 tot, len(dots[v]), out["seconds"], [(kk[:52], round(x, 6)) for kk, x in top])
    sweep41.git_commit([PROBES / f"{label}.json.gz"], f"k41 probe: {label}")
