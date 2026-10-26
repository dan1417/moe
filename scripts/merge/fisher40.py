#!/usr/bin/env python3
"""Per-micro-batch gradient projections on task vectors (gradient + Fisher in merge-coefficient space).

Same layer-streamed forward/backward as grad40.py, but instead of accumulating the full gradient the GPU holds
the task vectors V_v = K_a - K_b (bf16 for one vector, fp8-e4m3 with per-row scales for two or three) and, after
every micro-batch's backward, records d[m, v, t] = <grad L_m, V_{v,t}> for every tensor t (each routed expert
projection separately). From D = d[m] over M micro-batches of `micro` sequences:

  g      = mean_m d[m]                         first-order slope of the eval CE in every coefficient
  F      ~ (micro * T) * Cov_m(d[m])           empirical Fisher ~ Hessian at a trained optimum (T = 2047 tokens)

which grad_newton40.py turns into damped Newton (natural-gradient) coefficient steps. Validated on two tensors:
T * Var_seq(<grad, V>) matched the curvature measured by the sweep (head 0.041 vs 0.052, L44 shared 0.033 vs 0.027).

  python scripts/merge/fisher40.py --label F40_a --n 1024 --vecs D40=40-39,P39_36=39-36
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "scripts/merge"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import grad40 as g40  # noqa: E402

log = logging.getLogger("fisher40")
FP8 = torch.float8_e4m3fn


class VecStore:
    """Task vectors on the GPU, keyed like grad40's G: tensor name or (layer, proj) for expert stacks."""

    def __init__(self, keys: list, shapes: dict, vecs: dict, device, quant: bool, threads: int = 16):
        from concurrent.futures import ThreadPoolExecutor

        from src.parity.weights import Recipe, _Reader
        self.vecs, self.quant, self.device = list(vecs), quant, device
        self.data: dict = {v: {} for v in vecs}
        self.scale: dict = {v: {} for v in vecs}
        for v, (a, b) in vecs.items():
            reader = _Reader(Recipe.parse(g40.vec_recipe_spec(a, b)))
            started = time.time()

            def names_of(key):
                if isinstance(key, tuple):
                    layer, proj = key
                    return [f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight" for e in range(shapes[key][0])]
                return [key]

            def work(key):
                rows = []
                for name in names_of(key):
                    ta = reader.handle(0, name).get_tensor(name).to(device).float()
                    if b is not None:
                        ta.sub_(reader.handle(1, name).get_tensor(name).to(device).float())
                    rows.append(ta)
                d = torch.stack(rows) if isinstance(key, tuple) else rows[0]
                if quant:
                    flat = d.reshape(-1, d.shape[-1]) if d.dim() > 1 else d.reshape(1, -1)
                    s = flat.abs().amax(dim=1, keepdim=True).clamp_min(1e-30) / 448.0
                    q = (flat / s).to(FP8)
                    return key, q.view(d.shape) if d.dim() > 1 else q.view(-1), s.squeeze(1)
                return key, d.to(torch.bfloat16), None

            with ThreadPoolExecutor(threads) as pool:
                for key, q, s in pool.map(work, keys):
                    self.data[v][key] = q
                    if s is not None:
                        self.scale[v][key] = s
            torch.cuda.synchronize(device)
            log.info("vector %s = %s - %s on GPU (%s) in %.0fs; GPU used %.1f GiB", v, a, b,
                     "fp8+row-scale" if quant else "bf16", time.time() - started, torch.cuda.memory_allocated(device) / 2**30)

    def get(self, v: str, key) -> torch.Tensor:
        q = self.data[v][key]
        if not self.quant:
            return q
        s = self.scale[v][key]
        if q.dim() == 1:
            return q.float() * s
        flat = q.reshape(-1, q.shape[-1]).float() * s.unsqueeze(1)
        return flat.view(q.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--at", default=g40.K40)
    ap.add_argument("--sample", default="tag:G")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pass-seqs", type=int, default=32)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--vecs", default="D40=40-39")
    ap.add_argument("--out", type=Path, default=Path("/root/teutonic_grad"))
    ap.add_argument("--io-threads", type=int, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    assert args.pass_seqs % args.micro == 0 and args.n % args.pass_seqs == 0

    import parity_eval as pe
    from src.parity.sample import load_live_config
    from src.parity.weights import build_skeleton

    device = torch.device("cuda:0")
    live = load_live_config()
    sample = pe.resolve_sample(args.sample, live)
    recipe = pe.resolve_recipe(args.at)
    model = build_skeleton(Path(g40.K40), "flash_attention_4")
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = "flash_attention_4"
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    lazy_import_flash_attention("flash_attention_4")
    for name, buffer in model.named_buffers():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        owner._buffers[name.rsplit(".", 1)[1]] = buffer.to(device)
    st = g40.Streamer(model, recipe, device, threads=args.io_threads)
    st.load("model.embed_tokens", model.model.embed_tokens)
    st.load("model.norm", model.model.norm)
    st.load("lm_head", model.lm_head)
    layers = model.model.layers
    L = len(layers)

    # tensor keys (same convention as grad40's G) and their column in the record
    keys, shapes = [], {}
    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters(prefix=f"model.layers.{i}"):
            if g40.EXPERT.match(name) or g40.NO_GRAD.search(name):
                continue
            keys.append(name)
            shapes[name] = tuple(p.shape)
        if layer.mlp.__class__.__name__ == "MiMoV2MoE":
            for proj in g40.PROJS:
                shape = getattr(layer.mlp.experts[0], proj).weight.shape
                keys.append((i, proj))
                shapes[(i, proj)] = (len(layer.mlp.experts), *shape)
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        keys.append(name)
        shapes[name] = tuple(model.get_parameter(name).shape)
    columns, col = {}, 0
    names = []
    for key in keys:
        if isinstance(key, tuple):
            columns[key] = col
            names += [f"model.layers.{key[0]}.mlp.experts.{e}.{key[1]}.weight" for e in range(shapes[key][0])]
            col += shapes[key][0]
        else:
            columns[key] = col
            names.append(key)
            col += 1
    vecs = g40.parse_vecs(args.vecs)
    store = VecStore(keys, shapes, vecs, device, quant=len(vecs) > 1)
    n_micro = args.n // args.micro
    record = torch.zeros((n_micro, len(vecs), col), dtype=torch.float64, device=device)
    vlist = list(vecs)

    def project_leaf(m: int, key, grad: torch.Tensor) -> None:
        c = columns[key]
        for j, v in enumerate(vlist):
            if isinstance(key, tuple):  # expert stack: 16 experts at a time in fp32
                for e0 in range(0, grad.shape[0], 16):
                    e1 = min(grad.shape[0], e0 + 16)
                    q = store.data[v][key][e0:e1]
                    if store.quant:  # per-row scales of the whole stack: (E * out,) -> (E, out)[e0:e1]
                        V = q.float() * store.scale[v][key].view(grad.shape[0], -1)[e0:e1, :, None]
                    else:
                        V = q.float()
                    record[m, j, c + e0:c + e1] += (grad[e0:e1].float() * V).sum(dim=(1, 2)).double()
            else:
                record[m, j, c] += (grad.float() * store.get(v, key).float()).sum().double()

    captured = {}
    originals = [layer.forward for layer in layers]

    def capture(layer):
        def fwd(hidden_states, **kw):
            captured.setdefault(layer.attention_type, kw)
            return hidden_states
        return fwd
    for layer in layers:
        layer.forward = capture(layer)
    probe_ids = torch.as_tensor(np.asarray(sample.tokens[args.offset:args.offset + args.micro], dtype=np.int64), device=device)
    with torch.no_grad():
        model.model(input_ids=probe_ids, use_cache=False)
    for layer, f in zip(layers, originals):
        layer.forward = f

    head_norm, head = model.model.norm, model.lm_head
    t0 = time.time()
    losses = []
    n_pass = args.n // args.pass_seqs
    per_pass = args.pass_seqs // args.micro
    for p in range(n_pass):
        tp = time.time()
        start = args.offset + p * args.pass_seqs
        ids_all = torch.as_tensor(np.asarray(sample.tokens[start:start + args.pass_seqs], dtype=np.int64), device=device)
        micro_ids = list(ids_all.split(args.micro))
        mbase = p * per_pass
        H = []
        with torch.no_grad():
            H.append([model.model.embed_tokens(ids) for ids in micro_ids])
            for i, layer in enumerate(layers):
                st.get(f"model.layers.{i}", layer)
                if i + 1 < L:
                    st.prefetch(f"model.layers.{i + 1}", layers[i + 1])
                kw = captured[layer.attention_type]
                H.append([layer(h, **kw) for h in H[i]])
                if i < L - 1 and i != 0:
                    st.release(f"model.layers.{i}", layer)
        head_norm.weight.requires_grad_(True)
        head.weight.requires_grad_(True)
        grads = []
        for m, (h, ids) in enumerate(zip(H[L], micro_ids)):
            h = h.detach().requires_grad_(True)
            per_seq = g40.seq_ce_sum(head_norm(h), head, ids)
            per_seq.mean().backward()  # micro-batch mean loss: d[m] = <grad L_m, V>
            losses.extend(per_seq.detach().cpu().tolist())
            grads.append(h.grad)
            project_leaf(mbase + m, "model.norm.weight", head_norm.weight.grad)
            project_leaf(mbase + m, "lm_head.weight", head.weight.grad)
            head_norm.weight.grad = None
            head.weight.grad = None
        head_norm.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        H.pop()
        for i in reversed(range(L)):
            layer = layers[i]
            st.get(f"model.layers.{i}", layer)
            if i > 0:
                st.prefetch(f"model.layers.{i - 1}", layers[i - 1])
            leaves = g40.set_grad(f"model.layers.{i}", layer, True)
            kw = captured[layer.attention_type]
            for m in range(len(micro_ids)):
                h = H[i][m].detach().requires_grad_(True)
                out = layer(h, **kw)
                out.backward(grads[m])
                grads[m] = h.grad
                for name, leaf in leaves:
                    if leaf.grad is None:
                        continue
                    key = (i, name.rsplit(".", 1)[1]) if ".experts.*." in name else name
                    project_leaf(mbase + m, key, leaf.grad)
                    leaf.grad = None
            g40.set_grad(f"model.layers.{i}", layer, False)
            H.pop()
            if i != 0:
                st.release(f"model.layers.{i}", layer)
        # embedding: <grad_emb, V_emb> = sum over tokens of <dL/de_token, V_emb[id]>
        for m, (ids, gr) in enumerate(zip(micro_ids, grads)):
            for j, v in enumerate(vlist):
                Ve = store.get(v, "model.embed_tokens.weight")
                record[mbase + m, j, columns["model.embed_tokens.weight"]] += \
                    (gr.reshape(-1, gr.shape[-1]).float() * Ve[ids.reshape(-1)].float()).sum().double()
        del grads
        log.info("pass %d/%d | mean CE %.5f | %.0fs (elapsed %.0fs) peak %.1f GiB", p + 1, n_pass, float(np.mean(losses)),
                 time.time() - tp, time.time() - t0, torch.cuda.max_memory_allocated(device) / 2**30)
    D = record.cpu().numpy()
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / f"{args.label}.npz", D=D.astype(np.float32), names=np.array(names),
                        vecs=np.array(vlist), pairs=np.array([[str(x) for x in vecs[v]] for v in vlist]),
                        micro=args.micro, n=args.n, sample=args.sample, at=str(args.at), mean_ce=float(np.mean(losses)))
    g = D.mean(axis=0)
    for j, v in enumerate(vlist):
        log.info("FISHER %s %s: uniform slope dL/dc %+.4e (se %.1e)", args.label, v, g[j].sum(),
                 D[:, j, :].sum(axis=1).std(ddof=1) / np.sqrt(D.shape[0]))
    log.info("FISHER %s done in %.0fs -> %s", args.label, time.time() - t0, args.out / f"{args.label}.npz")


if __name__ == "__main__":
    main()
