#!/usr/bin/env python3
"""Full-gradient probe on ONE GPU by layer streaming (for learned merge coefficients).

G = d/dW  mean_seq CE_seq(W)  at a merge point W (default king #40), over sequences of a probe sample that
is disjoint from the screening sample. The GPU holds G (experts bf16, everything else fp32, ~213 GiB), the
embedding / norm / lm_head, two decoder layers at a time and the layer inputs of one pass:

  per pass of P sequences:  forward  layer 0..44 (weights streamed from the recipe, inputs kept)
                            loss + head grads (validator objective: mean over sequences of mean token CE)
                            backward layer 44..0 (weights streamed again, recompute, accumulate into G)

Afterwards g_{t,v} = <G_t, V_t> for task vectors V (differences of checkpoints, streamed from disk) gives the
first-order effect of every per-tensor / per-expert merge coefficient; G itself can be saved for later vectors.
Halfway through, projections are taken once more so that split-half noise can be estimated.

  python scripts/merge/grad40.py --label G40_a --n 1024 --vecs D40=40-39,D36=36-35,P40_36=40-36
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))
sys.path.insert(0, str(REPO / "scripts/merge"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

log = logging.getLogger("grad40")
K40 = "/opt/teutonic/models/7deb3bb45ea1bfaf0f6da7be5412e0bb303f0ba047d9fb33abd2afb08007c36c"
EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
NO_GRAD = re.compile(r"(\.mlp\.gate\.(weight|e_score_correction_bias)|attention_sink_bias)$")
PROJS = ("gate_proj", "up_proj", "down_proj")


class Streamer:
    """Materializes modules of a meta skeleton on the GPU from a recipe, one subtree at a time."""

    def __init__(self, model, recipe, device, threads: int = 24):
        from src.parity.weights import _Reader
        self.model, self.recipe, self.device = model, recipe, device
        self.reader = _Reader(recipe)
        self.pool = ThreadPoolExecutor(threads)
        self.prefetch_pool = ThreadPoolExecutor(1)
        self.pending: dict[str, object] = {}
        self.resident: set[str] = set()

    def alloc(self, prefix: str, root: nn.Module) -> dict[str, torch.Tensor]:
        from src.parity.moe import grouped_moe
        placement = {}
        for name, module in root.named_modules(prefix=prefix):
            if ".experts." in name or name.endswith(".experts"):
                continue
            for pname, p in list(module.named_parameters(recurse=False)):
                t = torch.empty(p.shape, dtype=p.dtype, device=self.device)
                module._parameters[pname] = nn.Parameter(t, requires_grad=False)
                placement[f"{name}.{pname}"] = t
            if module.__class__.__name__ == "MiMoV2MoE":
                stacked = {}
                for proj in PROJS:
                    shape = getattr(module.experts[0], proj).weight.shape
                    buf = torch.empty((len(module.experts), *shape), dtype=torch.bfloat16, device=self.device)
                    for idx, ex in enumerate(module.experts):
                        getattr(ex, proj)._parameters["weight"] = nn.Parameter(buf[idx], requires_grad=False)
                        placement[f"{name}.experts.{idx}.{proj}.weight"] = buf[idx]
                    stacked[proj] = buf
                module._parity_stacked = stacked
                module.moe = grouped_moe.__get__(module)
        return placement

    def fill(self, placement: dict[str, torch.Tensor]) -> None:
        from src.parity.weights import _fill

        def work(name):
            with torch.no_grad():
                _fill(placement[name], name, self.reader, self.recipe)
        list(self.pool.map(work, sorted(placement)))
        torch.cuda.synchronize(self.device)

    def load(self, prefix: str, root: nn.Module) -> nn.Module:
        placement = self.alloc(prefix, root)
        self.fill(placement)
        self.resident.add(prefix)
        return root

    def prefetch(self, prefix: str, root: nn.Module) -> None:
        if prefix not in self.pending and prefix not in self.resident:
            self.pending[prefix] = self.prefetch_pool.submit(self.load, prefix, root)

    def get(self, prefix: str, root: nn.Module) -> nn.Module:
        if prefix in self.pending:
            self.pending.pop(prefix).result()
        if prefix not in self.resident:
            self.load(prefix, root)
        return root

    def release(self, prefix: str, root: nn.Module) -> None:
        self.resident.discard(prefix)
        for _, module in root.named_modules():
            for pname, p in list(module.named_parameters(recurse=False)):
                module._parameters[pname] = nn.Parameter(torch.empty(p.shape, dtype=p.dtype, device="meta"),
                                                         requires_grad=False)
            if hasattr(module, "_parity_stacked"):
                del module._parity_stacked


def set_grad(prefix: str, root: nn.Module, on: bool) -> list[tuple[str, torch.Tensor]]:
    """Enable grads on a materialized layer; returns (name, leaf) pairs whose .grad is accumulated."""
    leaves = []
    for name, module in root.named_modules(prefix=prefix):
        if ".experts." in name or name.endswith(".experts"):
            continue
        if hasattr(module, "_parity_stacked"):
            for proj, buf in module._parity_stacked.items():
                buf.requires_grad_(on)
                leaves.append((f"{name}.experts.*.{proj}", buf))
        for pname, p in module.named_parameters(recurse=False):
            full = f"{name}.{pname}"
            if NO_GRAD.search(full):
                continue
            p.requires_grad_(on)
            leaves.append((full, p))
    return leaves


def seq_ce_sum(hidden: torch.Tensor, head: nn.Module, ids: torch.Tensor, chunk: int = 512) -> torch.Tensor:
    """Per-sequence mean next-token CE (bf16 logits upcast to fp32, as in the validator), shape [B]."""
    labels = ids[:, 1:]
    t = labels.shape[1]
    total = torch.zeros(ids.shape[0], device=hidden.device, dtype=torch.float32)
    for s in range(0, t, chunk):
        e = min(t, s + chunk)
        logits = head(hidden[:, s:e]).float()
        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels[:, s:e].reshape(-1),
                                         reduction="none").view(ids.shape[0], -1)
        total = total + ce.sum(dim=1)
    return total / t


def parse_vecs(spec: str) -> dict[str, tuple]:
    """NAME=a-b (king a minus king b) or NAME=@/path (a task-vector checkpoint, e.g. remote_delta40 output)."""
    out = {}
    for item in filter(None, spec.split(",")):
        name, pair = item.split("=", 1)
        if pair.startswith("@"):
            out[name] = (pair[1:], None)
            continue
        a, b = pair.split("-")
        out[name] = (int(a), int(b))
    return out


def vec_recipe_spec(a, b) -> str:
    return f"1:{a}" if b is None else f"1:{king_path(a)},-1:{king_path(b)}"


def king_path(n: int) -> str:
    import queue40
    return queue40.king_path(n)


def project(G: dict, vecs: dict[str, tuple[int, int]], device, threads: int = 16) -> tuple[dict, dict]:
    """dots[v][tensor] = <G_t, (K_a - K_b)_t> and vnorm2[v][tensor] = |V_t|^2 (per routed expert projection)."""
    from src.parity.weights import Recipe, _Reader
    dots = {v: {} for v in vecs}
    vnorm = {v: {} for v in vecs}
    lock = threading.Lock()
    for v, (a, b) in vecs.items():
        try:
            rec = Recipe.parse(vec_recipe_spec(a, b))
            if b is None and not (Path(a) / "DELTA.json").exists():
                raise SystemExit(f"delta checkpoint {a} not complete")
        except SystemExit as exc:  # king / delta not available (yet): skip this vector
            log.warning("skip projection on %s: %s", v, exc)
            dots.pop(v), vnorm.pop(v)
            continue
        reader = _Reader(rec)
        started = time.time()

        def work(key):
            g = G[key]
            if isinstance(key, tuple):  # (layer, proj) expert stack
                layer, proj = key
                names = [f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight" for e in range(g.shape[0])]
            else:
                names = [key]
            out_d, out_n = {}, {}
            for idx, name in enumerate(names):
                d = reader.handle(0, name).get_tensor(name).to(device, non_blocking=True).float()
                if b is not None:
                    d.sub_(reader.handle(1, name).get_tensor(name).to(device, non_blocking=True).float())
                gg = g[idx] if isinstance(key, tuple) else g
                out_d[name] = float((gg.float() * d).sum())
                out_n[name] = float(d.pow(2).sum())
            with lock:
                dots[v].update(out_d)
                vnorm[v].update(out_n)

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(work, list(G)))
        log.info("projected G on %s = %s - %s in %.0fs: total <G,V> %+.4e", v, a, b, time.time() - started,
                 sum(dots[v].values()))
    return dots, vnorm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--at", default=K40, help="merge point: model dir or recipe JSON")
    ap.add_argument("--sample", default="tag:G")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--pass-seqs", type=int, default=24)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--vecs", default="D40=40-39,D36=36-35,P40_36=40-36")
    ap.add_argument("--out", type=Path, default=Path("/root/teutonic_grad"))
    ap.add_argument("--g-dir", type=Path, default=Path("/opt/teutonic/grad"), help="where --save-g writes G")
    ap.add_argument("--save-g", action="store_true")
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
    model = build_skeleton(Path(K40), "flash_attention_4")
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = "flash_attention_4"
    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
    lazy_import_flash_attention("flash_attention_4")
    for name, buffer in model.named_buffers():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        owner._buffers[name.rsplit(".", 1)[1]] = buffer.to(device)
    st = Streamer(model, recipe, device, threads=args.io_threads)
    st.load("model.embed_tokens", model.model.embed_tokens)
    st.load("model.norm", model.model.norm)
    st.load("lm_head", model.lm_head)
    layers = model.model.layers
    L = len(layers)

    # gradient accumulators
    G: dict = {}
    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters(prefix=f"model.layers.{i}"):
            if EXPERT.match(name) or NO_GRAD.search(name):
                continue
            G[name] = torch.zeros(p.shape, dtype=torch.float32, device=device)
        if layer.mlp.__class__.__name__ == "MiMoV2MoE":
            for proj in PROJS:
                shape = getattr(layer.mlp.experts[0], proj).weight.shape
                G[(i, proj)] = torch.zeros((len(layer.mlp.experts), *shape), dtype=torch.bfloat16, device=device)
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        G[name] = torch.zeros(model.get_parameter(name).shape, dtype=torch.float32, device=device)
    log.info("G allocated: %.1f GiB; GPU used %.1f GiB", sum(g.numel() * g.element_size() for g in G.values()) / 2**30,
             torch.cuda.memory_allocated(device) / 2**30)

    def add_layer_grads(i: int, leaves) -> None:
        for name, leaf in leaves:
            if leaf.grad is None:
                continue
            if ".experts.*." in name:
                G[(i, name.rsplit(".", 1)[1])].add_(leaf.grad)
            else:
                G[name].add_(leaf.grad.float())
            leaf.grad = None

    # static per-layer kwargs for a micro-batch (no padding: they depend only on the shape)
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
    log.info("captured layer kwargs for %s", sorted(captured))

    head_norm = model.model.norm
    head = model.lm_head
    t0 = time.time()
    losses = []
    n_pass = args.n // args.pass_seqs
    vecs = parse_vecs(args.vecs)
    args.out.mkdir(parents=True, exist_ok=True)
    half = None
    for p in range(n_pass):
        tp = time.time()
        start = args.offset + p * args.pass_seqs
        ids_all = torch.as_tensor(np.asarray(sample.tokens[start:start + args.pass_seqs], dtype=np.int64), device=device)
        micro_ids = list(ids_all.split(args.micro))
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
        t_fwd = time.time() - tp
        # loss and head gradients
        head_norm.weight.requires_grad_(True)
        head.weight.requires_grad_(True)
        grads = []
        for h, ids in zip(H[L], micro_ids):
            h = h.detach().requires_grad_(True)
            per_seq = seq_ce_sum(head_norm(h), head, ids)
            (per_seq.sum() / args.n).backward()
            losses.extend(per_seq.detach().cpu().tolist())
            grads.append(h.grad)
        G["model.norm.weight"].add_(head_norm.weight.grad.float())
        G["lm_head.weight"].add_(head.weight.grad.float())
        head_norm.weight.grad = None
        head.weight.grad = None
        head_norm.weight.requires_grad_(False)
        head.weight.requires_grad_(False)
        H.pop()
        # backward through the layers
        for i in reversed(range(L)):
            layer = layers[i]
            st.get(f"model.layers.{i}", layer)  # resident (last layer / layer 0) or prefetched
            if i > 0:
                st.prefetch(f"model.layers.{i - 1}", layers[i - 1])
            leaves = set_grad(f"model.layers.{i}", layer, True)
            kw = captured[layer.attention_type]
            for m in range(len(micro_ids)):
                h = H[i][m].detach().requires_grad_(True)
                out = layer(h, **kw)
                out.backward(grads[m])
                grads[m] = h.grad
                add_layer_grads(i, leaves)
            set_grad(f"model.layers.{i}", layer, False)
            H.pop()
            if i != 0:
                st.release(f"model.layers.{i}", layer)
        # embedding gradient
        emb = G["model.embed_tokens.weight"]
        for ids, g in zip(micro_ids, grads):
            emb.index_add_(0, ids.reshape(-1), g.reshape(-1, g.shape[-1]).float())
        del grads
        log.info("pass %d/%d: %d seqs, mean CE so far %.5f | fwd %.0fs total %.0fs | elapsed %.0fs, GPU peak %.1f GiB",
                 p + 1, n_pass, args.pass_seqs, float(np.mean(losses)), t_fwd, time.time() - tp, time.time() - t0,
                 torch.cuda.max_memory_allocated(device) / 2**30)
        if p + 1 == n_pass // 2 and vecs:
            # first-half projections; G holds the first half's gradient scaled by 1/n (so x2 = half mean)
            half = project(G, vecs, device)
    full = project(G, vecs, device) if vecs else ({}, {})
    out = {"label": args.label, "at": str(args.at), "sample": args.sample, "n": args.n, "offset": args.offset,
           "mean_ce": float(np.mean(losses)), "vecs": {v: [str(x) for x in ab] for v, ab in vecs.items() if v in full[0]},
           "dots": full[0], "vnorm2": full[1], "dots_half1": half[0] if half else None,
           "seconds": round(time.time() - t0, 1)}
    (args.out / f"{args.label}.json").write_text(json.dumps(out))
    log.info("GRAD %s done in %.0fs: mean CE %.5f -> %s", args.label, time.time() - t0, out["mean_ce"],
             args.out / f"{args.label}.json")
    if args.save_g:
        from safetensors.torch import save_file
        gdir = args.g_dir / f"{args.label}_G"
        gdir.mkdir(parents=True, exist_ok=True)
        for i in range(L):
            part = {}
            for key, g in G.items():
                if isinstance(key, tuple) and key[0] == i:
                    part[f"model.layers.{i}.mlp.experts.{key[1]}"] = g.cpu()
                elif isinstance(key, str) and key.startswith(f"model.layers.{i}."):
                    part[key] = g.to(torch.bfloat16).cpu()
            save_file(part, str(gdir / f"layer{i:02d}.safetensors"))
        save_file({k: G[k].to(torch.bfloat16).cpu() for k in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight")},
                  str(gdir / "head.safetensors"))
        log.info("saved G to %s", gdir)


if __name__ == "__main__":
    main()
