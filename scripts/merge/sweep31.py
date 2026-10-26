#!/usr/bin/env python3
"""Fast merge sweep against king #31: task vectors held in page-locked RAM, candidates rebuilt on GPU.

The regular path (parity_eval.py duel on a recipe JSON) re-reads and merges ~660 GB of
safetensors for every candidate (~11 min, then ~3 min of scoring). Here the base (#31u)
and every task vector (fp32 difference of two same-basis checkpoints, stored as bf16) are
read once into page-locked host arenas laid out like the GPU model (each MoE layer's 256
experts of one projection form one contiguous block), the skeleton is allocated once, and
each candidate is rebuilt on all 8 GPUs by DMA in well under a minute:

    W = bf16( base + sum_v coef_v(name) * T_v(vec_v) [+ lambda * TIES(...)] )   (fp32 on GPU)

Linear recipes equal src/parity/weights.py's up to the bf16 rounding of each stored vector
(sub-ulp of the result); their equivalent standard recipe is recorded under `ship`.

Queue recipe (drop JSON files into --queue; processed by (priority, mtime)):
  {"label": "K_dm100", "description": "#31u + 1.0*Dm", "method": "task-arith",
   "vecs": {"Dm": 1.0},
   "overrides": [{"pattern": "\\\\.mlp\\\\.experts\\\\.", "vecs": {"Dm": 1.15}}],
   "transforms": {"Dm": [{"op": "trim", "density": 0.3}]},
   "ties": {"vecs": {"S31": 1.0, "Dm": 1.0}, "density": 0.3, "lambda": 1.0},
   "limit": 800, "priority": 100}
The first matching override wins; vectors it does not list keep the default coefficient.
Transforms act per original tensor (per expert): dare{p,seed}, trim{density}, drop_top{frac},
sign{ref, keep=agree|disagree}, orth{ref, beta}, clip{q}. `ties` adds lambda * TIES-merge
(trim to density, elect sign by weighted sum, disjoint mean) of the listed vectors.
Results go to scores.jsonl (read by update_31_md.py -> 31.md) and the loss cache.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/eval"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

log = logging.getLogger("sweep31")
SCORES = Path("/opt/teutonic/logs/scores.jsonl")
RECIPES = REPO / "configs/merge/k31"
CHUNK_BYTES = 512 << 20
EXPERT = re.compile(r"^(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


# ------------------------------------------------------------------ layout
@dataclass
class Unit:
    key: str
    names: list[str]
    target: torch.Tensor  # (E, *shape) for expert stacks; (1, *shape) view for single tensors
    per_numel: int
    index: int = 0

    @property
    def nbytes(self) -> int:
        return self.per_numel * len(self.names) * 2


def build_units(model, placement: dict[str, torch.Tensor]) -> list[Unit]:
    stacks: dict[tuple[str, str], dict[int, str]] = {}
    singles = []
    for name in sorted(placement):
        m = EXPERT.match(name)
        if m:
            stacks.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = name
        else:
            singles.append(name)
    units = []
    for (prefix, proj), found in sorted(stacks.items()):
        buffer = model.get_submodule(prefix)._parity_stacked[proj]
        names = [found[i] for i in range(buffer.shape[0])]
        units.append(Unit(f"{prefix}.experts.*.{proj}", names, buffer, buffer[0].numel()))
    for name in singles:
        tensor = placement[name]
        units.append(Unit(name, [name], tensor.unsqueeze(0), tensor.numel()))
    for i, u in enumerate(units):
        u.index = i
    return units


class HostArena:
    """One bf16 slot per nonzero unit, carved out of 16 GiB pinned blocks.

    Blocks come from cudaHostAlloc (torch pin_memory); cudaHostRegister is not permitted in
    this container. Pinned copies reach ~4 GB/s summed over the 8 GPUs here (pageable: ~1.6).
    """

    BLOCK = 1 << 34

    def __init__(self, label: str):
        self.label = label
        self.blocks: list[torch.Tensor] = []
        self.used = self.BLOCK
        self.slots: dict[str, torch.Tensor] = {}
        self.written: dict[str, set[int]] = {}
        self.lock = threading.Lock()

    def slot(self, u: Unit) -> torch.Tensor:
        with self.lock:
            if u.key not in self.slots:
                size = -(-u.nbytes // 4096) * 4096
                if size > self.BLOCK:
                    raise RuntimeError(f"{u.key} larger than a pinned block")
                if self.used + size > self.BLOCK:
                    self.blocks.append(torch.empty(self.BLOCK, dtype=torch.uint8, pin_memory=True))
                    self.used = 0
                raw = self.blocks[-1][self.used:self.used + u.nbytes]
                self.used += size
                self.slots[u.key] = raw.view(torch.bfloat16).view(u.target.shape)
                self.written[u.key] = set()
            return self.slots[u.key]

    def nonzero(self, u: Unit) -> bool:
        return u.key in self.slots

    def unit(self, u: Unit) -> torch.Tensor:
        return self.slots[u.key]

    def finish(self) -> float:
        for key, slot in self.slots.items():
            for i in set(range(slot.shape[0])) - self.written[key]:
                slot[i].zero_()
        return len(self.blocks) * self.BLOCK / 2**30


class Reader:
    """preadv-based BF16 tensor reader, one fd per shard per thread."""

    def __init__(self, root: Path):
        self.root = root
        self.map = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict[str, tuple[dict, int]] = {}
        self.local = threading.local()
        self.lock = threading.Lock()

    def _header(self, filename: str) -> tuple[dict, int]:
        with self.lock:
            if filename not in self.headers:
                with open(self.root / filename, "rb") as h:
                    (n,) = struct.unpack("<Q", h.read(8))
                    self.headers[filename] = (json.loads(h.read(n)), 8 + n)
            return self.headers[filename]

    def _fd(self, filename: str) -> int:
        fds = getattr(self.local, "fds", None)
        if fds is None:
            fds = self.local.fds = {}
        if filename not in fds:
            fds[filename] = os.open(self.root / filename, os.O_RDONLY)
        return fds[filename]

    def into(self, name: str, view: memoryview) -> None:
        filename = self.map[name]
        header, start = self._header(filename)
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise RuntimeError(f"{name}: {entry['dtype']}")
        begin, end = entry["data_offsets"]
        if end - begin != len(view):
            raise RuntimeError(f"{name}: size {end - begin} != slot {len(view)}")
        fd, done = self._fd(filename), 0
        while done < len(view):
            got = os.preadv(fd, [view[done:]], start + begin + done)
            if got <= 0:
                raise IOError(f"short read {name}")
            done += got


# ------------------------------------------------------------------ store
class Store:
    def __init__(self, base: Path, vec_specs: dict[str, tuple[Path, Path]], units: list[Unit], workers: int):
        self.base_path, self.vec_specs, self.units = base, vec_specs, units
        started = time.time()
        paths = [base] + sorted({p for pair in vec_specs.values() for p in pair} - {base}, key=str)
        readers = {p: Reader(p) for p in paths}
        for p, r in readers.items():
            if set(r.map) != {n for u in units for n in u.names}:
                raise SystemExit(f"{p}: tensor names differ from the model")
        # the base lives on the GPUs next to the working model (220 GB each; host->GPU is slow here)
        self.gpu_base = {u.key: torch.empty(u.target.shape, dtype=torch.bfloat16, device=u.target.device) for u in units}
        self.vecs = {v: HostArena(v) for v in vec_specs}
        jobs = [(u, i) for u in units for i in range(len(u.names))]
        done = [0]
        lock = threading.Lock()
        torch.set_num_threads(2)

        def work(job) -> None:
            u, i = job
            name = u.names[i]
            size = u.per_numel * 2
            got = {}
            for p in paths:
                tmp = bytearray(size)
                readers[p].into(name, memoryview(tmp))
                got[p] = torch.frombuffer(tmp, dtype=torch.bfloat16)
            self.gpu_base[u.key][i].view(-1).copy_(got[base])
            for v, (a, b) in vec_specs.items():
                diff = got[a].float() - got[b].float()
                if bool(diff.any()):
                    arena = self.vecs[v]
                    arena.slot(u)[i].view(-1).copy_(diff.to(torch.bfloat16))
                    with arena.lock:
                        arena.written[u.key].add(i)
            with lock:
                done[0] += 1
                if done[0] % 4000 == 0:
                    log.info("store %d/%d tensors (%.0fs)", done[0], len(jobs), time.time() - started)

        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(work, jobs))
        for gpu in sorted({u.target.device.index for u in units}):
            torch.cuda.synchronize(gpu)
        gib = 0.0
        for v, arena in self.vecs.items():
            gib += arena.finish()
            log.info("vector %s = %s - %s: %d/%d units nonzero", v, *vec_specs[v], len(arena.slots), len(units))
        log.info("store ready in %.0fs (base on GPU, %.0f GiB pinned host vectors)", time.time() - started, gib)


# ------------------------------------------------------------------ server
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/opt/teutonic/hf/king31u"))
    ap.add_argument("--vec", action="append", required=True, help="NAME=/path/a:/path/b  (vector = a - b)")
    ap.add_argument("--queue", type=Path, default=RECIPES / "queue")
    ap.add_argument("--sample", default="tag:A")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--attn", default="flash_attention_4")
    ap.add_argument("--lm-head-chunk", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--fill-threads", type=int, default=24)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import parity_eval as pe
    from src.parity.cache import LossCache, scorer_signature
    from src.parity.moe import grouped_moe
    from src.parity.pipeline import PipelineScorer
    from src.parity.sample import load_live_config
    from src.parity.verdict import duel as duel_verdict
    from src.parity.weights import LoadedModel, Recipe, _allocate, build_skeleton, model_identity, plan_layer_devices

    vec_specs = {}
    for spec in args.vec:
        name, pair = spec.split("=", 1)
        a, b = pair.split(":")
        vec_specs[name] = (Path(a), Path(b))
    vec_names = list(vec_specs)
    args.queue.mkdir(parents=True, exist_ok=True)
    (args.queue / "done").mkdir(exist_ok=True)

    live = load_live_config()
    sample = pe.resolve_sample(args.sample, live)
    cache = LossCache()
    signature = scorer_signature(args.attn, args.lm_head_chunk)
    base_id = model_identity(args.base)
    log.info("sample %s n=%d; king id %s", sample.sample_id, sample.n, base_id)

    gpus = list(range(torch.cuda.device_count()))
    model = build_skeleton(args.base, args.attn)
    layer_devices = plan_layer_devices(model, gpus)
    placement = _allocate(model, layer_devices)
    for module in model.modules():
        if module.__class__.__name__ == "MiMoV2MoE":
            module.moe = grouped_moe.__get__(module)
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = args.attn
    model.hf_device_map = {
        **{f"model.layers.{i}": d for i, d in enumerate(layer_devices)},
        "model.embed_tokens": layer_devices[0], "model.norm": layer_devices[-1], "lm_head": layer_devices[-1],
    }
    units = build_units(model, placement)
    store = Store(args.base, vec_specs, units, args.workers)

    # interleave units across GPUs so every device's copy engine is busy
    by_dev: dict[int, list[Unit]] = {}
    for u in units:
        by_dev.setdefault(u.target.device.index, []).append(u)
    order = []
    for k in range(max(len(v) for v in by_dev.values())):
        for dev in sorted(by_dev):
            if k < len(by_dev[dev]):
                order.append(by_dev[dev][k])

    loaded = LoadedModel(model, Recipe.single(args.base), layer_devices, args.attn, 0.0)
    if args.attn == "flash_attention_4":
        from transformers.modeling_flash_attention_utils import lazy_import_flash_attention
        lazy_import_flash_attention(args.attn)
    scorer = PipelineScorer(loaded, lm_head_chunk=args.lm_head_chunk)

    import importlib
    import types

    sys.path.insert(0, str(REPO / "scripts/merge"))
    import sweep31_ops as ops

    for u in units:
        u.leaf = u.target if EXPERT.match(u.names[0]) else placement[u.names[0]]
    ctx = types.SimpleNamespace(
        args=args, store=store, units=units, order=order, placement=placement, model=model, scorer=scorer,
        sample=sample, cache=cache, signature=signature, base_id=base_id, vec_specs=vec_specs, vec_names=vec_names,
        live=live, layer_devices=layer_devices, pe=pe, duel_verdict=duel_verdict, log=log, recipes=RECIPES,
        scores=SCORES, state={}, parked=None,
    )

    def base_of(u: Unit) -> torch.Tensor:
        return store.gpu_base[u.key] if store.gpu_base[u.key] is not None else ctx.parked.unit(u)

    def park_base() -> None:
        """Move the resident base to pinned host memory (frees ~220 GB of GPU for gradients)."""
        started = time.time()
        if ctx.parked is None:
            ctx.parked = HostArena("parked-base")
        for u in units:
            ctx.parked.slot(u).copy_(store.gpu_base[u.key])
            ctx.parked.written[u.key] = set(range(len(u.names)))
        for u in units:
            store.gpu_base[u.key] = None
        for gpu in sorted(set(layer_devices)):
            with torch.cuda.device(gpu):
                torch.cuda.empty_cache()
        log.info("parked base in host memory (%.0fs)", time.time() - started)

    def unpark_base() -> None:
        started = time.time()
        for u in units:
            t = torch.empty(u.target.shape, dtype=torch.bfloat16, device=u.target.device)
            t.copy_(ctx.parked.unit(u))
            store.gpu_base[u.key] = t
        log.info("restored base to GPU (%.0fs)", time.time() - started)

    ctx.base_of, ctx.park_base, ctx.unpark_base = base_of, park_base, unpark_base

    # self-test 1: the rebuilt king is bit-identical to the checkpoint on a spread of tensors
    ops.rebuild(ctx, {"label": "king31u", "vecs": {}})
    from safetensors import safe_open
    index = json.loads((args.base / "model.safetensors.index.json").read_text())["weight_map"]
    probe = [n for n in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                         "model.layers.0.self_attn.qkv_proj.weight", "model.layers.20.mlp.experts.0.gate_proj.weight",
                         "model.layers.20.mlp.experts.255.down_proj.weight", "model.layers.44.mlp.experts.77.up_proj.weight",
                         "model.layers.30.mlp.gate.weight", "model.layers.31.self_attn.o_proj.weight") if n in placement]
    bad = []
    for n in probe:
        with safe_open(str(args.base / index[n]), framework="pt", device="cpu") as h:
            ref = h.get_tensor(n)
        if not torch.equal(placement[n].cpu(), ref):
            bad.append(n)
    log.info("self-test: rebuilt king bit-identical on %d/%d probe tensors %s", len(probe) - len(bad), len(probe),
             f"MISMATCH {bad}" if bad else "")

    # baseline: king and challengers must be scored with the same batch size (MoE kernel rounding
    # depends on it). Losses cached by other runs (e.g. batch 2) are set aside.
    meta_path = cache._paths(sample.sample_id, base_id, signature)[1]
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta and meta.get("batch") != args.batch_size:
        old = cache.load(sample.sample_id, base_id, signature, sample.n)
        backup = meta_path.with_name(meta_path.stem + f".batch{meta.get('batch', 'unknown')}.npy")
        np.save(backup, old)
        cache.save(sample.sample_id, base_id, signature, np.full(sample.n, np.nan), {"batch": args.batch_size, "note": "reset"})
        fresh, _, secs = ops.losses_for(ctx, base_id, args.limit, {"label": "king31u", "vecs": {}})
        both = np.isfinite(old[:args.limit]) & np.isfinite(fresh[:args.limit])
        d = old[:args.limit][both] - fresh[:args.limit][both]
        log.info("king re-scored at batch %d on %d seqs (%.0fs); vs cached batch-%s losses: mean %+.2e sd %.2e",
                 args.batch_size, int(both.sum()), secs, meta.get("batch", "?"),
                 d.mean() if d.size else 0, d.std() if d.size else 0)

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

    log.info("sweep server ready; watching %s", args.queue)
    while True:
        if (args.queue / "STOP").exists():
            log.info("STOP file found; exiting")
            break
        path = next_item()
        if path is None:
            time.sleep(10)
            continue
        try:
            ops = importlib.reload(ops)  # pick up new methods without reloading 600 GB of weights
            ops.handle(ctx, path)
        except Exception as exc:  # keep serving the queue
            log.exception("FAIL %s: %s", path.name, exc)
            if ctx.store.gpu_base and next(iter(ctx.store.gpu_base.values())) is None:
                ctx.unpark_base()
            shutil.move(str(path), args.queue / "done" / (path.name + ".failed"))
            continue
        shutil.move(str(path), args.queue / "done" / path.name)
        subprocess.run([sys.executable, str(REPO / "scripts/merge/update_31_md.py")], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
