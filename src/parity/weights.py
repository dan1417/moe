"""Load (optionally merged) MiMo checkpoints layer-sharded across small GPUs.

The model skeleton is built exactly like the validator's ``load_eval_model`` (bf16
default dtype, empty weights, locked remote code, masking compat patch). Weights are
then streamed from safetensors straight into place on their target GPU. A *recipe*
of several checkpoints is combined per tensor in fp32 and rounded to bf16 once, so
merge candidates can be scored without ever writing them to disk.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import math
import os
import struct
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from src.parity import upstream
from src.parity.attention import FA4SIM, register_fa4sim
from src.parity.moe import grouped_moe

log = logging.getLogger(__name__)

MIB = 1024**2
GIB = 1024**3
# Merge arithmetic happens in fp32 on the target GPU in row chunks of this size, with at
# most MERGE_SLOTS_PER_GPU chunks in flight per GPU (the loader runs 32 I/O threads and a
# 5090 has only a few GiB free next to its share of the weights). A slot also covers the chunk's
# page-cache read and pageable upload, so on B300s (tens of GiB free) more slots = a faster load:
# PARITY_MERGE_SLOTS=8 there (~0.4 GiB of fp32 scratch per slot).
MERGE_CHUNK_BYTES = 128 * MIB
MERGE_SLOTS_PER_GPU = int(os.environ.get("PARITY_MERGE_SLOTS", "3"))
# Large preadv() reads instead of mmap page faults: ~2x faster on the FUSE-mounted /root.
FAST_READS = os.environ.get("PARITY_FAST_READS", "1") == "1"
_merge_slots: dict[str, threading.BoundedSemaphore] = {}
_merge_slots_lock = threading.Lock()
_SKELETON_LOCK = threading.Lock()


@contextlib.contextmanager
def _no_uninitialized_fill():
    """With torch.use_deterministic_algorithms(True), every torch.empty fills its memory with NaN: building the
    skeleton then memsets ~220 GB of CPU RAM that init_empty_weights throws away (~10 min per replica, all cores),
    and _allocate NaN-fills every GPU buffer. Every parameter is overwritten by _fill (checked: no meta left), so
    the fill is pure cost; scoring kernels keep deterministic mode."""
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill


def _merge_slot(device: torch.device) -> threading.BoundedSemaphore:
    with _merge_slots_lock:
        key = str(device)
        if key not in _merge_slots:
            limit = MERGE_SLOTS_PER_GPU if device.type == "cuda" else 64
            _merge_slots[key] = threading.BoundedSemaphore(limit)
        return _merge_slots[key]


# --------------------------------------------------------------------------- recipes
def model_identity(path: Path) -> str:
    """Stable identity for a checkpoint directory (R2 digest when known)."""
    info = path / "king_info.json"
    if info.exists():
        digest = json.loads(info.read_text()).get("digest")
        if digest:
            return f"sha256:{digest}"
    return _content_identity(str(path.resolve()), tuple(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.glob("*.safetensors"))))


@functools.lru_cache(maxsize=256)
def _content_identity(path: str, stats: tuple) -> str:
    """Index + a content fingerprint of every shard (3 x 1 MiB samples inside the tensor data).

    Name/size alone is not enough: every export of the same architecture has an identical index
    and identical shard sizes, so different training steps would share one loss-cache entry.
    """
    root = Path(path)
    index = root / "model.safetensors.index.json"
    digest = hashlib.sha256(index.read_bytes() if index.exists() else b"")
    chunk = 1 << 20
    for name, size, _ in stats:
        digest.update(f"{name}:{size}".encode())
        with open(root / name, "rb") as handle:
            for offset in (size // 3, size // 2, max(size - chunk, 0)):
                handle.seek(offset)
                digest.update(handle.read(chunk))
    return "local2:" + digest.hexdigest()


@dataclass
class Term:
    path: Path
    coef: float


@dataclass
class Transform:
    """Post-combine op on every tensor whose name matches ``pattern`` (applied in fp32, then rounded).

    op = "scale"         x <- s * x                               (e.g. lm_head logit temperature)
         "center_scale"  x <- mean(x) * keep_mean + g * (x - mean(x))   (router-bias gamma: g < 1 flattens
                         the load-balancing bias, g > 1 sharpens it; selection only sees differences)
         "shrink_to"     x <- x_ref + s * (x - x_ref) with x_ref = term ``ref`` of the recipe (partial revert)
         "match_norm"    x <- x * ||x_ref|| / ||x||  (keep the Frobenius norm of term ``ref``: extrapolate a
                         direction without growing the weights; per tensor, i.e. per expert matrix)
    """

    pattern: str
    op: str
    args: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._re = re.compile(self.pattern)
        if self.op not in ("scale", "center_scale", "shrink_to", "match_norm"):
            raise ValueError(f"unknown transform op {self.op!r}")

    def matches(self, name: str) -> bool:
        return bool(self._re.search(name))

    def describe(self) -> dict:
        return {"pattern": self.pattern, "op": self.op, "args": dict(self.args)}

    def apply(self, x: torch.Tensor, name: str, reader: "_Reader") -> torch.Tensor:
        if self.op == "scale":
            return x.mul_(float(self.args["s"]))
        if self.op == "center_scale":
            mean = x.mean()
            return x.sub_(mean).mul_(float(self.args["g"])).add_(mean * float(self.args.get("keep_mean", 1.0)))
        ref = reader.handle(int(self.args.get("ref", 0)), name).get_tensor(name).to(x.device).float()
        if self.op == "match_norm":
            return x.mul_(ref.norm() / x.norm().clamp_min(1e-30))
        return ref.add_(x.sub_(ref), alpha=float(self.args["s"]))


@dataclass
class Recipe:
    """weight(name) = sum_i coef_i(name) * checkpoint_i[name], computed in fp32."""

    terms: list[Term]
    overrides: list[tuple[str, list[float]]] = field(default_factory=list)
    label: str = ""
    # tensor name -> safetensors file holding that exact tensor (e.g. a tuned lm_head);
    # replaces the combined value for that name.
    replacements: dict[str, str] = field(default_factory=dict)
    # Combine modes (terms listed per mode; tensors matching mode_args["exclude"] stay = king):
    #   linear      sum_i coef_i·term_i  (+ per-tensor `tensor_coefs`, regex `overrides`)
    #   sign_agree  [king, donor, base]: king + beta·(donor-base)·[sign agrees with king-base]
    #   orthogonal  [king, donor, base]: king + beta·(tau_donor ⟂ tau_king), per tensor
    #   dare        [king, donor, base]: king + beta·Bernoulli(1-p)·(donor-base)/(1-p)
    #   ties        [king, donor, base]: king + beta·trim_density(donor-base) (top-|x| entries)
    #   dare_ties   [king, donor, base]: base + mergekit TIES over DARE'd deltas of king & donor
    #   slerp       [a, b]: per-tensor spherical interpolation at t
    #   model_stock [base, fine-tunes..., (on)]: Model Stock, per-tensor ratio t from the fine-tunes' angle
    #   delta_orthogonal_pair [king, parent, ckpt, start]: king + beta·((ckpt-start) ⟂ (king-parent)), per tensor
    mode: str = "linear"
    mode_args: dict = field(default_factory=dict)
    # exact tensor name -> coefficients per term (linear mode; takes precedence over overrides)
    tensor_coefs: dict[str, list[float]] = field(default_factory=dict)
    # lm_head rows (token ids) to take verbatim from the first term after any combine
    head_rows_from_king: list[int] = field(default_factory=list)
    # post-combine per-regex ops (see Transform), applied in order
    transforms: list[Transform] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.terms:
            raise ValueError("recipe needs at least one checkpoint")
        self._compiled = [(re.compile(p), coefs) for p, coefs in self.overrides]
        for pattern, coefs in self.overrides:
            if len(coefs) != len(self.terms):
                raise ValueError(f"override {pattern!r} needs {len(self.terms)} coefficients")

    @classmethod
    def single(cls, path: str | Path) -> "Recipe":
        return cls([Term(Path(path), 1.0)], label=Path(path).name)

    @classmethod
    def from_json(cls, path: str | Path) -> "Recipe":
        raw = json.loads(Path(path).read_text())
        return cls(
            terms=[Term(Path(t["model"]), float(t["coef"])) for t in raw["terms"]],
            overrides=[(o["pattern"], [float(c) for c in o["coefs"]]) for o in raw.get("overrides", [])],
            label=raw.get("label", Path(path).stem),
        )

    @classmethod
    def parse(cls, spec: str) -> "Recipe":
        """``/path`` or ``coef:/path,coef:/path`` or ``recipe.json``."""
        if spec.endswith(".json") and Path(spec).is_file():
            return cls.from_json(spec)
        if ":" not in spec or Path(spec).exists():
            return cls.single(spec)
        terms = []
        for part in spec.split(","):
            coef, path = part.split(":", 1)
            terms.append(Term(Path(path), float(coef)))
        return cls(terms, label=spec)

    def coefs_for(self, name: str) -> list[float]:
        if name in self.tensor_coefs:
            return self.tensor_coefs[name]
        for pattern, coefs in self._compiled:
            if pattern.search(name):
                return coefs
        return [term.coef for term in self.terms]

    @property
    def is_plain(self) -> bool:
        return len(self.terms) == 1 and self.terms[0].coef == 1.0 and not self.overrides

    def identity(self) -> str:
        material = {
            "terms": [[model_identity(t.path), t.coef] for t in self.terms],
            "overrides": self.overrides,
        }
        if self.replacements:
            material["replacements"] = {
                name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for name, path in sorted(self.replacements.items())
            }
        if self.mode != "linear":
            material["mode"] = [self.mode, self.mode_args]
        if self.head_rows_from_king:
            material["head_rows"] = hashlib.sha256(json.dumps(sorted(self.head_rows_from_king)).encode()).hexdigest()
        if self.tensor_coefs:
            material["tensor_coefs"] = hashlib.sha256(
                json.dumps(self.tensor_coefs, sort_keys=True).encode()
            ).hexdigest()
        if self.transforms:
            material["transforms"] = [t.describe() for t in self.transforms]
        if (self.is_plain and not self.replacements and self.mode == "linear" and not self.tensor_coefs
                and not self.transforms):
            return model_identity(self.terms[0].path)
        return "recipe:" + hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def describe(self) -> dict:
        return {
            "label": self.label,
            "terms": [{"model": str(t.path), "id": model_identity(t.path), "coef": t.coef} for t in self.terms],
            "overrides": [{"pattern": p, "coefs": c} for p, c in self.overrides],
            "replacements": dict(self.replacements),
            "mode": self.mode,
            "mode_args": dict(self.mode_args),
            "head_rows_from_king": len(self.head_rows_from_king),
            "transforms": [t.describe() for t in self.transforms],
        }


# ----------------------------------------------------------------------- placement
def _param_bytes(module: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in module.parameters())


def plan_layer_devices(
    model: nn.Module,
    gpu_ids: list[int],
    *,
    reserve_first_gib: float = 1.5,
    reserve_last_gib: float = 3.0,
) -> list[int]:
    """Contiguous layer->GPU assignment minimizing the largest per-GPU footprint.

    Embeddings sit on the first GPU and norm/lm_head on the last (as in the
    validator's balanced map); the reserves cover attention and lm-head/CE scratch.
    """
    layers = model.model.layers
    sizes = [_param_bytes(layer) for layer in layers]
    first_fixed = _param_bytes(model.model.embed_tokens) + int(reserve_first_gib * GIB)
    last_fixed = (
        _param_bytes(model.lm_head) + _param_bytes(model.model.norm) + int(reserve_last_gib * GIB)
    )
    n_gpus = len(gpu_ids)

    def budget(gpu: int, capacity: int) -> int:
        return capacity - (last_fixed if gpu == n_gpus - 1 else 0)

    def assign(capacity: int) -> list[int] | None:
        plan, gpu, load = [], 0, first_fixed
        for size in sizes:
            if load + size > budget(gpu, capacity):
                gpu, load = gpu + 1, 0
                if gpu >= n_gpus or size > budget(gpu, capacity):
                    return None
            plan.append(gpu)
            load += size
        return plan

    low, high = 1, sum(sizes) + first_fixed + last_fixed
    best = None
    while low <= high:
        middle = (low + high) // 2
        plan = assign(middle)
        if plan is not None:
            best, high = plan, middle - 1
        else:
            low = middle + 1
    if best is None:
        raise RuntimeError("could not place layers on the given GPUs")
    loads = [0] * n_gpus
    loads[0] += _param_bytes(model.model.embed_tokens)
    loads[-1] += _param_bytes(model.lm_head)
    for gpu, size in zip(best, sizes):
        loads[gpu] += size
    for index, gpu_id in enumerate(gpu_ids):
        total = torch.cuda.get_device_properties(gpu_id).total_memory
        if total - loads[index] < 2 * GIB:
            raise RuntimeError(
                f"GPU {gpu_id} would hold {loads[index] / GIB:.1f} GiB of weights with only "
                f"{(total - loads[index]) / GIB:.1f} GiB left for activations; use more GPUs"
            )
    log.info("weights per GPU (GiB): %s", [round(load / GIB, 2) for load in loads])
    return [gpu_ids[index] for index in best]


# ------------------------------------------------------------------------ skeleton
def build_skeleton(snapshot_dir: Path, attn_implementation: str) -> nn.Module:
    """Empty-weight MiMo model constructed exactly like the validator does."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    from transformers.initialization import no_init_weights

    engine = upstream.engine
    request = engine.EvalRequest(king_repo="king", challenger_repo="challenger")
    config, _ = engine.load_model_config(str(snapshot_dir), request, "parity")
    engine.validate_and_report_attention_config(config, "parity")
    config.use_cache = False
    config._attn_implementation = "eager"
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(), no_init_weights():
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    finally:
        torch.set_default_dtype(old_dtype)
    engine.patch_mimo_masking_compat(model)
    model.tie_weights()
    if attn_implementation == FA4SIM:
        register_fa4sim()
    elif attn_implementation not in ("eager", "flash_attention_4"):  # FA4 needs sm_90/sm_100 (e.g. B300)
        raise ValueError(f"unsupported attention implementation {attn_implementation!r}")
    model._parity_attn_implementation = attn_implementation
    return model


def _set_param(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    module._parameters[name] = nn.Parameter(tensor, requires_grad=False)


def _allocate(model: nn.Module, layer_devices: list[int]) -> dict[str, torch.Tensor]:
    """Allocate every parameter on its GPU; experts as views into stacked buffers."""
    first, last = layer_devices[0], layer_devices[-1]
    placement: dict[str, torch.Tensor] = {}

    def place(prefix: str, module: nn.Module, device: int) -> None:
        for name, param in list(module.named_parameters(recurse=False)):
            tensor = torch.empty(param.shape, dtype=param.dtype, device=f"cuda:{device}")
            _set_param(module, name, tensor)
            placement[f"{prefix}{name}"] = tensor

    for name, module in model.named_modules():
        if ".experts." in name or name.endswith(".experts"):
            continue
        match = re.match(r"model\.layers\.(\d+)(\.|$)", name)
        if match:
            device = layer_devices[int(match.group(1))]
        elif name.startswith("lm_head") or name == "model.norm":
            device = last
        else:
            device = first
        place(f"{name}." if name else "", module, device)
        if module.__class__.__name__ == "MiMoV2MoE":
            stacked = {}
            for proj in ("gate_proj", "up_proj", "down_proj"):
                shape = getattr(module.experts[0], proj).weight.shape
                buffer = torch.empty(
                    (len(module.experts), *shape), dtype=torch.bfloat16, device=f"cuda:{device}"
                )
                for index, expert in enumerate(module.experts):
                    linear = getattr(expert, proj)
                    _set_param(linear, "weight", buffer[index])
                    placement[f"{name}.experts.{index}.{proj}.weight"] = buffer[index]
                stacked[proj] = buffer
            module._parity_stacked = stacked
    for name, buffer in model.named_buffers():
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        owner._buffers[name.rsplit(".", 1)[1]] = buffer.to(f"cuda:{first}")
    return placement


# ------------------------------------------------------------------------- loading
# load_model's fill threads read chunks straight into a reusable per-thread pinned buffer (no fresh
# zero-filled bytearray + page faults per chunk, DMA upload instead of a pageable staging copy):
# ~2x faster loads, and less CPU taken from a replica set scoring next to a loading one. Opt-in per
# thread: the returned tensor is only valid until that thread's next read, which every GPU fill path
# honours (each chunk is uploaded before the next read); other users of _FastFile get fresh buffers.
_PINNED = threading.local()
PINNED_MAX_BYTES = 256 * MIB


@contextlib.contextmanager
def _pinned_reads():
    _PINNED.enabled = True
    try:
        yield
    finally:
        _PINNED.enabled = False


def _pinned_buffer(nbytes: int) -> torch.Tensor:
    buf = getattr(_PINNED, "buf", None)
    if buf is None or buf.numel() < nbytes:
        _PINNED.buf = buf = torch.empty(1 << max(nbytes - 1, 1).bit_length(), dtype=torch.uint8, pin_memory=True)
    return buf


class _FastSlice:
    """Row slices of one BF16 tensor read with a single ``preadv`` (no mmap faults)."""

    def __init__(self, file: "_FastFile", name: str):
        entry = file.header[name]
        if entry["dtype"] not in ("BF16", "I8"):
            raise RuntimeError(f"{name}: expected BF16 (or I8 delta), got {entry['dtype']}")
        self.file, self.shape, self.name = file, list(entry["shape"]), name
        self.int8 = entry["dtype"] == "I8"  # quantized task vector: int8 rows x fp32 row scales (sidecar file)
        self.begin, self.end = entry["data_offsets"]

    def get_shape(self) -> list[int]:
        return self.shape

    def __getitem__(self, key) -> torch.Tensor:
        rows = self.shape[0] if self.shape else 1
        row_bytes = (self.end - self.begin) // max(rows, 1)
        start = key.start or 0
        stop = rows if key.stop is None else min(key.stop, rows)
        nbytes = (stop - start) * row_bytes
        pinned = (not self.int8 and getattr(_PINNED, "enabled", False) and 0 < nbytes <= PINNED_MAX_BYTES
                  and nbytes % 2 == 0)
        host = _pinned_buffer(nbytes)[:nbytes] if pinned else None
        buffer = host.numpy() if pinned else bytearray(nbytes)
        offset = self.file.data_start + self.begin + start * row_bytes
        view, done = memoryview(buffer).cast("B"), 0
        while done < len(buffer):
            got = os.preadv(self.file.fd, [view[done:]], offset + done)
            if got <= 0:
                raise IOError(f"short read in {self.file.path}")
            done += got
        if self.int8:
            q = torch.frombuffer(buffer, dtype=torch.int8).view(stop - start, *self.shape[1:])
            scale = self.file.scales(self.name)[start:stop]
            return q.float().mul_(scale.view(-1, *([1] * (q.dim() - 1))))
        tensor = host.view(torch.bfloat16) if pinned else torch.frombuffer(buffer, dtype=torch.bfloat16)
        return tensor.view(stop - start, *self.shape[1:]) if self.shape else tensor.view(())


class _FastFile:
    def __init__(self, path: Path):
        self.path = path
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            self.header = json.loads(handle.read(length))
        self.data_start = 8 + length
        self.fd = os.open(path, os.O_RDONLY)

    def get_slice(self, name: str) -> _FastSlice:
        return _FastSlice(self, name)

    def __del__(self) -> None:  # without this every recipe leaked one fd per shard per thread
        try:
            os.close(self.fd)
        except (OSError, AttributeError):
            pass

    def scales(self, name: str) -> torch.Tensor:
        """fp32 row scales of an I8 tensor, from '<shard>.scales.safetensors' next to the shard."""
        cache = self.__dict__.setdefault("_scales", {})
        if name not in cache:
            with safe_open(str(self.path.with_suffix(".scales.safetensors")), framework="pt", device="cpu") as h:
                cache[name] = h.get_tensor(name).float()
        return cache[name]

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.get_slice(name)[:]


class _Reader:
    """Per-thread cache of safetensors handles for every recipe checkpoint."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.maps = []
        for term in recipe.terms:
            index = json.loads((term.path / "model.safetensors.index.json").read_text())
            self.maps.append(index["weight_map"])
        self.local = threading.local()

    def handle(self, term_index: int, name: str):
        handles = getattr(self.local, "handles", None)
        if handles is None:
            handles = self.local.handles = {}
        filename = self.maps[term_index][name]
        key = (term_index, filename)
        if key not in handles:
            path = self.recipe.terms[term_index].path / filename
            handles[key] = _FastFile(path) if FAST_READS else safe_open(str(path), framework="pt", device="cpu")
        return handles[key]


def _fill(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    _fill_inner(target, name, reader, recipe)
    ops = [t for t in recipe.transforms if t.matches(name)]
    if ops:
        with torch.no_grad():
            x = target.float()
            for op in ops:
                x = op.apply(x, name, reader)
            target.copy_(x)
    if name == "lm_head.weight" and recipe.head_rows_from_king:
        rows = torch.tensor(sorted(set(recipe.head_rows_from_king)), dtype=torch.long)
        king_rows = reader.handle(0, name).get_tensor(name)[rows]
        target[rows.to(target.device)] = king_rows.to(target.device, target.dtype)


def _fill_inner(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    if name in recipe.replacements:
        with safe_open(recipe.replacements[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise RuntimeError(f"{name}: replacement {tuple(tensor.shape)}/{tensor.dtype} != model {tuple(target.shape)}")
        target.copy_(tensor)
        return
    if recipe.mode == "sign_agree":
        _fill_sign_agree(target, name, reader, recipe)
        return
    if recipe.mode == "orthogonal":
        _fill_orthogonal(target, name, reader, recipe)
        return
    if recipe.mode in ("dare", "ties", "dare_ties", "slerp"):
        _fill_whole_tensor(target, name, reader, recipe)
        return
    if recipe.mode == "delta_agree":
        _fill_delta_agree(target, name, reader, recipe)
        return
    if recipe.mode == "delta_orthogonal":
        _fill_delta_orthogonal(target, name, reader, recipe)
        return
    if recipe.mode == "delta_orthogonal_pair":
        _fill_delta_orthogonal_pair(target, name, reader, recipe)
        return
    if recipe.mode in ("delta_ties", "delta_dare"):
        _fill_delta_sparse(target, name, reader, recipe)
        return
    if recipe.mode == "vec_ties":
        _fill_vec_ties(target, name, reader, recipe)
        return
    if recipe.mode == "model_stock":
        _fill_model_stock(target, name, reader, recipe)
        return
    coefs = recipe.coefs_for(name)
    active = [(i, c) for i, c in enumerate(coefs) if c != 0.0]
    if len(active) == 1 and active[0][1] == 1.0:
        tensor = reader.handle(active[0][0], name).get_tensor(name)
        if tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise RuntimeError(f"{name}: checkpoint {tuple(tensor.shape)}/{tensor.dtype} != model {tuple(target.shape)}")
        target.copy_(tensor)
        return
    rows = target.shape[0] if target.dim() else 1
    row_bytes = max(1, target.numel() // max(rows, 1)) * 4
    step = max(1, MERGE_CHUNK_BYTES // row_bytes)
    slices = [reader.handle(i, name).get_slice(name) for i, _ in active]
    with _merge_slot(target.device):
        for start in range(0, rows, step):
            end = min(rows, start + step)
            accumulator = None
            for (i, coef), part in zip(active, slices):
                chunk = part[start:end] if target.dim() else part[:]
                chunk = chunk.to(target.device, non_blocking=False).float()
                accumulator = chunk.mul_(coef) if accumulator is None else accumulator.add_(chunk, alpha=coef)
            view = target[start:end] if target.dim() else target
            view.copy_(accumulator)


def _fill_sign_agree(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """king + beta·(donor−base) wherever both moved the same way since ``base``."""
    if len(recipe.terms) != 3:
        raise ValueError("sign_agree needs terms [king, donor, base]")
    args = recipe.mode_args
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    beta = float(args["beta"])
    fill_untouched = bool(args.get("fill_untouched", False))
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(3)]
    with _merge_slot(target.device):
        for start in range(0, rows, step):
            end = min(rows, start + step)
            k, d, b = ((p[start:end] if target.dim() else p[:]).to(target.device).float() for p in parts)
            tau_king, tau_donor = k - b, d - b
            agree = (tau_king * tau_donor) > 0
            if fill_untouched:
                agree |= tau_king == 0
            view = target[start:end] if target.dim() else target
            view.copy_(k.add_(tau_donor.mul_(agree), alpha=beta))


def _fill_orthogonal(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """king + beta·(tau_donor minus its projection on tau_king), per tensor, tau = x − base.

    Keeps only what the donor lineage learned that the king's lineage did not (the
    shared component would just extrapolate the king along its own path).
    """
    if len(recipe.terms) != 3:
        raise ValueError("orthogonal needs terms [king, donor, base]")
    args = recipe.mode_args
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    beta = float(args["beta"])
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(3)]

    def chunks():
        for start in range(0, rows, step):
            end = min(rows, start + step)
            yield start, end, [(p[start:end] if target.dim() else p[:]).to(target.device).float() for p in parts]

    with _merge_slot(target.device):
        cross = norm = 0.0
        for _, _, (k, d, b) in chunks():  # pass 1: projection coefficient (fp64 accumulation)
            tau_king, tau_donor = k.sub_(b), d.sub_(b)
            cross += float((tau_king * tau_donor).sum(dtype=torch.float64))
            norm += float((tau_king * tau_king).sum(dtype=torch.float64))
        coef = cross / norm if norm > 0 else 0.0
        for start, end, (k, d, b) in chunks():  # pass 2: apply
            tau_king = k - b
            update = d.sub_(b).sub_(tau_king, alpha=coef)
            view = target[start:end] if target.dim() else target
            view.copy_(k.add_(update, alpha=beta))


def _seed_for(*parts) -> int:
    digest = hashlib.blake2b(":".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**63)


def _keep_mask(shape, keep: float, device, *seed_parts) -> torch.Tensor:
    """Deterministic Bernoulli(keep) mask (same draws in eval and in write_merge)."""
    generator = torch.Generator(device=device).manual_seed(_seed_for(*seed_parts))
    return torch.rand(shape, generator=generator, device=device) < keep


def _fill_whole_tensor(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """dare / ties / dare_ties / slerp, chunked with per-tensor statistics where needed."""
    mode, args = recipe.mode, recipe.mode_args
    n_terms = 2 if mode == "slerp" else 3
    if len(recipe.terms) != n_terms:
        raise ValueError(f"{mode} needs {n_terms} terms")
    first = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(first.get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(n_terms)]
    spans = [(s, min(rows, s + step)) for s in range(0, rows, step)]

    def load(start: int, end: int) -> list[torch.Tensor]:
        return [(p[start:end] if target.dim() else p[:]).to(device).float() for p in parts]

    def out(start: int, end: int) -> torch.Tensor:
        return target[start:end] if target.dim() else target

    with _merge_slot(device):
        if mode == "slerp":
            t = float(args["t"])
            dot = norm_a = norm_b = 0.0
            for start, end in spans:
                a, b = load(start, end)
                dot += float((a * b).sum(dtype=torch.float64))
                norm_a += float((a * a).sum(dtype=torch.float64))
                norm_b += float((b * b).sum(dtype=torch.float64))
            cos = max(-1.0, min(1.0, dot / math.sqrt(max(norm_a * norm_b, 1e-30))))
            theta = math.acos(cos)
            if theta < 1e-4:
                w_a, w_b = 1.0 - t, t
            else:
                w_a, w_b = math.sin((1 - t) * theta) / math.sin(theta), math.sin(t * theta) / math.sin(theta)
            for start, end in spans:
                a, b = load(start, end)
                out(start, end).copy_(a.mul_(w_a).add_(b, alpha=w_b))
        elif mode == "ties":
            density, beta = float(args["density"]), float(args["beta"])
            stride = max(1, target.numel() // 4_000_000)
            sample = []
            for start, end in spans:
                _, d, b = load(start, end)
                sample.append((d - b).abs().flatten()[::stride])
            pooled = torch.cat(sample)
            threshold = torch.quantile(pooled, 1.0 - density).item()
            top_drop = float(args.get("top_drop", 0.0))  # model breadcrumbs: also drop the largest entries
            ceiling = torch.quantile(pooled, 1.0 - top_drop).item() if top_drop > 0 else float("inf")
            for start, end in spans:
                k, d, b = load(start, end)
                tau = d.sub_(b)
                tau.mul_((tau.abs() >= threshold) & (tau.abs() < ceiling))
                out(start, end).copy_(k.add_(tau, alpha=beta))
        elif mode == "dare":
            drop, beta = float(args["drop"]), float(args["beta"])
            for start, end in spans:
                k, d, b = load(start, end)
                tau = d.sub_(b).mul_(_keep_mask(d.shape, 1.0 - drop, device, name, start, "dare"))
                out(start, end).copy_(k.add_(tau, alpha=beta / (1.0 - drop)))
        else:  # dare_ties, mergekit semantics: base + lambda·TIES(DARE(tau_king), DARE(tau_donor))
            weights = [float(w) for w in args["weights"]]
            densities = [float(x) for x in args["densities"]]
            lam = float(args.get("lambda", 1.0))
            normalize = bool(args.get("normalize", True))
            for start, end in spans:
                k, d, b = load(start, end)
                taus = []
                for index, (x, density) in enumerate(zip((k, d), densities)):
                    tau = x - b
                    if density < 1.0:
                        tau.mul_(_keep_mask(tau.shape, density, device, name, start, index)).div_(density)
                    taus.append(tau)
                elected = torch.sign(sum(w * tau for w, tau in zip(weights, taus)))
                numerator = torch.zeros_like(b)
                denominator = torch.zeros_like(b)
                for w, tau in zip(weights, taus):
                    agree = (torch.sign(tau) == elected) & (tau != 0)
                    numerator.add_(tau * agree, alpha=w)
                    denominator.add_(agree.float(), alpha=w)
                merged = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-12), 0.0) if normalize else numerator
                out(start, end).copy_(b.add_(merged, alpha=lam))


def _fill_delta_agree(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """terms = [king, delta1, delta2] (delta checkpoints). Per element, where the two sibling
    deltas agree in sign use (a_agree, b_agree), elsewhere (a_dis, b_dis)."""
    if len(recipe.terms) != 3:
        raise ValueError("delta_agree needs terms [king, delta1, delta2]")
    args = recipe.mode_args
    a1, a0 = float(args["a_agree"]), float(args.get("a_dis", 0.0))
    b1, b0 = float(args["b_agree"]), float(args.get("b_dis", 0.0))
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(3)]
    with _merge_slot(device):
        for start in range(0, rows, step):
            end = min(rows, start + step)
            k, d1, d2 = ((p[start:end] if target.dim() else p[:]).to(device).float() for p in parts)
            agree = ((d1 * d2) > 0).float()
            out = k.add_(d1 * (a0 + (a1 - a0) * agree)).add_(d2 * (b0 + (b1 - b0) * agree))
            (target[start:end] if target.dim() else target).copy_(out)


def _fill_delta_orthogonal(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """terms = [king, prev_king, delta1, delta2]: king + sum_i beta_i * (delta_i minus its projection
    on the king's own last step s = king - prev_king), per tensor. Keeps only what each sibling step
    adds beyond the direction the king just moved in."""
    if len(recipe.terms) != 4:
        raise ValueError("delta_orthogonal needs terms [king, prev_king, delta1, delta2]")
    args = recipe.mode_args
    betas = [float(args.get("beta1", 0.0)), float(args.get("beta2", 0.0))]
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(4)]

    def chunks():
        for start in range(0, rows, step):
            end = min(rows, start + step)
            yield start, end, [(p[start:end] if target.dim() else p[:]).to(device).float() for p in parts]

    with _merge_slot(device):
        dots, norm = [0.0, 0.0], 0.0
        for _, _, (k, pk, d1, d2) in chunks():
            s_ = k - pk
            norm += float((s_ * s_).sum(dtype=torch.float64))
            dots[0] += float((s_ * d1).sum(dtype=torch.float64))
            dots[1] += float((s_ * d2).sum(dtype=torch.float64))
        coefs = [d / norm if norm > 0 else 0.0 for d in dots]
        for start, end, (k, pk, d1, d2) in chunks():
            s_ = k - pk
            out = k.add_(d1.sub_(s_, alpha=coefs[0]), alpha=betas[0]).add_(d2.sub_(s_, alpha=coefs[1]), alpha=betas[1])
            (target[start:end] if target.dim() else target).copy_(out)


def _fill_delta_orthogonal_pair(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """terms = [king X, its parent P, checkpoint O, its start S]: X + beta * ((O - S) minus its projection on the
    king's own step X - P), per tensor. The transplant of a training delta onto a king without replaying the direction
    the king itself just moved in (extrapolating a king along its own step measured negative). mode_args: beta, exclude."""
    if len(recipe.terms) != 4:
        raise ValueError("delta_orthogonal_pair needs terms [king, parent, checkpoint, start]")
    args = recipe.mode_args
    beta = float(args.get("beta", 1.0))
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])) or beta == 0.0:
        target.copy_(king.get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(4)]

    def chunks():
        for start in range(0, rows, step):
            end = min(rows, start + step)
            yield start, end, [(p[start:end] if target.dim() else p[:]).to(device).float() for p in parts]

    with _merge_slot(device):
        dot, norm = 0.0, 0.0
        for _, _, (k, pk, o, s0) in chunks():
            s_ = k - pk
            norm += float((s_ * s_).sum(dtype=torch.float64))
            dot += float((s_ * (o - s0)).sum(dtype=torch.float64))
        coef = dot / norm if norm > 0 else 0.0
        for start, end, (k, pk, o, s0) in chunks():
            d = (o - s0).sub_(k - pk, alpha=coef)
            out = k.add_(d, alpha=beta)
            (target[start:end] if target.dim() else target).copy_(out)


def _fill_delta_sparse(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """terms = [king, delta1, ..., deltaN] (delta checkpoints); args betas=[...] plus
    delta_ties: density (keep top-|x| fraction per tensor per delta), top_drop (breadcrumbs)
    delta_dare: drop (Bernoulli drop prob, deterministic per tensor/chunk), rescale 1/(1-drop)."""
    args = recipe.mode_args
    betas = [float(b) for b in args["betas"]]
    if len(recipe.terms) != len(betas) + 1:
        raise ValueError("delta_ties/delta_dare need terms [king, delta1..deltaN] and betas of length N")
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    parts = [reader.handle(i, name).get_slice(name) for i in range(len(recipe.terms))]
    spans = [(s, min(rows, s + step)) for s in range(0, rows, step)]

    def load(i, start, end):
        return (parts[i][start:end] if target.dim() else parts[i][:]).to(device).float()

    with _merge_slot(device):
        bounds = []
        if recipe.mode == "delta_ties":
            density, top_drop = float(args["density"]), float(args.get("top_drop", 0.0))
            stride = max(1, target.numel() // 4_000_000)
            for i in range(1, len(parts)):
                sample = torch.cat([load(i, a, b).abs().flatten()[::stride] for a, b in spans])
                low = torch.quantile(sample, 1.0 - density).item() if density < 1.0 else 0.0
                high = torch.quantile(sample, 1.0 - top_drop).item() if top_drop > 0 else float("inf")
                bounds.append((low, high))
        else:
            drop = float(args["drop"])
        for start, end in spans:
            out = load(0, start, end)
            for i in range(1, len(parts)):
                tau = load(i, start, end)
                if recipe.mode == "delta_ties":
                    low, high = bounds[i - 1]
                    tau.mul_((tau.abs() >= low) & (tau.abs() < high))
                    out.add_(tau, alpha=betas[i - 1])
                else:
                    tau.mul_(_keep_mask(tau.shape, 1.0 - drop, device, name, start, i))
                    out.add_(tau, alpha=betas[i - 1] / (1.0 - drop))
            (target[start:end] if target.dim() else target).copy_(out)



def _fill_model_stock(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """Model Stock (Jang et al., 2024) over k fine-tunes of one base, per tensor:
      tau_i = term[m_i] - term[base];  cos = mean pairwise cosine of the tau_i;  t = k cos / (1 + (k-1) cos)
      W = term[on] + scale * t * mean_i(tau_i)     (on = base, scale 1: plain Model Stock, t avg(W_i) + (1-t) W_0)
    mode_args: base (term index, default 0), models (term indices), on (default base), scale (default 1), exclude."""
    args = recipe.mode_args
    base_i, on_i = int(args.get("base", 0)), int(args.get("on", args.get("base", 0)))
    ms, scale = [int(i) for i in args["models"]], float(args.get("scale", 1.0))
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(reader.handle(on_i, name).get_tensor(name))
        return
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4 * (len(ms) + 1)))
    parts = {i: reader.handle(i, name).get_slice(name) for i in sorted({base_i, on_i, *ms})}
    spans = [(s_, min(rows, s_ + step)) for s_ in range(0, rows, step)]

    def load(i, start, end):
        return (parts[i][start:end] if target.dim() else parts[i][:]).to(device).float()

    k = len(ms)
    with _merge_slot(device):
        dots = torch.zeros(k, k, dtype=torch.float64)
        for start, end in spans:
            w0 = load(base_i, start, end)
            taus = torch.stack([load(i, start, end).sub_(w0).flatten() for i in ms]).double()
            dots += (taus @ taus.T).cpu()
            del taus, w0
        norms = dots.diag().clamp_min(1e-30).sqrt()
        cosm = dots / (norms[:, None] * norms[None, :])
        cos = float((cosm.sum() - cosm.diag().sum()) / (k * (k - 1))) if k > 1 else 1.0
        cos = max(0.0, min(1.0, cos))
        t = k * cos / (1 + (k - 1) * cos) if k > 1 else 1.0
        for start, end in spans:
            w0 = load(base_i, start, end)
            delta = sum(load(i, start, end) for i in ms).div_(k).sub_(w0)
            dst = target[start:end] if target.dim() else target
            dst.copy_((w0 if on_i == base_i else load(on_i, start, end)).add_(delta, alpha=scale * t))


def _fill_vec_ties(target: torch.Tensor, name: str, reader: _Reader, recipe: Recipe) -> None:
    """TIES / DARE-TIES over task vectors relative to the king (term 0).

    mode_args: vecs = [[a, b, w], ...] (task vector V_i = term[a] - term[b], weight w_i), lambda (scale of the
    merged vector, default 1), density (TIES trim: keep the top-|x| fraction of each V_i per tensor, default 1),
    drop (DARE: Bernoulli drop with 1/(1-drop) rescale instead of trimming), elect ("sum": sign of sum_i w_i V_i,
    default), merge ("sum": sum of agreeing w_i V_i, "mean": disjoint mean, "max": elected sign x the largest agreeing
    |w_i V_i| as in EMR-merging), top_drop (breadcrumbs), della (with drop: DELLA, keep probability 1-drop +- della
    ranked by |x| within the chunk, small entries dropped more, rescaled by 1/p), exclude.
      W = king + lambda * merge_i{ w_i * keep_i(V_i) agreeing with the elected sign }
    """
    args = recipe.mode_args
    king = reader.handle(0, name)
    if any(re.search(p, name) for p in args.get("exclude", [])):
        target.copy_(king.get_tensor(name))
        return
    # each vec: [a, b, w] or [a, b, w, filter]; filter=False adds w*V_i unmasked (no trim / drop / election)
    vecs = [(int(v[0]), int(v[1]), float(v[2])) for v in args["vecs"]]
    filt = [bool(v[3]) if len(v) > 3 else True for v in args["vecs"]]
    ref = args.get("elect_ref")  # [a, b]: elect signs from term[a]-term[b] instead of sum_i w_i V_i
    lam = float(args.get("lambda", 1.0))
    density = float(args.get("density", 1.0))
    drop = float(args.get("drop", 0.0))
    della = float(args.get("della", 0.0))
    top_drop = float(args.get("top_drop", 0.0))
    merge = args.get("merge", "sum")
    device = target.device
    rows = target.shape[0] if target.dim() else 1
    step = max(1, MERGE_CHUNK_BYTES // (max(1, target.numel() // max(rows, 1)) * 4))
    # elect_ref: [a, b] or a list of such pairs (summed), e.g. [[p40_36, -1], [d36_35, -1]] = #40 - #35
    refs = ([[int(x) for x in r] for r in ref] if ref and isinstance(ref[0], (list, tuple)) else
            ([[int(ref[0]), int(ref[1])]] if ref else []))
    used = sorted(({0} | {i for a, b, _ in vecs for i in (a, b)} | {i for r in refs for i in r}) - {-1})
    parts = {i: reader.handle(i, name).get_slice(name) for i in used}
    spans = [(s_, min(rows, s_ + step)) for s_ in range(0, rows, step)]

    def load(i, start, end):
        return (parts[i][start:end] if target.dim() else parts[i][:]).to(device).float()

    def vec(a, b, start, end):  # b == -1: term a is itself a task-vector (delta) checkpoint
        return load(a, start, end) if b < 0 else load(a, start, end).sub_(load(b, start, end))

    with _merge_slot(device):
        bounds = []
        if density < 1.0 or top_drop > 0:
            stride = max(1, target.numel() // 4_000_000)
            for (a, b, _), f in zip(vecs, filt):
                if not f:
                    bounds.append((0.0, float("inf")))
                    continue
                sample = torch.cat([vec(a, b, x, y).abs().flatten()[::stride] for x, y in spans])
                low = torch.quantile(sample, 1.0 - density).item() if density < 1.0 else 0.0
                high = torch.quantile(sample, 1.0 - top_drop).item() if top_drop > 0 else float("inf")
                bounds.append((low, high))
        for start, end in spans:
            k = load(0, start, end)
            taus, plain = [], torch.zeros_like(k)
            for index, (a, b, w) in enumerate(vecs):
                tau = vec(a, b, start, end)
                if not filt[index]:
                    plain.add_(tau, alpha=w)
                    continue
                if bounds:
                    low, high = bounds[index]
                    tau.mul_((tau.abs() >= low) & (tau.abs() < high))
                if drop > 0 and della > 0:  # DELLA: magnitude-ranked keep probabilities
                    flat = tau.abs().flatten()
                    ranks = torch.empty_like(flat)
                    ranks[flat.argsort()] = torch.arange(flat.numel(), device=device, dtype=flat.dtype)
                    keep_p = ((1.0 - drop - della) + 2 * della * ranks / max(1, flat.numel() - 1)).clamp_(0.01, 1.0).view_as(tau)
                    gen = torch.Generator(device=device).manual_seed(_seed_for(name, start, "della", index))
                    tau.mul_(torch.rand(tau.shape, generator=gen, device=device) < keep_p).div_(keep_p)
                    del flat, ranks, keep_p
                elif drop > 0:
                    tau.mul_(_keep_mask(tau.shape, 1.0 - drop, device, name, start, "vec_ties", index)).div_(1.0 - drop)
                taus.append((w, tau))
            if refs:
                elected = torch.sign(sum((vec(a, b, start, end) for a, b in refs), torch.zeros_like(k)))
            else:
                elected = torch.sign(sum((w * tau for w, tau in taus), torch.zeros_like(k)))
            numerator = torch.zeros_like(k)
            count = torch.zeros_like(k)
            for w, tau in taus:
                agree = (torch.sign(tau) == elected) & (tau != 0)
                if merge == "max":
                    torch.maximum(numerator, (tau * w).abs_().mul_(agree), out=numerator)
                else:
                    numerator.add_(tau * agree, alpha=w)
                count.add_(agree.float())
            if merge == "max":
                merged = elected * numerator
            else:
                merged = numerator if merge == "sum" else torch.where(count > 0, numerator / count.clamp_min(1.0), 0.0)
            (target[start:end] if target.dim() else target).copy_(k.add_(merged, alpha=lam).add_(plain))


def _check_keys(model: nn.Module, reader: _Reader) -> None:
    expected = set(model.state_dict().keys())
    for index, weight_map in enumerate(reader.maps):
        present = set(weight_map)
        mtp = sorted(k for k in present if "mtp" in k.lower())
        if mtp:
            raise RuntimeError(f"{reader.recipe.terms[index].path}: MTP tensors present {mtp[:4]}")
        missing, unexpected = expected - present, present - expected
        if missing or unexpected:
            raise RuntimeError(
                f"{reader.recipe.terms[index].path}: missing={sorted(missing)[:4]} "
                f"unexpected={sorted(unexpected)[:4]}"
            )


@dataclass
class LoadedModel:
    model: nn.Module
    recipe: Recipe
    layer_devices: list[int]
    attn_implementation: str
    load_seconds: float


def shard_unit(name: str) -> str:
    """Exchange unit of a parameter: routed experts travel as their stacked (layer, proj) buffer."""
    m = re.match(r"(.*\.experts)\.\d+\.(gate_proj|up_proj|down_proj)\.weight$", name)
    return f"{m.group(1)}.*.{m.group(2)}" if m else name


def shard_owner(unit: str, n_shards: int) -> int:
    """Which of n replicas reads + merges this unit (whole decoder layers round-robin; embed / head / norm spread)."""
    m = re.match(r"model\.layers\.(\d+)\.", unit)
    if m:
        return int(m.group(1)) % n_shards
    return {"model.embed_tokens.weight": 0, "lm_head.weight": 1}.get(unit, 2) % n_shards


def shard_units(model: nn.Module) -> dict[str, torch.Tensor]:
    """Every distinct parameter storage of a loaded model, keyed by shard_unit()."""
    units: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        for proj, buffer in (getattr(module, "_parity_stacked", None) or {}).items():
            units[f"{name}.experts.*.{proj}"] = buffer
    for name, param in model.named_parameters():
        unit = shard_unit(name)
        if unit == name:
            units[name] = param
    return units


def load_model(
    recipe: Recipe,
    gpu_ids: list[int],
    *,
    attn_implementation: str = FA4SIM,
    snapshot_dir: Path | None = None,
    io_threads: int = 32,
    shard: tuple[int, int] | None = None,
) -> LoadedModel:
    """shard=(rank, n): fill only the units shard_owner() gives this rank (the caller exchanges the rest with the
    other n-1 replicas, see src.parity.mppool); every other parameter is left allocated but unfilled."""
    started = time.time()
    snapshot_dir = snapshot_dir or recipe.terms[0].path
    with _SKELETON_LOCK, _no_uninitialized_fill():  # replicas load concurrently; dynamic-module import is not thread-safe
        model = build_skeleton(snapshot_dir, attn_implementation)
        layer_devices = plan_layer_devices(model, gpu_ids)
    reader = _Reader(recipe)
    _check_keys(model, reader)
    with _no_uninitialized_fill():
        placement = _allocate(model, layer_devices)
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unallocated parameters: {meta[:4]}")

    names = sorted(placement)
    if shard is not None:
        names = [n for n in names if shard_owner(shard_unit(n), shard[1]) == shard[0]]
    done = [0]
    lock = threading.Lock()

    def work(name: str) -> None:
        with torch.no_grad(), _pinned_reads():
            _fill(placement[name], name, reader, recipe)
        with lock:
            done[0] += 1
            if done[0] % 4000 == 0:
                log.info("loaded %d/%d tensors (%.0fs)", done[0], len(names), time.time() - started)

    with ThreadPoolExecutor(max_workers=io_threads) as pool:
        list(pool.map(work, names))
    for gpu in sorted(set(layer_devices)):
        torch.cuda.synchronize(gpu)
        # Merge temporaries leave GBs cached in PyTorch's allocator; hand them back so
        # scoring threads can still cudaMalloc their cuBLAS workspaces (else
        # CUBLAS_STATUS_ALLOC_FAILED on the 5090's thin headroom).
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()

    for module in model.modules():
        if module.__class__.__name__ == "MiMoV2MoE":
            module.moe = grouped_moe.__get__(module)
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = attn_implementation
    model.hf_device_map = {
        **{f"model.layers.{i}": d for i, d in enumerate(layer_devices)},
        "model.embed_tokens": layer_devices[0],
        "model.norm": layer_devices[-1],
        "lm_head": layer_devices[-1],
    }
    elapsed = time.time() - started
    log.info(
        "loaded %s in %.0fs; layers per GPU: %s",
        recipe.label or recipe.identity(),
        elapsed,
        {gpu: layer_devices.count(gpu) for gpu in sorted(set(layer_devices))},
    )
    return LoadedModel(model, recipe, layer_devices, attn_implementation, elapsed)


def clone_model(source: LoadedModel, gpu_ids: list[int]) -> LoadedModel:
    """Copy an already-loaded model onto other GPUs (device-to-device over NVLink, no disk reads).

    Builds the same skeleton/placement as :func:`load_model` and copies every parameter from
    ``source`` (routed experts as whole stacked buffers), so the clone is bit-identical.
    """
    started = time.time()
    with _SKELETON_LOCK, _no_uninitialized_fill():
        model = build_skeleton(source.recipe.terms[0].path, source.attn_implementation)
        layer_devices = plan_layer_devices(model, gpu_ids)
    with _no_uninitialized_fill():
        placement = _allocate(model, layer_devices)
    src_modules = dict(source.model.named_modules())
    src_params = dict(source.model.named_parameters())
    copied: set[str] = set()
    with torch.no_grad():
        for name, module in model.named_modules():
            stacked = getattr(module, "_parity_stacked", None)
            if stacked:
                for proj, buffer in stacked.items():
                    buffer.copy_(src_modules[name]._parity_stacked[proj], non_blocking=True)
                    copied.update(f"{name}.experts.{i}.{proj}.weight" for i in range(buffer.shape[0]))
        for key, tensor in placement.items():
            if key not in copied:
                tensor.copy_(src_params[key], non_blocking=True)
    for gpu in sorted(set(layer_devices)):
        torch.cuda.synchronize(gpu)
    for module in model.modules():
        if module.__class__.__name__ == "MiMoV2MoE":
            module.moe = grouped_moe.__get__(module)
    model.eval()
    model.config.use_cache = False
    model.config._attn_implementation = source.attn_implementation
    model.hf_device_map = {
        **{f"model.layers.{i}": d for i, d in enumerate(layer_devices)},
        "model.embed_tokens": layer_devices[0],
        "model.norm": layer_devices[-1],
        "lm_head": layer_devices[-1],
    }
    elapsed = time.time() - started
    log.info("cloned %s to GPUs %s in %.0fs", source.recipe.label or source.recipe.identity(), gpu_ids, elapsed)
    return LoadedModel(model, source.recipe, layer_devices, source.attn_implementation, elapsed)


def free_model(loaded: LoadedModel) -> None:
    devices = sorted(set(loaded.layer_devices))
    loaded.model = None
    import gc

    gc.collect()
    for device in devices:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
