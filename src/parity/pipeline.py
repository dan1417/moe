"""Pipelined scoring over a layer-sharded model using the validator's CE function.

Each GPU is a pipeline stage guarded by a lock. Worker threads call the validator's
``compute_per_sequence_loss`` on one batch at a time; a forward pre-hook on the first
layer of each stage hands the thread over to the next stage's lock, so up to one batch
per GPU is in flight -- the N-GPU generalization of the validator's own
``TwoGpuSequencePipeline``.

Since validator 14a9bb2 that function scores document-masked: EOS-delimited fragments get
their own positions and attention, so ``position_ids``/``position_embeddings``/masks differ
from batch to batch (only ``cache_position`` is static), and samples of unequal length
(2048-token windows and 2049-8192-token documents) are packed into one row. Batches follow
the validator's ``scoring_batch_end`` token/attention budgets at our batch size.
"""

from __future__ import annotations

import logging
import threading
import time
from functools import partial

import numpy as np
import torch

from src.parity import upstream
from src.parity.weights import LoadedModel

log = logging.getLogger(__name__)

# Keyword inputs that are identical for every batch of the same width; their per-device copies
# are cached across batches. (Under document masking the positions, rotary embeddings and masks
# depend on where each batch's EOS tokens are: those are copied once per batch and stage instead.)
STATIC_KWARGS = frozenset({"cache_position"})
SEQ_LEN = 2048


def serialize_fa4_host_calls() -> None:
    """One thread at a time in FA4's host entry point (``flash_attn.cute.interface._flash_attn_fwd``).

    Its first call per configuration JIT-compiles the kernel with the CuTe DSL / MLIR, which is not thread-safe:
    4 replica threads compiling at once segfaulted in ``generate_mlir_function_types`` (B300, 2026-10-09), and 10
    pipeline threads crawled. Under the lock only the host-side setup and the asynchronous launch run (milliseconds);
    the GPU work of different replicas still overlaps. The wrapper shares the original's ``compile_cache``, which the
    function body looks up through its own global name.
    """
    import threading

    from flash_attn.cute import interface

    original = interface._flash_attn_fwd
    if getattr(original, "_parity_locked", False):
        return
    lock = threading.Lock()

    def locked(*args, **kwargs):
        with lock:
            return original(*args, **kwargs)

    locked._parity_locked = True
    locked.compile_cache = original.compile_cache
    interface._flash_attn_fwd = locked


def batch_end(lengths: list[int], start: int, batch_size: int, window_length: int = SEQ_LEN) -> int:
    """The validator's ``scoring_batch_end``: up to ``batch_size`` samples, capped by input tokens
    (batch_size windows) and by attention work (sum of squared lengths); one long document alone."""
    token_budget, attention_budget = batch_size * window_length, batch_size * window_length**2
    tokens = attention = 0
    end = start
    while end < min(start + batch_size, len(lengths)):
        length = lengths[end]
        if end > start and (tokens + length > token_budget or attention + length**2 > attention_budget):
            break
        tokens += length
        attention += length**2
        end += 1
    return end


class PipelineScorer:
    def __init__(self, loaded: LoadedModel, *, lm_head_chunk: int = 1024, verify_static: int = 4):
        self.model = loaded.model
        self.layer_devices = list(loaded.layer_devices)
        self.stage_devices = list(dict.fromkeys(self.layer_devices))
        self.stage_of_layer = [self.stage_devices.index(d) for d in self.layer_devices]
        self.lm_head_chunk = lm_head_chunk
        self.locks = [threading.Lock() for _ in self.stage_devices]
        self.local = threading.local()
        self.static: dict[tuple, torch.Tensor] = {}
        self.verify_static = verify_static
        self.hooks = [
            layer.register_forward_pre_hook(partial(self._pre_layer, index), with_kwargs=True, prepend=True)
            for index, layer in enumerate(self.model.model.layers)
        ]
        last = torch.device(f"cuda:{self.layer_devices[-1]}")
        self.hooks.append(
            self.model.model.norm.register_forward_pre_hook(
                lambda module, args: tuple(a.to(last) if isinstance(a, torch.Tensor) else a for a in args)
            )
        )

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    # ---------------------------------------------------------------- hooks
    def _pre_layer(self, index: int, module, args, kwargs):
        stage = self.stage_of_layer[index]
        state = getattr(self.local, "state", None)
        if state is not None and stage != state["held"]:
            self.locks[stage].acquire()
            self.locks[state["held"]].release()
            state["held"] = stage
        device = torch.device(f"cuda:{self.layer_devices[index]}")
        args = tuple(self._move(value, device, None) for value in args)
        memo = state["moved"] if state is not None else {}
        kwargs = {
            key: self._move(value, device, (key, module.attention_type) if key in STATIC_KWARGS else None, memo)
            for key, value in kwargs.items()
        }
        return args, kwargs

    def _move(self, value, device: torch.device, static_key, memo: dict | None = None):
        if isinstance(value, torch.Tensor):
            if value.device == device:
                return value
            if static_key is None:
                if memo is None:
                    return value.to(device)
                # per-batch inputs (positions, rotary embeddings, masks): one copy per stage per batch;
                # the memo holds the source tensor, so its id cannot be reused while the batch runs
                hit = memo.get((id(value), device))
                if hit is None:
                    hit = memo[(id(value), device)] = (value, value.to(device))
                return hit[1]
            key = (static_key, device, tuple(value.shape), value.dtype)
            cached = self.static.get(key)
            if cached is None:
                cached = self.static.setdefault(key, value.to(device))
            elif self.verify_static > 0 and not torch.equal(cached, value.to(device)):
                fresh = value.to(device)
                diff = (cached.float() - fresh.float()).abs()
                raise RuntimeError(
                    f"static input {static_key} changed between sequences: max |diff| {diff.max().item():.3e}, "
                    f"{int((diff > 0).sum())} of {diff.numel()} entries differ, first rows {diff.flatten(0, -2).amax(-1)[:4].tolist()}, "
                    f"cached[0,:2,:4] {cached.flatten(0, -3)[0, :2, :4].tolist()} fresh {fresh.flatten(0, -3)[0, :2, :4].tolist()}")
            return cached
        if isinstance(value, (tuple, list)):
            moved = [
                self._move(item, device, None if static_key is None else (*static_key, position), memo)
                for position, item in enumerate(value)
            ]
            return type(value)(moved)
        return value

    # --------------------------------------------------------------- scoring
    def score(
        self,
        tokens: np.ndarray,
        indices: list[int] | None = None,
        *,
        batch_size: int = 4,
        threads: int | None = None,
        log_every: int = 1000,
        on_result=None,
        fn=None,
        width: int = 1,
    ) -> np.ndarray:
        """Per-sample validator CE for ``tokens[indices]`` (NaN elsewhere): the masked mean over
        each sample's scored targets.

        ``tokens`` is an [n, 2048] array or a :class:`src.parity.sample.Sequences` of windows and
        whole documents. Batching only changes how many rows share each GEMM (per-row results
        differ at most by kernel rounding, as in the validator's own batch-40 scoring); batches
        are cut by the validator's token/attention budgets. ``fn(model, batches, lm_head_chunk)``
        may replace the validator's CE to collect other per-sample statistics (``width`` floats each).
        """
        n = int(tokens.shape[0])
        indices = list(range(n)) if indices is None else list(indices)
        losses = np.full((n, width) if width > 1 else n, np.nan, dtype=np.float64)
        # windows first, then documents by length: windows fill whole unpacked [batch, 2048] rows; only batches that
        # mix lengths are packed into one row, whose per-layer packed-sequence check syncs the host (results are
        # stored by index, so the order changes nothing but kernel rounding)
        indices.sort(key=lambda i: len(tokens[i]))
        lengths = [len(tokens[i]) for i in indices]
        cuts = [0]
        while cuts[-1] < len(indices):
            cuts.append(batch_end(lengths, cuts[-1], batch_size))
        batches = iter([indices[a:b] for a, b in zip(cuts, cuts[1:])])
        order_lock = threading.Lock()
        errors: list[BaseException] = []
        done = [0]
        started = time.time()
        engine = upstream.engine

        def worker() -> None:
            while not errors:
                with order_lock:
                    batch = next(batches, None)
                if batch is None:
                    return
                sequences = [np.asarray(tokens[i], dtype=np.int64).tolist() for i in batch]
                state = {"held": 0, "moved": {}}
                self.locks[0].acquire()
                self.local.state = state
                try:
                    if fn is None:
                        scored = engine.compute_per_sequence_loss(
                            self.model, sequences, self.lm_head_chunk, reset_peak_memory=False
                        )
                    else:
                        scored = fn(self.model, sequences, self.lm_head_chunk)
                except BaseException as exc:  # surface after joining
                    errors.append(exc)
                    return
                finally:
                    self.local.state = None
                    self.locks[state["held"]].release()
                for index, loss in zip(batch, scored, strict=True):
                    losses[index] = loss
                    if on_result is not None:
                        on_result(index, loss)
                with order_lock:
                    before = done[0]
                    done[0] += len(batch)
                    count = done[0]
                    if count >= self.verify_static:
                        self.verify_static = 0
                if log_every and count // log_every != before // log_every:
                    rate = count / max(time.time() - started, 1e-9)
                    log.info(
                        "scored %d/%d (%.2f seq/s, eta %.0fs)",
                        count, len(indices), rate, (len(indices) - count) / max(rate, 1e-9),
                    )

        workers = [
            threading.Thread(target=worker, name=f"parity-score-{i}", daemon=True)
            for i in range(threads or len(self.stage_devices) + 2)
        ]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()
        if errors:
            raise errors[0]
        elapsed = time.time() - started
        log.info("scored %d sequences in %.0fs (%.2f seq/s)", len(indices), elapsed, len(indices) / max(elapsed, 1e-9))
        return losses
