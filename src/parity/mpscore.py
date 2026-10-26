"""One process per model replica for parity scoring (no GIL contention between replicas).

With replicas as threads of one process, the host-side work of every forward (Python layer code, kernel launches,
packed-row syncs) competes for one GIL and the GPUs idle at 15-40 % (B300, masked scoring, 4 replicas x 2 GPUs).
Here every replica is its own spawned process: it loads the recipe onto its GPU group from disk (page cache), scores
its interleaved slice of each sample with :class:`src.parity.pipeline.PipelineScorer` and streams
(sample, index, loss) back; the parent fills and periodically saves the loss cache exactly like the threaded path.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
import traceback

import numpy as np

log = logging.getLogger(__name__)
FLUSH = 256  # losses per message


def _worker(rank: int, gpus: list[int], recipe, attn: str, lm_head_chunk: int, batch_size: int, deterministic: bool,
            jobs: list[tuple[str, list[int]]], out) -> None:
    try:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        import torch

        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.use_deterministic_algorithms(True, warn_only=False)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        from src.parity import upstream
        from src.parity.pipeline import PipelineScorer, serialize_fa4_host_calls
        from src.parity.sample import load_sample, sample_dir
        from src.parity.weights import free_model, load_model

        started = time.time()
        loaded = load_model(recipe, gpus, attn_implementation=attn)
        if attn == "flash_attention_4":
            from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

            lazy_import_flash_attention(attn)
            serialize_fa4_host_calls()
        out.put(("loaded", rank, round(time.time() - started, 1)))
        scorer = PipelineScorer(loaded, lm_head_chunk=lm_head_chunk)
        try:
            for sample_id, indices in jobs:
                sample = load_sample(sample_dir(upstream.DEFAULT_CACHE_ROOT, sample_id))
                buf: list[tuple[int, float]] = []
                lock = threading.Lock()

                def on_result(index, loss, buf=buf, lock=lock, sample_id=sample_id):
                    with lock:
                        buf.append((int(index), float(loss)))
                        if len(buf) >= FLUSH:
                            out.put(("losses", sample_id, buf.copy()))
                            buf.clear()

                scorer.score(sample.tokens, indices, batch_size=batch_size, on_result=on_result, log_every=0)
                with lock:
                    if buf:
                        out.put(("losses", sample_id, buf.copy()))
                        buf.clear()
        finally:
            scorer.close()
            free_model(loaded)
        out.put(("done", rank, None))
    except BaseException:  # reported to the parent, which stops every replica
        out.put(("error", rank, traceback.format_exc()))


def score_replicas(recipe, groups: list[list[int]], wanted: list, *, attn: str, lm_head_chunk: int, batch_size: int,
                   deterministic: bool, on_save) -> None:
    """Fill ``cached`` for every (sample, cached, missing) in ``wanted`` with one process per GPU group.

    ``on_save(sample, cached, elapsed)`` persists a sample's cache (called every ~2 minutes and at the end).
    """
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    n = len(groups)
    by_id = {sample.sample_id: (sample, cached) for sample, cached, _ in wanted}
    procs = []
    for rank, gpus in enumerate(groups):
        # interleaved slices: every replica sees the same mix of windows and documents
        jobs = [(sample.sample_id, missing[rank::n]) for sample, _, missing in wanted if missing[rank::n]]
        p = ctx.Process(target=_worker, args=(rank, gpus, recipe, attn, lm_head_chunk, batch_size, deterministic, jobs, out),
                        daemon=True)
        p.start()
        procs.append(p)
    total = sum(len(missing) for _, _, missing in wanted)
    done = scored = 0
    started = last_save = last_log = time.time()
    try:
        while done < n:
            try:
                kind, key, payload = out.get(timeout=60)
            except queue.Empty:
                dead = [r for r, p in enumerate(procs) if not p.is_alive() and p.exitcode not in (0, None)]
                if dead:
                    raise RuntimeError(f"scoring replicas {dead} exited without reporting")
                continue
            if kind == "error":
                raise RuntimeError(f"scoring replica {key} failed:\n{payload}")
            if kind == "loaded":
                log.info("replica %d loaded in %.0fs", key, payload)
                started = time.time() if scored == 0 else started
            elif kind == "done":
                done += 1
            elif kind == "losses":
                _, cached = by_id[key]
                for index, loss in payload:
                    cached[index] = loss
                scored += len(payload)
                now = time.time()
                if now - last_log > 30:
                    last_log = now
                    rate = scored / max(now - started, 1e-9)
                    log.info("total scored %d/%d (%.1f seq/s, eta %.0fs)", scored, total, rate, (total - scored) / max(rate, 1e-9))
                if now - last_save > 120:
                    last_save = now
                    for sample, cached in by_id.values():
                        on_save(sample, cached, now - started)
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=60)
    elapsed = time.time() - started
    log.info("scored %d sequences in %.0fs (%.1f seq/s) with %d replica processes", scored, elapsed, scored / max(elapsed, 1e-9), n)
    for sample, cached in by_id.values():
        on_save(sample, cached, elapsed)
    missing_after = sum(int(np.isnan(cached[missing]).sum()) for _, cached, missing in wanted)
    if missing_after:
        raise RuntimeError(f"{missing_after} requested sequences were not scored")
