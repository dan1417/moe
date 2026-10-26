"""Persistent replica processes in two sets on the same GPU groups ("ping-pong"): while one set scores recipe i,
the other set loads (reads + merges) recipe i+1 into the GPU memory the first set leaves free, so the GPUs never
wait for a load.

A 110B replica split over 2 B300s holds 103.5 GiB of weights per GPU and peaks at ~136 GiB while scoring, so a
second set's weights (103.5 GiB) fit next to it on the 268 GiB cards (~240 GiB peak). Processes stay alive across
recipes, so CUDA init and the FA4 JIT compile are paid once per process instead of once per recipe.

    pool = ReplicaPool(groups, attn=..., lm_head_chunk=..., batch_size=..., deterministic=True)
    pool.run(items, on_save, on_done)   # items: [(recipe, wanted)], wanted = [(sample, cached, missing)]
    pool.close()

Sharded loading (shard=True, default): the 4 replicas of a set each read + merge only a quarter of the decoder layers
(src.parity.weights.shard_owner) and then copy the other three quarters from their siblings GPU-to-GPU over NVLink
(CUDA IPC through torch.multiprocessing queues): a set reads each checkpoint once instead of four times (the page-cache
read rate of the box, ~12-28 GB/s, was the load bottleneck). After the exchange every replica sends a fingerprint of
all its weights; the parent refuses to score unless the replicas of a set agree.

Scoring is the same code as :mod:`src.parity.mpscore` (one PipelineScorer per replica, interleaved slices, losses
streamed back to the parent which fills ``cached``), so the losses are bit-identical to the single-recipe path.
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


def _fingerprint(units: dict) -> float:
    """Order-fixed checksum of every weight storage (strided sample, fp64): equal across replicas iff the
    exchange delivered every unit (an unfilled unit holds whatever the allocator handed out)."""
    import zlib

    total = 0.0
    for key in sorted(units):
        flat = units[key].reshape(-1)
        total += float(flat[:: max(1, flat.numel() // 65536)].double().sum()) * (1 + zlib.crc32(key.encode()) % 997 / 997)
    return total


def _exchange(model, rank: int, xqs: list) -> float:
    """Send this replica's units to its siblings (CUDA IPC) and copy theirs in over NVLink; returns seconds."""
    import torch

    from src.parity.weights import shard_owner, shard_units

    started = time.time()
    n = len(xqs)
    units = shard_units(model)
    mine = [(k, t) for k, t in units.items() if shard_owner(k, n) == rank]
    for other in range(n):
        if other != rank:
            xqs[other].put((rank, mine))
    devices = sorted({t.device.index for t in units.values()})
    for _ in range(n - 1):
        _, theirs = xqs[rank].get()
        with torch.no_grad():
            for key, tensor in theirs:
                units[key].copy_(tensor, non_blocking=True)
        for d in devices:
            torch.cuda.synchronize(d)
        theirs = tensor = None  # release the IPC mappings only after the copies completed
    return time.time() - started


def _pool_worker(set_id: int, rank: int, gpus: list[int], attn: str, lm_head_chunk: int, batch_size: int,
                 deterministic: bool, inq, out, xqs=None) -> None:
    try:
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        try:  # die with the parent (pm2 stop SIGKILLs it): an orphaned replica would keep ~100 GiB on two GPUs
            import ctypes
            import signal

            ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
        except OSError:
            pass
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

        fa4_ready = False
        samples: dict = {}
        loaded = scorer = None
        while True:
            cmd = inq.get()
            if cmd is None:
                break
            if cmd[0] == "load":
                started = time.time()
                loaded = load_model(cmd[1], gpus, attn_implementation=attn, shard=(rank, len(xqs)) if xqs else None)
                if xqs:
                    from src.parity.weights import shard_units

                    secs = _exchange(loaded.model, rank, xqs)
                    out.put(("fingerprint", set_id, rank, (_fingerprint(shard_units(loaded.model)), round(secs, 1))))
                if attn == "flash_attention_4" and not fa4_ready:
                    from transformers.modeling_flash_attention_utils import lazy_import_flash_attention

                    lazy_import_flash_attention(attn)
                    serialize_fa4_host_calls()
                    fa4_ready = True
                scorer = PipelineScorer(loaded, lm_head_chunk=lm_head_chunk)
                out.put(("loaded", set_id, rank, round(time.time() - started, 1)))
            elif cmd[0] == "score":
                item, jobs = cmd[1], cmd[2]
                for sample_id, indices in jobs:
                    if sample_id not in samples:
                        samples[sample_id] = load_sample(sample_dir(upstream.DEFAULT_CACHE_ROOT, sample_id))
                    buf: list[tuple[int, float]] = []
                    lock = threading.Lock()

                    def on_result(index, loss, buf=buf, lock=lock, sample_id=sample_id):
                        with lock:
                            buf.append((int(index), float(loss)))
                            if len(buf) >= FLUSH:
                                out.put(("losses", set_id, rank, (item, sample_id, buf.copy())))
                                buf.clear()

                    scorer.score(samples[sample_id].tokens, indices, batch_size=batch_size, on_result=on_result, log_every=0)
                    with lock:
                        if buf:
                            out.put(("losses", set_id, rank, (item, sample_id, buf.copy())))
                            buf.clear()
                # hand the memory back before reporting, so the parent can start the next load on this set
                scorer.close()
                free_model(loaded)
                loaded = scorer = None
                torch.cuda.ipc_collect()  # blocks siblings had mapped (they released them after copying)
                out.put(("done", set_id, rank, item))
    except BaseException:  # reported to the parent, which stops every process
        out.put(("error", set_id, rank, traceback.format_exc()))


class ReplicaPool:
    def __init__(self, groups: list[list[int]], *, attn: str, lm_head_chunk: int, batch_size: int, deterministic: bool,
                 sets: int = 2, worker=_pool_worker, shard: bool = True):
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.groups, self.sets = groups, sets
        self.out = ctx.Queue()
        self.inqs: dict[tuple[int, int], object] = {}
        self.procs = []
        self.fingerprints: dict[int, dict[int, tuple]] = {s: {} for s in range(sets)}
        self.xqs: dict[int, list | None] = {}
        for set_id in range(sets):
            # kept on self: a queue garbage-collected before a spawned child unpickles it takes its semaphore along
            xqs = self.xqs[set_id] = [ctx.Queue() for _ in groups] if shard and len(groups) > 1 else None
            for rank, gpus in enumerate(groups):
                inq = ctx.Queue()
                p = ctx.Process(target=worker, args=(set_id, rank, gpus, attn, lm_head_chunk, batch_size, deterministic,
                                                     inq, self.out, xqs), daemon=True)
                p.start()
                self.inqs[(set_id, rank)] = inq
                self.procs.append(p)

    def _send(self, set_id: int, msg) -> None:
        for rank in range(len(self.groups)):
            self.inqs[(set_id, rank)].put(msg)

    def run(self, items: list, on_save, on_done) -> None:
        """items: [(recipe, wanted)] with wanted = [(sample, cached, missing)] (non-empty). Scores the items in
        order; ``on_save(i, sample, cached, elapsed)`` persists a cache (every ~2 min and at the end of an item),
        ``on_done(i, load_seconds, score_seconds)`` is called when item i is complete."""
        n_ranks, n = len(self.groups), len(items)
        holder: dict[int, int] = {}  # set -> item it holds (loading or loaded)
        loaded = {s: 0 for s in range(self.sets)}
        load_secs = {s: 0.0 for s in range(self.sets)}
        nxt = [0]

        def kick(set_id: int) -> None:
            if nxt[0] < n:
                i = nxt[0]
                nxt[0] += 1
                holder[set_id], loaded[set_id], load_secs[set_id] = i, 0, 0.0
                self.fingerprints[set_id] = {}
                self._send(set_id, ("load", items[i][0]))
                log.info("set %d: loading item %d (%s)", set_id, i, items[i][0].label or items[i][0].identity())

        def get():
            while True:
                try:
                    msg = self.out.get(timeout=60)
                except queue.Empty:
                    dead = [p.pid for p in self.procs if not p.is_alive()]
                    if dead:
                        raise RuntimeError(f"replica processes {dead} exited without reporting")
                    continue
                if msg[0] == "error":
                    raise RuntimeError(f"replica set {msg[1]} rank {msg[2]} failed:\n{msg[3]}")
                if msg[0] == "fingerprint":
                    self.fingerprints[msg[1]][msg[2]] = msg[3]
                    continue
                if msg[0] == "loaded":  # bookkeeping here; returned too, so a waiting caller re-checks
                    loaded[msg[1]] += 1
                    load_secs[msg[1]] = max(load_secs[msg[1]], msg[3])
                    if loaded[msg[1]] == n_ranks:
                        log.info("set %d: item %d loaded in %.0fs", msg[1], holder[msg[1]], load_secs[msg[1]])
                return msg

        for set_id in range(self.sets):
            kick(set_id)
        for i in range(n):
            s = next(k for k, v in holder.items() if v == i)
            recipe, wanted = items[i]
            waited = time.time()
            while loaded[s] < n_ranks:
                msg = get()
                if msg[0] != "loaded":
                    raise RuntimeError(f"unexpected message while waiting for a load: {msg[:3]}")
            if time.time() - waited > 1:
                log.info("GPUs waited %.0fs for item %d's load", time.time() - waited, i)
            prints = self.fingerprints[s]
            if prints:  # sharded load: every replica of the set must hold the same weights
                values = {round(v[0], 6) for v in prints.values()}
                if len(prints) != n_ranks or len(values) != 1:
                    raise RuntimeError(f"item {i}: replicas of set {s} disagree after the exchange: {prints}")
                log.info("set %d: item %d replicas identical (fingerprint %.6e, exchange %.0fs)", s, i,
                         next(iter(prints.values()))[0], max(v[1] for v in prints.values()))
            by_id = {sample.sample_id: (sample, cached) for sample, cached, _ in wanted}
            self._send_jobs(s, i, wanted)
            total = sum(len(m) for _, _, m in wanted)
            log.info("scoring item %d (%s) on set %d: %d sequences", i, recipe.label or recipe.identity(), s, total)
            done = scored = 0
            started = last_save = last_log = time.time()
            while done < n_ranks:
                msg = get()
                if msg[0] == "loaded":  # the other set finished loading the next item
                    continue
                if msg[0] == "done":
                    if msg[1] != s or msg[3] != i:
                        raise RuntimeError(f"unexpected done {msg[:4]} while scoring item {i} on set {s}")
                    done += 1
                elif msg[0] == "losses":
                    item, sample_id, payload = msg[3]
                    if item != i:
                        raise RuntimeError(f"losses for item {item} while scoring item {i}")
                    cached = by_id[sample_id][1]
                    for index, loss in payload:
                        cached[index] = loss
                    scored += len(payload)
                    now = time.time()
                    if now - last_log > 30:
                        last_log = now
                        rate = scored / max(now - started, 1e-9)
                        log.info("total scored %d/%d (%.1f seq/s, eta %.0fs)", scored, total, rate,
                                 (total - scored) / max(rate, 1e-9))
                    if now - last_save > 120:
                        last_save = now
                        for sample, cached_ in by_id.values():
                            on_save(i, sample, cached_, now - started)
            elapsed = time.time() - started
            log.info("item %d (%s): scored %d sequences in %.0fs (%.1f seq/s) on set %d", i, recipe.label or recipe.identity(),
                     scored, elapsed, scored / max(elapsed, 1e-9), s)
            for sample, cached in by_id.values():
                on_save(i, sample, cached, elapsed)
            missing_after = sum(int(np.isnan(cached[m]).sum()) for _, cached, m in wanted)
            if missing_after:
                raise RuntimeError(f"item {i}: {missing_after} sequences left unscored")
            item_load = load_secs[s]
            del holder[s]
            kick(s)  # this set freed its model: start loading the next item while the other set scores
            on_done(i, item_load, elapsed)

    def _send_jobs(self, set_id: int, item: int, wanted: list) -> None:
        n_ranks = len(self.groups)
        for rank in range(n_ranks):
            jobs = [(sample.sample_id, missing[rank::n_ranks]) for sample, _, missing in wanted if missing[rank::n_ranks]]
            self.inqs[(set_id, rank)].put(("score", item, jobs))

    def close(self) -> None:
        for q in self.inqs.values():
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
        for p in self.procs:
            if p.is_alive():
                p.terminate()
