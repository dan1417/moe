"""Full-parameter CPT of the Teutonic king.

    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        -m src.cpt.train --config configs/cpt/b300.yaml

Score-only (per-sequence losses of any checkpoint on the config's val pool; used to get the
*real* king's reference losses when training starts from a merged body):

    python -m torch.distributed.run --standalone --nproc_per_node=8 \
        -m src.cpt.train --config cfg.yaml --score-model /dev/shm/teutonic/king30 \
        --score-out /workspace/teutonic/pools/val/king30_losses.npy

Run-time controls (checked by rank 0 every step, broadcast to all ranks):
    touch <output_dir>/COOLDOWN   -> start the linear LR cooldown now (decay_steps long), then stop
    touch <output_dir>/STOP       -> export (raw + EMA) after the current step and stop
    touch <output_dir>/EXPORT     -> one extra export after the current step (file is removed)
    <output_dir>/LR.json          -> {"scale": 1.5, "groups": {"<group label>": lr_mult}}: live LR change
                                     (applied at the next step; group labels are printed at start-up)

See docs/CPT.md for the recipe (why AdamW + fp32 master, frozen routers, EMA, lines).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from src.cpt.data import KDPool, MixedLoader, PoolLoader, WeightedPoolLoader
from src.cpt.export import export_checkpoint
from src.cpt.model import build_model, freeze, init_rotary_buffers, load_checkpoint_shards, load_config, set_attention

log = logging.getLogger("cpt")


# ------------------------------------------------------------------ helpers
class Schedule:
    """Warmup -> constant -> linear cooldown; the cooldown can be pulled forward at run time."""

    def __init__(self, cfg: dict):
        self.warmup = int(cfg["warmup_steps"])
        self.total = int(cfg["total_steps"])
        self.decay_steps = int(cfg["decay_steps"])
        self.floor = float(cfg.get("min_lr_ratio", 0.0))
        self.decay_start = self.total - self.decay_steps

    def trigger_cooldown(self, step: int) -> bool:
        """Begin the cooldown at ``step`` (no-op once it has started). Returns True if changed."""
        if step >= self.decay_start:
            return False
        self.decay_start = step
        self.total = step + self.decay_steps
        return True

    def multiplier(self, step: int) -> float:
        if step < self.warmup:
            return (step + 1) / max(self.warmup, 1)
        if step < self.decay_start:
            return 1.0
        progress = (step - self.decay_start) / max(self.total - self.decay_start, 1)
        return self.floor + (1.0 - self.floor) * max(0.0, 1.0 - progress)


def lr_multiplier(step: int, cfg: dict) -> float:
    return Schedule(cfg).multiplier(step)


def local(param: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return param.to_local() if isinstance(param, DTensor) else param.data


class DeviceEMA:
    """fp32 EMA of each rank's parameter shards, kept on the GPU next to the fp32 master shards.

    36 GB/GPU for the upper-layer recipe; one fused lerp per update (~50 ms). The old host-RAM EMA
    copied every shard to pageable memory and averaged on the CPU: ~23 s per update on this node."""

    def __init__(self, params: list[torch.Tensor], decay: float, every: int, power: float = 0.0):
        """``power`` > 0: power-function EMA (Karras et al. 2024), decay_t = (1 - 1/t)^(power+1) capped at ``decay``:
        the averaging window grows with training (~t/(power+1) updates) -- Polyak-style averaging for long runs
        instead of a fixed window."""
        self.params, self.every = params, max(1, every)
        self.decay = decay**self.every
        self.power, self.updates = float(power), 0
        self.shadow = [local(p).detach().clone() for p in params]
        self.pinned = False  # kept for log compatibility

    def current_decay(self) -> float:
        if self.power <= 0:
            return self.decay
        return min(self.decay, (1.0 - 1.0 / (self.updates + 1)) ** (self.power + 1))

    @torch.no_grad()
    def update(self, step: int) -> None:
        if (step + 1) % self.every:
            return
        self.updates += 1
        torch._foreach_lerp_(self.shadow, [local(p).detach() for p in self.params], 1.0 - self.current_decay())

    @torch.no_grad()
    def swap(self) -> None:
        """Exchange live weights and EMA weights (call twice to restore)."""
        for shadow, param in zip(self.shadow, self.params):
            live = local(param)
            tmp = live.clone()
            live.copy_(shadow)
            shadow.copy_(tmp)
            del tmp

    def state_dict(self) -> dict:
        return {"shadow": [t.cpu() for t in self.shadow], "decay": self.decay, "every": self.every}

    def load_state_dict(self, state: dict) -> None:
        for mine, saved in zip(self.shadow, state["shadow"]):
            mine.copy_(saved)


HostEMA = DeviceEMA  # old name


def sequence_losses(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Per-token CE [B, 2047] in fp32 over the full unpadded window (the eval objective).

    The model's forward is patched (``src.cpt.model.install_loss_head``) to return these directly,
    computing lm_head + CE in checkpointed chunks instead of materializing [B, 2047, 152576] logits."""
    return model(input_ids)


@torch.no_grad()
def evaluate(model, tokens: np.ndarray, sources: np.ndarray, *, rank: int, world: int, batch: int, device) -> dict:
    """Per-source mean CE on a fixed held-out set (every rank runs the same #forwards)."""
    model.eval()
    n = len(tokens)
    per_rank = math.ceil(n / world)
    mine = [min(rank * per_rank + i, n - 1) for i in range(per_rank)]
    losses = []
    for start in range(0, per_rank, batch):
        index = mine[start : start + batch]
        ids = torch.from_numpy(np.asarray(tokens[index], dtype=np.int64)).to(device)
        losses.append(sequence_losses(model, ids).mean(dim=1))
    mine_losses = torch.cat(losses)
    gathered = [torch.empty_like(mine_losses) for _ in range(world)]
    dist.all_gather(gathered, mine_losses)
    everything = torch.cat(gathered)[:n].cpu().numpy()
    model.train()
    out = {"all": float(everything.mean())}
    for name in sorted(set(sources.tolist())):
        out[name] = float(everything[sources == name].mean())
    out["_per_seq"] = everything
    return out


@torch.no_grad()
def score_tokens(model, tokens: np.ndarray, *, rank: int, world: int, batch: int, device) -> np.ndarray:
    """Per-token CE [n, 2047] on a fixed set (same rank striding as ``evaluate``)."""
    model.eval()
    n = len(tokens)
    per_rank = math.ceil(n / world)
    mine = [min(rank * per_rank + i, n - 1) for i in range(per_rank)]
    chunks = []
    for start in range(0, per_rank, batch):
        ids = torch.from_numpy(np.asarray(tokens[mine[start : start + batch]], dtype=np.int64)).to(device)
        chunks.append(sequence_losses(model, ids))
    local = torch.cat(chunks)
    gathered = [torch.empty_like(local) for _ in range(world)]
    dist.all_gather(gathered, local)
    model.train()
    return torch.cat(gathered)[:n].float().cpu().numpy()


def paired_vs(per_seq: np.ndarray, reference: np.ndarray, sources: np.ndarray) -> dict:
    """The validator's statistic on the held-out set: mean(reference − model) overall and per source."""
    diff = reference - per_seq
    se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    summary = {"mu_hat": round(float(diff.mean()), 6), "se": round(se, 7), "win_rate": round(float((diff > 0).mean()), 4)}
    for name in sorted(set(sources.tolist())):
        summary[name] = round(float(diff[sources == name].mean()), 6)
    return summary


def prune_exports(export_dir: Path, keep: int) -> None:
    """Rank 0: delete the oldest exports beyond ``keep`` (never dirs holding a KEEP file)."""
    if keep <= 0 or not export_dir.exists():
        return
    dirs = [d for d in export_dir.iterdir() if d.is_dir() and not (d / "KEEP").exists()]
    dirs.sort(key=lambda d: d.stat().st_mtime)
    for old in dirs[: max(0, len(dirs) - keep)]:
        log.info("pruning export %s", old)
        shutil.rmtree(old, ignore_errors=True)


def read_controls(run_dir: Path, *, rank: int, device) -> tuple[bool, bool, bool]:
    """(cooldown, stop, export) flags from trigger files, identical on every rank."""
    flags = torch.zeros(3, dtype=torch.int64, device=device)
    if rank == 0:
        flags[0] = int((run_dir / "COOLDOWN").exists())
        flags[1] = int((run_dir / "STOP").exists())
        export = run_dir / "EXPORT"
        if export.exists():
            flags[2] = 1
            export.unlink(missing_ok=True)
    dist.broadcast(flags, src=0)
    return bool(flags[0]), bool(flags[1]), bool(flags[2])


def read_lr_control(run_dir: Path, *, rank: int, current: dict) -> dict:
    """Live LR control from <run_dir>/LR.json: {"scale": 1.5, "groups": {"<label>": lr_mult}}.

    Rank 0 reads the file (if it changed) and broadcasts the parsed dict, so every rank applies the same
    multipliers at the same step. Malformed files are ignored (and logged) instead of crashing the run.
    """
    payload = [None]
    if rank == 0:
        path = run_dir / "LR.json"
        if path.exists():
            try:
                stamp = path.stat().st_mtime_ns
                if stamp != current.get("_stamp"):
                    parsed = json.loads(path.read_text())
                    payload[0] = {"scale": float(parsed.get("scale", 1.0)),
                                  "groups": {k: float(v) for k, v in parsed.get("groups", {}).items()},
                                  "_stamp": stamp}
            except (ValueError, OSError, AttributeError) as exc:
                log.warning("ignoring malformed LR.json: %s", exc)
    dist.broadcast_object_list(payload, src=0)
    return payload[0] if payload[0] is not None else current


def gpu_mem_gb() -> dict:
    return {
        "alloc": round(torch.cuda.memory_allocated() / 1e9, 1),
        "peak": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "reserved": round(torch.cuda.memory_reserved() / 1e9, 1),
    }


# ---------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--score-model", help="score-only: checkpoint dir to evaluate on the val pool")
    parser.add_argument("--score-out", help="score-only: where to write per-sequence losses (.npy)")
    parser.add_argument("--score-code-dir", help="score-only: build the model with this checkpoint's MiMo-V2 code and the "
                        "--score-model config's language-model fields (e.g. a MiMo-V2.5 teacher)")
    parser.add_argument("--score-per-token", action="store_true", help="score-only: also save per-token CE [n, 2047] (fp16)")
    parser.add_argument("--teacher-topk-out", help="with --score-model: write the model's top-k next-token log-probs for "
                        "pool sequences of --teacher-source (per-rank shards) and exit")
    parser.add_argument("--teacher-topk-k", type=int, default=64)
    parser.add_argument("--teacher-source", default="txt360-v2")
    parser.add_argument("--teacher-n", type=int, default=24000, help="sequences to label (first N of that source in the pool)")
    parser.add_argument("--max-steps", type=int, help="override schedule.total_steps (gate runs)")
    parser.add_argument("--profile-dir", help="torch.profiler trace of steps 2-3 (rank 0 table + chrome traces)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = Path(cfg["output_dir"])
    score_only = args.score_model is not None
    if score_only and not args.score_out:
        parser.error("--score-out is required with --score-model")

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format=f"%(asctime)s [rank{rank}] %(levelname)s %(name)s: %(message)s",
    )
    torch.manual_seed(int(cfg.get("seed", 0)))
    mesh = init_device_mesh("cuda", (world,))
    start_dir = Path(args.score_model) if score_only else Path(cfg["king_dir"])  # king_dir = start checkpoint

    # ---- model: meta skeleton -> FSDP2 -> fp32 shards loaded from the start checkpoint
    n_layers = cfg.get("n_layers")
    code_dir = Path(args.score_code_dir) if score_only and args.score_code_dir else None
    config = load_config(start_dir, n_layers=n_layers, code_dir=code_dir)
    model = build_model(config, activation_checkpointing=bool(cfg.get("activation_checkpointing", True)))
    trainable, frozen = freeze(model, list(cfg.get("freeze", [])))
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=mp)
    fully_shard(model.lm_head, mesh=mesh, mp_policy=mp)
    fully_shard(model, mesh=mesh, mp_policy=mp)
    model.to_empty(device=device)
    init_rotary_buffers(model, device)
    started = time.time()
    # external MiMo-V2 checkpoints (teachers) store fused qkv interleaved by TP shard; ours use the HF layout
    qkv_tp = int(config.num_key_value_heads) if code_dir is not None and getattr(config, "attention_projection_layout", "") == "fused_qkv" else None
    load_checkpoint_shards(model, start_dir, qkv_ckpt_tp=qkv_tp)
    set_attention(model, cfg.get("attention", "flash_attention_4"))
    model.train()
    log.info(
        "model ready in %.0fs from %s: %.2fB trainable, %.2fB frozen params, %d ranks, gpu %s",
        time.time() - started, start_dir, trainable / 1e9, frozen / 1e9, world, json.dumps(gpu_mem_gb()),
    )

    data_cfg = cfg["data"]
    micro_batch, accum = int(data_cfg["micro_batch"]), int(data_cfg["grad_accum"])
    eval_batch = int(data_cfg.get("eval_batch", micro_batch))  # forward-only: larger batches amortize the all-gathers
    val_dir = Path(data_cfg["val_dir"]) if data_cfg.get("val_dir") else None
    val_tokens = val_sources = None
    if val_dir is not None:
        val_tokens = np.load(val_dir / "tokens.npy", mmap_mode="r")[: int(data_cfg.get("val_n", 1024))]
        val_sources = np.load(val_dir / "sources.npy")[: len(val_tokens)]

    # ---- teacher extraction: top-k log-probs on training sequences of one source, then exit
    if score_only and args.teacher_topk_out:
        pool_tokens = np.load(Path(data_cfg["pool_dir"]) / "tokens.npy", mmap_mode="r")
        pool_sources = np.load(Path(data_cfg["pool_dir"]) / "sources.npy")
        # --teacher-source: "a", "a,b" or "!a" (every source except a)
        spec = [x.strip() for x in args.teacher_source.split(",") if x.strip()]
        if all(x.startswith("!") for x in spec):
            mask = ~np.isin(pool_sources, [x[1:] for x in spec])
        else:
            mask = np.isin(pool_sources, spec)
        selected = np.flatnonzero(mask)[: args.teacher_n]
        chunk = math.ceil(len(selected) / world)
        mine = selected[rank * chunk : (rank + 1) * chunk]
        padded = np.concatenate([mine, np.repeat(selected[-1:], chunk - len(mine))])  # equal #forwards on every rank
        model.lm_head._cpt_topk = int(args.teacher_topk_k)
        model.eval()
        started = time.time()
        vals, ids_out = [], []
        with torch.no_grad():
            for start in range(0, chunk, eval_batch):
                ids = torch.from_numpy(np.asarray(pool_tokens[np.sort(padded[start : start + eval_batch])], dtype=np.int64))
                order = np.argsort(np.argsort(padded[start : start + eval_batch]))  # back to padded order
                v, i = model(ids.to(device))
                vals.append(v[order].half().cpu())
                ids_out.append(i[order].int().cpu())
                if rank == 0 and (start // eval_batch) % 20 == 0:
                    done = start + eval_batch
                    log.info("teacher top-k: %d/%d sequences per rank (%.0f tok/s total)", done, chunk,
                             done * world * 2048 / max(time.time() - started, 1e-9))
        out = Path(args.teacher_topk_out) / f"rank{rank}"
        out.mkdir(parents=True, exist_ok=True)
        n_mine = len(mine)
        np.save(out / "pool_index.npy", mine.astype(np.int64))
        np.save(out / "tokens.npy", np.asarray(pool_tokens[np.sort(mine)])[np.argsort(np.argsort(mine))] if n_mine else np.zeros((0, 2048), np.uint32))
        np.save(out / "topk_logp.npy", torch.cat(vals)[:n_mine].numpy())
        np.save(out / "topk_idx.npy", torch.cat(ids_out)[:n_mine].numpy())
        dist.barrier()
        if rank == 0:
            (Path(args.teacher_topk_out) / "meta.json").write_text(json.dumps({
                "teacher": str(start_dir), "source": args.teacher_source, "k": int(args.teacher_topk_k),
                "n": int(len(selected)), "pool_dir": str(data_cfg["pool_dir"]), "ranks": world,
                "seconds": round(time.time() - started, 1)}, indent=1))
            log.info("teacher top-k for %d sequences written to %s in %.0fs", len(selected), args.teacher_topk_out, time.time() - started)
        dist.barrier()
        dist.destroy_process_group()
        return

    # ---- score-only: per-sequence losses of --score-model on the val pool, then exit
    if score_only:
        if val_tokens is None:
            raise SystemExit("score-only needs data.val_dir")
        started = time.time()
        if args.score_per_token:
            per_token = score_tokens(model, val_tokens, rank=rank, world=world, batch=eval_batch, device=device)
            if rank == 0:
                Path(args.score_out).parent.mkdir(parents=True, exist_ok=True)
                np.save(Path(args.score_out).with_suffix(".tokens.npy"), per_token.astype(np.float16))
        result = evaluate(model, val_tokens, val_sources, rank=rank, world=world, batch=eval_batch, device=device)
        per_seq = result.pop("_per_seq")
        if rank == 0:
            out = Path(args.score_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, per_seq)
            out.with_suffix(".json").write_text(json.dumps({
                "model": str(start_dir), "val_dir": str(val_dir), "n": int(len(per_seq)),
                "attention": cfg.get("attention", "flash_attention_4"), "micro_batch": micro_batch,
                "mean": {k: round(v, 6) for k, v in result.items()}, "seconds": round(time.time() - started, 1),
            }, indent=1))
            log.info("scored %d sequences in %.0fs: %s -> %s", len(per_seq), time.time() - started,
                     json.dumps({k: round(v, 5) for k, v in result.items()}), out)
        dist.barrier()
        dist.destroy_process_group()
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = cfg["optimizer"]
    group_specs = [(re.compile(g["pattern"]), float(g.get("lr_mult", 1.0)), g["pattern"]) for g in opt_cfg.get("groups", [])]
    grouped: dict[str, dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        mult, label = 1.0, "base"
        for pattern, group_mult, group_label in group_specs:
            if pattern.search(name):
                mult, label = group_mult, group_label
                break
        entry = grouped.setdefault(label, {"params": [], "lr_mult": mult, "label": label})
        entry["params"].append(param)
    param_groups = [{"params": g["params"], "lr": float(opt_cfg["lr"]) * g["lr_mult"], "lr_mult": g["lr_mult"], "label": g["label"]}
                    for g in grouped.values()]
    fused = bool(opt_cfg.get("fused", True))  # foreach AdamW allocates full-size temporaries (sqrt(v): +36 GB/GPU)
    optimizer = torch.optim.AdamW(
        param_groups, lr=float(opt_cfg["lr"]), betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
        eps=float(opt_cfg.get("eps", 1e-8)), weight_decay=float(opt_cfg.get("weight_decay", 0.0)),
        **({"fused": True} if fused else {"foreach": True}),
    )
    if rank == 0:
        for g in param_groups:
            log.info("param group %s: %.2fB params, lr x%.3f", g["label"], sum(p.numel() for p in g["params"]) / 1e9, g["lr_mult"])
    sched_cfg = dict(cfg["schedule"])
    if args.max_steps is not None:
        sched_cfg["total_steps"] = int(args.max_steps)
        sched_cfg["decay_steps"] = min(int(sched_cfg["decay_steps"]), int(args.max_steps))
    schedule = Schedule(sched_cfg)
    ema_cfg = cfg.get("ema", {})
    ema = (DeviceEMA(params, float(ema_cfg["decay"]), int(ema_cfg.get("every", 1)), float(ema_cfg.get("power", 0.0)))
           if ema_cfg.get("enabled") else None)
    if ema is not None:
        log.info("EMA ready on GPU (decay %.4f per %d steps, power %.1f), gpu %s", ema.decay, ema.every, ema.power,
                 json.dumps(gpu_mem_gb()))

    if data_cfg.get("kd_dir"):
        kd_pool = KDPool(Path(data_cfg["kd_dir"]), Path(data_cfg["kd_student_dir"]) if data_cfg.get("kd_student_dir") else None)
        if data_cfg.get("slm_keep"):
            model.lm_head._cpt_slm = float(data_cfg["slm_keep"])
            log.info("selective LM on teacher-scored rows: CE on the top %.0f%% tokens by excess loss (ours - teacher)",
                     100 * model.lm_head._cpt_slm)
        if kd_pool.student:
            model.lm_head._cpt_kd = (float(data_cfg.get("kd_alpha", 0.5)), float(data_cfg.get("kd_beta", 1.0)),
                                     float(data_cfg.get("kd_mix", 0.5)))
            log.info("KD mixture target: alpha %.2f * CE + beta %.2f * soft-CE((1-%.2f) * frozen student + %.2f * teacher)",
                     model.lm_head._cpt_kd[0], model.lm_head._cpt_kd[1], model.lm_head._cpt_kd[2], model.lm_head._cpt_kd[2])
        loader = MixedLoader(Path(data_cfg["pool_dir"]), kd_pool, kd_source=data_cfg.get("kd_source", "txt360-v2"),
                             kd_frac=float(data_cfg.get("kd_frac", 0.25)), kd_lambda=float(data_cfg.get("kd_lambda", 1.0)),
                             rank=rank, world=world, micro_batch=micro_batch, seed=int(data_cfg.get("seed", 0)), device=device)
        log.info("KD: %d teacher-labelled %s sequences (top-%d), %d of %d per micro-batch, lambda %.2f", kd_pool.n,
                 loader.kd_source, kd_pool.k, loader.n_kd, micro_batch, loader.kd_lambda)
    elif data_cfg.get("source_weights"):
        loader = WeightedPoolLoader(Path(data_cfg["pool_dir"]), source_weights=dict(data_cfg["source_weights"]), rank=rank,
                                    world=world, micro_batch=micro_batch, seed=int(data_cfg.get("seed", 0)), device=device,
                                    source_pools=dict(data_cfg.get("source_pools") or {}))
        log.info("training mixture per micro-batch: %s", loader.describe())
    else:
        loader = PoolLoader(Path(data_cfg["pool_dir"]), rank=rank, world=world, micro_batch=micro_batch,
                            seed=int(data_cfg.get("seed", 0)), device=device)

    start_step = 0
    ckpt_cfg = cfg.get("checkpoint", {})
    latest = run_dir / "checkpoints" / "latest"
    if ckpt_cfg.get("resume", True) and (latest / ".metadata").exists():
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        model_sd, optim_sd = get_state_dict(model, optimizer)
        state = {"model": model_sd, "optim": optim_sd}
        dcp.load(state, checkpoint_id=str(latest))
        set_state_dict(model, optimizer, model_state_dict=state["model"], optim_state_dict=state["optim"])
        extra = torch.load(latest / f"extra_rank{rank}.pt", weights_only=False)
        start_step = int(extra["step"]) + 1
        loader.load_state_dict(extra["loader"])
        if ema is not None and "ema" in extra:
            ema.load_state_dict(extra["ema"])
        log.info("resumed from %s at step %d", latest, start_step)

    base_lr = float(opt_cfg["lr"])
    tokens_per_step = world * micro_batch * accum * 2048
    history_path = run_dir / "history.jsonl"
    export_dir = Path(cfg.get("export_dir") or (run_dir / "exports"))
    export_keep = int(cfg.get("export_keep", 0))
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        export_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, run_dir / "config.yaml")
        for stale in ("COOLDOWN", "STOP", "EXPORT"):
            (run_dir / stale).unlink(missing_ok=True)
        log.info("schedule: warmup %d, cooldown starts %d, total %d; %d tokens/step; exports -> %s (keep %s)",
                 schedule.warmup, schedule.decay_start, schedule.total, tokens_per_step, export_dir, export_keep or "all")
    dist.barrier()

    if cfg.get("export_at_start") and start_step == 0:
        # round-trip gate: must reproduce the start checkpoint's tensors bit-for-bit
        export_checkpoint(model, start_dir, export_dir / "step000000_start", rank=rank, n_layers=n_layers)

    # ---- held-out references: the real king (precomputed) and the start checkpoint (step 0)
    references: dict[str, np.ndarray] = {}
    if val_tokens is not None:
        king_path = data_cfg.get("king_val_losses")
        if king_path:
            king_val = np.load(king_path)
            if len(king_val) < len(val_tokens):
                raise SystemExit(f"{king_path} has {len(king_val)} losses < val_n {len(val_tokens)}")
            references["king"] = king_val[: len(val_tokens)]
        start_val_path = run_dir / "start_val_losses.npy"
        if start_val_path.exists():
            references["start"] = np.load(start_val_path)[: len(val_tokens)]
        elif start_step == 0:
            baseline = evaluate(model, val_tokens, val_sources, rank=rank, world=world, batch=eval_batch, device=device)
            references["start"] = baseline.pop("_per_seq")
            if rank == 0:
                np.save(start_val_path, references["start"])
                record = {"step": 0, "val": {k: round(v, 6) for k, v in baseline.items()}}
                if "king" in references:
                    record["start_vs_king"] = paired_vs(references["start"], references["king"], val_sources)
                log.info("step 0 (start checkpoint) val: %s", json.dumps(record))
                with history_path.open("a") as out:
                    out.write(json.dumps(record) + "\n")
        if "king" not in references and rank == 0:
            log.warning("no data.king_val_losses: vs_king will not be reported (only vs_start)")

    def run_eval(step_done: int) -> None:
        result = {"step": step_done, "lr": base_lr * schedule.multiplier(max(step_done - 1, 0))}
        for tag, swap in (("val", False), ("val_ema", True)):
            if swap:
                if ema is None:
                    continue
                ema.swap()
            scores = evaluate(model, val_tokens, val_sources, rank=rank, world=world, batch=eval_batch, device=device)
            if swap:
                ema.swap()
            per_seq = scores.pop("_per_seq")
            result[tag] = {k: round(v, 6) for k, v in scores.items()}
            for name, reference in references.items():
                result[f"{'ema_' if swap else ''}vs_{name}"] = paired_vs(per_seq, reference, val_sources)
        if rank == 0:
            log.info("val: %s", json.dumps(result))
            with history_path.open("a") as out:
                out.write(json.dumps(result) + "\n")

    def run_export(step_done: int, *, raw: bool) -> None:
        tag = f"step{step_done:06d}"
        if raw:
            export_checkpoint(model, start_dir, export_dir / f"{tag}_raw", rank=rank, n_layers=n_layers)
        if ema is not None:
            ema.swap()
            export_checkpoint(model, start_dir, export_dir / f"{tag}_ema", rank=rank, n_layers=n_layers)
            ema.swap()
        if rank == 0:
            prune_exports(export_dir, export_keep)
        dist.barrier()

    profiler = None
    if args.profile_dir:
        from torch.profiler import ProfilerActivity, profile, schedule as profile_schedule

        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)

        def on_trace(prof) -> None:
            prof.export_chrome_trace(str(profile_dir / f"trace_rank{rank}.json"))
            if rank == 0:
                table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=60)
                (profile_dir / "top_cuda.txt").write_text(table)
                log.info("profile written to %s\n%s", profile_dir, table)

        profiler = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                           schedule=profile_schedule(wait=1, warmup=1, active=2, repeat=1), on_trace_ready=on_trace)
        profiler.start()

    lr_control = {"scale": 1.0, "groups": {}, "_stamp": None}
    log_every = int(cfg.get("log_every", 10))
    noise_every = int(cfg.get("noise_every", log_every))  # gradient-noise-scale measurement cadence (0 = off)
    step_time = time.time()
    last_logged = start_step - 1
    step = start_step
    while step < schedule.total:
        cooldown, stop, export_now = read_controls(run_dir, rank=rank, device=device)
        if cooldown and schedule.trigger_cooldown(step) and rank == 0:
            log.info("COOLDOWN triggered at step %d: linear decay over %d steps, total now %d",
                     step, schedule.decay_steps, schedule.total)
        updated = read_lr_control(run_dir, rank=rank, current=lr_control)
        if updated is not lr_control:
            lr_control = updated
            for group in optimizer.param_groups:
                if group["label"] in lr_control["groups"]:
                    group["lr_mult"] = lr_control["groups"][group["label"]]
            if rank == 0:
                log.info("LR.json applied at step %d: scale %.3f, groups %s", step, lr_control["scale"],
                         {g["label"]: g["lr_mult"] for g in optimizer.param_groups})
        multiplier = schedule.multiplier(step) * lr_control["scale"]
        for group in optimizer.param_groups:
            group["lr"] = base_lr * group.get("lr_mult", 1.0) * multiplier
        loss_sum = torch.zeros((), device=device)
        by_source: dict[str, list[float]] = {}
        log_step = step % log_every == 0 or step == schedule.total - 1
        seq_losses, seq_sources = [], []
        measure_noise = noise_every and step % noise_every == 0
        small_norm = None
        kd_sum = torch.zeros((), device=device)
        kd_student_sum = torch.zeros((), device=device)
        for micro in range(accum):
            batch = loader.next()
            kd = getattr(batch, "kd", None)
            if kd is not None:
                outputs = model(batch.input_ids, kd=kd)  # CE, total[, soft-CE vs teacher, soft-CE vs frozen student]
                per_token, total = outputs[0], outputs[1]
                (total.mean() / accum).backward()
                kd_rows = kd[-1] > 0
                if len(outputs) == 4:
                    kd_sum += outputs[2].detach()[kd_rows].mean() / accum
                    kd_student_sum += outputs[3].detach()[kd_rows].mean() / accum
                else:
                    kd_sum += ((total - per_token).detach()[kd_rows].mean() / kd[-1][kd_rows].mean()) / accum
            else:
                per_token = sequence_losses(model, batch.input_ids)
                (per_token.mean() / accum).backward()
            loss_sum += per_token.detach().mean() / accum
            if log_step:  # keep per-sequence values on the GPU: a .tolist() here would drain the CUDA queue
                seq_losses.append(per_token.detach().mean(dim=1))
                seq_sources.extend(batch.sources.tolist())
            if measure_noise and micro == 0:
                first = torch.nn.utils.get_total_norm([p.grad for p in params if p.grad is not None])
                first = first.full_tensor() if hasattr(first, "full_tensor") else first
                small_norm = float(first) * accum  # gradient of the first (world x micro_batch) sequences alone
        grad_norm = torch.nn.utils.clip_grad_norm_(params, float(opt_cfg.get("grad_clip", 1.0)))
        grad_norm = grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm
        noise_record = None
        if small_norm is not None:
            # simple noise scale B_noise = tr(Sigma) / |G|^2 from two batch sizes (b_small, b_full)
            b_small, b_full = world * micro_batch * 2048, tokens_per_step
            g2_full, g2_small = float(grad_norm) ** 2, small_norm ** 2
            true_g2 = (b_full * g2_full - b_small * g2_small) / (b_full - b_small)
            trace = (g2_small - g2_full) / (1.0 / b_small - 1.0 / b_full)
            noise_record = {"grad_norm_small": round(small_norm, 4), "grad_norm_full": round(float(grad_norm), 4),
                            "true_grad_norm": round(math.sqrt(max(true_g2, 0.0)), 4),
                            "b_noise_tokens": round(trace / true_g2) if true_g2 > 0 else None}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema_seconds = None
        if ema is not None and (step + 1) % ema.every == 0:
            ema_started = time.time()
            ema.update(step)
            ema_seconds = round(time.time() - ema_started, 2)
        if seq_losses:
            for name, value in zip(seq_sources, torch.cat(seq_losses).tolist()):
                by_source.setdefault(name, []).append(value)
        if profiler is not None:
            profiler.step()

        dist.all_reduce(loss_sum, op=dist.ReduceOp.AVG)
        last = step == schedule.total - 1 or stop
        if rank == 0 and (step % log_every == 0 or last):
            elapsed = time.time() - step_time
            steps_done = max(step - last_logged, 1)
            record = {
                "step": step, "loss": round(float(loss_sum), 5), "lr": base_lr * multiplier,
                "grad_norm": round(float(grad_norm), 4),
                "tok_per_s": round(tokens_per_step * steps_done / max(elapsed, 1e-9)),
                "sec_per_step": round(elapsed / steps_done, 2),
                "tokens_seen": tokens_per_step * (step + 1),
                "rank0_sources": {k: round(float(np.mean(v)), 4) for k, v in by_source.items()},
                "gpu_gb": gpu_mem_gb(),
            }
            if noise_record:
                record["noise"] = noise_record
            if ema_seconds is not None:
                record["ema_s"] = ema_seconds
            if ema is not None and ema.power > 0:
                record["ema_decay"] = round(ema.current_decay(), 5)
            if data_cfg.get("kd_dir"):
                record["kd_soft_ce"] = round(float(kd_sum), 5)
                if data_cfg.get("kd_student_dir"):
                    record["kd_soft_ce_student"] = round(float(kd_student_sum), 5)
            if len(optimizer.param_groups) > 1 or lr_control["scale"] != 1.0:
                record["group_lr"] = {g["label"]: g["lr"] for g in optimizer.param_groups}
            log.info(json.dumps(record))
            with history_path.open("a") as out:
                out.write(json.dumps(record) + "\n")
        if step % log_every == 0 or last:
            last_logged, step_time = step, time.time()

        if val_tokens is not None and ((step + 1) % int(cfg.get("eval_every", 100)) == 0 or last):
            run_eval(step + 1)
            last_logged, step_time = step, time.time()  # keep eval time out of tok/s

        export_every = int(cfg.get("export_every", 0))
        final_export = last and bool(cfg.get("export_final", True))
        if (export_every and (step + 1) % export_every == 0) or final_export or export_now:
            run_export(step + 1, raw=bool(cfg.get("export_raw", True)) or stop or last)  # final export: raw + EMA
            last_logged, step_time = step, time.time()

        every = int(ckpt_cfg.get("every", 0))
        if every and ((step + 1) % every == 0 or last):
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import get_state_dict

            target = run_dir / "checkpoints" / f"step{step + 1:06d}"
            model_sd, optim_sd = get_state_dict(model, optimizer)
            dcp.save({"model": model_sd, "optim": optim_sd}, checkpoint_id=str(target))
            extra = {"step": step, "loader": loader.state_dict()}
            if ema is not None:
                extra["ema"] = ema.state_dict()
            torch.save(extra, target / f"extra_rank{rank}.pt")
            dist.barrier()
            if rank == 0:
                if latest.is_symlink() or latest.exists():
                    previous = latest.resolve()
                    latest.unlink()
                    if previous != target.resolve() and int(ckpt_cfg.get("keep", 1)) <= 1:
                        shutil.rmtree(previous, ignore_errors=True)
                latest.symlink_to(target.name)
            dist.barrier()

        if stop:
            if rank == 0:
                log.info("STOP triggered: exported after step %d, exiting", step + 1)
            break
        step += 1

    if profiler is not None:
        profiler.stop()
    if rank == 0:
        (run_dir / "DONE").write_text(json.dumps({"last_step": step + 1, "tokens": tokens_per_step * (step + 1)}))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
