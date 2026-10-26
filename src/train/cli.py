#!/usr/bin/env python3
"""Teutonic SN3 distributed training entry point."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import yaml
from transformers import AutoTokenizer

from src.checkpoint.averaging import run_pre_decay_soup
from src.data import CollatedBatch, EvalDataset, Manifest, build_eval_dataset, build_train_loader
from src.checkpoint.dcp import (
    load_distributed_checkpoint,
    save_distributed_checkpoint,
)
from src.model.lora import apply_lora, should_save_full_checkpoint
from src.model import (
    apply_fsdp,
    compile_decoder_layers,
    enable_gradient_checkpointing,
    enable_low_precision_training,
    load_challenger_model,
    load_challenger_model_distributed,
    load_state_dict_into_fsdp,
    load_state_dict_strict_except_mtp,
    low_precision_context,
    maybe_compile_optimizer,
    print_model_summary,
)
from src.moe.utils import (
    clip_grad_norms_per_module,
    compute_seq_auxiliary_loss,
    compute_z_loss,
    log_expert_utilization,
    register_router_hooks,
    router_stats_summary,
    update_expert_biases,
)
from src.eval.offline import (
    DEFAULT_SAFETY_MARGIN,
    build_paired_eval_result,
    log_paired_eval_summary,
    log_paired_source_row,
    score_source_distributed,
)
from src.optim.optimizer import CombinedOptimizer, build_optimizer
from src.data.packing import pack_batch_with_masking
from src.dist.process_groups import (
    ProcessGroups,
    init_process_groups,
    log_topology,
    resolve_gpus_per_node,
)
from src.model.qk_norm import check_and_inject_qk_norm
from src.optim.scheduler import build_scheduler
from src.optim.sparse_loco import SparseLoCo
from src.submit.export import (
    export_submission,
    load_checkpoint_meta,
    load_checkpoint_state,
    resolve_checkpoint,
    strip_mtp_tensors,
)

log = logging.getLogger(__name__)


class DatasetLossTracker:
    """Exponential moving average of loss per dataset source."""

    def __init__(self, alpha: float = 0.98) -> None:
        self.alpha = alpha
        self._values: dict[str, float] = {}

    def __repr__(self) -> str:
        return f"DatasetLossTracker(datasets={list(self._values.keys())})"

    def update(self, dataset_names: list[str], loss_value: float) -> None:
        for name in set(dataset_names):
            previous = self._values.get(name, loss_value)
            self._values[name] = self.alpha * previous + (1.0 - self.alpha) * loss_value

    def summary(self) -> dict[str, float]:
        return dict(self._values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Teutonic SN3 challenger model")
    parser.add_argument("--config", default="configs/b300/train.yaml", help="Training config path")
    parser.add_argument("--export", action="store_true", help="Export submission model from checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Run offline paired-bootstrap eval vs king")
    parser.add_argument("--trigger-decay", action="store_true", help="Run pre-decay soup then decay phase")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint directory (default: latest)")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training from checkpoint directory (loads model, optimizer, scheduler)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Challenger weights path: merged model dir, model.pt, or training checkpoint dir",
    )
    parser.add_argument(
        "--weights-sweep",
        nargs="+",
        default=None,
        help=(
            "Evaluate multiple challenger dirs in one FSDP session (load skeleton once, "
            "swap weights per path). Much faster than separate eval.sh launches."
        ),
    )
    parser.add_argument(
        "--eval-output",
        default=None,
        help="Write offline eval JSON to this path (default: checkpoints/offline_eval_result.json)",
    )
    parser.add_argument(
        "--eval-n",
        type=int,
        default=None,
        help="Override manifest eval_n (total sequences across sources). Use a small value for smoke tests.",
    )
    parser.add_argument("--submission-dir", default=None, help="Submission output directory")
    parser.add_argument("--genesis-dir", default=None, help="Genesis directory for contract verification")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def setup_distributed(cfg: dict[str, Any]) -> ProcessGroups:
    """Initialize distributed backend and build EP/DP topology from torchrun world size."""
    if "CUDA_TOOLKIT_PATH" not in os.environ:
        for candidate in (Path("/usr/local/cuda-13.0"), Path("/usr/local/cuda"), Path("/usr")):
            if (candidate / "bin" / "nvcc").exists():
                os.environ["CUDA_TOOLKIT_PATH"] = str(candidate.resolve())
                os.environ.setdefault("CUDA_HOME", str(candidate.resolve()))
                break
    # FA4 CuTe JIT on SM120 needs a real toolkit path; fail loudly if missing.
    if not os.environ.get("CUDA_TOOLKIT_PATH"):
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "CUDA_TOOLKIT_PATH unset; FlashAttention-4 SM120 JIT may miscompile"
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    timeout_sec = int(os.environ.get("NCCL_TIMEOUT", "1800"))
    smoke_mode = int(cfg.get("training", {}).get("smoke_steps", 0)) > 0
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    backend = os.environ.get("TEUTONIC_DIST_BACKEND", "").strip().lower()
    if not backend:
        # Single-GPU smoke: gloo avoids NCCL's nvmlInit when driver/userspace NVML mismatch.
        backend = "gloo" if smoke_mode and world_size == 1 else "nccl"
    if backend not in {"nccl", "gloo"}:
        raise ValueError(f"Unsupported TEUTONIC_DIST_BACKEND={backend!r}; use nccl or gloo")
    if backend == "gloo" and world_size > 1:
        raise RuntimeError(
            "TEUTONIC_DIST_BACKEND=gloo is only supported for single-process smoke "
            f"(WORLD_SIZE=1, got {world_size})"
        )
    init_kwargs: dict[str, Any] = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout_sec),
    }
    if backend == "nccl":
        init_kwargs["device_id"] = torch.device("cuda", local_rank)
    dist.init_process_group(**init_kwargs)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    hardware_cfg = cfg.get("hardware", {})
    configured = int(hardware_cfg.get("gpus_per_node", 8))
    gpus_per_node = resolve_gpus_per_node(configured)
    if gpus_per_node != configured and not smoke_mode:
        raise RuntimeError(
            f"hardware.gpus_per_node={configured} does not match runtime {gpus_per_node}"
        )
    if gpus_per_node != configured and local_rank == 0:
        log.warning("Smoke mode overrides configured GPU count %d -> %d", configured, gpus_per_node)
    pg = init_process_groups(gpus_per_node=gpus_per_node)
    configured_nodes = int(hardware_cfg.get("num_nodes", 1))
    if pg.dp_size != configured_nodes:
        raise RuntimeError(
            f"hardware.num_nodes={configured_nodes} does not match runtime {pg.dp_size}"
        )
    if pg.dp_size != 1:
        raise RuntimeError(
            "Multi-node training is disabled until cross-node gradient synchronization "
            "and SparseLoCo resume are validated"
        )
    log_topology(pg)
    if pg.is_rank0:
        teutonic_nodes = os.environ.get("TEUTONIC_NODES", "unset")
        log.info(
            "TEUTONIC_NODES=%s world_size=%d gpus_per_node=%d dist_backend=%s "
            "(FSDP eval/train; DDP unfit for 110B)",
            teutonic_nodes,
            pg.world_size,
            pg.ep_size,
            backend,
        )
    return pg


def _batch_iterator(loader) -> Iterator[CollatedBatch]:
    while True:
        yield from loader


def _synthetic_smoke_batch_iterator(
    cfg: dict[str, Any],
    *,
    global_rank: int = 0,
    start_step: int = 0,
) -> Iterator[CollatedBatch]:
    """Deterministic local batches so pipeline tests are not blocked on shard downloads.

    ``smoke_patterned_data: true`` emits repeating cycles (learnable). Default random
    tokens stay near log(V) CE and only validate the train loop.
    """
    training_cfg = cfg.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 1))
    seq_len = int(training_cfg.get("smoke_seq_len") or 256)
    vocab_size = int(training_cfg.get("smoke_vocab_size", 152576))
    patterned = bool(training_cfg.get("smoke_patterned_data", False))
    period = max(2, min(int(training_cfg.get("smoke_pattern_period", 32)), vocab_size))
    step = start_step
    while True:
        g = torch.Generator()
        g.manual_seed(int(training_cfg.get("base_seed", 42)) + global_rank + step)
        if patterned:
            # Fixed cycle shifted by step → next-token prediction is deterministic.
            cycle = torch.arange(period, dtype=torch.long)
            pos = torch.arange(seq_len, dtype=torch.long)
            row = cycle[(pos + step + global_rank) % period]
            input_ids = row.unsqueeze(0).repeat(micro_batch_size, 1)
            dataset_name = "synthetic_patterned"
        else:
            input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_len), generator=g)
            dataset_name = "synthetic_smoke"
        labels = input_ids.clone()
        step += 1
        yield CollatedBatch(
            input_ids=input_ids,
            labels=labels,
            datasets=[dataset_name] * micro_batch_size,
        )


def _load_tokenizer(king_dir: str | Path) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(king_dir, trust_remote_code=True)


def _atomic_torch_save(value: Any, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_json(value: dict[str, Any], path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2))
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: CombinedOptimizer,
    scheduler: Any,
    step: int,
    output_dir: Path,
    pg: ProcessGroups,
    meta: dict[str, Any],
    suffix: str = "",
    sparse_loco: SparseLoCo | None = None,
) -> None:
    """Save one committed FSDP2/DCP checkpoint without rank-0 consolidation."""
    ckpt_dir = save_distributed_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        output_dir,
        pg,
        meta,
        suffix=suffix,
        sparse_loco=sparse_loco,
    )
    if pg.is_rank0:
        log.info("Saved distributed checkpoint to %s", ckpt_dir)


def _prepare_batch(
    batch: CollatedBatch,
    cfg: dict[str, Any],
    eos_id: int,
    device: torch.device,
) -> dict[str, Any]:
    packing_enabled = bool(cfg.get("packing", {}).get("enabled", False))
    if packing_enabled:
        packed = pack_batch_with_masking(batch, eos_id=eos_id)
    else:
        packed = {
            "input_ids": batch.input_ids,
            "labels": batch.labels,
            "datasets": batch.datasets,
        }
    smoke_seq_len = int(cfg.get("training", {}).get("smoke_seq_len", 0) or 0)
    if smoke_seq_len > 0:
        packed["input_ids"] = packed["input_ids"][:, :smoke_seq_len]
        packed["labels"] = packed["labels"][:, :smoke_seq_len]
        if "position_ids" in packed:
            packed["position_ids"] = packed["position_ids"][:, :smoke_seq_len]
    packed["input_ids"] = packed["input_ids"].to(device, non_blocking=True)
    packed["labels"] = packed["labels"].to(device, non_blocking=True)
    if "position_ids" in packed:
        packed["position_ids"] = packed["position_ids"].to(device, non_blocking=True)
    if "cu_seqlens" in packed:
        packed["cu_seqlens"] = packed["cu_seqlens"].to(device, non_blocking=True)
    return packed


def _load_training_resume(
    resume_dir: Path,
    model: torch.nn.Module,
    optimizer: CombinedOptimizer,
    scheduler: Any,
    sparse_loco: SparseLoCo | None,
    pg: ProcessGroups,
) -> int:
    """Load checkpoint state and return the step to continue from."""
    meta = load_distributed_checkpoint(
        resume_dir,
        model,
        optimizer,
        scheduler,
        pg,
        sparse_loco=sparse_loco,
    )
    start_step = int(meta.get("next_step", int(meta.get("step", 0)) + 1))
    if pg.is_rank0:
        log.info("Resumed from %s at step %d", resume_dir, start_step)
    return start_step


def _forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "labels": batch["labels"],
        "use_cache": False,
    }
    if "position_ids" in batch:
        kwargs["position_ids"] = batch["position_ids"]
    return model(**kwargs)


def _validate_packing_mode(model: torch.nn.Module, cfg: dict[str, Any]) -> None:
    """Only allow packing on a backend that enforces document isolation."""
    if not bool(cfg.get("packing", {}).get("enabled", False)):
        return
    model_cfg = getattr(model, "config", None)
    implementation = getattr(model_cfg, "_attn_implementation", None)
    if implementation != "eager":
        raise RuntimeError(
            "packing.enabled requires eager attention. FlashAttention packed isolation "
            "is not wired for micro_batch_size > 1; set packing.enabled=false or "
            "TEUTONIC_ATTN=eager."
        )


def _validate_router_hooks(
    model: torch.nn.Module,
    hook_manager: RouterHookManager,
    moe_cfg: dict[str, Any],
) -> None:
    """Fail before training when configured MoE losses cannot observe every router."""
    model_cfg = getattr(model, "config", None)
    layer_flags = getattr(model_cfg, "moe_layer_freq", None)
    expected = sum(bool(flag) for flag in layer_flags) if layer_flags is not None else None
    losses_enabled = (
        float(moe_cfg.get("z_loss_coeff", 0.0)) > 0
        or float(moe_cfg.get("seq_aux_alpha", 0.0)) > 0
        or float(moe_cfg.get("bias_update_speed", 0.0)) > 0
    )
    if losses_enabled and hook_manager.router_count == 0:
        raise RuntimeError("MoE losses are enabled, but no router modules were discovered")
    if expected is not None and hook_manager.router_count != expected:
        raise RuntimeError(
            f"Router hook count mismatch: found={hook_manager.router_count}, expected={expected}"
        )


def _load_challenger_from_source(
    king_dir: str | Path,
    source: Path | None,
    pg: ProcessGroups,
) -> torch.nn.Module:
    """Load challenger weights from king, checkpoint, merged dir, or model.pt."""
    from src.submit.merge import is_safetensors_model_dir

    king_path = Path(king_dir)
    if source is None:
        model = load_challenger_model(king_path)
        if pg.is_rank0:
            log.info("Challenger initialized from king (no override)")
        return model

    source = source.resolve()
    if source.is_file() and source.suffix == ".pt":
        model = load_challenger_model(king_path)
        state = strip_mtp_tensors(torch.load(source, map_location="cpu", weights_only=True))
        load_state_dict_strict_except_mtp(model, state)
        if pg.is_rank0:
            log.info("Loaded challenger weights from %s", source)
        return model

    if is_safetensors_model_dir(source) or (source / "model.pt").exists():
        model = load_challenger_model(king_path)
        state = _load_challenger_state_dict(source)
        load_state_dict_strict_except_mtp(model, state)
        if pg.is_rank0:
            log.info("Loaded challenger state dict from %s (via king skeleton)", source)
        return model

    if (source / "config.json").exists():
        if pg.is_rank0:
            log.info("Loading challenger via from_pretrained(%s)", source)
        return load_challenger_model(source)

    raise FileNotFoundError(
        f"Unsupported --weights path {source}. Expected HF model dir, checkpoint dir, or model.pt"
    )


def _load_challenger_state_dict(source: Path) -> dict[str, torch.Tensor]:
    """Load challenger weights as a CPU state dict for in-place FSDP swap."""
    from src.submit.merge import is_safetensors_model_dir, load_model_state

    source = source.resolve()
    if source.is_file() and source.suffix == ".pt":
        return strip_mtp_tensors(torch.load(source, map_location="cpu", weights_only=True))
    if (source / "model.pt").exists():
        return strip_mtp_tensors(
            torch.load(source / "model.pt", map_location="cpu", weights_only=True)
        )
    if is_safetensors_model_dir(source):
        return strip_mtp_tensors(load_model_state(source))
    raise FileNotFoundError(f"No loadable weights found at {source}")


KING_CE_CACHE_DIRNAME = "king_ce_cache"


def _resolve_king_digest(king_dir: Path) -> str:
    info_path = king_dir / "king_info.json"
    if info_path.exists():
        digest = json.loads(info_path.read_text()).get("digest")
        if digest:
            return str(digest)
    index_path = king_dir / "model.safetensors.index.json"
    if index_path.exists():
        import hashlib

        return hashlib.sha256(index_path.read_bytes()).hexdigest()
    import hashlib

    return hashlib.sha256(str(king_dir.resolve()).encode()).hexdigest()


def _attn_cache_tag() -> str:
    """King CE depends on attention backend; keep FA4/eager caches separate."""
    import os

    from src.model import _resolve_attn_implementation

    return str(_resolve_attn_implementation(os.environ.get("TEUTONIC_ATTN") or None) or "eager")


def _eval_sequence_fingerprint(
    eval_dataset: EvalDataset,
    sources: list,
    eval_n: int,
) -> str:
    """Hash the exact token rows used for this eval_n (binds king CE cache to data)."""
    import hashlib

    h = hashlib.sha256()
    h.update(str(getattr(eval_dataset, "seed", "")).encode())
    for source in sources:
        n_source = max(1, round(eval_n * source.proportion))
        batch = eval_dataset.get_batch(source.name)[:n_source]
        h.update(source.name.encode())
        h.update(str(n_source).encode())
        h.update(batch.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()[:32]


def _king_ce_cache_root(
    cache_dir: Path,
    king_digest: str,
    config_version: str,
    attn: str | None = None,
    seq_fp: str | None = None,
) -> Path:
    import hashlib

    attn_tag = attn or _attn_cache_tag()
    fp = seq_fp or "noseq"
    key = hashlib.sha256(f"{king_digest}:{config_version}:{attn_tag}:{fp}".encode()).hexdigest()[:24]
    return cache_dir / KING_CE_CACHE_DIRNAME / key


def _load_king_ce_cache(
    cache_dir: Path,
    king_dir: Path,
    manifest: Manifest,
    sources: list,
    eval_n: int,
    *,
    is_rank0: bool,
    eval_dataset: EvalDataset | None = None,
) -> dict[str, torch.Tensor] | None:
    if not is_rank0:
        return None

    attn = _attn_cache_tag()
    seq_fp = (
        _eval_sequence_fingerprint(eval_dataset, sources, eval_n)
        if eval_dataset is not None
        else None
    )
    root = _king_ce_cache_root(
        cache_dir, _resolve_king_digest(king_dir), manifest.config_version, attn, seq_fp
    )
    meta_path = root / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if (
        meta.get("eval_n") != eval_n
        or meta.get("config_version") != manifest.config_version
        or meta.get("attn_implementation", "eager") != attn
        or (seq_fp is not None and meta.get("sequence_fingerprint") != seq_fp)
    ):
        return None
    expected = {source.name: max(1, round(eval_n * source.proportion)) for source in sources}
    if meta.get("sources") != expected:
        return None
    ces: dict[str, torch.Tensor] = {}
    for name, n_seq in expected.items():
        path = root / f"{name}.pt"
        if not path.exists():
            return None
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tensor.shape[0] != n_seq:
            return None
        ces[name] = tensor
    log.info(
        "Loaded king CE cache from %s (attn=%s, seq_fp=%s, skip king forward)",
        root,
        attn,
        (seq_fp or "?")[:12],
    )
    return ces


def pg_is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _save_king_ce_cache(
    cache_dir: Path,
    king_dir: Path,
    manifest: Manifest,
    sources: list,
    eval_n: int,
    king_ces: dict[str, torch.Tensor],
    *,
    eval_dataset: EvalDataset | None = None,
) -> None:
    if not pg_is_rank0():
        return
    attn = _attn_cache_tag()
    seq_fp = (
        _eval_sequence_fingerprint(eval_dataset, sources, eval_n)
        if eval_dataset is not None
        else "unknown"
    )
    root = _king_ce_cache_root(
        cache_dir, _resolve_king_digest(king_dir), manifest.config_version, attn, seq_fp
    )
    root.mkdir(parents=True, exist_ok=True)
    for name, tensor in king_ces.items():
        torch.save(tensor, root / f"{name}.pt")
    meta = {
        "eval_n": eval_n,
        "config_version": manifest.config_version,
        "king_digest": _resolve_king_digest(king_dir),
        "attn_implementation": attn,
        "sequence_fingerprint": seq_fp,
        "eval_seed": getattr(eval_dataset, "seed", None),
        "sources": {source.name: max(1, round(eval_n * source.proportion)) for source in sources},
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Saved king CE cache to %s (attn=%s, seq_fp=%s)", root, attn, seq_fp[:12])



def _score_all_sources(
    model: torch.nn.Module,
    eval_dataset: EvalDataset,
    sources: list,
    eval_n: int,
    eval_batch_size: int,
    device: torch.device,
    pg: ProcessGroups,
    *,
    label: str = "model",
) -> dict[str, torch.Tensor]:
    ces: dict[str, torch.Tensor] = {}
    for source in sources:
        n_source = max(1, round(eval_n * source.proportion))
        ces[source.name] = score_source_distributed(
            model,
            eval_dataset,
            source.name,
            n_source,
            eval_batch_size,
            device,
            pg,
            progress_desc=f"paired_eval/{label}/{source.name}",
        )
        if pg.is_rank0:
            log.info(
                "Scored %s on %s: mean_ce=%.4f (n=%d)",
                label,
                source.name,
                float(ces[source.name].mean().item()),
                n_source,
            )
    return ces


def _run_paired_dual_eval(
    cfg: dict[str, Any],
    pg: ProcessGroups,
    *,
    eval_dataset: EvalDataset,
    sources: list,
    eval_n: int,
    eval_batch_size: int,
    device: torch.device,
    king_dir: Path,
    challenger_source: Path | None,
    cache_dir: Path,
    manifest: Manifest,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], bool]:
    """Validator-style paired eval: identical sequences, king CE then challenger CE."""
    same_as_king = challenger_source is None or challenger_source.resolve() == king_dir.resolve()
    can_swap = _can_swap_challenger_weights(challenger_source)
    king_ces: dict[str, torch.Tensor] | None = None
    if pg.is_rank0:
        king_ces = _load_king_ce_cache(
            cache_dir,
            king_dir,
            manifest,
            sources,
            eval_n,
            is_rank0=True,
            eval_dataset=eval_dataset,
        )
    king_cache_hit = _broadcast_king_cache_hit(pg, device, king_ces is not None)

    if pg.is_rank0:
        log.info(
            "Paired dual eval (validator-style): king=%s vs challenger=%s on identical sequences",
            king_dir,
            challenger_source if challenger_source else king_dir,
        )

    eval_model: torch.nn.Module | None = None
    challenger_ces: dict[str, torch.Tensor] = {}

    if king_cache_hit:
        if pg.is_rank0:
            log.info("King CE cache hit — reusing cached king scores for paired comparison")
            for source in sources:
                n_source = max(1, round(eval_n * source.proportion))
                k_ce = king_ces[source.name]  # type: ignore[index]
                log.info(
                    "Cached king %s: mean_ce=%.4f (n=%d)",
                    source.name,
                    float(k_ce.mean().item()),
                    n_source,
                )
        if same_as_king:
            if pg.is_rank0:
                challenger_ces = king_ces  # type: ignore[assignment]
            dist.barrier()
        elif can_swap:
            if pg.is_rank0:
                log.info(
                    "Scoring challenger from %s (king skeleton + FSDP weight swap)",
                    challenger_source,
                )
            # Fast path: rank0 loads king once, FSDP syncs, then swap challenger weights.
            eval_model = _load_eval_model(cfg, pg, king_dir, "king-skeleton")
            state = _load_challenger_state_dict(challenger_source) if pg.is_rank0 else {}
            load_state_dict_into_fsdp(eval_model, state)
            eval_model.eval()
            dist.barrier()
            challenger_ces = _score_all_sources(
                eval_model,
                eval_dataset,
                sources,
                eval_n,
                eval_batch_size,
                device,
                pg,
                label="challenger",
            )
            _free_eval_model(eval_model, pg, "challenger")
        else:
            eval_model = _load_eval_model(cfg, pg, challenger_source, "challenger")  # type: ignore[arg-type]
            challenger_ces = _score_all_sources(
                eval_model,
                eval_dataset,
                sources,
                eval_n,
                eval_batch_size,
                device,
                pg,
                label="challenger",
            )
            _free_eval_model(eval_model, pg, "challenger")
    else:
        if pg.is_rank0:
            log.info("Pass 1/2: scoring king on eval sequences")
        eval_model = _load_eval_model(cfg, pg, king_dir, "king")
        king_ces = _score_all_sources(
            eval_model,
            eval_dataset,
            sources,
            eval_n,
            eval_batch_size,
            device,
            pg,
            label="king",
        )
        _save_king_ce_cache(
            cache_dir,
            king_dir,
            manifest,
            sources,
            eval_n,
            king_ces,
            eval_dataset=eval_dataset,
        )

        if same_as_king:
            challenger_ces = king_ces
            _free_eval_model(eval_model, pg, "king")
        elif can_swap:
            if pg.is_rank0:
                log.info("Pass 2/2: swapping to challenger weights (same FSDP shell, paired sequences)")
            state = _load_challenger_state_dict(challenger_source) if pg.is_rank0 else {}
            load_state_dict_into_fsdp(eval_model, state)
            eval_model.eval()
            dist.barrier()
            challenger_ces = _score_all_sources(
                eval_model,
                eval_dataset,
                sources,
                eval_n,
                eval_batch_size,
                device,
                pg,
                label="challenger",
            )
            _free_eval_model(eval_model, pg, "challenger")
        else:
            _free_eval_model(eval_model, pg, "king")
            if pg.is_rank0:
                log.info("Pass 2/2: loading challenger for paired scoring")
            eval_model = _load_eval_model(cfg, pg, challenger_source, "challenger")  # type: ignore[arg-type]
            challenger_ces = _score_all_sources(
                eval_model,
                eval_dataset,
                sources,
                eval_n,
                eval_batch_size,
                device,
                pg,
                label="challenger",
            )
            _free_eval_model(eval_model, pg, "challenger")

    if pg.is_rank0:
        assert king_ces is not None
        for source in sources:
            n_source = max(1, round(eval_n * source.proportion))
            k_mean = float(king_ces[source.name].mean().item())
            c_mean = float(challenger_ces[source.name].mean().item())
            mu = k_mean - c_mean
            k_tensor = king_ces[source.name]
            c_tensor = challenger_ces[source.name]
            wins_pct = float((k_tensor - c_tensor > 0).float().mean().item() * 100.0)
            log_paired_source_row(source.name, n_source, k_mean, c_mean, mu, challenger_wins_pct=wins_pct)

    return king_ces, challenger_ces, king_cache_hit


def _can_swap_challenger_weights(source: Path | None) -> bool:
    """True when challenger weights can be loaded via king skeleton + state dict."""
    if source is None:
        return False
    from src.submit.merge import is_safetensors_model_dir

    source = source.resolve()
    if source.is_file() and source.suffix == ".pt":
        return True
    return is_safetensors_model_dir(source) or (source / "model.pt").exists()


def _weights_sweep_reject_reason(source: Path) -> str | None:
    """Explain why a path cannot be used with --weights-sweep (None if OK)."""
    if not source.exists():
        return "path does not exist"
    if source.is_file() and source.suffix == ".pt":
        return None
    if not source.is_dir():
        return "not a directory"
    if _can_swap_challenger_weights(source):
        return None
    has_shards = any(source.glob("*.safetensors"))
    missing = []
    if not (source / "model.safetensors.index.json").exists() and not (source / "model.safetensors").exists():
        missing.append("model.safetensors.index.json")
    if not (source / "config.json").exists():
        missing.append("config.json")
    if has_shards:
        return (
            f"incomplete weight dir (shards present, missing {', '.join(missing)}); "
            "re-run the merge for this alpha"
        )
    return "no model.safetensors.index.json / model.safetensors / model.pt"


def _broadcast_king_cache_hit(pg: ProcessGroups, device: torch.device, hit: bool) -> bool:
    flag = torch.tensor([1 if hit else 0], device=device, dtype=torch.long)
    dist.broadcast(flag, src=0)
    return bool(flag.item())


def _load_eval_model(
    cfg: dict[str, Any],
    pg: ProcessGroups,
    model_dir: str | Path,
    label: str,
) -> torch.nn.Module:
    """Load one HF checkpoint (rank0 only) and wrap with FSDP for distributed eval."""
    if pg.is_rank0:
        log.info(
            "Loading %s from %s for FSDP eval (%d GPU(s), rank0-only disk read)",
            label,
            model_dir,
            pg.ep_size,
        )
    model = load_challenger_model_distributed(model_dir, pg, attn_implementation="eager")
    model = apply_fsdp(model, pg, cfg)
    model.eval()
    dist.barrier()
    return model


def _free_eval_model(model: torch.nn.Module, pg: ProcessGroups, label: str) -> None:
    """Drop an FSDP model and reclaim CUDA memory before loading the next one."""
    if pg.is_rank0:
        log.info("Releasing %s to free GPU memory before next eval load", label)
    del model
    torch.cuda.empty_cache()
    dist.barrier()


def _write_eval_result_json(
    output_path: Path,
    result: Any,
    *,
    king_dir: Path,
    challenger_source: Path | None,
    king_cache_hit: bool,
    safety_margin: float,
    pg: ProcessGroups,
) -> None:
    safe = result.is_safe_to_submit(safety_margin=safety_margin)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "eval_mode": "paired_dual",
                "mu_hat": result.mu_hat,
                "lcb": result.lcb,
                "accepted": result.accepted,
                "safe_to_submit": safe,
                "delta": result.delta,
                "safety_margin": safety_margin,
                "threshold": result.delta + safety_margin,
                "n_sequences": result.n_sequences,
                "per_source": result.per_source,
                "king_dir": str(king_dir),
                "challenger_weights": str(challenger_source) if challenger_source else None,
                "weights": str(challenger_source) if challenger_source else None,
                "checkpoint": str(challenger_source) if challenger_source else None,
                "king_ce_cached": king_cache_hit,
                "parallel": "fsdp",
                "world_size": pg.world_size,
            },
            indent=2,
        )
    )
    log.info("Wrote eval result to %s", output_path)


def _run_eval_weights_sweep(
    cfg: dict[str, Any],
    pg: ProcessGroups,
    args: argparse.Namespace,
    *,
    eval_dataset: EvalDataset,
    sources: list,
    eval_n: int,
    eval_batch_size: int,
    device: torch.device,
    king_dir: Path,
    cache_dir: Path,
    manifest: Manifest,
    safety_margin: float,
) -> int:
    """Score many challengers in one FSDP session (one skeleton load, many swaps)."""
    weight_paths = [Path(p).resolve() for p in (args.weights_sweep or [])]
    for path in weight_paths:
        reason = _weights_sweep_reject_reason(path)
        if reason is not None:
            raise ValueError(f"--weights-sweep cannot use {path}: {reason}")

    if pg.is_rank0:
        log.info(
            "Fast weights sweep: %d challengers, one FSDP shell (king CE cache + weight swaps)",
            len(weight_paths),
        )

    king_ces: dict[str, torch.Tensor] | None = None
    if pg.is_rank0:
        king_ces = _load_king_ce_cache(
            cache_dir,
            king_dir,
            manifest,
            sources,
            eval_n,
            is_rank0=True,
            eval_dataset=eval_dataset,
        )
    king_cache_hit = _broadcast_king_cache_hit(pg, device, king_ces is not None)
    eval_model: torch.nn.Module | None = None
    if not king_cache_hit:
        if pg.is_rank0:
            log.info("Pass 1: scoring king once for sweep")
        eval_model = _load_eval_model(cfg, pg, king_dir, "king")
        king_ces = _score_all_sources(
            eval_model,
            eval_dataset,
            sources,
            eval_n,
            eval_batch_size,
            device,
            pg,
            label="king",
        )
        _save_king_ce_cache(
            cache_dir,
            king_dir,
            manifest,
            sources,
            eval_n,
            king_ces,
            eval_dataset=eval_dataset,
        )
    else:
        if pg.is_rank0:
            log.info("King CE cache hit — skipping king forward for sweep")
            for source in sources:
                n_source = max(1, round(eval_n * source.proportion))
                k_ce = king_ces[source.name]  # type: ignore[index]
                log.info(
                    "Cached king %s: mean_ce=%.4f (n=%d)",
                    source.name,
                    float(k_ce.mean().item()),
                    n_source,
                )
        eval_model = _load_eval_model(cfg, pg, king_dir, "king-skeleton")

    assert eval_model is not None
    delta = manifest.delta_threshold

    for index, challenger_source in enumerate(weight_paths):
        if pg.is_rank0:
            log.info(
                "Sweep %d/%d: swapping weights from %s",
                index + 1,
                len(weight_paths),
                challenger_source,
            )
        state = _load_challenger_state_dict(challenger_source) if pg.is_rank0 else {}
        load_state_dict_into_fsdp(eval_model, state)
        eval_model.eval()
        dist.barrier()
        challenger_ces = _score_all_sources(
            eval_model,
            eval_dataset,
            sources,
            eval_n,
            eval_batch_size,
            device,
            pg,
            label=f"challenger[{index}]",
        )
        dist.barrier()
        if pg.is_rank0:
            assert king_ces is not None
            result = build_paired_eval_result(king_ces, challenger_ces, sources, eval_n, delta)
            log_paired_eval_summary(
                result,
                king_label=str(king_dir),
                challenger_label=str(challenger_source),
            )
            print(result)
            if args.eval_output is not None and len(weight_paths) == 1:
                out = Path(args.eval_output)
            else:
                stem = challenger_source.name
                out_dir = (
                    Path(args.eval_output).parent
                    if args.eval_output is not None
                    else Path("./merge_eval_results")
                )
                out = out_dir / f"{stem}_eval.json"
            _write_eval_result_json(
                out,
                result,
                king_dir=king_dir,
                challenger_source=challenger_source,
                king_cache_hit=king_cache_hit,
                safety_margin=safety_margin,
                pg=pg,
            )
        dist.barrier()

    _free_eval_model(eval_model, pg, "sweep")
    return 0


def run_eval_only(cfg: dict[str, Any], pg: ProcessGroups, args: argparse.Namespace) -> int:
    """Run validator-style paired dual eval: king vs challenger on identical sequences."""
    training_cfg = cfg["training"]
    submission_cfg = cfg.get("submission", {})
    device = torch.device(f"cuda:{pg.local_rank}")
    safety_margin = float(submission_cfg.get("safety_margin", DEFAULT_SAFETY_MARGIN))
    eval_batch_size = int(training_cfg.get("eval_batch_size", 4))
    king_dir = Path(training_cfg["king_dir"])

    if pg.is_rank0:
        log.info(
            "Eval parallel strategy: FSDP FULL_SHARD on %d GPU(s) (DDP unsuitable for 110B)",
            pg.world_size,
        )
        log.info("Loading evaluation dataset (seed=99999)...")
        eval_dataset = build_eval_dataset(cfg)
    else:
        from src.data import ShardCache

        log.info("Waiting for rank 0 to load eval dataset...")
        eval_dataset = EvalDataset(
            manifest=Manifest(manifest_url=cfg.get("data", {}).get("manifest_url")),
            shard_cache=ShardCache(cfg.get("data", {}).get("cache_dir", "./shard_cache")),
        )
    dist.barrier()

    manifest = Manifest(manifest_url=cfg.get("data", {}).get("manifest_url"))
    sources = manifest.fetch(force=True)
    eval_n = int(args.eval_n) if args.eval_n is not None else manifest.eval_n
    if args.eval_n is not None:
        if args.eval_n < 1:
            raise ValueError(f"--eval-n must be >= 1, got {args.eval_n}")
        if pg.is_rank0:
            log.warning(
                "Overriding manifest eval_n=%d -> %d (smoke/fast path; LCB less reliable)",
                manifest.eval_n,
                eval_n,
            )
    cache_dir = Path(cfg.get("data", {}).get("cache_dir", "./shard_cache"))

    if getattr(args, "weights_sweep", None):
        return _run_eval_weights_sweep(
            cfg,
            pg,
            args,
            eval_dataset=eval_dataset,
            sources=sources,
            eval_n=eval_n,
            eval_batch_size=eval_batch_size,
            device=device,
            king_dir=king_dir,
            cache_dir=cache_dir,
            manifest=manifest,
            safety_margin=safety_margin,
        )

    challenger_source: Path | None = None
    if args.weights is not None:
        challenger_source = Path(args.weights)
        if not challenger_source.exists():
            raise FileNotFoundError(f"--weights path not found: {challenger_source}")
    else:
        checkpoint_dir = resolve_checkpoint(training_cfg["output_dir"], args.checkpoint)
        challenger_source = checkpoint_dir
        if checkpoint_dir is None and pg.is_rank0:
            log.warning("No checkpoint found; evaluating king against itself (expect mu_hat≈0)")

    delta = manifest.delta_threshold
    king_ces, challenger_ces, king_cache_hit = _run_paired_dual_eval(
        cfg,
        pg,
        eval_dataset=eval_dataset,
        sources=sources,
        eval_n=eval_n,
        eval_batch_size=eval_batch_size,
        device=device,
        king_dir=king_dir,
        challenger_source=challenger_source,
        cache_dir=cache_dir,
        manifest=manifest,
    )

    dist.barrier()
    if not pg.is_rank0:
        return 0

    result = build_paired_eval_result(king_ces, challenger_ces, sources, eval_n, delta)
    log_paired_eval_summary(
        result,
        king_label=str(king_dir),
        challenger_label=str(challenger_source if challenger_source else king_dir),
    )
    print(result)
    safe = result.is_safe_to_submit(safety_margin=safety_margin)
    log.info(
        "Submit threshold: LCB > %.3f (delta %.3f + margin %.3f) -> %s",
        result.delta + safety_margin,
        result.delta,
        safety_margin,
        "PASS" if safe else "FAIL",
    )
    output_path = (
        Path(args.eval_output)
        if args.eval_output is not None
        else Path(training_cfg["output_dir"]) / "offline_eval_result.json"
    )
    _write_eval_result_json(
        output_path,
        result,
        king_dir=king_dir,
        challenger_source=challenger_source,
        king_cache_hit=king_cache_hit,
        safety_margin=safety_margin,
        pg=pg,
    )
    return 0 if safe else 1



def run_export(cfg: dict[str, Any], pg: ProcessGroups, args: argparse.Namespace) -> int:
    """Export a validator-ready submission directory."""
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    from src.checkpoint.dcp import is_committed_checkpoint

    training_cfg = cfg["training"]
    submission_cfg = cfg.get("submission", {})
    output_dir = Path(args.submission_dir or submission_cfg.get("output_dir", "./submit_model"))
    genesis_dir = Path(args.genesis_dir or submission_cfg.get("genesis_dir", "./genesis"))
    checkpoint_dir = resolve_checkpoint(training_cfg["output_dir"], args.checkpoint)
    if checkpoint_dir is None:
        if pg.is_rank0:
            log.error("No checkpoint found in %s; pass --checkpoint", training_cfg["output_dir"])
        return 2

    model = load_challenger_model(training_cfg["king_dir"])
    model = apply_fsdp(model, pg, cfg)
    if is_committed_checkpoint(checkpoint_dir):
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        model_state = get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True),
        )
        dcp.load(
            {"model": model_state},
            checkpoint_id=checkpoint_dir / "dcp",
            process_group=pg.ep_group,
        )
        set_model_state_dict(
            model,
            model_state,
            options=StateDictOptions(full_state_dict=False, cpu_offload=False, strict=True),
        )
    else:
        # Legacy format-v2 model.pt path (pre-FSDP2).
        state = strip_mtp_tensors(load_checkpoint_state(checkpoint_dir)) if pg.is_rank0 else {}
        load_state_dict_into_fsdp(model, state)

    dist.barrier()
    full_state = get_model_state_dict(
        model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            strict=True,
        ),
    )
    state_dict = strip_mtp_tensors(full_state) if pg.is_rank0 else {}
    if pg.is_rank0:
        meta = load_checkpoint_meta(checkpoint_dir)
        meta["checkpoint"] = str(checkpoint_dir)
        export_submission(
            state_dict=state_dict,
            king_dir=training_cfg["king_dir"],
            output_dir=output_dir,
            genesis_dir=genesis_dir,
            rank=pg.global_rank,
            meta=meta,
        )
        log.info("Submission ready at %s", output_dir)
    dist.barrier()
    return 0


def train(cfg: dict[str, Any], pg: ProcessGroups, args: argparse.Namespace) -> None:
    training_cfg = cfg["training"]
    device = torch.device(f"cuda:{pg.local_rank}")
    king_dir = training_cfg["king_dir"]
    output_dir = Path(training_cfg["output_dir"])
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    total_steps = int(training_cfg["total_steps"])
    log_every = int(training_cfg.get("log_every", 10))
    base_seed = int(training_cfg.get("base_seed", 42))
    pool_enabled = bool(cfg.get("data", {}).get("sequence_pool", {}).get("enabled", False))
    synthetic_data = bool(training_cfg.get("smoke_synthetic_data", False))
    if args.resume and not pool_enabled and not synthetic_data:
        raise RuntimeError(
            "Deterministic resume requires data.sequence_pool.enabled=true; "
            "streaming sampler position is not checkpointable"
        )
    if bool(cfg.get("sparse_loco", {}).get("enabled", False)):
        raise RuntimeError(
            "SparseLoCo is disabled until multi-node synchronization and name-keyed "
            "checkpoint resume are validated"
        )
    torch.manual_seed(base_seed + pg.global_rank)

    tokenizer = _load_tokenizer(king_dir)
    eos_id = int(tokenizer.eos_token_id)

    attn_implementation = training_cfg.get("attn_implementation") or training_cfg.get(
        "attention_implementation"
    )
    model = load_challenger_model_distributed(
        king_dir,
        pg,
        attn_implementation=str(attn_implementation) if attn_implementation else None,
    )
    _validate_packing_mode(model, cfg)
    apply_lora(model, cfg)
    qk_norm_active = check_and_inject_qk_norm(model, cfg)
    if pg.is_rank0:
        log.info("QK norm active: %s", qk_norm_active)

    if bool(training_cfg.get("activation_checkpointing", False)):
        from src.model import _find_decoder_layer_class

        ac_mode = str(training_cfg.get("activation_checkpointing_mode", "selective"))
        enable_gradient_checkpointing(
            model,
            _find_decoder_layer_class(model),
            mode=ac_mode,
        )

    low_precision = enable_low_precision_training(model, pg, cfg)
    print_model_summary(model, low_precision=low_precision)

    moe_cfg = cfg.get("moe", {})
    if bool(moe_cfg.get("delay_wgrad", False)):
        raise RuntimeError("moe.delay_wgrad is not implemented; keep it false")
    if int(moe_cfg.get("node_limit", 1)) != 1:
        raise RuntimeError("moe.node_limit values other than 1 are unsupported")
    if not bool(cfg.get("parallelism", {}).get("fsdp", True)):
        raise RuntimeError("parallelism.fsdp=false is unsupported for the 110B model")
    # Custom EP dispatch was incorrect (not truly async, broken combine, Python token
    # loops). Keep native MiMo MoE under FSDP — do not enable dispatch_overlap.
    if bool(moe_cfg.get("dispatch_overlap", False)):
        raise RuntimeError(
            "moe.dispatch_overlap is disabled: the custom ExpertDispatcher path is "
            "unsafe/incorrect for this MiMo MoE. Keep dispatch_overlap: false and use "
            "native MoE + FSDP."
        )

    hook_manager = register_router_hooks(model)
    _validate_router_hooks(model, hook_manager, moe_cfg)
    if pg.is_rank0:
        log.info("Registered %d validated MoE router hooks", hook_manager.router_count)

    # FSDP2 keeps logical parameter shapes as DTensors. Build optimizers after
    # sharding so their persistent states inherit the same placements.
    model = apply_fsdp(model, pg, cfg)
    optimizer = build_optimizer(model, cfg, pg)
    maybe_compile_optimizer(optimizer, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    compile_decoder_layers(model, pg, cfg)

    sparse_cfg = cfg.get("sparse_loco", {})
    sparse_loco = SparseLoCo(
        model=model,
        dp_group=pg.dp_group,
        inner_steps=int(sparse_cfg.get("inner_steps", 200)),
        sparsity_k=float(sparse_cfg.get("sparsity_k", 0.01)),
        quant_bits=int(sparse_cfg.get("quant_bits", 2)),
        outer_lr=float(sparse_cfg.get("outer_lr", 0.7)),
        outer_momentum=float(sparse_cfg.get("outer_momentum", 0.9)),
        enabled=bool(sparse_cfg.get("enabled", True)) and pg.dp_size > 1,
    )
    sparse_loco.snapshot_params()

    start_step = 0
    if args.resume:
        start_step = _load_training_resume(
            Path(args.resume),
            model,
            optimizer,
            scheduler,
            sparse_loco,
            pg,
        )

    initial_num_workers_override: int | None = None
    if pool_enabled and args.resume:
        # For exact step-boundary continuity after resume, keep the first step single-process.
        initial_num_workers_override = 0

    if synthetic_data:
        if pg.is_rank0:
            patterned = bool(training_cfg.get("smoke_patterned_data", False))
            log.info(
                "Using %s smoke batches (seq_len=%s); skipping live shard downloads",
                "patterned" if patterned else "random synthetic",
                training_cfg.get("smoke_seq_len", 256),
            )
        train_loader = None
        train_iter = _synthetic_smoke_batch_iterator(
            cfg,
            global_rank=pg.global_rank,
            start_step=start_step,
        )
    else:
        train_loader = build_train_loader(
            cfg,
            rank=pg.global_rank,
            world_size=pg.world_size,
            start_step=start_step,
            num_workers_override=initial_num_workers_override,
        )
        train_iter = iter(train_loader) if pool_enabled else _batch_iterator(train_loader)
    numworkers_switched_after_resume = False
    dataset_tracker = DatasetLossTracker()
    opt_cfg = cfg.get("optimizer", {})
    clip_cfg = opt_cfg.get("gradient_clip", {})
    ckpt_avg_cfg = cfg.get("checkpoint_averaging", {})
    decay_triggered = False
    smoke_steps = int(training_cfg.get("smoke_steps", 0))
    if smoke_steps > 0:
        total_steps = min(total_steps, start_step + smoke_steps)
        if pg.is_rank0:
            log.info("Smoke mode: limiting to %d steps (smoke_steps=%d)", total_steps, smoke_steps)
    dist.barrier()

    model.train()
    step_start_time = time.monotonic()
    mean_lm_loss = 0.0
    # Apply the scheduled LR for start_step *before* the first optimizer update
    # (otherwise step 0 would run at the constructed max LR, skipping warmup).
    if hasattr(scheduler, "_set_lr"):
        lr = scheduler._set_lr(scheduler.get_lr(getattr(scheduler, "step_count", start_step)))
    else:
        lr = scheduler.get_lr(getattr(scheduler, "step_count", start_step))
    if pg.is_rank0:
        attn_name = getattr(getattr(model, "config", None), "_attn_implementation", "?")
        log.info(
            "Entering train loop at step=%d total_steps=%d attn=%s "
            "(FA4 first-step JIT can take 10–20+ min under 8-GPU contention; later steps are fast)",
            start_step,
            total_steps,
            attn_name,
        )
    for step in range(start_step, total_steps):
        accum_lm_loss = 0.0
        accum_z_loss = 0.0
        accum_aux_loss = 0.0
        batch_datasets: list[str] = []
        if pg.is_rank0:
            log.info("step=%d starting forward/backward", step)

        for _ in range(grad_accum_steps):
            raw_batch = next(train_iter)
            batch = _prepare_batch(raw_batch, cfg, eos_id, device)
            batch_datasets.extend(batch["datasets"])

            with low_precision_context(low_precision):
                # Keep TE autocast active through backward so activation-checkpoint
                # recompute sees MXFP8/NVFP4 (otherwise dtype/shape metadata mismatches).
                outputs = _forward_model(model, batch)

                if (
                    (
                        float(moe_cfg.get("z_loss_coeff", 0.0)) > 0
                        or float(moe_cfg.get("seq_aux_alpha", 0.0)) > 0
                    )
                    and hook_manager.record_count == 0
                ):
                    raise RuntimeError(
                        "MoE losses are enabled, but no router hook records were captured "
                        "during the model forward"
                    )
                lm_loss = outputs.loss
                z_loss = compute_z_loss(hook_manager, float(moe_cfg.get("z_loss_coeff", 1e-4)))
                aux_loss = compute_seq_auxiliary_loss(
                    hook_manager, float(moe_cfg.get("seq_aux_alpha", 1e-4))
                )
                if not isinstance(z_loss, torch.Tensor):
                    z_loss = torch.zeros((), device=device)
                if not isinstance(aux_loss, torch.Tensor):
                    aux_loss = torch.zeros((), device=device)
                if z_loss.device != device:
                    z_loss = z_loss.to(device)
                if aux_loss.device != device:
                    aux_loss = aux_loss.to(device)

                total_loss = lm_loss + z_loss + aux_loss
                (total_loss / grad_accum_steps).backward()

            accum_lm_loss += float(lm_loss.detach().item())
            accum_z_loss += float(z_loss.detach().item())
            accum_aux_loss += float(aux_loss.detach().item())
            # Fold detached routing counts for bias; drop live records so the next
            # microbatch does not retain a completed autograd graph.
            hook_manager.accumulate_selection_counts()
            hook_manager.clear()

        grad_norms = clip_grad_norms_per_module(
            model,
            pg,
            expert_max_norm=float(clip_cfg.get("expert_ffn", 1.0)),
            attention_max_norm=float(clip_cfg.get("attention", 1.0)),
            router_max_norm=float(clip_cfg.get("router", 0.5)),
        )
        mean_lm_loss = accum_lm_loss / grad_accum_steps
        optimizer.step()

        if sparse_loco.should_sync(step):
            sync_stats = sparse_loco.outer_step(step)
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                output_dir,
                pg,
                meta={
                    "lm_loss": mean_lm_loss,
                    "lr": scheduler.get_lr(getattr(scheduler, "step_count", step)),
                    "phase": getattr(scheduler, "phase", "unknown"),
                    "dataset_losses": dataset_tracker.summary(),
                    "sparse_loco": {
                        "compression_ratio": sync_stats.compression_ratio,
                        "sync_time_ms": sync_stats.sync_time_ms,
                        "pseudo_grad_norm": sync_stats.pseudo_grad_norm,
                    },
                    "outer_sync": True,
                },
                suffix="_outer",
                sparse_loco=sparse_loco,
            )

        lr = scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        dataset_tracker.update(batch_datasets, mean_lm_loss)
        # Stats must be read before bias update clears accumulated load counts.
        router_stats = router_stats_summary(hook_manager, process_group=pg.ep_group)
        if pg.is_rank0 and step % log_every == 0:
            log_expert_utilization(hook_manager, step, summary=router_stats)
        update_expert_biases(
            model,
            hook_manager,
            bias_update_speed=float(moe_cfg.get("bias_update_speed", 0.001)),
            process_group=pg.ep_group,
        )
        hook_manager.clear_all()

        # After the first resumed step, switch to the configured DataLoader
        # parallelism. For pool-mode, this preserves exact sequence continuity
        # at step boundaries because the pool dataset is start_step-addressable.
        if (
            pool_enabled
            and args.resume
            and not numworkers_switched_after_resume
            and step == start_step
        ):
            train_loader = build_train_loader(
                cfg,
                rank=pg.global_rank,
                world_size=pg.world_size,
                start_step=step + 1,
                num_workers_override=None,
            )
            train_iter = iter(train_loader)
            numworkers_switched_after_resume = True

        if (
            should_save_full_checkpoint(cfg)
            and bool(ckpt_avg_cfg.get("enabled", False))
            and getattr(scheduler, "phase", "") == "stable"
            and step > 0
            and step % int(ckpt_avg_cfg.get("save_every_stable", 100)) == 0
        ):
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                step,
                output_dir,
                pg,
                meta={
                    "lm_loss": mean_lm_loss,
                    "lr": lr,
                    "phase": scheduler.phase,
                    "dataset_losses": dataset_tracker.summary(),
                    "sparse_loco": {
                        "compression_ratio": sparse_loco.last_stats.compression_ratio,
                        "sync_time_ms": sparse_loco.last_stats.sync_time_ms,
                        "pseudo_grad_norm": sparse_loco.last_stats.pseudo_grad_norm,
                    },
                },
                sparse_loco=sparse_loco,
            )

        if args.trigger_decay and not decay_triggered and getattr(scheduler, "phase", "") == "stable":
            run_pre_decay_soup(
                model=model,
                output_dir=output_dir,
                scheduler=scheduler,
                rank=pg.global_rank,
                n_checkpoints=int(ckpt_avg_cfg.get("n_checkpoints", 5)),
                min_step_gap=int(ckpt_avg_cfg.get("min_step_gap", 50)),
                process_group=pg.ep_group,
            )
            decay_triggered = True

        if pg.is_rank0 and step % log_every == 0:
            elapsed = max(time.monotonic() - step_start_time, 1e-6)
            tokens = (
                grad_accum_steps
                * int(training_cfg["micro_batch_size"])
                * int(training_cfg.get("smoke_seq_len") or 2048)
                * pg.world_size
            )
            tokens_per_sec = tokens / elapsed
            log.info(
                "step=%d lm_loss=%.4f z_loss=%.6f aux_loss=%.6f lr=%.2e phase=%s "
                "grad_total=%.3f dead=%.0f max_load=%.4f tokens/s=%.0f low_precision=%s "
                "dp=%d ep=%d sparse_loco=%s compression=%.4f datasets=%s",
                step,
                mean_lm_loss,
                accum_z_loss / grad_accum_steps,
                accum_aux_loss / grad_accum_steps,
                lr,
                getattr(scheduler, "phase", "unknown"),
                grad_norms["total"],
                router_stats["dead_experts"],
                router_stats["max_expert_load"],
                tokens_per_sec,
                low_precision.mode,
                pg.dp_size,
                pg.ep_size,
                sparse_loco.enabled,
                sparse_loco.last_stats.compression_ratio,
                dataset_tracker.summary(),
            )
            step_start_time = time.monotonic()

    if should_save_full_checkpoint(cfg):
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            total_steps - 1,
            output_dir,
            pg,
            meta={
                "lm_loss": mean_lm_loss,
                "lr": lr,
                "phase": getattr(scheduler, "phase", "unknown"),
                "dataset_losses": dataset_tracker.summary(),
                "sparse_loco": {
                    "compression_ratio": sparse_loco.last_stats.compression_ratio,
                    "sync_time_ms": sparse_loco.last_stats.sync_time_ms,
                    "pseudo_grad_norm": sparse_loco.last_stats.pseudo_grad_norm,
                },
                "final": True,
            },
            sparse_loco=sparse_loco,
        )
    elif pg.is_rank0:
        log.info("Skipping full FSDP checkpoint save (lora.enabled and save_full_checkpoint=false)")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [rank=%(process)s] %(levelname)s %(message)s",
    )
    cfg = load_config(args.config)

    pg = setup_distributed(cfg)
    try:
        if args.eval_only:
            return run_eval_only(cfg, pg, args)
        if args.export:
            return run_export(cfg, pg, args)
        train(cfg, pg, args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
