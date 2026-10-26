"""MoE-aware merging of two or more model checkpoints."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import Tensor

from src.submit.export import CONTRACT_FILENAMES, CONTRACT_GLOBS, strip_mtp_tensors

log = logging.getLogger(__name__)

ROUTED_EXPERT_KEYWORDS = (".experts.", "experts.", "expert_", "mlp.experts")
SHARED_EXPERT_KEYWORDS = ("shared_expert", "shared_experts")
ROUTER_KEYWORDS = ("router", "gate", "expert_bias", "e_score_correction", "moe_gate")
ATTENTION_KEYWORDS = ("self_attn", ".attn.", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "qkv")
EMBED_LM_KEYWORDS = ("embed_tokens", "lm_head", "word_embeddings", "wte", "output")
MTP_KEY_SUBSTRING = "mtp"


class ParamGroup(str, Enum):
    """Tensor groups for strategy-specific blending."""

    ROUTED_EXPERT = "routed_expert"
    SHARED_EXPERT = "shared_expert"
    ROUTER = "router"
    ATTENTION = "attention"
    EMBED_LM = "embed_lm"
    OTHER = "other"


class MergeStrategy(str, Enum):
    """Supported merge strategies."""

    LINEAR = "linear"
    SMA = "sma"
    AOE = "aoe"
    EXPERTS_ONLY = "experts_only"


@dataclass
class MergeConfig:
    """Configuration for MoE model merging."""

    strategy: MergeStrategy = MergeStrategy.AOE
    alpha: float = 0.5
    dtype: torch.dtype = torch.bfloat16
    group_alphas: dict[str, float] = field(default_factory=dict)
    strict_keys: bool = True

    def __repr__(self) -> str:
        return (
            f"MergeConfig(strategy={self.strategy.value}, alpha={self.alpha}, "
            f"group_alphas={self.group_alphas})"
        )

    def alpha_for(self, group: ParamGroup) -> float:
        if group.value in self.group_alphas:
            return float(self.group_alphas[group.value])
        if self.strategy == MergeStrategy.EXPERTS_ONLY:
            if group in (ParamGroup.ROUTED_EXPERT, ParamGroup.SHARED_EXPERT):
                return self.alpha
            return 0.0
        if self.strategy == MergeStrategy.AOE:
            defaults = {
                ParamGroup.ROUTED_EXPERT: self.alpha,
                ParamGroup.SHARED_EXPERT: self.alpha,
                ParamGroup.ROUTER: 0.0,
                ParamGroup.ATTENTION: 0.0,
                ParamGroup.EMBED_LM: 0.0,
                ParamGroup.OTHER: 0.0,
            }
            return defaults[group]
        return self.alpha


@dataclass
class MergeReport:
    """Summary of a merge operation."""

    strategy: str
    n_tensors: int
    n_from_a_only: int
    n_blended: int
    group_counts: dict[str, int]
    sources: list[str]
    elapsed_sec: float = 0.0
    mode: str = "cpu"
    alpha: float | None = None

    def __repr__(self) -> str:
        return (
            f"MergeReport(strategy={self.strategy!r}, mode={self.mode}, alpha={self.alpha}, "
            f"n_tensors={self.n_tensors}, blended={self.n_blended}, elapsed={self.elapsed_sec:.1f}s)"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "mode": self.mode,
            "alpha": self.alpha,
            "n_tensors": self.n_tensors,
            "n_from_a_only": self.n_from_a_only,
            "n_blended": self.n_blended,
            "group_counts": self.group_counts,
            "sources": self.sources,
            "elapsed_sec": self.elapsed_sec,
        }


def classify_param(key: str) -> ParamGroup:
    """Classify a state-dict key into an MoE merge group."""
    lower = key.lower()
    if any(keyword in lower for keyword in SHARED_EXPERT_KEYWORDS):
        return ParamGroup.SHARED_EXPERT
    if any(keyword in lower for keyword in ROUTED_EXPERT_KEYWORDS):
        return ParamGroup.ROUTED_EXPERT
    if any(keyword in lower for keyword in ROUTER_KEYWORDS):
        return ParamGroup.ROUTER
    if any(keyword in lower for keyword in ATTENTION_KEYWORDS):
        return ParamGroup.ATTENTION
    if any(keyword in lower for keyword in EMBED_LM_KEYWORDS):
        return ParamGroup.EMBED_LM
    return ParamGroup.OTHER


def summarize_groups(state_dict: dict[str, Tensor]) -> dict[str, int]:
    """Count tensors per merge group."""
    counts: dict[str, int] = {group.value: 0 for group in ParamGroup}
    for key in state_dict:
        counts[classify_param(key).value] += 1
    return counts


def _blend_tensors(base: Tensor, donor: Tensor, alpha: float, dtype: torch.dtype) -> Tensor:
    """Blend ``(1 - alpha) * base + alpha * donor``.

    Supports extrapolation: ``alpha < 0`` steps away from donor past base;
    ``alpha > 1`` steps past donor. Exact copies for ``alpha in {0, 1}``.
    """
    if alpha == 0.0:
        return base.to(dtype=dtype)
    if alpha == 1.0:
        return donor.to(dtype=dtype)
    return (
        base.to(dtype=torch.float32) * (1.0 - alpha) + donor.to(dtype=torch.float32) * alpha
    ).to(dtype=dtype)


def _count_blend(alpha: float) -> tuple[int, int]:
    """Return ``(n_from_a_only_delta, n_blended_delta)`` for one tensor."""
    if alpha == 0.0:
        return 1, 0
    if alpha == 1.0:
        return 0, 0
    return 0, 1



def _filter_mtp_keys(weight_map: dict[str, str]) -> dict[str, str]:
    return {key: shard for key, shard in weight_map.items() if MTP_KEY_SUBSTRING not in key.lower()}


def is_safetensors_model_dir(path: Path) -> bool:
    """Return True if path contains sharded or single safetensors weights."""
    return (path / "model.safetensors.index.json").exists() or (path / "model.safetensors").exists()


def get_safetensors_weight_map(path: Path) -> dict[str, str]:
    """Return key -> shard filename for a HuggingFace safetensors model."""
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return _filter_mtp_keys(index.get("weight_map", {}))
    single = path / "model.safetensors"
    if single.exists():
        from safetensors import safe_open

        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys() if MTP_KEY_SUBSTRING not in key.lower()}
    raise FileNotFoundError(f"No safetensors weights in {path}")


def list_safetensor_shards(path: Path) -> list[str]:
    """Return sorted unique shard filenames for a safetensors model directory."""
    weight_map = get_safetensors_weight_map(path)
    return sorted(set(weight_map.values()))


def copy_contract_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy HuggingFace contract files needed for from_pretrained loading."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in CONTRACT_FILENAMES:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    for pattern in CONTRACT_GLOBS:
        for src in sorted(src_dir.glob(pattern)):
            shutil.copy2(src, dst_dir / src.name)


def merge_two_state_dicts(
    base: dict[str, Tensor],
    donor: dict[str, Tensor],
    config: MergeConfig,
) -> tuple[dict[str, Tensor], MergeReport]:
    """Merge donor into base using the configured strategy."""
    base = strip_mtp_tensors(base)
    donor = strip_mtp_tensors(donor)

    base_keys = set(base.keys())
    donor_keys = set(donor.keys())
    if base_keys != donor_keys:
        missing_in_donor = sorted(base_keys - donor_keys)
        missing_in_base = sorted(donor_keys - base_keys)
        if config.strict_keys:
            raise KeyError(
                "State dict keys mismatch. "
                f"missing_in_donor={missing_in_donor[:5]} missing_in_base={missing_in_base[:5]}"
            )
        shared = base_keys & donor_keys
        log.warning(
            "Key mismatch: using %d shared keys (%d only in base, %d only in donor)",
            len(shared),
            len(base_keys - donor_keys),
            len(donor_keys - base_keys),
        )
        keys = sorted(shared)
    else:
        keys = sorted(base_keys)

    merged: dict[str, Tensor] = {}
    group_counts: dict[str, int] = {group.value: 0 for group in ParamGroup}
    n_blended = 0
    n_from_a_only = 0

    for key in keys:
        group = classify_param(key)
        group_counts[group.value] += 1
        alpha = config.alpha_for(group)
        tensor_a = base[key]
        tensor_b = donor[key]
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}")

        merged[key] = _blend_tensors(tensor_a, tensor_b, alpha, config.dtype)
        da, db = _count_blend(alpha)
        n_from_a_only += da
        n_blended += db

    report = MergeReport(
        strategy=config.strategy.value,
        n_tensors=len(merged),
        n_from_a_only=n_from_a_only,
        n_blended=n_blended,
        group_counts=group_counts,
        sources=["base", "donor"],
        mode="cpu_dict",
        alpha=config.alpha,
    )
    return merged, report


def merge_state_dicts(
    states: list[dict[str, Tensor]],
    config: MergeConfig,
) -> tuple[dict[str, Tensor], MergeReport]:
    """Merge multiple state dicts. SMA averages all; others fold pairwise left-to-right."""
    if len(states) < 2:
        raise ValueError("At least two state dicts are required")

    cleaned = [strip_mtp_tensors(state) for state in states]
    if config.strategy == MergeStrategy.SMA:
        keys = sorted(cleaned[0].keys())
        for state in cleaned[1:]:
            if config.strict_keys and set(state.keys()) != set(keys):
                raise KeyError("All checkpoints must share identical keys for SMA merge")
        merged: dict[str, Tensor] = {}
        for key in keys:
            stacked = torch.stack(
                [state[key].to(dtype=torch.float32) for state in cleaned],
                dim=0,
            )
            merged[key] = stacked.mean(dim=0).to(dtype=config.dtype)
        report = MergeReport(
            strategy=config.strategy.value,
            n_tensors=len(merged),
            n_from_a_only=0,
            n_blended=len(merged),
            group_counts=summarize_groups(merged),
            sources=[f"model_{index}" for index in range(len(states))],
            mode="cpu_dict",
            alpha=config.alpha,
        )
        return merged, report

    merged = cleaned[0]
    report: MergeReport | None = None
    for donor in cleaned[1:]:
        merged, report = merge_two_state_dicts(merged, donor, config)
    assert report is not None
    report.sources = [f"model_{index}" for index in range(len(states))]
    return merged, report


class _TensorSource:
    """Lazy tensor reader for safetensors dirs or model.pt checkpoints."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._pt_state: dict[str, Tensor] | None = None
        self._weight_map: dict[str, str] | None = None
        self._open_shards: dict[str, Any] = {}

    def weight_map(self) -> dict[str, str]:
        if self._weight_map is None:
            if is_safetensors_model_dir(self.root):
                self._weight_map = get_safetensors_weight_map(self.root)
            elif (self.root / "model.pt").exists():
                self._pt_state = strip_mtp_tensors(
                    torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
                )
                self._weight_map = {key: "model.pt" for key in self._pt_state}
            else:
                raise FileNotFoundError(f"No supported weights in {self.root}")
        return self._weight_map

    def get_tensor(self, key: str) -> Tensor:
        weight_map = self.weight_map()
        if key not in weight_map:
            raise KeyError(f"Missing key {key} in {self.root}")
        shard = weight_map[key]
        if shard == "model.pt":
            if self._pt_state is None:
                self._pt_state = strip_mtp_tensors(
                    torch.load(self.root / "model.pt", map_location="cpu", weights_only=True)
                )
            return self._pt_state[key]

        from safetensors import safe_open

        shard_path = self.root / shard
        if shard not in self._open_shards:
            self._open_shards[shard] = safe_open(shard_path, framework="pt", device="cpu")
        return self._open_shards[shard].get_tensor(key)

    def close(self) -> None:
        self._open_shards.clear()
        self._pt_state = None


def _merge_shard_tensors(
    keys: list[str],
    source_a: _TensorSource,
    source_b: _TensorSource,
    config: MergeConfig,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, int], int, int]:
    from safetensors.torch import save_file

    del save_file
    outputs: dict[str, Tensor] = {}
    group_counts: dict[str, int] = {group.value: 0 for group in ParamGroup}
    n_blended = 0
    n_from_a_only = 0

    for key in keys:
        group = classify_param(key)
        group_counts[group.value] += 1
        alpha = config.alpha_for(group)
        tensor_a = source_a.get_tensor(key)
        tensor_b = source_b.get_tensor(key)
        if tensor_a.shape != tensor_b.shape:
            raise ValueError(f"Shape mismatch for {key}: {tensor_a.shape} vs {tensor_b.shape}")
        if device.type == "cuda":
            tensor_a = tensor_a.to(device, non_blocking=True)
            tensor_b = tensor_b.to(device, non_blocking=True)
        blended = _blend_tensors(tensor_a, tensor_b, alpha, config.dtype)
        outputs[key] = blended.cpu()
        da, db = _count_blend(alpha)
        n_from_a_only += da
        n_blended += db
    return outputs, group_counts, n_blended, n_from_a_only


def merge_safetensors_streaming(
    path_a: Path,
    path_b: Path,
    output_dir: Path,
    config: MergeConfig,
    device: torch.device | None = None,
    copy_contracts_from: Path | None = None,
) -> MergeReport:
    """Shard-at-a-time merge without loading the full model into RAM."""
    from safetensors.torch import save_file

    start = time.monotonic()
    device = device or torch.device("cpu")
    weight_map = get_safetensors_weight_map(path_a)
    source_b_probe = _TensorSource(path_b)
    weight_map_b = source_b_probe.weight_map()
    source_b_probe.close()
    if config.strict_keys and set(weight_map.keys()) != set(weight_map_b.keys()):
        raise KeyError("Weight key sets differ between model-a and model-b")

    output_dir.mkdir(parents=True, exist_ok=True)
    source_a = _TensorSource(path_a)
    source_b = _TensorSource(path_b)

    group_counts: dict[str, int] = {group.value: 0 for group in ParamGroup}
    n_blended = 0
    n_from_a_only = 0
    n_tensors = 0

    for shard_name in list_safetensor_shards(path_a):
        keys = sorted(key for key, shard in weight_map.items() if shard == shard_name)
        outputs, shard_groups, shard_blended, shard_from_a = _merge_shard_tensors(
            keys,
            source_a,
            source_b,
            config,
            device,
        )
        save_file(outputs, output_dir / shard_name)
        n_tensors += len(outputs)
        n_blended += shard_blended
        n_from_a_only += shard_from_a
        for group_name, count in shard_groups.items():
            group_counts[group_name] += count
        log.info("Merged shard %s (%d tensors) on %s", shard_name, len(outputs), device)

    source_a.close()
    source_b.close()

    index_src = path_a / "model.safetensors.index.json"
    if index_src.exists():
        shutil.copy2(index_src, output_dir / "model.safetensors.index.json")
    if copy_contracts_from is not None:
        copy_contract_files(copy_contracts_from, output_dir)

    elapsed = time.monotonic() - start
    return MergeReport(
        strategy=config.strategy.value,
        n_tensors=n_tensors,
        n_from_a_only=n_from_a_only,
        n_blended=n_blended,
        group_counts=group_counts,
        sources=[str(path_a.resolve()), str(path_b.resolve())],
        elapsed_sec=elapsed,
        mode=f"streaming_{device.type}",
        alpha=config.alpha,
    )


def merge_safetensors_distributed(
    path_a: Path,
    path_b: Path,
    output_dir: Path,
    config: MergeConfig,
    gpus_per_node: int = 8,
    copy_contracts_from: Path | None = None,
) -> MergeReport:
    """Parallel shard merge across GPUs (intended for 8× Pro 6000 / similar)."""
    from safetensors.torch import save_file

    from src.dist.process_groups import ProcessGroups, init_process_groups

    if not dist.is_initialized():
        raise RuntimeError("merge_safetensors_distributed requires torchrun / init_process_group")

    pg = init_process_groups(gpus_per_node=gpus_per_node)
    device = torch.device(f"cuda:{pg.local_rank}")
    start = time.monotonic()

    weight_map = get_safetensors_weight_map(path_a)
    source_b_probe = _TensorSource(path_b)
    weight_map_b = source_b_probe.weight_map()
    source_b_probe.close()
    if config.strict_keys and set(weight_map.keys()) != set(weight_map_b.keys()):
        raise KeyError("Weight key sets differ between model-a and model-b")

    if pg.is_rank0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    shards = list_safetensor_shards(path_a)
    my_shards = [shard for index, shard in enumerate(shards) if index % pg.world_size == pg.global_rank]

    source_a = _TensorSource(path_a)
    source_b = _TensorSource(path_b)
    group_counts: dict[str, int] = {group.value: 0 for group in ParamGroup}
    n_blended = 0
    n_from_a_only = 0
    n_tensors = 0

    for shard_name in my_shards:
        keys = sorted(key for key, shard in weight_map.items() if shard == shard_name)
        outputs, shard_groups, shard_blended, shard_from_a = _merge_shard_tensors(
            keys,
            source_a,
            source_b,
            config,
            device,
        )
        save_file(outputs, output_dir / shard_name)
        n_tensors += len(outputs)
        n_blended += shard_blended
        n_from_a_only += shard_from_a
        for group_name, count in shard_groups.items():
            group_counts[group_name] += count
        if pg.is_rank0 or shard_name == my_shards[0]:
            log.info(
                "rank=%d merged shard %s (%d tensors)",
                pg.global_rank,
                shard_name,
                len(outputs),
            )

    source_a.close()
    source_b.close()
    dist.barrier()

    stats = torch.tensor(
        [n_tensors, n_blended, n_from_a_only, *[group_counts[group.value] for group in ParamGroup]],
        dtype=torch.long,
        device=device,
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if pg.is_rank0:
        index_src = path_a / "model.safetensors.index.json"
        if index_src.exists():
            shutil.copy2(index_src, output_dir / "model.safetensors.index.json")
        if copy_contracts_from is not None:
            copy_contract_files(copy_contracts_from, output_dir)

    elapsed = time.monotonic() - start
    group_totals = {
        group.value: int(stats[3 + index].item()) for index, group in enumerate(ParamGroup)
    }
    report = MergeReport(
        strategy=config.strategy.value,
        n_tensors=int(stats[0].item()),
        n_from_a_only=int(stats[2].item()),
        n_blended=int(stats[1].item()),
        group_counts=group_totals,
        sources=[str(path_a.resolve()), str(path_b.resolve())],
        elapsed_sec=elapsed,
        mode=f"distributed_{pg.world_size}gpu",
        alpha=config.alpha,
    )
    if pg.is_rank0:
        (output_dir / "merge_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        log.info("Distributed merge complete: %s", report)
    dist.barrier()
    return report


def load_model_state(path: str | Path, timeout: float = 120.0) -> dict[str, Tensor]:
    """Load a model state dict from a training checkpoint or HF model directory."""
    del timeout
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Model path not found: {root}")

    model_pt = root / "model.pt"
    if model_pt.exists():
        return strip_mtp_tensors(torch.load(model_pt, map_location="cpu", weights_only=True))

    if is_safetensors_model_dir(root):
        source = _TensorSource(root)
        weight_map = source.weight_map()
        state = {key: source.get_tensor(key) for key in sorted(weight_map)}
        source.close()
        return strip_mtp_tensors(state)

    pytorch_bin = root / "pytorch_model.bin"
    if pytorch_bin.exists():
        return strip_mtp_tensors(torch.load(pytorch_bin, map_location="cpu", weights_only=True))

    raise FileNotFoundError(
        f"No supported weights found in {root}. Expected model.pt, model.safetensors, "
        "model.safetensors.index.json, or pytorch_model.bin"
    )


def save_merged_state(
    state_dict: dict[str, Tensor],
    output_dir: str | Path,
    report: MergeReport | None = None,
    timeout: float = 120.0,
) -> Path:
    """Save merged weights and optional merge report."""
    del timeout
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cleaned = strip_mtp_tensors(state_dict)
    torch.save(cleaned, out / "model.pt")
    if report is not None:
        (out / "merge_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    log.info("Saved merged model.pt to %s (%d tensors)", out, len(cleaned))
    return out


def export_merged_hf(
    state_dict: dict[str, Tensor],
    contract_dir: str | Path,
    output_dir: str | Path,
    genesis_dir: str | Path | None = None,
    meta: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Export merged weights as a HuggingFace safetensors directory."""
    from src.submit.export import export_submission

    return export_submission(
        state_dict=state_dict,
        king_dir=contract_dir,
        output_dir=output_dir,
        genesis_dir=genesis_dir or contract_dir,
        rank=0,
        meta=meta or {},
        timeout=timeout,
    )


def merge_model_paths(
    model_paths: Iterable[str | Path],
    output_dir: str | Path,
    config: MergeConfig,
    contract_dir: str | Path | None = None,
    genesis_dir: str | Path | None = None,
    export_hf: bool = False,
    device: torch.device | None = None,
    distributed: bool = False,
    gpus_per_node: int = 8,
    timeout: float = 120.0,
) -> tuple[Path, MergeReport]:
    """Load, merge, and save models from disk paths."""
    del timeout, genesis_dir
    paths = [Path(path) for path in model_paths]
    if len(paths) != 2:
        if distributed:
            raise ValueError("Distributed fast merge supports exactly two models")
        log.info("Loading %d models for %s merge", len(paths), config.strategy.value)
        states = [load_model_state(path) for path in paths]
        merged, report = merge_state_dicts(states, config)
        report.sources = [str(path.resolve()) for path in paths]
        out = save_merged_state(merged, output_dir, report=report)
        if export_hf:
            if contract_dir is None:
                contract_dir = paths[0]
            export_merged_hf(
                state_dict=merged,
                contract_dir=contract_dir,
                output_dir=output_dir,
                meta={"merge_strategy": report.strategy, "merge_sources": report.sources},
            )
        return out, report

    path_a, path_b = paths
    out = Path(output_dir)
    copy_from = Path(contract_dir) if contract_dir is not None else path_a

    if distributed and is_safetensors_model_dir(path_a):
        report = merge_safetensors_distributed(
            path_a=path_a,
            path_b=path_b,
            output_dir=out,
            config=config,
            gpus_per_node=gpus_per_node,
            copy_contracts_from=copy_from,
        )
        return out, report

    if is_safetensors_model_dir(path_a):
        report = merge_safetensors_streaming(
            path_a=path_a,
            path_b=path_b,
            output_dir=out,
            config=config,
            device=device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            copy_contracts_from=copy_from,
        )
        (out / "merge_report.json").write_text(json.dumps(report.to_dict(), indent=2))
        return out, report

    log.info("Fallback CPU dict merge for non-safetensors inputs")
    states = [load_model_state(path) for path in paths]
    merged, report = merge_state_dicts(states, config)
    report.sources = [str(path.resolve()) for path in paths]
    out = save_merged_state(merged, out, report=report)
    if export_hf and contract_dir is not None:
        export_merged_hf(
            state_dict=merged,
            contract_dir=contract_dir,
            output_dir=output_dir,
            meta={"merge_strategy": report.strategy, "merge_sources": report.sources},
        )
    return out, report
