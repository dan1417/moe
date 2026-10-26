"""Submission export utilities for Teutonic SN3 challengers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.model import load_state_dict_strict_except_mtp

log = logging.getLogger(__name__)

CHECKPOINT_PATTERN = re.compile(r"step_(\d+)(?:_outer)?$")
CONTRACT_FILENAMES = (
    "config.json",
    "modeling_mimo_v2.py",
    "configuration_mimo_v2.py",
    "chat_template.jinja.txt",
)
CONTRACT_GLOBS = ("tokenizer*",)
MTP_KEY_SUBSTRING = "mtp"


def sha256_file(path: Path, timeout: float = 60.0) -> str:
    """Compute SHA256 of a file."""
    del timeout
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def strip_mtp_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove MTP tensors from a state dict and assert none remain."""
    filtered = {key: value for key, value in state_dict.items() if MTP_KEY_SUBSTRING not in key.lower()}
    removed = len(state_dict) - len(filtered)
    if removed:
        log.info("Stripped %d MTP tensors from state dict", removed)
    remaining = [key for key in filtered if MTP_KEY_SUBSTRING in key.lower()]
    if remaining:
        raise RuntimeError(f"MTP keys remain after stripping: {remaining[:5]}")
    return filtered


def _contract_files(king_dir: Path) -> list[Path]:
    files: list[Path] = []
    for name in CONTRACT_FILENAMES:
        path = king_dir / name
        if path.exists():
            files.append(path)
    for pattern in CONTRACT_GLOBS:
        files.extend(sorted(king_dir.glob(pattern)))
    return files


def verify_contract_files(
    king_dir: str | Path,
    genesis_dir: str | Path,
    timeout: float = 60.0,
) -> dict[str, str]:
    """Verify contract files in king_dir are byte-identical to genesis."""
    king_path = Path(king_dir)
    genesis_path = Path(genesis_dir)
    if not genesis_path.exists():
        raise FileNotFoundError(f"Genesis directory not found: {genesis_path}")

    results: dict[str, str] = {}
    for path in _contract_files(king_path):
        rel = path.name
        genesis_file = genesis_path / rel
        if not genesis_file.exists():
            raise FileNotFoundError(f"Genesis contract file missing: {genesis_file}")
        king_hash = sha256_file(path, timeout=timeout)
        genesis_hash = sha256_file(genesis_file, timeout=timeout)
        if king_hash != genesis_hash:
            raise RuntimeError(f"Contract file hash mismatch for {rel}: king={king_hash} genesis={genesis_hash}")
        results[rel] = king_hash
    return results


def copy_contract_files(king_dir: str | Path, output_dir: str | Path, timeout: float = 60.0) -> None:
    """Copy contract files byte-identical from king_dir to output_dir."""
    del timeout
    king_path = Path(king_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for path in _contract_files(king_path):
        shutil.copy2(path, out_path / path.name)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Return the latest committed training checkpoint directory."""
    from src.checkpoint.dcp import is_committed_checkpoint

    root = Path(output_dir)
    if not root.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = CHECKPOINT_PATTERN.match(path.name)
        if match and not path.name.endswith("_outer"):
            if is_committed_checkpoint(path) or (path / "model.pt").exists():
                candidates.append((int(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_checkpoint(
    output_dir: str | Path,
    checkpoint: str | Path | None,
) -> Path | None:
    """Resolve explicit or latest checkpoint path."""
    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    return find_latest_checkpoint(output_dir)


def load_checkpoint_state(checkpoint_dir: Path, timeout: float = 60.0) -> dict[str, torch.Tensor]:
    """Load model state dict from a checkpoint directory."""
    del timeout
    model_path = checkpoint_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in checkpoint: {checkpoint_dir}")
    return torch.load(model_path, map_location="cpu", weights_only=True)


def load_checkpoint_meta(checkpoint_dir: Path, timeout: float = 60.0) -> dict[str, Any]:
    """Load meta.json from a checkpoint directory."""
    del timeout
    meta_path = checkpoint_dir / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def export_submission(
    state_dict: dict[str, torch.Tensor],
    king_dir: str | Path,
    output_dir: str | Path,
    genesis_dir: str | Path,
    rank: int,
    meta: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Path:
    """Export a validator-ready submission directory."""
    if rank != 0:
        raise RuntimeError("export_submission must be called on rank 0 only")

    king_path = Path(king_dir)
    out_path = Path(output_dir)
    genesis_path = Path(genesis_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    verify_contract_files(king_path, genesis_path, timeout=timeout)
    cleaned = strip_mtp_tensors(state_dict)

    reference = AutoModelForCausalLM.from_pretrained(
        king_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    load_state_dict_strict_except_mtp(reference, cleaned)
    reference.save_pretrained(out_path, safe_serialization=True, max_shard_size="4GB")

    copy_contract_files(king_path, out_path, timeout=timeout)
    manifest_path = out_path / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()

    submission_info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": meta.get("step") if meta else None,
        "lm_loss": meta.get("lm_loss") if meta else None,
        "phase": meta.get("phase") if meta else None,
        "king_dir": str(king_path.resolve()),
        "checkpoint": meta.get("checkpoint") if meta else None,
        "n_tensors": len(cleaned),
    }
    (out_path / "submission_info.json").write_text(json.dumps(submission_info, indent=2))
    log.info("Exported submission to %s (%d tensors)", out_path, len(cleaned))
    return out_path
