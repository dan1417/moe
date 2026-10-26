"""Pinned access to the live Teutonic validator code.

The validator (https://github.com/unarbos/teutonic) is installed into the eval venv
at ``UPSTREAM_COMMIT`` (see scripts/setup/setup_eval_venv.sh). Import its modules
through this file so cache/env configuration happens before the validator reads its
environment at import time.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import urllib.request
from pathlib import Path
from types import ModuleType

import torch

UPSTREAM_REPO = "unarbos/teutonic"
UPSTREAM_COMMIT = "5118c326a10fde4bdf485e1963c0f09f9c759645"
# Commit of the code that produces a loss (engine.compute_per_sequence_loss, masking, document_masking and our port in
# src/parity/pipeline.py). Loss caches are keyed by it, not by UPSTREAM_COMMIT: 14a9bb2 -> 5118c32 only changed how
# shards are fetched (HTTP range reads) and cache locking; the sampled tokens and every scoring file are byte-identical,
# so cached king/candidate losses on unchanged samples stay valid. Bump it only when a pin move changes a loss.
SCORING_CODE_COMMIT = "14a9bb2cd8f28b645fa6abb146fe00a1593e1c96"
TRANSFORMERS_FORK = "tony-dendrite/transformers@9e8a1817d44502602b5b50e09b2a5ab0271bdb39"

# Files whose content can change a verdict. `upstream_drift()` compares the installed
# copies against the validator's main branch so protocol changes are noticed early.
SCORING_FILES = (
    "chain.toml",
    "teutonic/evaluation/policy.py",
    "teutonic/evaluation/categories.py",
    "teutonic/evaluation/dataset_categories.json",
    "teutonic/evaluation/configuration.py",
    "teutonic/evaluation/early_stopping.py",
    "teutonic/evaluation/protocol_v2.py",
    "teutonic/evaluation/masking.py",
    "teutonic/evaluation/long_documents.py",
    "teutonic/evaluation/index_manifest.py",
    "teutonic/evaluator/engine.py",
    "teutonic/evaluator/sources.py",
    "teutonic/evaluator/document_masking.py",
    "teutonic/evaluator/document_index.py",
    "teutonic/evaluator/long_documents.py",
    "teutonic/evaluator/token_shards.py",
    "teutonic/evaluator/range_http.py",
    "teutonic/validator/runtime.py",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = Path(os.environ.get("TEUTONIC_EVAL_CACHE", REPO_ROOT / "eval_cache"))


def _configure_environment(cache_root: Path) -> None:
    shard_dir = cache_root / "shards"
    os.environ.setdefault("TEUTONIC_PRETOKENIZED_CACHE_DIR", str(shard_dir))
    os.environ.setdefault("TEUTONIC_SHARD_CACHE_DIR", str(shard_dir))
    # The engine module creates nothing at import, but keep its defaults out of /tmp.
    os.environ.setdefault("TEUTONIC_MODEL_CACHE_DIR", str(cache_root / "validator_model_cache"))
    os.environ.setdefault("TEUTONIC_EVAL_RECORD_DIR", str(cache_root / "validator_records"))
    os.environ.setdefault("TEUTONIC_KERNEL_CACHE_DIR", str(cache_root / "kernel_cache"))


def _install_foreach_mm_shim() -> None:
    """Provide ``torch._foreach_mm`` when this PyTorch build lacks it.

    The validator's grouped MoE calls ``torch._foreach_mm(inputs, weights)``; its
    semantics are one independent matmul per list entry. Only used when comparing
    against the validator's own grouped MoE; scoring uses :mod:`src.parity.moe`.
    """
    if hasattr(torch, "_foreach_mm"):
        return

    def _foreach_mm(inputs, weights):
        return [torch.mm(a, b) for a, b in zip(inputs, weights, strict=True)]

    torch._foreach_mm = _foreach_mm  # type: ignore[attr-defined]


_configure_environment(DEFAULT_CACHE_ROOT)
_install_foreach_mm_shim()


def module(name: str) -> ModuleType:
    """Import a validator module, e.g. ``module("teutonic.evaluator.engine")``."""
    return importlib.import_module(name)


engine = module("teutonic.evaluator.engine")
sources = module("teutonic.evaluator.sources")
policy = module("teutonic.evaluation.policy")
categories = module("teutonic.evaluation.categories")
configuration = module("teutonic.evaluation.configuration")
early_stopping = module("teutonic.evaluation.early_stopping")
masking = module("teutonic.evaluation.masking")
long_documents = module("teutonic.evaluation.long_documents")
document_masking = module("teutonic.evaluator.document_masking")
chain_config = module("chain_config")


def installed_file(relative: str) -> Path:
    if relative == "chain.toml":
        return Path(chain_config.CONFIG_PATH)
    top = Path(module("teutonic").__file__).resolve().parent.parent
    return top / relative


def upstream_drift(ref: str = "main", timeout: float = 30.0) -> dict[str, str]:
    """Return {file: status} for scoring files that differ from ``ref`` upstream."""
    changed: dict[str, str] = {}
    for relative in SCORING_FILES:
        url = f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/{ref}/{relative}"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                remote = response.read()
        except Exception as exc:  # network errors are reported, not fatal
            changed[relative] = f"fetch_failed: {exc}"
            continue
        local_path = installed_file(relative)
        local = local_path.read_bytes() if local_path.exists() else b""
        if hashlib.sha256(local).digest() != hashlib.sha256(remote).digest():
            changed[relative] = "differs"
    return changed
