"""Training entry package: ``python -m src.train``."""

from src.train.cli import _validate_packing_mode, main

__all__ = ["_validate_packing_mode", "main"]
