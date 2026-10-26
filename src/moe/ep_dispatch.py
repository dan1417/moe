"""Expert-parallel dispatcher (disabled).

DEPRECATED / UNSAFE: Training must not use this module.
``moe.dispatch_overlap`` must remain false. The previous implementation was
not truly async, combined tokens incorrectly across ranks, and used Python
per-token loops that would slow training. Prefer native MiMo MoE + FSDP.
"""

from __future__ import annotations

from typing import Any

_EP_DISABLED_MSG = (
    "ExpertDispatcher is disabled: the custom EP path is incorrect for MiMo MoE. "
    "Keep moe.dispatch_overlap=false and use native MoE + FSDP."
)


class ExpertDispatcher:
    """Disabled placeholder — raises on construction."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(_EP_DISABLED_MSG)
