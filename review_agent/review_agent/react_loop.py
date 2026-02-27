"""Deprecated module retained for compatibility.

The procedural ReAct for-loop has been replaced by the PydanticAI agent runtime
in :mod:`review_agent.orchestrator`.
"""

from __future__ import annotations


class ReActLoop:
    """Removed in favor of LLM-driven tool calling."""

    def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - compatibility only
        _ = (args, kwargs)
        raise RuntimeError("ReActLoop is deprecated; use ReviewOrchestrator.run()")
