"""Optional MCP transport adapter placeholder.

The v1 implementation uses direct HTTP skill executors.
This module exists to allow drop-in MCP transport later without changing skill contracts.
"""

from __future__ import annotations

from typing import Any


class MCPPassthroughAdapter:
    """Stub MCP adapter for future transport migration."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def call(self, _tool_name: str, _args: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("MCP adapter is not enabled in v1")

