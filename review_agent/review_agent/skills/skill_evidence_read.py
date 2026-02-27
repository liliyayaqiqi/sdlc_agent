"""Evidence-read skill: retrieve file snippets and compile command context."""

from __future__ import annotations

import time
from typing import Any

from review_agent.models import ToolCallRecord


def run_evidence_read(state: dict[str, Any], *, file_keys: list[str], **_kwargs: Any) -> dict[str, Any]:
    """Fetch source snippets and compile metadata for selected files."""
    client = state["client"]
    tool_usage: list[ToolCallRecord] = state.setdefault("tool_usage", [])
    evidence: dict[str, dict[str, Any]] = state.setdefault("file_evidence", {})

    for file_key in file_keys:
        if not file_key or file_key in evidence:
            continue
        t0 = time.perf_counter()
        try:
            read_resp = client.explore_read_file(file_key=file_key, start_line=1, end_line=220)
            tool_usage.append(
                ToolCallRecord(skill="skill_evidence_read", tool="explore.read_file", success=True, elapsed_ms=(time.perf_counter() - t0) * 1000.0)
            )
        except Exception as exc:
            tool_usage.append(
                ToolCallRecord(
                    skill="skill_evidence_read",
                    tool="explore.read_file",
                    success=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    note=str(exc),
                )
            )
            continue

        t1 = time.perf_counter()
        try:
            compile_resp = client.explore_get_compile_command(file_key=file_key)
            tool_usage.append(
                ToolCallRecord(
                    skill="skill_evidence_read",
                    tool="explore.get_compile_command",
                    success=True,
                    elapsed_ms=(time.perf_counter() - t1) * 1000.0,
                )
            )
        except Exception as exc:
            compile_resp = {"warnings": [str(exc)], "match_type": "missing"}
            tool_usage.append(
                ToolCallRecord(
                    skill="skill_evidence_read",
                    tool="explore.get_compile_command",
                    success=False,
                    elapsed_ms=(time.perf_counter() - t1) * 1000.0,
                    note=str(exc),
                )
            )

        evidence[file_key] = {"read_file": read_resp, "compile_command": compile_resp}

    return {"file_evidence": evidence}

