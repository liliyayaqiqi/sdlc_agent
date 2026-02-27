"""Symbol-impact skill: run semantic primitive sequence for one symbol."""

from __future__ import annotations

import time
from typing import Any

from review_agent.models import SymbolImpact, ToolCallRecord


def run_symbol_impact(state: dict[str, Any], *, symbol: str, **_kwargs: Any) -> dict[str, Any]:
    """Investigate one symbol with list->freshness->parse->fetch->confidence flow."""
    client = state["client"]
    request = state["request"]
    tool_usage: list[ToolCallRecord] = state.setdefault("tool_usage", [])

    def call(tool: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            tool_usage.append(
                ToolCallRecord(skill="skill_symbol_impact", tool=tool, success=True, elapsed_ms=(time.perf_counter() - t0) * 1000.0)
            )
            return result
        except Exception as exc:
            tool_usage.append(
                ToolCallRecord(
                    skill="skill_symbol_impact",
                    tool=tool,
                    success=False,
                    elapsed_ms=(time.perf_counter() - t0) * 1000.0,
                    note=str(exc),
                )
            )
            raise

    listed = call(
        "explore.list_candidates",
        client.explore_list_candidates,
        symbol=symbol,
        max_files=request.max_candidates_per_symbol,
        include_rg=True,
        entry_repos=state.get("changed_repos", []),
        max_repo_hops=2,
    )
    candidates = list(listed.get("candidates", []) or [])
    deleted = list(listed.get("deleted_file_keys", []) or [])

    freshness = call(
        "explore.classify_freshness",
        client.explore_classify_freshness,
        candidate_file_keys=candidates,
        max_files=request.max_candidates_per_symbol,
    )
    stale = list(freshness.get("stale", []) or [])
    unparsed = list(freshness.get("unparsed", []) or [])
    fresh = list(freshness.get("fresh", []) or [])
    overlay_mode = str(freshness.get("overlay_mode", listed.get("overlay_mode", "sparse")))

    parsed = call(
        "explore.parse_file",
        client.explore_parse_file,
        file_keys=stale,
        max_parse_workers=request.parse_workers,
        timeout_s=request.parse_timeout_s,
        skip_if_fresh=True,
    )
    parsed_keys = list(parsed.get("parsed_file_keys", []) or [])
    parse_failed = list(parsed.get("failed_file_keys", []) or [])

    sym_rows = call(
        "explore.fetch_symbols",
        client.explore_fetch_symbols,
        symbol=symbol,
        candidate_file_keys=candidates,
        excluded_file_keys=deleted,
        limit=request.max_fetch_limit,
    )
    ref_rows = call(
        "explore.fetch_references",
        client.explore_fetch_references,
        symbol=symbol,
        candidate_file_keys=candidates,
        excluded_file_keys=deleted,
        limit=request.max_fetch_limit,
    )
    edge_rows = call(
        "explore.fetch_call_edges",
        client.explore_fetch_call_edges,
        symbol=symbol,
        direction="both",
        candidate_file_keys=candidates,
        excluded_file_keys=deleted,
        limit=request.max_fetch_limit,
    )
    warnings = sorted(
        set(
            list(listed.get("warnings", []) or [])
            + list(freshness.get("warnings", []) or [])
            + list(parsed.get("parse_warnings", []) or [])
            + list(sym_rows.get("warnings", []) or [])
            + list(ref_rows.get("warnings", []) or [])
            + list(edge_rows.get("warnings", []) or [])
        )
    )
    confidence = call(
        "explore.get_confidence",
        client.explore_get_confidence,
        verified_files=sorted(set(fresh + parsed_keys + list(parsed.get("skipped_fresh_file_keys", []) or []))),
        stale_files=sorted(set(parse_failed)),
        unparsed_files=sorted(set(unparsed + list(parsed.get("unparsed_file_keys", []) or []))),
        warnings=warnings,
        overlay_mode=overlay_mode,
    )

    impact = SymbolImpact(
        symbol=symbol,
        candidate_file_keys=candidates,
        deleted_file_keys=deleted,
        fresh=fresh,
        stale=stale,
        unparsed=unparsed,
        parsed_file_keys=parsed_keys,
        parse_failed_file_keys=parse_failed,
        symbols=list(sym_rows.get("symbols", []) or []),
        references=list(ref_rows.get("references", []) or []),
        call_edges=list(edge_rows.get("edges", []) or []),
        confidence=dict(confidence.get("confidence", {}) or {}),
        warnings=warnings,
    )
    impacts: list[SymbolImpact] = state.setdefault("symbol_impacts", [])
    impacts.append(impact)
    return {"symbol_impacts": impacts}

