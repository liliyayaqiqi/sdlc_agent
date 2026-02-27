"""Architecture-risk skill using manifest dependency constraints."""

from __future__ import annotations

from collections import deque
from typing import Any


def run_architecture_risk(state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    """Compute dependency-direction violations from symbol evidence."""
    dep_map: dict[str, set[str]] = state.get("dependency_map", {})
    impacts = state.get("symbol_impacts", [])
    violations: list[dict[str, Any]] = []

    for impact in impacts:
        owner_repos = _repos_from_rows(impact.symbols)
        caller_repos = _repos_from_rows(impact.references) | _repos_from_rows(impact.call_edges)
        if not owner_repos:
            owner_repos = set(impact.repos_involved)
        for caller in sorted(caller_repos):
            for owner in sorted(owner_repos):
                if caller == owner:
                    continue
                reachable = _reachable_from(caller, dep_map)
                if owner not in reachable:
                    violations.append(
                        {
                            "symbol": impact.symbol,
                            "caller_repo": caller,
                            "owner_repo": owner,
                            "reason": "dependency_direction_violation",
                        }
                    )

    return {"architecture_signals": violations}


def _repos_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for row in rows:
        fk = str(row.get("file_key", ""))
        if ":" in fk:
            out.add(fk.split(":", 1)[0])
    return out


def _reachable_from(start: str, dep_map: dict[str, set[str]]) -> set[str]:
    if start not in dep_map:
        return set()
    seen: set[str] = set()
    queue: deque[str] = deque([start])
    while queue:
        repo = queue.popleft()
        if repo in seen:
            continue
        seen.add(repo)
        for nxt in dep_map.get(repo, set()):
            if nxt not in seen:
                queue.append(nxt)
    return seen

