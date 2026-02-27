"""Cross-repo breakage skill: derive impact signals from semantic evidence."""

from __future__ import annotations

from typing import Any


def run_cross_repo_breakage(state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    """Generate cross-repo signal rows used by final analyzers."""
    changed_repos = set(state.get("changed_repos", []))
    impacts = state.get("symbol_impacts", [])
    signals: list[dict[str, Any]] = []

    for impact in impacts:
        symbol = impact.symbol
        ref_repos: set[str] = set()
        edge_repos: set[str] = set()
        for ref in impact.references:
            fk = str(ref.get("file_key", ""))
            if ":" in fk:
                ref_repos.add(fk.split(":", 1)[0])
        for edge in impact.call_edges:
            fk = str(edge.get("file_key", ""))
            if ":" in fk:
                edge_repos.add(fk.split(":", 1)[0])

        all_repos = sorted(ref_repos | edge_repos | set(impact.repos_involved))
        external_repos = sorted(set(all_repos) - changed_repos)
        signals.append(
            {
                "symbol": symbol,
                "all_repos": all_repos,
                "external_repos": external_repos,
                "incoming_ref_count": len(impact.references),
                "incoming_edge_count": len(impact.call_edges),
                "parse_failed_count": len(impact.parse_failed_file_keys),
                "unparsed_count": len(impact.unparsed),
            }
        )
    return {"cross_repo_signals": signals}

