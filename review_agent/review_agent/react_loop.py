"""ReAct execution loop over native skill actions."""

from __future__ import annotations

from typing import Any

from review_agent.models import PatchChange, ReviewRequest, SeedSymbol, SymbolImpact


class ReActLoop:
    """Budgeted skill-execution loop for semantic review."""

    def __init__(self, *, registry) -> None:
        self._registry = registry

    def run(
        self,
        *,
        request: ReviewRequest,
        client,
        patch_changes: list[PatchChange],
        changed_file_keys: list[str],
        changed_repos: list[str],
        dependency_map: dict[str, set[str]],
    ) -> dict[str, Any]:
        """Execute the planned skill sequence and return aggregate state."""
        state: dict[str, Any] = {
            "request": request,
            "client": client,
            "patch_changes": patch_changes,
            "changed_file_keys": changed_file_keys,
            "changed_repos": changed_repos,
            "dependency_map": dependency_map,
            "symbol_impacts": [],
            "tool_usage": [],
            "loop_warnings": [],
        }

        self._apply(state, self._registry.execute("skill_patch_intake", state))
        seed_symbols: list[SeedSymbol] = list(state.get("seed_symbols", []))

        rounds = 0
        total_tool_calls = len(state.get("tool_usage", []))
        for seed in seed_symbols:
            if rounds >= request.max_tool_rounds:
                state["loop_warnings"].append("max_tool_rounds_reached")
                break
            if total_tool_calls >= request.max_total_tool_calls:
                state["loop_warnings"].append("max_total_tool_calls_reached")
                break
            try:
                self._apply(
                    state,
                    self._registry.execute("skill_symbol_impact", state, symbol=seed.symbol),
                )
            except Exception as exc:
                state["loop_warnings"].append(f"skill_symbol_impact_failed:{seed.symbol}:{exc}")
            rounds += 1
            total_tool_calls = len(state.get("tool_usage", []))

        top_file_keys = _collect_top_file_keys(
            impacts=state.get("symbol_impacts", []),
            changed_file_keys=changed_file_keys,
            limit=12,
        )
        if top_file_keys:
            try:
                self._apply(
                    state,
                    self._registry.execute("skill_evidence_read", state, file_keys=top_file_keys),
                )
            except Exception as exc:
                state["loop_warnings"].append(f"skill_evidence_read_failed:{exc}")

        try:
            self._apply(state, self._registry.execute("skill_cross_repo_breakage", state))
        except Exception as exc:
            state["loop_warnings"].append(f"skill_cross_repo_breakage_failed:{exc}")
        try:
            self._apply(state, self._registry.execute("skill_architecture_risk", state))
        except Exception as exc:
            state["loop_warnings"].append(f"skill_architecture_risk_failed:{exc}")

        return state

    @staticmethod
    def _apply(state: dict[str, Any], updates: dict[str, Any] | None) -> None:
        if not updates:
            return
        state.update(updates)


def _collect_top_file_keys(
    *,
    impacts: list[SymbolImpact],
    changed_file_keys: list[str],
    limit: int,
) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()

    for fk in changed_file_keys:
        if fk and fk not in seen:
            seen.add(fk)
            ranked.append(fk)
    for impact in impacts:
        for fk in impact.parsed_file_keys + impact.candidate_file_keys:
            if fk and fk not in seen:
                seen.add(fk)
                ranked.append(fk)
            if len(ranked) >= limit:
                return ranked[:limit]
    return ranked[:limit]

