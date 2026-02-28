"""Typed LLM adapter services."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Protocol

from review_agent.models import (
    EvidenceRef,
    ExplorationResult,
    FindingCategory,
    ModelContractError,
    ReviewFactSheet,
    ReviewFinding,
    ReviewPlan,
    Severity,
    SynthesisDraft,
)
from review_agent.prompting import (
    EXPLORATION_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_exploration_prompt,
    build_planner_prompt,
    build_synthesis_prompt,
)

logger = logging.getLogger("review_agent.adapters.llm")


class PlannerService(Protocol):
    """Planner contract."""

    def plan(self, *, context, prepass, budgets: dict[str, int]) -> ReviewPlan:
        """Build a bounded plan."""


class ExplorationService(Protocol):
    """Exploration contract."""

    def explore(
        self,
        *,
        fact_sheet: ReviewFactSheet,
        prepass,
        remaining_calls: int,
        remaining_rounds: int,
        client,
        budget,
        tool_usage: list,
        analysis_context: dict[str, Any],
    ) -> ExplorationResult:
        """Perform follow-up exploration."""


class SynthesisService(Protocol):
    """Synthesis contract."""

    def synthesize(self, *, fact_sheet: ReviewFactSheet, fail_threshold: str) -> SynthesisDraft:
        """Generate findings from fact sheet."""


class PydanticAiPlannerService:
    """Planner backed by PydanticAI."""

    def __init__(self, model_name: str) -> None:
        from pydantic_ai import Agent  # type: ignore

        self._agent = Agent(model_name, system_prompt=PLANNER_SYSTEM_PROMPT, result_type=ReviewPlan)

    def plan(self, *, context, prepass, budgets: dict[str, int]) -> ReviewPlan:
        try:
            prompt = build_planner_prompt(context=context, prepass=prepass, budgets=budgets)
            return self._agent.run_sync(prompt).data
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc


class PydanticAiSynthesisService:
    """Synthesis backed by PydanticAI."""

    def __init__(self, model_name: str) -> None:
        from pydantic_ai import Agent  # type: ignore

        self._agent = Agent(model_name, system_prompt=SYNTHESIS_SYSTEM_PROMPT, result_type=SynthesisDraft)

    def synthesize(self, *, fact_sheet: ReviewFactSheet, fail_threshold: str) -> SynthesisDraft:
        try:
            prompt = build_synthesis_prompt(fact_sheet=fact_sheet, fail_threshold=fail_threshold)
            return self._agent.run_sync(prompt).data
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc


class PydanticAiExplorationService:
    """Exploration backed by PydanticAI and real tools."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

    def explore(
        self,
        *,
        fact_sheet: ReviewFactSheet,
        prepass,
        remaining_calls: int,
        remaining_rounds: int,
        client,
        budget,
        tool_usage: list,
        analysis_context: dict[str, Any],
    ) -> ExplorationResult:
        prompt = build_exploration_prompt(
            fact_sheet=fact_sheet,
            prepass=prepass,
            remaining_calls=remaining_calls,
            remaining_rounds=remaining_rounds,
        )
        try:
            agent, deps = _build_exploration_agent(
                self._model_name,
                client=client,
                budget=budget,
                tool_usage=tool_usage,
                analysis_context=analysis_context,
            )
            result = agent.run_sync(prompt, deps=deps)
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc
        data = result.data
        if isinstance(data, ExplorationResult):
            return data
        if isinstance(data, str):
            return ExplorationResult(summary=data)
        raise ModelContractError(f"unexpected exploration result type: {type(data).__name__}")


@dataclass
class _FixtureProfile:
    name: str


class FixturePlannerService:
    """Deterministic planner for tests and local validation."""

    def __init__(self, profile: _FixtureProfile) -> None:
        self._profile = profile

    def plan(self, *, context, prepass, budgets: dict[str, int]) -> ReviewPlan:
        prioritized = [seed.symbol for seed in prepass.seed_symbols[: budgets.get("max_symbols", 24)]]
        if self._profile.name == "empty":
            prioritized = []
        return ReviewPlan(
            prioritized_symbols=prioritized,
            require_merge_preview=bool(context.merge_preview_sha),
            notes=[f"fixture_profile:{self._profile.name}"],
        )


class FixtureExplorationService:
    """Deterministic exploration for tests and local validation."""

    def __init__(self, profile: _FixtureProfile) -> None:
        self._profile = profile

    def explore(
        self,
        *,
        fact_sheet: ReviewFactSheet,
        prepass,
        remaining_calls: int,
        remaining_rounds: int,
        client,
        budget,
        tool_usage: list,
        analysis_context: dict[str, Any],
    ) -> ExplorationResult:
        if self._profile.name == "empty":
            return ExplorationResult()
        summary = f"fixture_exploration:{self._profile.name}:calls={remaining_calls}:rounds={remaining_rounds}"
        return ExplorationResult(summary=summary)


class FixtureSynthesisService:
    """Deterministic synthesis for tests and local validation."""

    def __init__(self, profile: _FixtureProfile) -> None:
        self._profile = profile

    def synthesize(self, *, fact_sheet: ReviewFactSheet, fail_threshold: str) -> SynthesisDraft:
        notes = [f"fixture_profile:{self._profile.name}"]
        if self._profile.name in {"empty", "no-findings"}:
            return SynthesisDraft(summary="Fixture synthesis completed with no findings.", global_notes=notes)

        evidence = list(fact_sheet.evidence_anchors[:1])
        if not evidence and fact_sheet.changed_files:
            evidence = [
                EvidenceRef(
                    tool="fixture",
                    description="fallback_changed_file",
                    file_key=fact_sheet.changed_files[0],
                    line=1,
                )
            ]
        severity = Severity.HIGH if self._profile.name in {"blocking", "gitlab-inline"} else Severity.MEDIUM
        title = "Fixture review finding"
        if self._profile.name == "cross-repo":
            category = FindingCategory.CROSS_REPO_BREAKAGE
            title = "Cross-repo contract drift"
        else:
            category = FindingCategory.ARCHITECTURE_RISK
        finding = ReviewFinding(
            id=f"fixture-{self._profile.name}",
            severity=severity,
            category=category,
            title=title,
            impact="Deterministic fixture finding for end-to-end validation.",
            recommendation="Inspect the mapped location and evidence payload.",
            evidence=evidence,
            confidence=0.85,
            related_symbols=[evidence[0].symbol] if evidence and evidence[0].symbol else [],
        )
        summary = f"Fixture synthesis completed with {1 if finding else 0} finding."
        return SynthesisDraft(summary=summary, findings=[finding], global_notes=notes)


def build_model_services(model_name: str) -> tuple[PlannerService, ExplorationService, SynthesisService]:
    """Build typed services for the configured model provider."""
    normalized = (model_name or "").strip() or "openai:gpt-4o"
    provider, _, remainder = normalized.partition(":")
    if provider == "fixture":
        profile = _FixtureProfile(remainder or "default")
        return (
            FixturePlannerService(profile),
            FixtureExplorationService(profile),
            FixtureSynthesisService(profile),
        )
    return (
        PydanticAiPlannerService(normalized),
        PydanticAiExplorationService(normalized),
        PydanticAiSynthesisService(normalized),
    )


def _build_exploration_agent(
    model_name: str,
    *,
    client,
    budget,
    tool_usage: list,
    analysis_context: dict[str, Any],
):
    """Build a PydanticAI exploration agent with bounded tools."""
    from pydantic_ai import Agent, RunContext  # type: ignore

    @dataclass
    class ExploreDeps:
        client: Any
        budget: Any
        tool_usage: list
        analysis_context: dict[str, Any]

    agent = Agent(
        model_name,
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        deps_type=ExploreDeps,
        result_type=ExplorationResult,
    )

    def _record(deps: ExploreDeps, tool_name: str, success: bool, elapsed_ms: float, note: str = "") -> None:
        from review_agent.models import ToolCallRecord

        deps.tool_usage.append(
            ToolCallRecord(skill="explorer", tool=tool_name, success=success, elapsed_ms=elapsed_ms, note=note)
        )

    def _timed_call(deps: ExploreDeps, tool_name: str, fn, **kwargs: Any) -> dict[str, Any]:
        from time import perf_counter

        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted"}
        t0 = perf_counter()
        try:
            result = fn(**kwargs)
            _record(deps, tool_name, True, (perf_counter() - t0) * 1000)
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            _record(deps, tool_name, False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc)}

    @agent.tool  # type: ignore[misc]
    def explore_list_candidates(ctx: RunContext[ExploreDeps], symbol: str, max_files: int = 50) -> dict[str, Any]:
        deps = ctx.deps
        capped = min(max_files, deps.budget.max_candidates_per_symbol)
        return _timed_call(
            deps,
            "explore.list_candidates",
            deps.client.explore_list_candidates,
            symbol=symbol,
            max_files=capped,
            include_rg=True,
            analysis_context=deps.analysis_context,
        )

    @agent.tool  # type: ignore[misc]
    def explore_fetch_symbols(ctx: RunContext[ExploreDeps], symbol: str, candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        deps = ctx.deps
        return _timed_call(
            deps,
            "explore.fetch_symbols",
            deps.client.explore_fetch_symbols,
            symbol=symbol,
            candidate_file_keys=candidate_file_keys or [],
            excluded_file_keys=[],
            limit=deps.budget.max_fetch_limit,
            analysis_context=deps.analysis_context,
        )

    @agent.tool  # type: ignore[misc]
    def explore_fetch_references(ctx: RunContext[ExploreDeps], symbol: str, candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        deps = ctx.deps
        return _timed_call(
            deps,
            "explore.fetch_references",
            deps.client.explore_fetch_references,
            symbol=symbol,
            candidate_file_keys=candidate_file_keys or [],
            excluded_file_keys=[],
            limit=deps.budget.max_fetch_limit,
            analysis_context=deps.analysis_context,
        )

    @agent.tool  # type: ignore[misc]
    def explore_fetch_call_edges(ctx: RunContext[ExploreDeps], symbol: str, direction: str = "both", candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        deps = ctx.deps
        return _timed_call(
            deps,
            "explore.fetch_call_edges",
            deps.client.explore_fetch_call_edges,
            symbol=symbol,
            direction=direction,
            candidate_file_keys=candidate_file_keys or [],
            excluded_file_keys=[],
            limit=deps.budget.max_fetch_limit,
            analysis_context=deps.analysis_context,
        )

    @agent.tool  # type: ignore[misc]
    def explore_read_file(ctx: RunContext[ExploreDeps], file_key: str, start_line: int = 1, end_line: int = 120) -> dict[str, Any]:
        deps = ctx.deps
        return _timed_call(
            deps,
            "explore.read_file",
            deps.client.explore_read_file,
            file_key=file_key,
            start_line=start_line,
            end_line=end_line,
            max_bytes=32_000,
        )

    @agent.tool  # type: ignore[misc]
    def explore_rg_search(ctx: RunContext[ExploreDeps], query: str, mode: str = "symbol", max_hits: int = 50) -> dict[str, Any]:
        deps = ctx.deps
        return _timed_call(
            deps,
            "explore.rg_search",
            deps.client.explore_rg_search,
            query=query,
            mode=mode,
            max_hits=min(max_hits, 200),
            max_files=min(50, deps.budget.max_candidates_per_symbol),
            timeout_s=min(deps.budget.parse_timeout_s, 30),
            context_lines=1,
            analysis_context=deps.analysis_context,
        )

    deps = ExploreDeps(
        client=client,
        budget=budget,
        tool_usage=tool_usage,
        analysis_context=analysis_context,
    )
    return agent, deps
