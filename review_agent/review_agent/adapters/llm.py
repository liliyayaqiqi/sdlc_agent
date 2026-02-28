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
    ReviewRequest,
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


@dataclass(frozen=True)
class LlmEndpoint:
    """Resolved provider configuration for one review run."""

    provider: str
    model_name: str
    base_url: str = ""
    api_key: str = ""
    app_url: str = ""
    app_title: str = ""


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

    def __init__(self, endpoint: LlmEndpoint) -> None:
        from pydantic_ai import Agent  # type: ignore

        self._agent = Agent(_build_agent_model(endpoint), system_prompt=PLANNER_SYSTEM_PROMPT, output_type=ReviewPlan)

    def plan(self, *, context, prepass, budgets: dict[str, int]) -> ReviewPlan:
        try:
            prompt = build_planner_prompt(context=context, prepass=prepass, budgets=budgets)
            return _result_output(self._agent.run_sync(prompt), ReviewPlan)
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc


class PydanticAiSynthesisService:
    """Synthesis backed by PydanticAI."""

    def __init__(self, endpoint: LlmEndpoint) -> None:
        from pydantic_ai import Agent  # type: ignore

        self._agent = Agent(_build_agent_model(endpoint), system_prompt=SYNTHESIS_SYSTEM_PROMPT, output_type=SynthesisDraft)

    def synthesize(self, *, fact_sheet: ReviewFactSheet, fail_threshold: str) -> SynthesisDraft:
        try:
            prompt = build_synthesis_prompt(fact_sheet=fact_sheet, fail_threshold=fail_threshold)
            return _result_output(self._agent.run_sync(prompt), SynthesisDraft)
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc


class PydanticAiExplorationService:
    """Exploration backed by PydanticAI and real tools."""

    def __init__(self, endpoint: LlmEndpoint) -> None:
        self._endpoint = endpoint

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
                self._endpoint,
                client=client,
                budget=budget,
                tool_usage=tool_usage,
                analysis_context=analysis_context,
            )
            result = agent.run_sync(prompt, deps=deps)
        except Exception as exc:
            raise ModelContractError(str(exc)) from exc
        data = _result_output(result, ExplorationResult | str)
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


def build_model_services(
    request_or_model: ReviewRequest | str,
    *,
    base_url: str = "",
    api_key: str = "",
    app_url: str = "",
    app_title: str = "",
) -> tuple[PlannerService, ExplorationService, SynthesisService]:
    """Build typed services for the configured model provider."""
    endpoint = resolve_llm_endpoint(
        request_or_model,
        base_url=base_url,
        api_key=api_key,
        app_url=app_url,
        app_title=app_title,
    )
    provider = endpoint.provider
    if provider == "fixture":
        profile = _FixtureProfile(endpoint.model_name or "default")
        return (
            FixturePlannerService(profile),
            FixtureExplorationService(profile),
            FixtureSynthesisService(profile),
        )
    return (
        PydanticAiPlannerService(endpoint),
        PydanticAiExplorationService(endpoint),
        PydanticAiSynthesisService(endpoint),
    )


def _build_exploration_agent(
    endpoint: LlmEndpoint,
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
        _build_agent_model(endpoint),
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        deps_type=ExploreDeps,
        output_type=ExplorationResult,
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


def resolve_llm_endpoint(
    request_or_model: ReviewRequest | str,
    *,
    base_url: str = "",
    api_key: str = "",
    app_url: str = "",
    app_title: str = "",
) -> LlmEndpoint:
    """Normalize request/config input into one provider endpoint."""
    if isinstance(request_or_model, ReviewRequest):
        raw_model = request_or_model.llm_model
        base_url = request_or_model.llm_base_url
        api_key = request_or_model.llm_api_key
        app_url = request_or_model.llm_app_url
        app_title = request_or_model.llm_app_title
    else:
        raw_model = request_or_model
    normalized = (raw_model or "").strip() or "openai:gpt-4o"
    if ":" not in normalized:
        raise ValueError("llm_model must be in '<provider>:<model>' format")
    provider, _, model_name = normalized.partition(":")
    normalized_base_url = (base_url or "").strip()
    if provider in {"gateway", "openai-compatible"} and not normalized_base_url:
        raise ValueError("llm_base_url is required for gateway/openai-compatible llm providers")
    return LlmEndpoint(
        provider=(provider or "openai").strip(),
        model_name=model_name.strip() or "gpt-4o",
        base_url=normalized_base_url,
        api_key=(api_key or "").strip(),
        app_url=(app_url or "").strip(),
        app_title=(app_title or "").strip(),
    )


def endpoint_cache_key(request_or_model: ReviewRequest | str, **kwargs: str) -> str:
    """Stable non-secret cache key fragment for provider selection."""
    endpoint = resolve_llm_endpoint(request_or_model, **kwargs)
    return "|".join(
        [
            endpoint.provider,
            endpoint.model_name,
            endpoint.base_url,
            endpoint.app_url,
            endpoint.app_title,
        ]
    )


def _build_agent_model(endpoint: LlmEndpoint):
    """Construct a provider-specific PydanticAI model or model id."""
    if endpoint.provider in {"gateway", "openai-compatible"}:
        from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
        from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore

        provider = OpenAIProvider(
            base_url=endpoint.base_url or None,
            api_key=endpoint.api_key or None,
        )
        return OpenAIChatModel(endpoint.model_name, provider=provider)

    if endpoint.provider == "openrouter":
        from pydantic_ai.models.openrouter import OpenRouterModel  # type: ignore
        from pydantic_ai.providers.openrouter import OpenRouterProvider  # type: ignore

        provider = OpenRouterProvider(
            api_key=endpoint.api_key or None,
            app_url=endpoint.app_url or None,
            app_title=endpoint.app_title or None,
        )
        return OpenRouterModel(endpoint.model_name, provider=provider)

    if endpoint.provider == "openai" and (endpoint.base_url or endpoint.api_key):
        from pydantic_ai.models.openai import OpenAIChatModel  # type: ignore
        from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore

        provider = OpenAIProvider(
            base_url=endpoint.base_url or None,
            api_key=endpoint.api_key or None,
        )
        return OpenAIChatModel(endpoint.model_name, provider=provider)

    return f"{endpoint.provider}:{endpoint.model_name}"


def _result_output(result: Any, expected_type: Any) -> Any:
    """Extract run output across pydantic-ai result API versions."""
    output = getattr(result, "output", getattr(result, "data", None))
    if output is None:
        raise ModelContractError("model returned no output payload")
    return output
