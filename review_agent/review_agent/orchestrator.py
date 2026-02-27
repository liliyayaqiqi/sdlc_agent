"""Context-driven orchestration for semantic PR review."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
import logging
import os
from pathlib import Path
from typing import Any, Callable
from time import perf_counter, sleep

from review_agent.context_ingestion import IngestedReviewContext, ReviewContextIngestor
from review_agent.models import (
    CoverageSummary,
    EvidenceRef,
    FindingCategory,
    ReviewDecision,
    ReviewFactSheet,
    ReviewFinding,
    ReviewPlan,
    ReviewReport,
    ReviewRequest,
    SEVERITY_RANK,
    Severity,
    SymbolFact,
    SymbolImpact,
    TestImpact,
    ToolCallRecord,
    ViewContextMaterialization,
)
from review_agent.patch_parser import build_prepass_result, parse_unified_diff
from review_agent.prompting import (
    EXPLORATION_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_exploration_prompt,
    build_planner_prompt,
    build_synthesis_prompt,
)
from review_agent.report_renderer import render_markdown
from review_agent.review_cache import ReviewTraceCache
from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient

logger = logging.getLogger("review_agent.orchestrator")


# ---------------------------------------------------------------------------
# Timeout helper
# ---------------------------------------------------------------------------

class _ReviewTimeout(RuntimeError):
    """Raised when the wall-clock review timeout is exceeded."""


def _check_deadline(deadline: float, step_name: str) -> None:
    """Raise *_ReviewTimeout* if the wall-clock deadline has passed."""
    if deadline > 0 and perf_counter() > deadline:
        raise _ReviewTimeout(f"review wall-clock timeout exceeded during {step_name}")


# ---------------------------------------------------------------------------
# Budget tracking
# ---------------------------------------------------------------------------

@dataclass
class BudgetTracker:
    """Mutable budget counters shared across the entire review run.

    *max_symbol_slots* -- maximum number of top-level symbol investigations
    (one slot is consumed per symbol in the evidence loop).

    *max_total_tool_calls* -- absolute cap on individual backend tool calls.
    """

    max_symbol_slots: int
    max_total_tool_calls: int
    max_symbols: int
    max_candidates_per_symbol: int
    max_fetch_limit: int
    parse_timeout_s: int
    parse_workers: int

    slots_used: int = 0
    calls_used: int = 0

    @property
    def remaining_slots(self) -> int:
        return max(0, self.max_symbol_slots - self.slots_used)

    @property
    def remaining_calls(self) -> int:
        return max(0, self.max_total_tool_calls - self.calls_used)

    @property
    def exhausted(self) -> bool:
        return self.remaining_calls <= 0

    def consume_call(self) -> bool:
        """Try to consume one call.  Returns False if budget exhausted."""
        if self.calls_used >= self.max_total_tool_calls:
            return False
        self.calls_used += 1
        return True

    def consume_slot(self) -> bool:
        """Try to consume one symbol slot.  Returns False if exhausted."""
        if self.slots_used >= self.max_symbol_slots:
            return False
        self.slots_used += 1
        return True


# ---------------------------------------------------------------------------
# View context container
# ---------------------------------------------------------------------------

@dataclass
class ViewContexts:
    baseline: dict[str, Any]
    head: dict[str, Any]
    merge_preview: dict[str, Any] | None
    status: ViewContextMaterialization


# ---------------------------------------------------------------------------
# Agent factories
# ---------------------------------------------------------------------------

def build_planner_agent(model_name: str):
    from pydantic_ai import Agent  # type: ignore

    return Agent(model_name, system_prompt=PLANNER_SYSTEM_PROMPT, result_type=ReviewPlan)


def build_synthesis_agent(model_name: str):
    from pydantic_ai import Agent  # type: ignore

    return Agent(model_name, system_prompt=SYNTHESIS_SYSTEM_PROMPT, result_type=ReviewReport)


def build_exploration_agent(
    model_name: str,
    *,
    client: CxxtractHttpClient,
    budget: BudgetTracker,
    tool_usage: list[ToolCallRecord],
    analysis_context: dict[str, Any],
):
    """Build a PydanticAI agent with the 6 explore tools registered."""
    from pydantic_ai import Agent, RunContext  # type: ignore

    @dataclass
    class ExploreDeps:
        client: CxxtractHttpClient
        budget: BudgetTracker
        tool_usage: list[ToolCallRecord]
        analysis_context: dict[str, Any]

    agent = Agent(
        model_name,
        system_prompt=EXPLORATION_SYSTEM_PROMPT,
        deps_type=ExploreDeps,
        result_type=str,
    )

    def _record(deps: ExploreDeps, tool_name: str, success: bool, elapsed_ms: float, note: str = ""):
        deps.tool_usage.append(
            ToolCallRecord(skill="explorer", tool=tool_name, success=success, elapsed_ms=elapsed_ms, note=note)
        )

    @agent.tool  # type: ignore[misc]
    def explore_list_candidates(ctx: RunContext[ExploreDeps], symbol: str, max_files: int = 50) -> dict[str, Any]:
        """Find candidate files that may contain a symbol."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "candidates": []}
        t0 = perf_counter()
        try:
            capped = min(max_files, deps.budget.max_candidates_per_symbol)
            result = deps.client.explore_list_candidates(
                symbol=symbol, max_files=capped, include_rg=True,
                analysis_context=deps.analysis_context,
            )
            _record(deps, "explore.list_candidates", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.list_candidates", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "candidates": []}

    @agent.tool  # type: ignore[misc]
    def explore_fetch_symbols(ctx: RunContext[ExploreDeps], symbol: str, candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        """Look up symbol definitions across candidate files."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "symbols": []}
        t0 = perf_counter()
        try:
            result = deps.client.explore_fetch_symbols(
                symbol=symbol,
                candidate_file_keys=candidate_file_keys or [],
                excluded_file_keys=[],
                limit=deps.budget.max_fetch_limit,
                analysis_context=deps.analysis_context,
            )
            _record(deps, "explore.fetch_symbols", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.fetch_symbols", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "symbols": []}

    @agent.tool  # type: ignore[misc]
    def explore_fetch_references(ctx: RunContext[ExploreDeps], symbol: str, candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        """Trace references to a symbol across candidate files."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "references": []}
        t0 = perf_counter()
        try:
            result = deps.client.explore_fetch_references(
                symbol=symbol,
                candidate_file_keys=candidate_file_keys or [],
                excluded_file_keys=[],
                limit=deps.budget.max_fetch_limit,
                analysis_context=deps.analysis_context,
            )
            _record(deps, "explore.fetch_references", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.fetch_references", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "references": []}

    @agent.tool  # type: ignore[misc]
    def explore_fetch_call_edges(ctx: RunContext[ExploreDeps], symbol: str, direction: str = "both", candidate_file_keys: list[str] | None = None) -> dict[str, Any]:
        """Trace callers and callees of a symbol."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "edges": []}
        t0 = perf_counter()
        try:
            result = deps.client.explore_fetch_call_edges(
                symbol=symbol,
                direction=direction,
                candidate_file_keys=candidate_file_keys or [],
                excluded_file_keys=[],
                limit=deps.budget.max_fetch_limit,
                analysis_context=deps.analysis_context,
            )
            _record(deps, "explore.fetch_call_edges", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.fetch_call_edges", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "edges": []}

    @agent.tool  # type: ignore[misc]
    def explore_read_file(ctx: RunContext[ExploreDeps], file_key: str, start_line: int = 1, end_line: int = 120) -> dict[str, Any]:
        """Read source code from a file."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "content": ""}
        t0 = perf_counter()
        try:
            result = deps.client.explore_read_file(
                file_key=file_key,
                start_line=start_line,
                end_line=end_line,
                max_bytes=32_000,
            )
            _record(deps, "explore.read_file", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.read_file", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "content": ""}

    @agent.tool  # type: ignore[misc]
    def explore_rg_search(ctx: RunContext[ExploreDeps], query: str, mode: str = "symbol", max_hits: int = 50) -> dict[str, Any]:
        """Search for text/symbols across the workspace."""
        deps = ctx.deps
        if not deps.budget.consume_call():
            return {"warning": "budget_exhausted", "hits": []}
        t0 = perf_counter()
        try:
            result = deps.client.explore_rg_search(
                query=query,
                mode=mode,
                max_hits=min(max_hits, 200),
                max_files=min(50, deps.budget.max_candidates_per_symbol),
                timeout_s=min(deps.budget.parse_timeout_s, 30),
                context_lines=1,
                analysis_context=deps.analysis_context,
            )
            _record(deps, "explore.rg_search", True, (perf_counter() - t0) * 1000)
            return result
        except Exception as exc:
            _record(deps, "explore.rg_search", False, (perf_counter() - t0) * 1000, str(exc))
            return {"warning": str(exc), "hits": []}

    deps = ExploreDeps(
        client=client,
        budget=budget,
        tool_usage=tool_usage,
        analysis_context=analysis_context,
    )
    return agent, deps


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ReviewOrchestrator:
    def __init__(
        self,
        *,
        client: CxxtractHttpClient | None = None,
        planner_factory: Callable[[str], Any] | None = None,
        synthesis_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._client = client
        self._planner_factory = planner_factory or build_planner_agent
        self._synthesis_factory = synthesis_factory or build_synthesis_agent
        self._planner_cache: dict[str, Any] = {}
        self._synthesis_cache: dict[str, Any] = {}

    def run(self, request: ReviewRequest) -> ReviewReport:
        owns_client = self._client is None
        client = self._client or CxxtractHttpClient(
            base_url=request.cxxtract_base_url,
            workspace_id=request.workspace_id,
            timeout_s=90.0,
        )

        # Wall-clock deadline (0 means disabled)
        deadline = 0.0
        if request.review_timeout_s > 0:
            deadline = perf_counter() + request.review_timeout_s

        # Context IDs created during this run, for cleanup
        created_context_ids: list[str] = []

        try:
            return self._run_inner(request, client, deadline, created_context_ids)
        except _ReviewTimeout:
            logger.warning("review timed out after %ds", request.review_timeout_s)
            return ReviewReport(
                workspace_id=request.workspace_id,
                summary=f"Review timed out after {request.review_timeout_s}s; partial results.",
                findings=[],
                coverage=CoverageSummary(),
                decision=ReviewDecision(
                    fail_threshold=request.fail_on_severity,
                    blocking_findings=0,
                    should_block=False,
                ),
                tool_usage=[],
            )
        finally:
            # Expire any contexts we created
            for ctx_id in created_context_ids:
                try:
                    client.context_expire(context_id=ctx_id)
                    logger.debug("expired context %s", ctx_id)
                except Exception as exc:
                    logger.debug("failed to expire context %s: %s", ctx_id, exc)
            # Close the HTTP client if we own it
            if owns_client:
                client.close()

    def _run_inner(
        self,
        request: ReviewRequest,
        client: CxxtractHttpClient,
        deadline: float,
        created_context_ids: list[str],
    ) -> ReviewReport:
        # Health check
        try:
            workspace = client.workspace_info()
        except Exception as exc:
            raise RuntimeError(
                f"CXXtract backend unreachable at {client.base_url}; is the service running? ({exc})"
            ) from exc

        ws_id = str(workspace.get("workspace_id", "")).strip() or request.workspace_id
        if ws_id != request.workspace_id:
            raise RuntimeError("workspace info mismatch for requested workspace_id")

        runtime = ReviewContextIngestor.ingest(request)
        bundle = runtime.bundle
        changes = parse_unified_diff(bundle.patch_text)
        prepass = build_prepass_result(changes, max_symbols=runtime.max_symbols)
        if not bundle.changed_files:
            bundle.changed_files = list(prepass.changed_files)
        if not bundle.changed_hunks:
            bundle.changed_hunks = [
                {
                    "file_path": change.effective_path,
                    "old_start": h.old_start,
                    "old_count": h.old_count,
                    "new_start": h.new_start,
                    "new_count": h.new_count,
                }
                for change in changes
                for h in change.hunks
            ][:1000]

        _check_deadline(deadline, "context_ingestion")

        views = self._prepare_views(
            client=client, runtime_context=bundle,
            lock_root=runtime.cache_dir, created_context_ids=created_context_ids,
        )
        tool_usage: list[ToolCallRecord] = []

        budget = BudgetTracker(
            max_symbol_slots=runtime.max_symbol_slots,
            max_total_tool_calls=runtime.max_total_tool_calls,
            max_symbols=runtime.max_symbols,
            max_candidates_per_symbol=runtime.max_candidates_per_symbol,
            max_fetch_limit=runtime.max_fetch_limit,
            parse_timeout_s=runtime.parse_timeout_s,
            parse_workers=runtime.parse_workers,
        )

        cache = ReviewTraceCache(runtime.cache_dir) if runtime.enable_cache else None
        cache_key = ""
        if cache is not None:
            cache_key = cache.make_key(
                workspace_id=bundle.workspace_id,
                base_sha=bundle.base_sha,
                head_sha=bundle.head_sha,
                target_sha=bundle.target_branch_head_sha,
                merge_sha=bundle.merge_preview_sha,
                patch_text=bundle.patch_text,
                policy={
                    "fail_on_severity": runtime.fail_on_severity.value,
                    "llm_model": request.llm_model,
                    "max_symbols": runtime.max_symbols,
                    "max_symbol_slots": runtime.max_symbol_slots,
                    "max_total_tool_calls": runtime.max_total_tool_calls,
                    "parse_timeout_s": runtime.parse_timeout_s,
                    "parse_workers": runtime.parse_workers,
                    "max_candidates_per_symbol": runtime.max_candidates_per_symbol,
                    "max_fetch_limit": runtime.max_fetch_limit,
                },
            )
            cached = cache.load_report(cache_key)
            if cached is not None:
                logger.info("cache hit for key %s", cache_key[:16])
                return cached

        # --- Step 1: LLM Planner ---
        _check_deadline(deadline, "planner")
        planner = self._planner_for(request.llm_model)
        planner_prompt = build_planner_prompt(
            context=bundle,
            prepass=prepass,
            budgets={
                "max_symbols": runtime.max_symbols,
                "max_symbol_slots": runtime.max_symbol_slots,
                "max_total_tool_calls": runtime.max_total_tool_calls,
                "parse_timeout_s": runtime.parse_timeout_s,
                "parse_workers": runtime.parse_workers,
                "max_candidates_per_symbol": runtime.max_candidates_per_symbol,
                "max_fetch_limit": runtime.max_fetch_limit,
            },
        )
        try:
            plan = planner.run_sync(planner_prompt).data
        except Exception as exc:
            logger.warning("planner failed, using seed-symbol fallback: %s", exc)
            plan = ReviewPlan(prioritized_symbols=[s.symbol for s in prepass.seed_symbols[: runtime.max_symbols]])

        # --- Step 2: Deterministic base evidence collection ---
        _check_deadline(deadline, "evidence_collection")
        impacts, symbol_facts, coverage = self._collect_evidence(
            client, runtime, views, prepass, plan, tool_usage, budget, deadline,
        )

        # --- Step 3: Build merge delta signals ---
        merge_delta_signals = _build_merge_delta_signals(symbol_facts, views)

        # --- Step 4: Build initial fact sheet ---
        fact_sheet = ReviewFactSheet(
            changed_files=prepass.changed_files,
            changed_hunk_count=prepass.changed_hunk_count,
            seed_symbols=[s.symbol for s in prepass.seed_symbols],
            suspicious_anchors=prepass.suspicious_anchors,
            changed_methods=prepass.changed_methods,
            added_call_sites=prepass.added_call_sites,
            removed_call_sites=prepass.removed_call_sites,
            include_macro_config_changes=prepass.include_macro_config_changes,
            symbol_facts=symbol_facts,
            evidence_anchors=[ev for impact in impacts for ev in _anchors_from_impact(impact)][:400],
            coverage=coverage,
            view_contexts=views.status,
            merge_delta_signals=merge_delta_signals,
            warnings=sorted(set(prepass.warnings + views.status.warnings + coverage.warnings)),
        )

        # --- Step 5: Agent-driven follow-up exploration ---
        _check_deadline(deadline, "exploration")
        if budget.remaining_calls > 0 and budget.remaining_slots > 0:
            followup_evidence = self._agent_follow_up(
                model_name=request.llm_model,
                client=client,
                budget=budget,
                tool_usage=tool_usage,
                fact_sheet=fact_sheet,
                prepass=prepass,
                views=views,
            )
            # Merge follow-up evidence anchors into fact sheet
            if followup_evidence:
                existing_anchors = list(fact_sheet.evidence_anchors)
                existing_anchors.extend(followup_evidence[:100])
                fact_sheet = fact_sheet.model_copy(update={
                    "evidence_anchors": existing_anchors[:500],
                    "warnings": sorted(set(fact_sheet.warnings + ["agent_followup_executed"])),
                })

        # --- Step 6: Test impact analysis ---
        _check_deadline(deadline, "test_impact")
        test_impact = _analyze_test_impact(
            prepass=prepass, impacts=impacts, client=client, views=views,
            budget=budget, tool_usage=tool_usage,
        )

        # --- Step 7: LLM Synthesis ---
        _check_deadline(deadline, "synthesis")
        synth = self._synthesis_for(request.llm_model)
        try:
            draft = synth.run_sync(
                build_synthesis_prompt(fact_sheet=fact_sheet, fail_threshold=runtime.fail_on_severity.value)
            ).data
        except Exception as exc:
            logger.warning("synthesis failed, using deterministic fallback: %s", exc)
            draft = ReviewReport(
                workspace_id=bundle.workspace_id,
                summary="LLM synthesis failed; using deterministic fallback.",
                findings=[],
                coverage=coverage,
                decision=ReviewDecision(
                    fail_threshold=runtime.fail_on_severity,
                    blocking_findings=0,
                    should_block=False,
                ),
                tool_usage=tool_usage,
            )

        # --- Step 8: Policy gate ---
        final = _policy_gate(
            report=draft,
            fact_sheet=fact_sheet,
            test_impact=test_impact,
            fail_threshold=runtime.fail_on_severity,
            tool_usage=tool_usage,
            workspace_id=bundle.workspace_id,
        )
        if cache is not None and cache_key:
            cache.save(cache_key, {"review_report": final.model_dump(mode="json")})
        return final

    def _planner_for(self, model_name: str):
        key = (model_name or "").strip() or "openai:gpt-4o"
        if key not in self._planner_cache:
            self._planner_cache[key] = self._planner_factory(key)
        return self._planner_cache[key]

    def _synthesis_for(self, model_name: str):
        key = (model_name or "").strip() or "openai:gpt-4o"
        if key not in self._synthesis_cache:
            self._synthesis_cache[key] = self._synthesis_factory(key)
        return self._synthesis_cache[key]

    def _prepare_views(
        self,
        *,
        client: CxxtractHttpClient,
        runtime_context,
        lock_root: str,
        created_context_ids: list[str],
    ) -> ViewContexts:
        ws_id = runtime_context.workspace_id
        baseline_id = f"{ws_id}:baseline"
        pr_id = str(runtime_context.pr_metadata.get("pr_id", "") or runtime_context.pr_metadata.get("mr_id", "") or "review")
        head_id = f"{ws_id}:pr:{pr_id}:head"
        merge_id = f"{ws_id}:pr:{pr_id}:merge"
        warnings: list[str] = []

        baseline_materialized = False
        head_materialized = False
        merge_materialized = False

        try:
            with _workspace_lock(lock_root=lock_root, workspace_id=ws_id, timeout_s=20.0):
                if runtime_context.primary_repo_id and _is_sha(runtime_context.target_branch_head_sha):
                    try:
                        client.sync_repo(
                            repo_id=runtime_context.primary_repo_id,
                            commit_sha=runtime_context.target_branch_head_sha,
                            branch=str(runtime_context.pr_metadata.get("target_branch", "")),
                            force_clean=True,
                        )
                        baseline_materialized = True
                    except Exception as exc:
                        logger.warning("baseline sync failed: %s", exc)
                        warnings.append(f"baseline_sync_failed:{exc}")
                else:
                    # No sync needed -- use workspace as-is
                    baseline_materialized = True

                if _is_sha(runtime_context.head_sha):
                    try:
                        created = client.context_create_pr_overlay(
                            pr_id=pr_id,
                            base_ref=runtime_context.target_branch_head_sha or runtime_context.base_sha,
                            head_ref=runtime_context.head_sha,
                            context_id=head_id,
                        )
                        head_id = str(created.get("context_id", head_id))
                        created_context_ids.append(head_id)
                        head_materialized = True
                        if bool(created.get("partial_overlay", False)):
                            warnings.append("head_partial_overlay")
                    except Exception as exc:
                        logger.warning("head context creation failed: %s", exc)
                        warnings.append(f"head_context_failed:{exc}")
                else:
                    warnings.append("head_sha_missing_or_invalid")

                if _is_sha(runtime_context.merge_preview_sha):
                    try:
                        created = client.context_create_pr_overlay(
                            pr_id=f"{pr_id}-merge",
                            base_ref=runtime_context.target_branch_head_sha or runtime_context.base_sha,
                            head_ref=runtime_context.merge_preview_sha,
                            context_id=merge_id,
                        )
                        merge_id = str(created.get("context_id", merge_id))
                        created_context_ids.append(merge_id)
                        merge_materialized = True
                        if bool(created.get("partial_overlay", False)):
                            warnings.append("merge_partial_overlay")
                    except Exception as exc:
                        logger.warning("merge preview context creation failed: %s", exc)
                        warnings.append(f"merge_preview_context_failed:{exc}")
                elif runtime_context.merge_preview_sha:
                    warnings.append("merge_preview_sha_invalid")
                else:
                    warnings.append("merge_preview_not_materialized")
        except TimeoutError as exc:
            logger.warning("workspace lock timed out: %s", exc)
            warnings.append(f"workspace_lock_timeout:{exc}")

        status = ViewContextMaterialization(
            baseline_context_id=baseline_id,
            head_context_id=head_id,
            merge_preview_context_id=merge_id if runtime_context.merge_preview_sha else "",
            baseline_materialized=baseline_materialized,
            head_materialized=head_materialized,
            merge_preview_materialized=merge_materialized,
            warnings=warnings,
        )
        return ViewContexts(
            baseline={"mode": "baseline", "context_id": baseline_id},
            head={"mode": "pr", "context_id": head_id, "pr_id": pr_id},
            merge_preview={"mode": "pr", "context_id": merge_id, "pr_id": f"{pr_id}-merge"} if runtime_context.merge_preview_sha else None,
            status=status,
        )

    def _collect_evidence(
        self,
        client: CxxtractHttpClient,
        runtime: IngestedReviewContext,
        views: ViewContexts,
        prepass,
        plan: ReviewPlan,
        tool_usage: list[ToolCallRecord],
        budget: BudgetTracker,
        deadline: float = 0.0,
    ) -> tuple[list[SymbolImpact], list[SymbolFact], CoverageSummary]:
        symbols = list(dict.fromkeys(plan.prioritized_symbols + [s.symbol for s in prepass.seed_symbols]))[: budget.max_symbols]
        impacts: list[SymbolImpact] = []
        facts: list[SymbolFact] = []
        for idx, symbol in enumerate(symbols):
            if not budget.consume_slot() or budget.exhausted:
                logger.debug("evidence collection stopped at symbol %d/%d (budget)", idx, len(symbols))
                break
            _check_deadline(deadline, f"evidence_collection_symbol_{idx}")
            impact = _collect_symbol(client, runtime, views.head, symbol, tool_usage, budget)
            impacts.append(impact)
            if budget.exhausted:
                break
            base_refs, base_edges = _fetch_counts(client, runtime, views.baseline, symbol, impact, tool_usage, budget)
            merge_refs, merge_edges = (0, 0)
            if views.merge_preview is not None and not budget.exhausted:
                merge_refs, merge_edges = _fetch_counts(client, runtime, views.merge_preview, symbol, impact, tool_usage, budget)
            facts.append(
                SymbolFact(
                    symbol=symbol,
                    candidate_file_keys=impact.candidate_file_keys,
                    parsed_file_keys=impact.parsed_file_keys,
                    head_reference_count=len(impact.references),
                    baseline_reference_count=base_refs,
                    merge_preview_reference_count=merge_refs,
                    head_call_edge_count=len(impact.call_edges),
                    baseline_call_edge_count=base_edges,
                    merge_preview_call_edge_count=merge_edges,
                    reference_delta_vs_baseline=len(impact.references) - base_refs,
                    call_edge_delta_vs_baseline=len(impact.call_edges) - base_edges,
                    warnings=impact.warnings,
                )
            )
        return impacts, facts, _coverage(impacts)

    def _agent_follow_up(
        self,
        *,
        model_name: str,
        client: CxxtractHttpClient,
        budget: BudgetTracker,
        tool_usage: list[ToolCallRecord],
        fact_sheet: ReviewFactSheet,
        prepass,
        views: ViewContexts,
    ) -> list[EvidenceRef]:
        """Run agent-driven follow-up exploration with remaining budget."""
        if budget.remaining_calls < 2 or budget.remaining_slots < 1:
            return []

        prompt = build_exploration_prompt(
            fact_sheet=fact_sheet,
            prepass=prepass,
            remaining_calls=budget.remaining_calls,
            remaining_rounds=budget.remaining_slots,
        )

        try:
            agent, deps = build_exploration_agent(
                model_name,
                client=client,
                budget=budget,
                tool_usage=tool_usage,
                analysis_context=views.head,
            )
            result = agent.run_sync(prompt, deps=deps)
            summary_text = str(result.data or "")
        except Exception as exc:
            logger.warning("exploration follow-up failed: %s", exc)
            return []

        # Convert the exploration summary to evidence anchors
        evidence: list[EvidenceRef] = []
        if summary_text.strip():
            evidence.append(
                EvidenceRef(
                    tool="agent.exploration_followup",
                    description="agent_exploration_summary",
                    snippet=summary_text[:500],
                )
            )
        return evidence

    @staticmethod
    def write_report_files(report: ReviewReport, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "review_report.md"
        json_path = out / "review_report.json"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path


# ---------------------------------------------------------------------------
# Deterministic evidence collection (per-symbol)
# ---------------------------------------------------------------------------

def _collect_symbol(client, runtime, analysis_context, symbol: str, tool_usage: list[ToolCallRecord], budget: BudgetTracker) -> SymbolImpact:
    # --- Phase 1: Candidate discovery ---
    rg = _call(
        tool_usage,
        budget,
        "collector",
        "explore.rg_search",
        client.explore_rg_search,
        query=symbol,
        mode="symbol",
        analysis_context=analysis_context,
        max_hits=min(budget.max_fetch_limit, 200),
        max_files=min(budget.max_candidates_per_symbol, 200),
        timeout_s=min(budget.parse_timeout_s, 60),
        context_lines=1,
    )
    rg_hits = list(rg.get("hits", []) or [])
    rg_candidates = [str(row.get("file_key", "")).strip() for row in rg_hits if str(row.get("file_key", "")).strip()]

    if budget.exhausted:
        return _empty_impact(symbol, rg_hits=rg_hits)

    listed = _call(
        tool_usage,
        budget,
        "collector",
        "explore.list_candidates",
        client.explore_list_candidates,
        symbol=symbol,
        max_files=budget.max_candidates_per_symbol,
        include_rg=True,
        analysis_context=analysis_context,
    )
    listed_candidates = list(listed.get("candidates", []) or [])
    candidates = _dedupe_file_keys(rg_candidates + listed_candidates)[: budget.max_candidates_per_symbol]
    deleted = _dedupe_file_keys(list(listed.get("deleted_file_keys", []) or []))

    if budget.exhausted:
        return _empty_impact(symbol, rg_hits=rg_hits, candidates=candidates, deleted=deleted)

    # --- Phase 2: Context reads ---
    read_contexts: list[dict[str, Any]] = []
    for hit in rg_hits[:2]:
        if budget.exhausted:
            break
        file_key = str(hit.get("file_key", "")).strip()
        if not file_key:
            continue
        line = int(hit.get("line", 1) or 1)
        row = _call(
            tool_usage,
            budget,
            "collector",
            "explore.read_file",
            client.explore_read_file,
            file_key=file_key,
            start_line=max(1, line - 8),
            end_line=line + 8,
            max_bytes=32_000,
        )
        if row.get("content"):
            read_contexts.append(row)
    if not read_contexts and candidates and not budget.exhausted:
        row = _call(
            tool_usage,
            budget,
            "collector",
            "explore.read_file",
            client.explore_read_file,
            file_key=candidates[0],
            start_line=1,
            end_line=120,
            max_bytes=32_000,
        )
        if row.get("content"):
            read_contexts.append(row)

    if budget.exhausted:
        return _empty_impact(symbol, rg_hits=rg_hits, candidates=candidates, deleted=deleted, read_contexts=read_contexts)

    # --- Phase 3: Parse & freshness ---
    freshness: dict[str, Any] = {}
    stale: list[str] = []
    fresh: list[str] = []
    unparsed: list[str] = []
    parsed: dict[str, Any] = {}
    parsed_keys: list[str] = []
    parse_failed: list[str] = []

    if candidates:
        freshness = _call(tool_usage, budget, "collector", "explore.classify_freshness", client.explore_classify_freshness, candidate_file_keys=candidates, max_files=max(1, len(candidates)), analysis_context=analysis_context)
        stale = list(freshness.get("stale", []) or [])
        fresh = list(freshness.get("fresh", []) or [])
        unparsed = list(freshness.get("unparsed", []) or [])

    if stale and not budget.exhausted:
        parsed = _call(tool_usage, budget, "collector", "explore.parse_file", client.explore_parse_file, file_keys=stale, max_parse_workers=budget.parse_workers, timeout_s=budget.parse_timeout_s, skip_if_fresh=True, analysis_context=analysis_context)
        parsed_keys = list(parsed.get("parsed_file_keys", []) or [])
        parse_failed = list(parsed.get("failed_file_keys", []) or [])

    if budget.exhausted:
        return _empty_impact(
            symbol, rg_hits=rg_hits, candidates=candidates, deleted=deleted,
            read_contexts=read_contexts, fresh=fresh, stale=stale, unparsed=unparsed,
            parsed_keys=parsed_keys, parse_failed=parse_failed,
        )

    # --- Phase 4: Semantic fetch ---
    sym_rows = _call(tool_usage, budget, "collector", "explore.fetch_symbols", client.explore_fetch_symbols, symbol=symbol, candidate_file_keys=candidates, excluded_file_keys=deleted, limit=budget.max_fetch_limit, analysis_context=analysis_context)

    ref_rows: dict[str, Any] = {}
    if not budget.exhausted:
        ref_rows = _call(tool_usage, budget, "collector", "explore.fetch_references", client.explore_fetch_references, symbol=symbol, candidate_file_keys=candidates, excluded_file_keys=deleted, limit=budget.max_fetch_limit, analysis_context=analysis_context)

    edge_rows: dict[str, Any] = {}
    if not budget.exhausted:
        edge_rows = _call(tool_usage, budget, "collector", "explore.fetch_call_edges", client.explore_fetch_call_edges, symbol=symbol, direction="both", candidate_file_keys=candidates, excluded_file_keys=deleted, limit=budget.max_fetch_limit, analysis_context=analysis_context)

    conf: dict[str, Any] = {}
    if not budget.exhausted:
        conf = _call(tool_usage, budget, "collector", "explore.get_confidence", client.explore_get_confidence, verified_files=sorted(set(fresh + parsed_keys)), stale_files=sorted(set(parse_failed)), unparsed_files=sorted(set(unparsed + list(parsed.get("unparsed_file_keys", []) or []))), warnings=[], overlay_mode=str(freshness.get("overlay_mode", "sparse")))

    macro_summary = ""
    if not budget.exhausted and not list(ref_rows.get("references", []) or []) and not list(edge_rows.get("edges", []) or []):
        macro = _call(
            tool_usage,
            budget,
            "collector",
            "agent.investigate_symbol",
            client.agent_investigate_symbol,
            symbol=symbol,
        )
        macro_summary = str(macro.get("summary_markdown", "") or macro.get("summary", ""))

    warnings = sorted(
        set(
            list(rg.get("warnings", []) or [])
            + list(listed.get("warnings", []) or [])
            + list(freshness.get("warnings", []) or [])
            + list(parsed.get("parse_warnings", []) or [])
            + list(sym_rows.get("warnings", []) or [])
            + list(ref_rows.get("warnings", []) or [])
            + list(edge_rows.get("warnings", []) or [])
            + (["candidates_truncated"] if len(rg_candidates + listed_candidates) > len(candidates) else [])
            + (["macro_fallback_used"] if macro_summary else [])
            + (["budget_exhausted_mid_symbol"] if budget.exhausted else [])
        )
    )
    return SymbolImpact(
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
        rg_hits=rg_hits[:200],
        read_contexts=read_contexts[:20],
        confidence=dict(conf.get("confidence", {}) or {}),
        macro_summary=macro_summary[:2000],
        warnings=warnings,
    )


def _empty_impact(
    symbol: str,
    *,
    rg_hits: list[dict[str, Any]] | None = None,
    candidates: list[str] | None = None,
    deleted: list[str] | None = None,
    read_contexts: list[dict[str, Any]] | None = None,
    fresh: list[str] | None = None,
    stale: list[str] | None = None,
    unparsed: list[str] | None = None,
    parsed_keys: list[str] | None = None,
    parse_failed: list[str] | None = None,
) -> SymbolImpact:
    """Build a partial SymbolImpact when budget is exhausted mid-symbol."""
    return SymbolImpact(
        symbol=symbol,
        candidate_file_keys=candidates or [],
        deleted_file_keys=deleted or [],
        fresh=fresh or [],
        stale=stale or [],
        unparsed=unparsed or [],
        parsed_file_keys=parsed_keys or [],
        parse_failed_file_keys=parse_failed or [],
        rg_hits=(rg_hits or [])[:200],
        read_contexts=(read_contexts or [])[:20],
        warnings=["budget_exhausted_mid_symbol"],
    )


def _fetch_counts(client, runtime, analysis_context, symbol: str, impact: SymbolImpact, tool_usage: list[ToolCallRecord], budget: BudgetTracker) -> tuple[int, int]:
    if budget.exhausted:
        return 0, 0
    refs = _call(tool_usage, budget, "collector", "explore.fetch_references", client.explore_fetch_references, symbol=symbol, candidate_file_keys=impact.candidate_file_keys, excluded_file_keys=impact.deleted_file_keys, limit=budget.max_fetch_limit, analysis_context=analysis_context)
    if budget.exhausted:
        return len(list(refs.get("references", []) or [])), 0
    edges = _call(tool_usage, budget, "collector", "explore.fetch_call_edges", client.explore_fetch_call_edges, symbol=symbol, direction="both", candidate_file_keys=impact.candidate_file_keys, excluded_file_keys=impact.deleted_file_keys, limit=budget.max_fetch_limit, analysis_context=analysis_context)
    return len(list(refs.get("references", []) or [])), len(list(edges.get("edges", []) or []))


def _dedupe_file_keys(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        key = str(raw).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _call(tool_usage: list[ToolCallRecord], budget: BudgetTracker, skill: str, tool: str, fn, **kwargs: Any) -> dict[str, Any]:
    if budget.exhausted:
        return {"warnings": [f"max_total_tool_calls_reached:{budget.max_total_tool_calls}"]}
    if not budget.consume_call():
        return {"warnings": [f"max_total_tool_calls_reached:{budget.max_total_tool_calls}"]}
    t0 = perf_counter()
    try:
        data = fn(**kwargs)
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=True, elapsed_ms=(perf_counter() - t0) * 1000.0))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("tool call %s failed: %s", tool, exc)
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=False, elapsed_ms=(perf_counter() - t0) * 1000.0, note=str(exc)))
        return {"warnings": [f"{tool}_failed:{exc}"]}


# ---------------------------------------------------------------------------
# Evidence helpers
# ---------------------------------------------------------------------------

def _anchors_from_impact(impact: SymbolImpact) -> list[EvidenceRef]:
    lexical = [
        EvidenceRef(
            tool="explore.rg_search",
            file_key=str(hit.get("file_key", "")),
            line=int(hit.get("line", 0) or 0),
            symbol=impact.symbol,
            description="lexical_hit",
            snippet=str(hit.get("line_text", ""))[:220],
        )
        for hit in impact.rg_hits[:2]
    ]
    reads = [
        EvidenceRef(
            tool="explore.read_file",
            file_key=str(row.get("file_key", "")),
            line=int((row.get("line_range") or [0])[0] if isinstance(row.get("line_range"), list) and row.get("line_range") else 0),
            symbol=impact.symbol,
            description="file_context",
            snippet=str(row.get("content", ""))[:220],
        )
        for row in impact.read_contexts[:2]
    ]
    refs = [
        EvidenceRef(tool="explore.fetch_references", file_key=str(r.get("file_key", "")), line=int(r.get("line", 0) or 0), symbol=impact.symbol, description="reference")
        for r in impact.references[:2]
    ]
    edges = [
        EvidenceRef(tool="explore.fetch_call_edges", file_key=str(e.get("file_key", "")), line=int(e.get("line", 0) or 0), symbol=impact.symbol, description=f"{e.get('caller', '')}->{e.get('callee', '')}")
        for e in impact.call_edges[:2]
    ]
    macro = [EvidenceRef(tool="agent.investigate_symbol", symbol=impact.symbol, description="macro_fallback", snippet=impact.macro_summary[:220])] if impact.macro_summary else []
    return lexical + reads + refs + edges + macro


def _coverage(impacts: list[SymbolImpact]) -> CoverageSummary:
    verified: set[str] = set()
    stale: set[str] = set()
    unparsed: set[str] = set()
    warnings: set[str] = set()
    ratio_sum = 0.0
    weight = 0
    for impact in impacts:
        conf = dict(impact.confidence or {})
        total = int(conf.get("total_candidates", 0) or 0)
        ratio = float(conf.get("verified_ratio", 0.0) or 0.0)
        ratio_sum += ratio * max(total, 1)
        weight += max(total, 1)
        verified.update([str(x) for x in conf.get("verified_files", []) or []])
        stale.update([str(x) for x in conf.get("stale_files", []) or []])
        unparsed.update([str(x) for x in conf.get("unparsed_files", []) or []])
        warnings.update(impact.warnings)
    return CoverageSummary(
        verified_ratio=round((ratio_sum / weight) if weight else 0.0, 4),
        total_candidates=len(verified | stale | unparsed),
        verified_files=sorted(verified),
        stale_files=sorted(stale),
        unparsed_files=sorted(unparsed),
        warnings=sorted(warnings),
    )


# ---------------------------------------------------------------------------
# Merge delta signals
# ---------------------------------------------------------------------------

def _build_merge_delta_signals(symbol_facts: list[SymbolFact], views: ViewContexts) -> list[dict[str, Any]]:
    """Flag symbols where merge-preview counts differ from both head and baseline."""
    if views.merge_preview is None or not views.status.merge_preview_materialized:
        return []
    signals: list[dict[str, Any]] = []
    for sf in symbol_facts:
        if not sf.merge_preview_reference_count and not sf.merge_preview_call_edge_count:
            continue
        merge_ref_delta_vs_head = sf.merge_preview_reference_count - sf.head_reference_count
        merge_edge_delta_vs_head = sf.merge_preview_call_edge_count - sf.head_call_edge_count
        # Signal when merge introduces changes not present in head
        if merge_ref_delta_vs_head != 0 or merge_edge_delta_vs_head != 0:
            signals.append({
                "symbol": sf.symbol,
                "head_refs": sf.head_reference_count,
                "baseline_refs": sf.baseline_reference_count,
                "merge_refs": sf.merge_preview_reference_count,
                "merge_ref_delta_vs_head": merge_ref_delta_vs_head,
                "head_edges": sf.head_call_edge_count,
                "baseline_edges": sf.baseline_call_edge_count,
                "merge_edges": sf.merge_preview_call_edge_count,
                "merge_edge_delta_vs_head": merge_edge_delta_vs_head,
                "risk": "merge_introduces_new_interaction" if (
                    merge_ref_delta_vs_head != 0
                    and sf.reference_delta_vs_baseline != merge_ref_delta_vs_head
                ) else "merge_count_shift",
            })
    return signals[:50]


# ---------------------------------------------------------------------------
# Test impact analysis
# ---------------------------------------------------------------------------

def _analyze_test_impact(
    *,
    prepass,
    impacts: list[SymbolImpact],
    client: CxxtractHttpClient,
    views: ViewContexts,
    budget: BudgetTracker,
    tool_usage: list[ToolCallRecord],
) -> TestImpact:
    """Build test impact using path heuristics + call-edge traversal + semantic lookup."""
    # 1. Directly impacted: changed files that are tests
    direct = sorted({p for p in prepass.changed_files if _looks_like_test_path(p)})

    # 2. Likely impacted via call edges from already-collected impacts
    likely: set[str] = set()
    dependency_edges: list[dict[str, Any]] = []
    for impact in impacts:
        for row in impact.references + impact.call_edges:
            fk = str(row.get("file_key", ""))
            rel = fk.split(":", 1)[1] if ":" in fk else ""
            if rel and _looks_like_test_path(rel):
                likely.add(rel)
                dependency_edges.append({
                    "symbol": impact.symbol,
                    "test_file": rel,
                    "file_key": fk,
                    "source": "call_edge_or_reference",
                    "line": int(row.get("line", 0) or 0),
                })

    # 3. Semantic test dependency: for top changed methods, fetch references to find test files
    semantic_test_files: set[str] = set()
    methods_to_check = prepass.changed_methods[:10]
    for method in methods_to_check:
        if budget.exhausted:
            break
        refs = _call(
            tool_usage, budget, "test_impact", "explore.fetch_references",
            client.explore_fetch_references,
            symbol=method,
            candidate_file_keys=[],
            excluded_file_keys=[],
            limit=min(budget.max_fetch_limit, 100),
            analysis_context=views.head,
        )
        for ref in list(refs.get("references", []) or []):
            fk = str(ref.get("file_key", ""))
            rel = fk.split(":", 1)[1] if ":" in fk else ""
            if rel and _looks_like_test_path(rel) and rel not in likely:
                semantic_test_files.add(rel)
                dependency_edges.append({
                    "symbol": method,
                    "test_file": rel,
                    "file_key": fk,
                    "source": "semantic_method_reference",
                    "line": int(ref.get("line", 0) or 0),
                })

    likely.update(semantic_test_files)

    # 4. Suggested scopes with confidence
    scopes = ["smoke"]
    if direct:
        scopes.append("unit")
    if likely:
        scopes.append("integration")
    if any(len(impact.repos_involved) > 1 for impact in impacts):
        scopes.append("e2e")

    # 5. Confidence scoring
    if direct and likely and semantic_test_files:
        confidence = 0.9
    elif direct and likely:
        confidence = 0.75
    elif direct:
        confidence = 0.6
    elif likely:
        confidence = 0.5
    else:
        confidence = 0.3

    rationale = ["deterministic_test_impact"]
    if semantic_test_files:
        rationale.append(f"semantic_method_trace:{len(semantic_test_files)}_test_files")
    if dependency_edges:
        rationale.append(f"call_edge_trace:{len(dependency_edges)}_edges")

    return TestImpact(
        directly_impacted_tests=direct[:200],
        likely_impacted_tests=sorted([p for p in likely if p not in set(direct)])[:300],
        suggested_scopes=list(dict.fromkeys(scopes)),
        rationale=rationale,
        confidence=round(confidence, 2),
        test_dependency_edges=dependency_edges[:200],
    )


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------

def _policy_gate(*, report: ReviewReport, fact_sheet: ReviewFactSheet, test_impact: TestImpact, fail_threshold: Severity, tool_usage: list[ToolCallRecord], workspace_id: str) -> ReviewReport:
    findings: list[ReviewFinding] = []
    for finding in report.findings:
        if finding.severity in {Severity.HIGH, Severity.CRITICAL} and not finding.evidence:
            finding = finding.model_copy(update={"severity": Severity.MEDIUM, "tags": sorted(set(finding.tags + ["auto_downgraded_missing_evidence"]))})
        findings.append(finding)
    low_cov = fact_sheet.coverage.verified_ratio < 0.4
    if low_cov:
        findings.append(
            ReviewFinding(
                id=_fid("cov", str(fact_sheet.coverage.verified_ratio)),
                severity=Severity.MEDIUM,
                category=FindingCategory.CONFIDENCE_GAP,
                title="Low semantic coverage reduces confidence",
                impact="Coverage is partial; strict blocking can be overconfident.",
                recommendation="Increase parse/tool budget and rerun review.",
                evidence=[EvidenceRef(tool="policy_gate", description=f"verified_ratio={fact_sheet.coverage.verified_ratio:.3f}")],
                confidence=0.95,
                tags=["coverage_policy"],
            )
        )
    findings = sorted(findings, key=lambda f: (-SEVERITY_RANK[f.severity], f.title, f.id))
    blocking = len([f for f in findings if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[fail_threshold]])
    should_block = blocking > 0 and not (low_cov and not any(f.severity == Severity.CRITICAL and f.evidence for f in findings))
    summary = report.summary.strip() or f"Reviewed {len(fact_sheet.changed_files)} files; findings={len(findings)}."
    return report.model_copy(update={"workspace_id": workspace_id, "summary": summary, "findings": findings, "coverage": fact_sheet.coverage, "decision": ReviewDecision(fail_threshold=fail_threshold, blocking_findings=blocking, should_block=should_block), "tool_usage": tool_usage, "fact_sheet": fact_sheet, "test_impact": test_impact})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fid(*parts: str) -> str:
    return sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _looks_like_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    # Normalize to always have a leading slash for prefix matching
    lp = "/" + p if not p.startswith("/") else p
    return (
        "/test/" in lp
        or "/tests/" in lp
        or p.endswith("_test.cpp")
        or p.endswith("_tests.cpp")
        or p.endswith("_unittest.cpp")
        or "/ut/" in lp
        or "/it/" in lp
        or "/e2e/" in lp
    )


def _pid_alive(pid: int) -> bool:
    """Check whether *pid* refers to a running process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it
        return True
    except OSError:
        return False


class _workspace_lock:
    """Best-effort cross-process lock to avoid concurrent sync/context collisions.

    Includes stale-lock detection: if the lock file exists and contains a PID
    that is no longer alive, the lock is removed and re-acquired.
    """

    def __init__(self, *, lock_root: str, workspace_id: str, timeout_s: float) -> None:
        self._lock_root = Path(lock_root).resolve() / ".workspace_locks"
        self._workspace_id = workspace_id
        self._timeout_s = max(1.0, timeout_s)
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in workspace_id)
        self._lock_path = self._lock_root / f"{safe_id}.lock"
        self._fd = -1

    def _try_remove_stale(self) -> bool:
        """If the lock file is stale (owner PID dead), remove it."""
        try:
            raw = self._lock_path.read_text(encoding="utf-8").strip()
            pid = int(raw)
        except Exception:
            return False
        if _pid_alive(pid):
            return False
        logger.info("removing stale lock for workspace %s (dead pid %d)", self._workspace_id, pid)
        try:
            self._lock_path.unlink(missing_ok=True)
        except Exception:
            return False
        return True

    def __enter__(self):
        self._lock_root.mkdir(parents=True, exist_ok=True)
        deadline = perf_counter() + self._timeout_s
        stale_checked = False
        while True:
            try:
                self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                if not stale_checked:
                    stale_checked = True
                    if self._try_remove_stale():
                        continue
                if perf_counter() >= deadline:
                    raise TimeoutError(self._workspace_id)
                sleep(0.2)

    def __exit__(self, exc_type, exc, tb):
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
        try:
            self._lock_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


def _is_sha(value: str) -> bool:
    v = (value or "").strip()
    return len(v) == 40 and all(ch in "0123456789abcdefABCDEF" for ch in v)
