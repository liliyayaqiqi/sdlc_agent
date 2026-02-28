"""Context-driven orchestration for semantic PR review."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from time import perf_counter, sleep, time
from uuid import uuid4

from review_agent.adapters.llm import build_model_services, endpoint_cache_key
from review_agent.application.pipeline import ReviewPipeline
from review_agent.context_ingestion import IngestedReviewContext, ReviewContextIngestor
from review_agent.domain.location_mapper import FindingLocationMapper
from review_agent.domain.policy import finalize_report, indeterminate_report
from review_agent.manifest_resolver import (
    load_workspace_manifest,
    repo_for_file_key,
    resolve_file_key,
    resolve_repo_id_for_project_path,
)
from review_agent.models import (
    AGENT_VERSION,
    CoverageSummary,
    EvidenceRef,
    InfrastructureError,
    InputNormalizationError,
    ModelContractError,
    PARSER_VERSION,
    PrepassDebug,
    PROMPT_VERSION,
    RepoRevisionContext,
    ReviewExecutionStatus,
    ReviewFactSheet,
    ReviewPlan,
    ReviewReport,
    ReviewRequest,
    ReviewTestImpact,
    RunMetadata,
    Severity,
    SymbolFact,
    SymbolImpact,
    SynthesisDraft,
    ToolCallRecord,
    ViewContextMaterialization,
)
from review_agent.patch_parser import build_prepass_result, parse_unified_diff
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


@dataclass(frozen=True)
class RetrievalStage:
    """One candidate-discovery widening stage."""

    name: str
    entry_repos: list[str]
    max_repo_hops: int
    path_prefixes: list[str]


# ---------------------------------------------------------------------------
# Service compatibility helpers
# ---------------------------------------------------------------------------

def build_planner_agent(model_name: str, **endpoint_kwargs: str):
    planner, _, _ = build_model_services(model_name, **endpoint_kwargs)
    return planner


def build_synthesis_agent(model_name: str, **endpoint_kwargs: str):
    _, _, synthesis = build_model_services(model_name, **endpoint_kwargs)
    return synthesis


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ReviewOrchestrator:
    def __init__(
        self,
        *,
        client: CxxtractHttpClient | None = None,
        service_factory: Callable[[Any], tuple[Any, Any, Any]] | None = None,
    ) -> None:
        self._client = client
        self._service_factory = service_factory or build_model_services
        self._service_cache: dict[str, tuple[Any, Any, Any]] = {}
        self._pipeline = ReviewPipeline(orchestrator=self)

    def run(self, request: ReviewRequest) -> ReviewReport:
        return self._pipeline.run(request)

    def _run_request(self, request: ReviewRequest) -> ReviewReport:
        run_id = uuid4().hex[:12]
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

        # Determine infra fail behaviour
        infra_should_block = request.infra_fail_mode != "pass"

        try:
            return self._run_inner(request, client, deadline, created_context_ids, run_id)
        except _ReviewTimeout:
            logger.warning("[%s] review timed out after %ds", run_id, request.review_timeout_s)
            return _indeterminate_report(
                workspace_id=request.workspace_id,
                reason=f"review_wall_clock_timeout_after_{request.review_timeout_s}s",
                summary=f"Review timed out after {request.review_timeout_s}s; partial results.",
                fail_threshold=request.fail_on_severity,
                should_block=infra_should_block,
                run_id=run_id,
                run_metadata=RunMetadata(
                    input_mode="unknown",
                    run_id=run_id,
                    backend_base_url=request.cxxtract_base_url,
                ),
            )
        except InputNormalizationError as exc:
            logger.error("[%s] input normalization failed: %s", run_id, exc)
            return _indeterminate_report(
                workspace_id=request.workspace_id,
                reason=f"input_normalization_error:{exc}",
                summary=f"Input normalization failed: {exc}",
                fail_threshold=request.fail_on_severity,
                should_block=True,
                run_id=run_id,
                run_metadata=RunMetadata(
                    input_mode="unknown",
                    run_id=run_id,
                    backend_base_url=request.cxxtract_base_url,
                ),
            )
        except InfrastructureError as exc:
            logger.error("[%s] infrastructure error: %s", run_id, exc)
            return _indeterminate_report(
                workspace_id=request.workspace_id,
                reason=f"infrastructure_error:{exc}",
                summary=f"Infrastructure error: {exc}",
                fail_threshold=request.fail_on_severity,
                should_block=infra_should_block,
                run_id=run_id,
                run_metadata=RunMetadata(
                    input_mode="unknown",
                    run_id=run_id,
                    backend_base_url=request.cxxtract_base_url,
                ),
            )
        finally:
            # Expire any contexts we created
            for ctx_id in created_context_ids:
                try:
                    client.context_expire(context_id=ctx_id)
                    logger.debug("[%s] expired context %s", run_id, ctx_id)
                except Exception as exc:
                    logger.debug("[%s] failed to expire context %s: %s", run_id, ctx_id, exc)
            # Close the HTTP client if we own it
            if owns_client:
                client.close()

    def _run_inner(
        self,
        request: ReviewRequest,
        client: CxxtractHttpClient,
        deadline: float,
        created_context_ids: list[str],
        run_id: str,
    ) -> ReviewReport:
        # Determine input mode for metadata
        input_mode = _detect_input_mode(request)
        logger.info("[%s] review start workspace=%s input_mode=%s", run_id, request.workspace_id, input_mode)

        run_metadata = RunMetadata(
            input_mode=input_mode,
            run_id=run_id,
            backend_base_url=request.cxxtract_base_url,
        )

        # Health check
        try:
            workspace = client.workspace_info()
        except Exception as exc:
            raise InfrastructureError(
                f"CXXtract backend unreachable at {client.base_url}; is the service running? ({exc})"
            ) from exc
        logger.info(
            "[%s] workspace connected backend=%s workspace_id=%s root=%s",
            run_id,
            client.base_url,
            workspace.get("workspace_id", request.workspace_id),
            workspace.get("root_path", ""),
        )

        ws_id = str(workspace.get("workspace_id", "")).strip() or request.workspace_id
        if ws_id != request.workspace_id:
            raise InfrastructureError("workspace info mismatch for requested workspace_id")

        runtime = ReviewContextIngestor.ingest(request)
        bundle = runtime.bundle
        changes = parse_unified_diff(bundle.patch_text)

        # Fail-fast invariant: non-empty patch input must produce ≥1 PatchChange
        if bundle.patch_text.strip() and not changes:
            raise InputNormalizationError(
                f"patch input contained {len(bundle.patch_text)} bytes but "
                f"parse_unified_diff produced 0 changes; the diff format may "
                f"be incompatible (missing 'diff --git' headers?)"
            )

        prepass = build_prepass_result(changes, max_symbols=runtime.max_symbols)
        logger.info(
            "[%s] prepass complete changed_files=%d hunks=%d declarations=%d seeds=%d suspicious=%d",
            run_id,
            len(prepass.changed_files),
            prepass.changed_hunk_count,
            len(prepass.changed_declarations),
            len(prepass.seed_symbols),
            len(prepass.suspicious_anchors),
        )
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

        bundle, entry_repos, scope_warnings = _normalize_bundle_and_scope_repos(
            workspace=workspace,
            bundle=bundle,
            changed_files=prepass.changed_files,
        )
        if scope_warnings:
            prepass.warnings.extend(scope_warnings)
        run_metadata = run_metadata.model_copy(
            update={
                "prepass_debug": PrepassDebug(
                    ranked_seed_candidates=prepass.seed_symbols[:24],
                    changed_declarations=prepass.changed_declarations[:40],
                    member_call_sites_top=prepass.member_call_sites[:40],
                    diff_excerpt_reasons=[excerpt.reason for excerpt in prepass.diff_excerpts[:20]],
                    retrieval_widening_events=[],
                )
            }
        )
        retrieval_stages = _build_retrieval_stages(
            workspace=workspace,
            bundle=bundle,
            prepass=prepass,
            entry_repos=entry_repos,
        )
        logger.info(
            "[%s] scope resolved entry_repos=%s retrieval_stages=%s",
            run_id,
            entry_repos or [],
            [stage.name for stage in retrieval_stages],
        )
        retrieval_widening_events: list[dict[str, Any]] = []

        # --- Cache lookup BEFORE view materialization ---
        cache = ReviewTraceCache(runtime.cache_dir) if runtime.enable_cache else None
        cache_key = ""
        if cache is not None:
            cache_key = cache.make_key(
                workspace_id=bundle.workspace_id,
                base_sha=bundle.base_sha,
                head_sha=bundle.head_sha,
                target_sha=bundle.target_branch_head_sha,
                merge_sha=bundle.merge_preview_sha,
                workspace_fingerprint=bundle.workspace_fingerprint,
                patch_text=bundle.patch_text,
                policy={
                    "fail_on_severity": runtime.fail_on_severity.value,
                    "llm_endpoint": endpoint_cache_key(request),
                    "max_symbols": runtime.max_symbols,
                    "max_symbol_slots": runtime.max_symbol_slots,
                    "max_total_tool_calls": runtime.max_total_tool_calls,
                    "parse_timeout_s": runtime.parse_timeout_s,
                    "parse_workers": runtime.parse_workers,
                    "max_candidates_per_symbol": runtime.max_candidates_per_symbol,
                    "max_fetch_limit": runtime.max_fetch_limit,
                    "agent_version": AGENT_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "parser_version": PARSER_VERSION,
                },
            )
            try:
                cached = cache.load_report(cache_key)
            except Exception as exc:
                logger.warning("[%s] cache load failed, treating as miss: %s", run_id, exc)
                cached = None
            if cached is not None:
                logger.info("[%s] cache hit for key %s", run_id, cache_key[:16])
                return cached
            logger.info("[%s] cache miss for key %s", run_id, cache_key[:16])

        # --- View materialization (only on cache miss) ---
        logger.info("[%s] preparing views", run_id)
        views = self._prepare_views(
            client=client, runtime_context=bundle,
            lock_root=runtime.cache_dir, created_context_ids=created_context_ids, run_id=run_id,
        )
        logger.info(
            "[%s] views ready baseline=%s head=%s merge=%s warnings=%d",
            run_id,
            views.status.baseline_materialized,
            views.status.head_materialized,
            views.status.merge_preview_materialized,
            len(views.status.warnings),
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

        # --- Step 1: LLM Planner ---
        _check_deadline(deadline, "planner")
        logger.info("[%s] planner step start", run_id)
        planner, exploration, synthesis = self._services_for(request)
        try:
            plan = planner.plan(
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
        except Exception as exc:
            logger.warning("[%s] planner failed, using seed-symbol fallback: %s", run_id, exc)
            plan = ReviewPlan(prioritized_symbols=_fallback_prioritized_symbols(prepass, runtime.max_symbols))
        logger.info(
            "[%s] planner step complete prioritized_symbols=%d sample=%s",
            run_id,
            len(plan.prioritized_symbols),
            plan.prioritized_symbols[:5],
        )

        # --- Step 2: Deterministic base evidence collection ---
        _check_deadline(deadline, "evidence_collection")
        logger.info(
            "[%s] evidence collection start max_calls=%d max_slots=%d",
            run_id,
            budget.max_total_tool_calls,
            budget.max_symbol_slots,
        )
        impacts, symbol_facts, coverage = self._collect_evidence(
            client,
            runtime,
            views,
            prepass,
            plan,
            tool_usage,
            budget,
            deadline,
            retrieval_stages=retrieval_stages,
            widening_events=retrieval_widening_events,
        )
        if run_metadata.prepass_debug is not None:
            run_metadata = run_metadata.model_copy(
                update={
                    "prepass_debug": run_metadata.prepass_debug.model_copy(
                        update={"retrieval_widening_events": retrieval_widening_events[:100]}
                    )
                }
            )
        logger.info(
            "[%s] evidence collection complete impacts=%d symbol_facts=%d coverage_candidates=%d budget_used=%d/%d",
            run_id,
            len(impacts),
            len(symbol_facts),
            coverage.total_candidates,
            budget.calls_used,
            budget.max_total_tool_calls,
        )

        # --- Step 3: Build merge delta signals ---
        merge_delta_signals = _build_merge_delta_signals(symbol_facts, views)

        # Determine if merge analysis is degraded
        merge_analysis_degraded = _is_merge_degraded(bundle, views)

        # --- Step 4: Build initial fact sheet ---
        fact_sheet = ReviewFactSheet(
            changed_files=prepass.changed_files,
            changed_hunk_count=prepass.changed_hunk_count,
            seed_symbols=[s.symbol for s in prepass.seed_symbols],
            suspicious_anchors=prepass.suspicious_anchors,
            changed_declarations=prepass.changed_declarations,
            changed_containers=prepass.changed_containers,
            member_call_sites=prepass.member_call_sites[:100],
            changed_methods=prepass.changed_methods,
            added_call_sites=prepass.added_call_sites,
            removed_call_sites=prepass.removed_call_sites,
            include_macro_config_changes=prepass.include_macro_config_changes,
            symbol_facts=symbol_facts,
            evidence_anchors=[ev for impact in impacts for ev in _anchors_from_impact(impact)][:400],
            coverage=coverage,
            view_contexts=views.status,
            merge_delta_signals=merge_delta_signals,
            merge_analysis_degraded=merge_analysis_degraded,
            warnings=sorted(set(prepass.warnings + views.status.warnings + coverage.warnings)),
        )

        # --- Step 5: Agent-driven follow-up exploration ---
        _check_deadline(deadline, "exploration")
        if budget.remaining_calls > 0 and budget.remaining_slots > 0:
            logger.info(
                "[%s] exploration follow-up start remaining_calls=%d remaining_slots=%d",
                run_id,
                budget.remaining_calls,
                budget.remaining_slots,
            )
            try:
                followup = exploration.explore(
                    fact_sheet=fact_sheet,
                    prepass=prepass,
                    remaining_calls=budget.remaining_calls,
                    remaining_rounds=budget.remaining_slots,
                    client=client,
                    budget=budget,
                    tool_usage=tool_usage,
                    analysis_context=views.head,
                )
            except Exception as exc:
                logger.warning("[%s] exploration follow-up failed: %s", run_id, exc)
                followup = None
            if followup is not None:
                existing_anchors = list(fact_sheet.evidence_anchors)
                existing_anchors.extend(followup.new_evidence[:100])
                warnings = list(fact_sheet.warnings)
                warnings.extend(followup.warnings)
                if followup.summary.strip():
                    existing_anchors.append(
                        EvidenceRef(
                            tool="agent.exploration_followup",
                            description="agent_exploration_summary",
                            snippet=followup.summary[:500],
                        )
                    )
                    warnings.append("agent_followup_executed")
                fact_sheet = fact_sheet.model_copy(update={
                    "evidence_anchors": existing_anchors[:500],
                    "warnings": sorted(set(warnings)),
                })
                logger.info(
                    "[%s] exploration follow-up complete new_evidence=%d warnings=%d",
                    run_id,
                    len(followup.new_evidence),
                    len(followup.warnings),
                )
        else:
            logger.info(
                "[%s] exploration follow-up skipped remaining_calls=%d remaining_slots=%d",
                run_id,
                budget.remaining_calls,
                budget.remaining_slots,
            )

        # --- Step 6: Test impact analysis ---
        _check_deadline(deadline, "test_impact")
        logger.info("[%s] test impact analysis start", run_id)
        test_impact = _analyze_test_impact(
            prepass=prepass, impacts=impacts, client=client, views=views,
            budget=budget, tool_usage=tool_usage,
        )
        logger.info(
            "[%s] test impact analysis complete direct=%d likely=%d confidence=%.2f",
            run_id,
            len(test_impact.directly_impacted_tests),
            len(test_impact.likely_impacted_tests),
            test_impact.confidence,
        )

        # --- Step 7: LLM Synthesis ---
        _check_deadline(deadline, "synthesis")
        logger.info("[%s] synthesis step start", run_id)
        try:
            draft = synthesis.synthesize(
                fact_sheet=fact_sheet,
                fail_threshold=runtime.fail_on_severity.value,
            )
        except ModelContractError as exc:
            logger.error("[%s] synthesis contract failed: %s", run_id, exc)
            return _indeterminate_report(
                workspace_id=bundle.workspace_id,
                reason=f"synthesis_contract_error:{exc}",
                summary=f"Synthesis contract failed: {exc}",
                fail_threshold=runtime.fail_on_severity,
                should_block=True,
                run_id=run_id,
                run_metadata=run_metadata,
            ).model_copy(update={
                "fact_sheet": fact_sheet,
                "test_impact": test_impact,
                "tool_usage": tool_usage,
                "coverage": fact_sheet.coverage,
            })
        except Exception as exc:
            logger.error("[%s] synthesis failed: %s", run_id, exc)
            return _indeterminate_report(
                workspace_id=bundle.workspace_id,
                reason=f"synthesis_failed:{exc}",
                summary=f"Synthesis failed: {exc}",
                fail_threshold=runtime.fail_on_severity,
                should_block=True,
                run_id=run_id,
                run_metadata=run_metadata,
            ).model_copy(update={
                "fact_sheet": fact_sheet,
                "test_impact": test_impact,
                "tool_usage": tool_usage,
                "coverage": fact_sheet.coverage,
            })

        # --- Step 8: Policy gate ---
        location_mapper = FindingLocationMapper(changes=changes, bundle=bundle)
        draft = draft.model_copy(update={"findings": [location_mapper.apply(finding) for finding in draft.findings]})
        final = finalize_report(
            draft=draft,
            fact_sheet=fact_sheet,
            test_impact=test_impact,
            fail_threshold=runtime.fail_on_severity,
            tool_usage=tool_usage,
            workspace_id=bundle.workspace_id,
            run_id=run_id,
            run_metadata=run_metadata,
        )
        logger.info(
            "[%s] review complete findings=%d blocking=%d status=%s confidence=%s",
            run_id,
            len(final.findings),
            final.decision.blocking_findings,
            final.decision.execution_status.value,
            final.decision.review_confidence,
        )
        if cache is not None and cache_key:
            try:
                cache.save(cache_key, {"review_report": final.model_dump(mode="json")})
            except Exception as exc:
                logger.warning("[%s] failed to persist cache: %s", run_id, exc)
        return final

    def _services_for(self, request: ReviewRequest) -> tuple[Any, Any, Any]:
        key = endpoint_cache_key(request)
        if key not in self._service_cache:
            self._service_cache[key] = self._service_factory(request)
        return self._service_cache[key]

    def _prepare_views(
        self,
        *,
        client: CxxtractHttpClient,
        runtime_context,
        lock_root: str,
        created_context_ids: list[str],
        run_id: str,
    ) -> ViewContexts:
        ws_id = runtime_context.workspace_id
        baseline_id = f"{ws_id}:baseline"
        pr_id = str(runtime_context.pr_metadata.get("pr_id", "") or runtime_context.pr_metadata.get("mr_id", "") or "review")
        head_id = f"{ws_id}:pr:{pr_id}:head:{run_id}"
        merge_id = f"{ws_id}:pr:{pr_id}:merge:{run_id}"
        warnings: list[str] = []

        baseline_materialized = False
        head_materialized = False
        merge_materialized = False
        repo_revisions: list[RepoRevisionContext] = list(runtime_context.repo_revisions or [])

        try:
            with _workspace_lock(lock_root=lock_root, workspace_id=ws_id, timeout_s=20.0, run_id=run_id):
                revisions_to_sync = [row for row in repo_revisions if _is_sha(row.target_sha)]
                if revisions_to_sync:
                    baseline_materialized = True
                    for row in revisions_to_sync:
                        try:
                            client.sync_repo(
                                repo_id=row.repo_id,
                                commit_sha=row.target_sha,
                                branch=str(runtime_context.pr_metadata.get("target_branch", "")),
                                force_clean=True,
                            )
                        except Exception as exc:
                            logger.warning("baseline sync failed for %s: %s", row.repo_id, exc)
                            baseline_materialized = False
                            warnings.append(f"baseline_sync_failed:{row.repo_id}:{exc}")
                elif runtime_context.primary_repo_id and _is_sha(runtime_context.target_branch_head_sha):
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
        *,
        retrieval_stages: list[RetrievalStage] | None = None,
        widening_events: list[dict[str, Any]] | None = None,
    ) -> tuple[list[SymbolImpact], list[SymbolFact], CoverageSummary]:
        fallback_symbols = _fallback_prioritized_symbols(prepass, budget.max_symbols)
        symbols = _normalize_investigation_symbols(
            plan.prioritized_symbols + fallback_symbols,
            prepass=prepass,
            max_symbols=budget.max_symbols,
        )
        logger.info(
            "evidence symbol queue prepared planned=%d fallback=%d final=%d",
            len(plan.prioritized_symbols),
            len(fallback_symbols),
            len(symbols),
        )
        impacts: list[SymbolImpact] = []
        facts: list[SymbolFact] = []
        for idx, symbol in enumerate(symbols):
            if not budget.consume_slot() or budget.exhausted:
                logger.debug("evidence collection stopped at symbol %d/%d (budget)", idx, len(symbols))
                break
            _check_deadline(deadline, f"evidence_collection_symbol_{idx}")
            logger.info(
                "evidence symbol start index=%d total=%d symbol=%s remaining_calls=%d",
                idx + 1,
                len(symbols),
                symbol,
                budget.remaining_calls,
            )
            impact = _collect_symbol(
                client,
                runtime,
                views.head,
                symbol,
                tool_usage,
                budget,
                retrieval_stages=retrieval_stages or [],
                widening_events=widening_events,
            )
            impacts.append(impact)
            warnings = list(impact.warnings)
            if views.status.baseline_materialized and not budget.exhausted:
                base_refs, base_edges = _fetch_counts(client, runtime, views.baseline, symbol, impact, tool_usage, budget)
            else:
                base_refs, base_edges = (0, 0)
                warnings.append("baseline_counts_skipped")
            merge_refs, merge_edges = (0, 0)
            if views.merge_preview is not None and views.status.merge_preview_materialized and not budget.exhausted:
                merge_refs, merge_edges = _fetch_counts(client, runtime, views.merge_preview, symbol, impact, tool_usage, budget)
            elif views.merge_preview is not None:
                warnings.append("merge_counts_skipped")
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
                    warnings=sorted(set(warnings)),
                )
            )
            logger.info(
                "evidence symbol complete index=%d total=%d symbol=%s candidates=%d refs=%d edges=%d warnings=%d",
                idx + 1,
                len(symbols),
                symbol,
                len(impact.candidate_file_keys),
                len(impact.references),
                len(impact.call_edges),
                len(warnings),
            )
        return impacts, facts, _coverage(impacts)

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

def _collect_symbol(
    client,
    runtime,
    analysis_context,
    symbol: str,
    tool_usage: list[ToolCallRecord],
    budget: BudgetTracker,
    *,
    retrieval_stages: list[RetrievalStage] | None = None,
    widening_events: list[dict[str, Any]] | None = None,
) -> SymbolImpact:
    rg_hits: list[dict[str, Any]] = []
    rg_candidates: list[str] = []
    listed_candidates: list[str] = []
    deleted: list[str] = []
    discovery_warnings: list[str] = []
    selected_stage_name = ""
    stages = retrieval_stages or [RetrievalStage(name="repo_wide", entry_repos=[], max_repo_hops=0, path_prefixes=[])]

    for idx, stage in enumerate(stages):
        logger.info(
            "candidate discovery stage start symbol=%s stage=%s repos=%s prefixes=%s",
            symbol,
            stage.name,
            stage.entry_repos,
            stage.path_prefixes,
        )
        stage_rg, stage_listed, stage_deleted, stage_warnings = _discover_candidates_for_stage(
            client=client,
            analysis_context=analysis_context,
            symbol=symbol,
            tool_usage=tool_usage,
            budget=budget,
            stage=stage,
        )
        rg_hits = _merge_rg_hits(rg_hits, stage_rg)
        rg_candidates = _dedupe_file_keys(rg_candidates + [str(row.get("file_key", "")).strip() for row in stage_rg if str(row.get("file_key", "")).strip()])
        listed_candidates = _dedupe_file_keys(listed_candidates + stage_listed)
        deleted = _dedupe_file_keys(deleted + stage_deleted)
        discovery_warnings.extend(stage_warnings)
        selected_stage_name = stage.name

        candidate_count = len(_dedupe_file_keys(rg_candidates + listed_candidates))
        logger.info(
            "candidate discovery stage complete symbol=%s stage=%s rg_hits=%d listed=%d candidates=%d deleted=%d warnings=%d",
            symbol,
            stage.name,
            len(stage_rg),
            len(stage_listed),
            candidate_count,
            len(stage_deleted),
            len(stage_warnings),
        )
        if idx > 0 and widening_events is not None:
            widening_events.append(
                {
                    "symbol": symbol,
                    "stage": stage.name,
                    "candidate_count": candidate_count,
                    "entry_repos": list(stage.entry_repos),
                    "path_prefixes": list(stage.path_prefixes),
                }
            )
        if budget.exhausted:
            return _empty_impact(symbol, rg_hits=rg_hits)
        if candidate_count >= 2 or (not stage.path_prefixes and stage.max_repo_hops == 0):
            break

    candidates = _dedupe_file_keys(rg_candidates + listed_candidates)[: budget.max_candidates_per_symbol]

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
            discovery_warnings
            + list(freshness.get("warnings", []) or [])
            + list(parsed.get("parse_warnings", []) or [])
            + list(sym_rows.get("warnings", []) or [])
            + list(ref_rows.get("warnings", []) or [])
            + list(edge_rows.get("warnings", []) or [])
            + (["candidates_truncated"] if len(rg_candidates + listed_candidates) > len(candidates) else [])
            + (["macro_fallback_used"] if macro_summary else [])
            + ([f"retrieval_scope:{selected_stage_name}"] if selected_stage_name else [])
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


def _discover_candidates_for_stage(
    *,
    client,
    analysis_context,
    symbol: str,
    tool_usage: list[ToolCallRecord],
    budget: BudgetTracker,
    stage: RetrievalStage,
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    scope = {
        "entry_repos": stage.entry_repos,
        "max_repo_hops": stage.max_repo_hops,
        "path_prefixes": stage.path_prefixes,
    }
    rg = _call(
        tool_usage,
        budget,
        "collector",
        "explore.rg_search",
        client.explore_rg_search,
        query=symbol,
        mode="symbol",
        analysis_context=analysis_context,
        scope=scope,
        max_hits=min(budget.max_fetch_limit, 200),
        max_files=min(budget.max_candidates_per_symbol, 200),
        timeout_s=min(budget.parse_timeout_s, 60),
        context_lines=1,
    )
    rg_hits = list(rg.get("hits", []) or [])
    if budget.exhausted:
        return rg_hits, [], [], list(rg.get("warnings", []) or [])

    listed = _call(
        tool_usage,
        budget,
        "collector",
        "explore.list_candidates",
        client.explore_list_candidates,
        symbol=symbol,
        max_files=budget.max_candidates_per_symbol,
        include_rg=True,
        entry_repos=stage.entry_repos,
        max_repo_hops=stage.max_repo_hops,
        path_prefixes=stage.path_prefixes,
        analysis_context=analysis_context,
    )
    stage_listed = list(listed.get("candidates", []) or [])
    stage_deleted = list(listed.get("deleted_file_keys", []) or [])
    warnings = list(rg.get("warnings", []) or []) + list(listed.get("warnings", []) or [])
    return rg_hits, stage_listed, stage_deleted, warnings


def _merge_rg_hits(existing: list[dict[str, Any]], new_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for row in existing + new_hits:
        key = (str(row.get("file_key", "")), int(row.get("line", 0) or 0))
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
    return merged[:200]


def _normalize_bundle_and_scope_repos(
    *,
    workspace: dict[str, Any],
    bundle,
    changed_files: list[str],
) -> tuple[Any, list[str], list[str]]:
    """Resolve GitLab repo ids into manifest repo ids and derive changed repo scope."""
    workspace_root = str(workspace.get("root_path", "") or "").strip()
    manifest_path = str(workspace.get("manifest_path", "") or "").strip()
    workspace_repos = {str(repo_id).strip() for repo_id in workspace.get("repos", []) if str(repo_id).strip()}
    warnings: list[str] = []

    if not workspace_root or not manifest_path:
        entry_repos = [bundle.primary_repo_id] if bundle.primary_repo_id in workspace_repos else []
        return bundle, entry_repos, warnings

    try:
        manifest = load_workspace_manifest(manifest_path)
    except Exception as exc:
        warnings.append(f"manifest_load_failed:{exc}")
        entry_repos = [bundle.primary_repo_id] if bundle.primary_repo_id in workspace_repos else []
        return bundle, entry_repos, warnings

    normalized_primary_repo_id = resolve_repo_id_for_project_path(bundle.primary_repo_id, manifest) or bundle.primary_repo_id
    normalized_repo_revisions: list[RepoRevisionContext] = []
    for row in bundle.repo_revisions:
        normalized_repo_id = resolve_repo_id_for_project_path(row.repo_id, manifest) or row.repo_id
        normalized_repo_revisions.append(row.model_copy(update={"repo_id": normalized_repo_id}))

    normalized_per_repo_shas: dict[str, str] = {}
    for repo_id, sha in (bundle.per_repo_shas or {}).items():
        normalized_repo_id = resolve_repo_id_for_project_path(repo_id, manifest) or repo_id
        normalized_per_repo_shas[normalized_repo_id] = sha

    changed_repo_ids: set[str] = set()
    for path in changed_files:
        file_key = resolve_file_key(changed_path=path, workspace_root=workspace_root, manifest=manifest)
        if file_key:
            changed_repo_ids.add(repo_for_file_key(file_key))
    if not changed_repo_ids and normalized_primary_repo_id in manifest.repo_map():
        changed_repo_ids.add(normalized_primary_repo_id)
    elif not changed_repo_ids and bundle.primary_repo_id:
        warnings.append(f"changed_repo_resolution_failed:{bundle.primary_repo_id}")

    updated_bundle = bundle.model_copy(
        update={
            "primary_repo_id": normalized_primary_repo_id,
            "repo_revisions": normalized_repo_revisions,
            "per_repo_shas": normalized_per_repo_shas,
        }
    )
    return updated_bundle, sorted(changed_repo_ids), warnings


def _fallback_prioritized_symbols(prepass, max_symbols: int) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    def _append(values: list[str]) -> None:
        for raw in values:
            text = str(raw or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            ordered.append(text)
            if len(ordered) >= max_symbols:
                return

    _append([decl.symbol for decl in prepass.changed_declarations])
    _append(prepass.changed_containers)
    _append([seed.symbol for seed in prepass.seed_symbols if seed.relevance_tier == "receiver_owned"])
    _append([seed.symbol for seed in prepass.seed_symbols if seed.relevance_tier == "qualified"])
    _append([
        seed.symbol
        for seed in prepass.seed_symbols
        if any(reason.startswith("suspicious_anchor:") for reason in seed.reasons)
    ])
    _append([
        seed.symbol
        for seed in prepass.seed_symbols
        if seed.relevance_tier == "generic_fallback"
        and (
            "changed_declaration:function" in seed.reasons
            or "changed_declaration:method" in seed.reasons
            or "changed_declaration:constructor" in seed.reasons
            or "changed_declaration:destructor" in seed.reasons
            or "qualified_occurrence" in seed.reasons
        )
    ])
    return ordered[:max_symbols]


def _normalize_investigation_symbols(symbols: list[str], *, prepass, max_symbols: int) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in symbols:
        symbol = _canonicalize_investigation_symbol(raw, prepass)
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        normalized.append(symbol)
        if len(normalized) >= max_symbols:
            break
    return normalized


def _canonicalize_investigation_symbol(symbol: str, prepass) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    known = {seed.symbol for seed in prepass.seed_symbols}
    known.update(decl.symbol for decl in prepass.changed_declarations)
    known.update(prepass.changed_containers)
    if text in known:
        return text
    if "->" in text or "." in text:
        for separator in ("->", "."):
            if separator in text:
                left = text.split(separator, 1)[0].strip()
                right = text.split(separator, 1)[1].strip()
                if left in prepass.changed_containers:
                    return left
                if left in known:
                    return left
                if right in known:
                    return right
                return left or right
    return text


def _build_retrieval_stages(
    *,
    workspace: dict[str, Any],
    bundle,
    prepass,
    entry_repos: list[str],
) -> list[RetrievalStage]:
    stages: list[RetrievalStage] = []
    repo_rel_paths = _resolve_repo_relative_changed_paths(
        workspace=workspace,
        changed_files=prepass.changed_files,
        preferred_repos=entry_repos,
    )
    exact_prefixes = _derive_exact_prefixes(repo_rel_paths)
    module_prefixes = _derive_module_prefixes(exact_prefixes)

    if entry_repos and exact_prefixes:
        stages.append(RetrievalStage(name="exact_paths", entry_repos=entry_repos, max_repo_hops=0, path_prefixes=exact_prefixes))
    if entry_repos and module_prefixes and module_prefixes != exact_prefixes:
        stages.append(RetrievalStage(name="module_paths", entry_repos=entry_repos, max_repo_hops=0, path_prefixes=module_prefixes))
    if entry_repos:
        stages.append(RetrievalStage(name="repo_wide", entry_repos=entry_repos, max_repo_hops=0, path_prefixes=[]))

    if _is_interface_like_change(prepass):
        stages.append(RetrievalStage(name="dependency_graph", entry_repos=entry_repos, max_repo_hops=1, path_prefixes=[]))

    if not stages:
        stages.append(RetrievalStage(name="repo_wide", entry_repos=entry_repos, max_repo_hops=0, path_prefixes=[]))
    return stages


def _resolve_repo_relative_changed_paths(
    *,
    workspace: dict[str, Any],
    changed_files: list[str],
    preferred_repos: list[str],
) -> dict[str, list[str]]:
    manifest_path = str(workspace.get("manifest_path", "") or "").strip()
    workspace_root = str(workspace.get("root_path", "") or "").strip()
    if not manifest_path or not workspace_root:
        return {}
    try:
        manifest = load_workspace_manifest(manifest_path)
    except Exception:
        return {}

    resolved: dict[str, list[str]] = defaultdict(list)
    preferred = set(preferred_repos)
    for path in changed_files:
        file_key = resolve_file_key(changed_path=path, workspace_root=workspace_root, manifest=manifest)
        if not file_key or ":" not in file_key:
            continue
        repo_id, rel_path = file_key.split(":", 1)
        if preferred and repo_id not in preferred:
            continue
        if rel_path not in resolved[repo_id]:
            resolved[repo_id].append(rel_path)
    return resolved


def _derive_exact_prefixes(repo_rel_paths: dict[str, list[str]]) -> list[str]:
    prefixes: list[str] = []
    for paths in repo_rel_paths.values():
        for rel_path in paths:
            parent = str(Path(rel_path).parent).replace("\\", "/").strip(".").strip("/")
            if parent and parent not in prefixes:
                prefixes.append(parent)
    return prefixes[:8]


def _derive_module_prefixes(exact_prefixes: list[str]) -> list[str]:
    widened: list[str] = []
    for prefix in exact_prefixes:
        parts = [part for part in prefix.split("/") if part]
        if len(parts) > 4:
            candidate = "/".join(parts[:-1])
        else:
            candidate = prefix
        if candidate and candidate not in widened:
            widened.append(candidate)
    common = _common_prefix_path(widened)
    if common and len(common.split("/")) >= 4:
        return [common]
    return widened[:8]


def _common_prefix_path(paths: list[str]) -> str:
    if not paths:
        return ""
    split_paths = [[part for part in path.split("/") if part] for path in paths]
    common: list[str] = []
    for index in range(min(len(parts) for parts in split_paths)):
        token = split_paths[0][index]
        if all(parts[index] == token for parts in split_paths[1:]):
            common.append(token)
        else:
            break
    return "/".join(common)


def _is_interface_like_change(prepass) -> bool:
    if prepass.include_macro_config_changes:
        return True
    if any(path.endswith((".h", ".hh", ".hpp", ".hxx")) for path in prepass.changed_files):
        return True
    if any(decl.kind in {"class", "struct", "enum"} for decl in prepass.changed_declarations):
        return True
    if any(anchor.kind == "abi_or_dispatch" for anchor in prepass.suspicious_anchors):
        return True
    return False


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
    logger.debug("tool call start tool=%s remaining_calls=%d", tool, budget.remaining_calls)
    try:
        data = fn(**kwargs)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=True, elapsed_ms=elapsed_ms))
        if elapsed_ms >= 1000.0:
            logger.info("tool call complete tool=%s elapsed_ms=%.1f success=true", tool, elapsed_ms)
        else:
            logger.debug("tool call complete tool=%s elapsed_ms=%.1f success=true", tool, elapsed_ms)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("tool call %s failed: %s", tool, exc)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        logger.info("tool call complete tool=%s elapsed_ms=%.1f success=false error=%s", tool, elapsed_ms, exc)
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=False, elapsed_ms=elapsed_ms, note=str(exc)))
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
# Merge degradation check
# ---------------------------------------------------------------------------

def _is_merge_degraded(bundle, views: ViewContexts) -> bool:
    """Returns True when merge-aware analysis is unreliable."""
    if not bundle.merge_preview_sha:
        return False  # no merge requested, not degraded
    if not views.status.merge_preview_materialized:
        return True
    if not bundle.repo_revisions and not bundle.primary_repo_id and not bundle.per_repo_shas:
        return True
    return False


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
) -> ReviewTestImpact:
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
    methods_to_check = prepass.changed_methods[:10] or [
        decl.symbol
        for decl in prepass.changed_declarations
        if decl.kind in {"function", "method", "constructor", "destructor"}
    ][:10]
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

    return ReviewTestImpact(
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

def _policy_gate(
    *,
    report: ReviewReport,
    fact_sheet: ReviewFactSheet,
    test_impact: ReviewTestImpact,
    fail_threshold: Severity,
    tool_usage: list[ToolCallRecord],
    workspace_id: str,
    run_id: str = "",
    run_metadata: RunMetadata | None = None,
    infra_fail_mode: str = "block",
) -> ReviewReport:
    draft = SynthesisDraft(summary=report.summary, findings=report.findings)
    return finalize_report(
        draft=draft,
        fact_sheet=fact_sheet,
        test_impact=test_impact,
        fail_threshold=fail_threshold,
        tool_usage=tool_usage,
        workspace_id=workspace_id,
        run_id=run_id,
        run_metadata=run_metadata,
    )


# ---------------------------------------------------------------------------
# Indeterminate report builder
# ---------------------------------------------------------------------------

def _indeterminate_report(
    *,
    workspace_id: str,
    reason: str,
    summary: str,
    fail_threshold: Severity,
    should_block: bool,
    run_id: str = "",
    run_metadata: RunMetadata | None = None,
) -> ReviewReport:
    """Build a report with INDETERMINATE execution status."""
    return indeterminate_report(
        workspace_id=workspace_id,
        reason=reason,
        summary=summary,
        fail_threshold=fail_threshold,
        should_block=should_block,
        run_id=run_id,
        run_metadata=run_metadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fid(*parts: str) -> str:
    return sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _detect_input_mode(request: ReviewRequest) -> str:
    """Detect the input mode from the request shape."""
    if request.context_bundle is not None:
        meta = request.context_bundle.pr_metadata or {}
        if meta.get("web_url", ""):
            return "gitlab_mr"
        return "context_bundle"
    return "patch_file"


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


class _workspace_lock:
    """Best-effort cross-process lock with TTL-based stale cleanup."""

    def __init__(self, *, lock_root: str, workspace_id: str, timeout_s: float, run_id: str) -> None:
        self._lock_root = Path(lock_root).resolve() / ".workspace_locks"
        self._workspace_id = workspace_id
        self._timeout_s = max(1.0, timeout_s)
        self._run_id = run_id
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in workspace_id)
        self._lock_path = self._lock_root / f"{safe_id}.lock"
        self._fd = -1

    def _try_remove_stale(self) -> bool:
        """If the lock file TTL has expired, remove it."""
        try:
            raw = json.loads(self._lock_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        expires_at = float(raw.get("expires_at", 0.0) or 0.0)
        if expires_at and expires_at > time():
            return False
        logger.info("removing stale lock for workspace %s", self._workspace_id)
        try:
            self._lock_path.unlink(missing_ok=True)
        except Exception:
            return False
        return True

    def __enter__(self):
        self._lock_root.mkdir(parents=True, exist_ok=True)
        deadline = perf_counter() + self._timeout_s
        payload = {
            "workspace_id": self._workspace_id,
            "run_id": self._run_id,
            "created_at": time(),
            "expires_at": time() + self._timeout_s,
        }
        stale_checked = False
        while True:
            try:
                self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, json.dumps(payload, ensure_ascii=True).encode("utf-8"))
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
