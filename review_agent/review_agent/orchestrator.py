"""Context-driven orchestration for semantic PR review."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
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
    PLANNER_SYSTEM_PROMPT,
    SYNTHESIS_SYSTEM_PROMPT,
    build_planner_prompt,
    build_synthesis_prompt,
)
from review_agent.report_renderer import render_markdown
from review_agent.review_cache import ReviewTraceCache
from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient


@dataclass
class ViewContexts:
    baseline: dict[str, Any]
    head: dict[str, Any]
    merge_preview: dict[str, Any] | None
    status: ViewContextMaterialization


def build_planner_agent(model_name: str):
    from pydantic_ai import Agent  # type: ignore

    return Agent(model_name, system_prompt=PLANNER_SYSTEM_PROMPT, result_type=ReviewPlan)


def build_synthesis_agent(model_name: str):
    from pydantic_ai import Agent  # type: ignore

    return Agent(model_name, system_prompt=SYNTHESIS_SYSTEM_PROMPT, result_type=ReviewReport)


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
        client = self._client or CxxtractHttpClient(
            base_url=request.cxxtract_base_url,
            workspace_id=request.workspace_id,
            timeout_s=90.0,
        )
        workspace = client.workspace_info()
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
        views = self._prepare_views(client=client, runtime_context=bundle, lock_root=runtime.cache_dir)
        tool_usage: list[ToolCallRecord] = []

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
                    "max_symbols": runtime.max_symbols,
                    "max_tool_rounds": runtime.max_tool_rounds,
                    "max_total_tool_calls": runtime.max_total_tool_calls,
                    "parse_timeout_s": runtime.parse_timeout_s,
                    "parse_workers": runtime.parse_workers,
                    "max_candidates_per_symbol": runtime.max_candidates_per_symbol,
                    "max_fetch_limit": runtime.max_fetch_limit,
                },
            )
            cached = cache.load_report(cache_key)
            if cached is not None:
                return cached

        planner = self._planner_for(request.llm_model)
        planner_prompt = build_planner_prompt(
            context=bundle,
            prepass=prepass,
            budgets={
                "max_symbols": runtime.max_symbols,
                "max_tool_rounds": runtime.max_tool_rounds,
                "max_total_tool_calls": runtime.max_total_tool_calls,
                "parse_timeout_s": runtime.parse_timeout_s,
                "parse_workers": runtime.parse_workers,
                "max_candidates_per_symbol": runtime.max_candidates_per_symbol,
                "max_fetch_limit": runtime.max_fetch_limit,
            },
        )
        try:
            plan = planner.run_sync(planner_prompt).data
        except Exception:
            plan = ReviewPlan(prioritized_symbols=[s.symbol for s in prepass.seed_symbols[: runtime.max_symbols]])

        impacts, symbol_facts, coverage = self._collect_evidence(client, runtime, views, prepass, plan, tool_usage)
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
            warnings=sorted(set(prepass.warnings + views.status.warnings + coverage.warnings)),
        )
        test_impact = _build_test_impact(prepass=prepass, impacts=impacts)

        synth = self._synthesis_for(request.llm_model)
        try:
            draft = synth.run_sync(
                build_synthesis_prompt(fact_sheet=fact_sheet, fail_threshold=runtime.fail_on_severity.value)
            ).data
        except Exception:
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

    def _prepare_views(self, *, client: CxxtractHttpClient, runtime_context, lock_root: str) -> ViewContexts:
        ws_id = runtime_context.workspace_id
        baseline_id = f"{ws_id}:baseline"
        pr_id = str(runtime_context.pr_metadata.get("pr_id", "") or runtime_context.pr_metadata.get("mr_id", "") or "review")
        head_id = f"{ws_id}:pr:{pr_id}:head"
        merge_id = f"{ws_id}:pr:{pr_id}:merge"
        warnings: list[str] = []

        baseline_materialized = True
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
                    except Exception as exc:
                        warnings.append(f"baseline_sync_failed:{exc}")

                if _is_sha(runtime_context.head_sha):
                    try:
                        created = client.context_create_pr_overlay(
                            pr_id=pr_id,
                            base_ref=runtime_context.target_branch_head_sha or runtime_context.base_sha,
                            head_ref=runtime_context.head_sha,
                            context_id=head_id,
                        )
                        head_id = str(created.get("context_id", head_id))
                        head_materialized = True
                        if bool(created.get("partial_overlay", False)):
                            warnings.append("head_partial_overlay")
                    except Exception as exc:
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
                        merge_materialized = True
                        if bool(created.get("partial_overlay", False)):
                            warnings.append("merge_partial_overlay")
                    except Exception as exc:
                        warnings.append(f"merge_preview_context_failed:{exc}")
                elif runtime_context.merge_preview_sha:
                    warnings.append("merge_preview_sha_invalid")
                else:
                    warnings.append("merge_preview_not_materialized")
        except TimeoutError as exc:
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
    ) -> tuple[list[SymbolImpact], list[SymbolFact], CoverageSummary]:
        symbols = list(dict.fromkeys(plan.prioritized_symbols + [s.symbol for s in prepass.seed_symbols]))[: runtime.max_symbols]
        impacts: list[SymbolImpact] = []
        facts: list[SymbolFact] = []
        for idx, symbol in enumerate(symbols):
            if idx >= runtime.max_tool_rounds or len(tool_usage) >= runtime.max_total_tool_calls:
                break
            impact = _collect_symbol(client, runtime, views.head, symbol, tool_usage)
            impacts.append(impact)
            base_refs, base_edges = _fetch_counts(client, runtime, views.baseline, symbol, impact, tool_usage)
            merge_refs, merge_edges = (0, 0)
            if views.merge_preview is not None:
                merge_refs, merge_edges = _fetch_counts(client, runtime, views.merge_preview, symbol, impact, tool_usage)
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

    @staticmethod
    def write_report_files(report: ReviewReport, out_dir: str | Path) -> tuple[Path, Path]:
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "review_report.md"
        json_path = out / "review_report.json"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path


def _collect_symbol(client, runtime, analysis_context, symbol: str, tool_usage: list[ToolCallRecord]) -> SymbolImpact:
    rg = _call(
        tool_usage,
        runtime.max_total_tool_calls,
        "collector",
        "explore.rg_search",
        client.explore_rg_search,
        query=symbol,
        mode="symbol",
        analysis_context=analysis_context,
        max_hits=min(runtime.max_fetch_limit, 200),
        max_files=min(runtime.max_candidates_per_symbol, 200),
        timeout_s=min(runtime.parse_timeout_s, 60),
        context_lines=1,
    )
    rg_hits = list(rg.get("hits", []) or [])
    rg_candidates = [str(row.get("file_key", "")).strip() for row in rg_hits if str(row.get("file_key", "")).strip()]

    listed = _call(
        tool_usage,
        runtime.max_total_tool_calls,
        "collector",
        "explore.list_candidates",
        client.explore_list_candidates,
        symbol=symbol,
        max_files=runtime.max_candidates_per_symbol,
        include_rg=True,
        analysis_context=analysis_context,
    )
    listed_candidates = list(listed.get("candidates", []) or [])
    candidates = _dedupe_file_keys(rg_candidates + listed_candidates)[: runtime.max_candidates_per_symbol]
    deleted = _dedupe_file_keys(list(listed.get("deleted_file_keys", []) or []))

    read_contexts: list[dict[str, Any]] = []
    for hit in rg_hits[:2]:
        file_key = str(hit.get("file_key", "")).strip()
        if not file_key:
            continue
        line = int(hit.get("line", 1) or 1)
        row = _call(
            tool_usage,
            runtime.max_total_tool_calls,
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
    if not read_contexts and candidates:
        row = _call(
            tool_usage,
            runtime.max_total_tool_calls,
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

    freshness = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.classify_freshness", client.explore_classify_freshness, candidate_file_keys=candidates, max_files=max(1, len(candidates)), analysis_context=analysis_context)
    stale = list(freshness.get("stale", []) or [])
    fresh = list(freshness.get("fresh", []) or [])
    unparsed = list(freshness.get("unparsed", []) or [])
    parsed = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.parse_file", client.explore_parse_file, file_keys=stale, max_parse_workers=runtime.parse_workers, timeout_s=runtime.parse_timeout_s, skip_if_fresh=True, analysis_context=analysis_context)
    parsed_keys = list(parsed.get("parsed_file_keys", []) or [])
    parse_failed = list(parsed.get("failed_file_keys", []) or [])
    sym_rows = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.fetch_symbols", client.explore_fetch_symbols, symbol=symbol, candidate_file_keys=candidates, excluded_file_keys=deleted, limit=runtime.max_fetch_limit, analysis_context=analysis_context)
    ref_rows = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.fetch_references", client.explore_fetch_references, symbol=symbol, candidate_file_keys=candidates, excluded_file_keys=deleted, limit=runtime.max_fetch_limit, analysis_context=analysis_context)
    edge_rows = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.fetch_call_edges", client.explore_fetch_call_edges, symbol=symbol, direction="both", candidate_file_keys=candidates, excluded_file_keys=deleted, limit=runtime.max_fetch_limit, analysis_context=analysis_context)
    conf = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.get_confidence", client.explore_get_confidence, verified_files=sorted(set(fresh + parsed_keys)), stale_files=sorted(set(parse_failed)), unparsed_files=sorted(set(unparsed + list(parsed.get("unparsed_file_keys", []) or []))), warnings=[], overlay_mode=str(freshness.get("overlay_mode", "sparse")))
    macro_summary = ""
    if not list(ref_rows.get("references", []) or []) and not list(edge_rows.get("edges", []) or []):
        macro = _call(
            tool_usage,
            runtime.max_total_tool_calls,
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


def _fetch_counts(client, runtime, analysis_context, symbol: str, impact: SymbolImpact, tool_usage: list[ToolCallRecord]) -> tuple[int, int]:
    refs = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.fetch_references", client.explore_fetch_references, symbol=symbol, candidate_file_keys=impact.candidate_file_keys, excluded_file_keys=impact.deleted_file_keys, limit=runtime.max_fetch_limit, analysis_context=analysis_context)
    edges = _call(tool_usage, runtime.max_total_tool_calls, "collector", "explore.fetch_call_edges", client.explore_fetch_call_edges, symbol=symbol, direction="both", candidate_file_keys=impact.candidate_file_keys, excluded_file_keys=impact.deleted_file_keys, limit=runtime.max_fetch_limit, analysis_context=analysis_context)
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


def _call(tool_usage: list[ToolCallRecord], max_calls: int, skill: str, tool: str, fn, **kwargs: Any) -> dict[str, Any]:
    if len(tool_usage) >= max_calls:
        return {"warnings": [f"max_total_tool_calls_reached:{max_calls}"]}
    t0 = perf_counter()
    try:
        data = fn(**kwargs)
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=True, elapsed_ms=(perf_counter() - t0) * 1000.0))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        tool_usage.append(ToolCallRecord(skill=skill, tool=tool, success=False, elapsed_ms=(perf_counter() - t0) * 1000.0, note=str(exc)))
        return {"warnings": [f"{tool}_failed:{exc}"]}


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


def _build_test_impact(*, prepass, impacts: list[SymbolImpact]) -> TestImpact:
    direct = sorted({p for p in prepass.changed_files if _looks_like_test_path(p)})
    likely: set[str] = set()
    for impact in impacts:
        for row in impact.references + impact.call_edges:
            fk = str(row.get("file_key", ""))
            rel = fk.split(":", 1)[1] if ":" in fk else ""
            if rel and _looks_like_test_path(rel):
                likely.add(rel)
    scopes = ["smoke"]
    if direct:
        scopes.append("unit")
    if likely:
        scopes.append("integration")
    if any(len(impact.repos_involved) > 1 for impact in impacts):
        scopes.append("e2e")
    return TestImpact(
        directly_impacted_tests=direct[:200],
        likely_impacted_tests=sorted([p for p in likely if p not in set(direct)])[:300],
        suggested_scopes=list(dict.fromkeys(scopes)),
        rationale=["deterministic_test_impact"],
    )


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


def _fid(*parts: str) -> str:
    return sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _looks_like_test_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return "/test/" in p or "/tests/" in p or p.endswith("_test.cpp") or p.endswith("_tests.cpp") or p.endswith("_unittest.cpp") or "/ut/" in p or "/it/" in p or "/e2e/" in p


class _workspace_lock:
    """Best-effort cross-process lock to avoid concurrent sync/context collisions."""

    def __init__(self, *, lock_root: str, workspace_id: str, timeout_s: float) -> None:
        self._lock_root = Path(lock_root).resolve() / ".workspace_locks"
        self._workspace_id = workspace_id
        self._timeout_s = max(1.0, timeout_s)
        safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in workspace_id)
        self._lock_path = self._lock_root / f"{safe_id}.lock"
        self._fd = -1

    def __enter__(self):
        self._lock_root.mkdir(parents=True, exist_ok=True)
        deadline = perf_counter() + self._timeout_s
        while True:
            try:
                self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
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
