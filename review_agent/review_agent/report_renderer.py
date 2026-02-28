"""Render review reports into markdown."""

from __future__ import annotations

import logging

from review_agent.models import ReviewExecutionStatus, ReviewReport

logger = logging.getLogger("review_agent.report_renderer")


def render_markdown(report: ReviewReport) -> str:
    """Render a human-readable markdown report."""
    lines: list[str] = []
    lines.append("# CXXtract2 Semantic PR Review")
    lines.append("")
    lines.append(f"- Workspace: `{report.workspace_id}`")
    lines.append(f"- Generated at: `{report.generated_at}`")
    if report.run_id:
        lines.append(f"- Run ID: `{report.run_id}`")

    exec_status = report.decision.execution_status
    if exec_status == ReviewExecutionStatus.INDETERMINATE:
        lines.append(f"- Decision: `INDETERMINATE` ({report.decision.indeterminate_reason})")
    else:
        lines.append(f"- Decision: `{'BLOCK' if report.decision.should_block else 'PASS'}`")
    lines.append(
        f"- Threshold: `{report.decision.fail_threshold.value}` (blocking findings: {report.decision.blocking_findings})"
    )
    lines.append(f"- Execution status: `{exec_status.value}`")
    lines.append(f"- Review confidence: `{report.decision.review_confidence}`")

    if report.run_metadata:
        lines.append(f"- Agent version: `{report.run_metadata.agent_version}`")
        lines.append(f"- Input mode: `{report.run_metadata.input_mode}`")
        if report.run_metadata.prepass_debug is not None:
            lines.append(
                f"- Pre-pass debug: `{len(report.run_metadata.prepass_debug.ranked_seed_candidates)}` seeds / "
                f"`{len(report.run_metadata.prepass_debug.changed_declarations)}` declarations"
            )

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(report.summary)
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if not report.findings:
        lines.append("No findings.")
    else:
        for finding in report.findings:
            lines.append(f"### [{finding.severity.value.upper()}] {finding.title}")
            lines.append("")
            lines.append(f"- Category: `{finding.category.value}`")
            lines.append(f"- Impact: {finding.impact}")
            lines.append(f"- Recommendation: {finding.recommendation}")
            lines.append(f"- Confidence: `{finding.confidence:.2f}`")
            if finding.related_symbols:
                lines.append(f"- Symbols: `{', '.join(finding.related_symbols)}`")
            if finding.related_repos:
                lines.append(f"- Repos: `{', '.join(finding.related_repos)}`")
            if finding.location is not None:
                lines.append(
                    f"- Location: `{finding.location.path}:{finding.location.line}`"
                    f" ({finding.location.side})"
                )
            elif finding.diff_path and finding.diff_line > 0:
                lines.append(f"- Location: `{finding.diff_path}:{finding.diff_line}`")
            if finding.evidence:
                lines.append("- Evidence:")
                for ev in finding.evidence:
                    desc = ev.description or "evidence"
                    loc = ""
                    if ev.file_key:
                        loc = ev.file_key
                        if ev.line > 0:
                            loc = f"{loc}:{ev.line}"
                    elif ev.symbol:
                        loc = ev.symbol
                    lines.append(f"  - `{ev.tool}` {loc} {desc}".rstrip())
    lines.append("")

    if report.fact_sheet is not None:
        lines.append("## Fact Sheet")
        lines.append("")
        lines.append(f"- Changed files: `{len(report.fact_sheet.changed_files)}`")
        lines.append(f"- Changed hunks: `{report.fact_sheet.changed_hunk_count}`")
        lines.append(f"- Seed symbols: `{len(report.fact_sheet.seed_symbols)}`")
        lines.append(f"- Suspicious anchors: `{len(report.fact_sheet.suspicious_anchors)}`")
        lines.append(f"- Changed declarations: `{len(report.fact_sheet.changed_declarations)}`")
        lines.append(f"- Member call sites: `{len(report.fact_sheet.member_call_sites)}`")

        if report.fact_sheet.merge_analysis_degraded:
            lines.append("- ⚠ Merge-aware analysis: `degraded`")

        # Merge delta signals
        if report.fact_sheet.merge_delta_signals:
            lines.append("")
            lines.append("### Merge Delta Signals")
            lines.append("")
            for sig in report.fact_sheet.merge_delta_signals[:20]:
                sym = sig.get("symbol", "?")
                risk = sig.get("risk", "unknown")
                merge_ref = sig.get("merge_ref_delta_vs_head", 0)
                merge_edge = sig.get("merge_edge_delta_vs_head", 0)
                lines.append(
                    f"- `{sym}` -- {risk} "
                    f"(merge-vs-head refs: {merge_ref:+d}, edges: {merge_edge:+d})"
                )

        if report.fact_sheet.warnings:
            lines.append("- Fact-sheet warnings:")
            for warn in report.fact_sheet.warnings[:20]:
                lines.append(f"  - {warn}")
        lines.append("")

    if report.run_metadata and report.run_metadata.prepass_debug is not None:
        debug = report.run_metadata.prepass_debug
        lines.append("## Pre-pass Debug")
        lines.append("")
        if debug.ranked_seed_candidates:
            lines.append("- Top ranked seeds:")
            for seed in debug.ranked_seed_candidates[:10]:
                lines.append(
                    f"  - `{seed.symbol}` [{seed.relevance_tier}] score={seed.score:.2f} "
                    f"reasons={', '.join(seed.reasons[:3])}"
                )
        if debug.diff_excerpt_reasons:
            lines.append("- Diff excerpt reasons:")
            for reason in debug.diff_excerpt_reasons[:10]:
                lines.append(f"  - {reason}")
        if debug.retrieval_widening_events:
            lines.append("- Retrieval widening:")
            for event in debug.retrieval_widening_events[:10]:
                lines.append(
                    f"  - `{event.get('symbol', '?')}` -> `{event.get('stage', '?')}` "
                    f"(candidates={event.get('candidate_count', 0)})"
                )
        lines.append("")

    if report.test_impact is not None:
        lines.append("## Test Impact")
        lines.append("")
        lines.append(f"- Directly impacted tests: `{len(report.test_impact.directly_impacted_tests)}`")
        lines.append(f"- Likely impacted tests: `{len(report.test_impact.likely_impacted_tests)}`")
        if report.test_impact.suggested_scopes:
            lines.append(f"- Suggested scopes: `{', '.join(report.test_impact.suggested_scopes)}`")
        if report.test_impact.rationale:
            lines.append("- Rationale:")
            for row in report.test_impact.rationale[:10]:
                lines.append(f"  - {row}")

        # Test dependency edges
        if report.test_impact.test_dependency_edges:
            lines.append("")
            lines.append("### Test Dependency Edges")
            lines.append("")
            for edge in report.test_impact.test_dependency_edges[:30]:
                sym = edge.get("symbol", "?")
                tf = edge.get("test_file", "?")
                src = edge.get("source", "?")
                ln = edge.get("line", 0)
                lines.append(f"- `{sym}` -> `{tf}` ({src}, line {ln})")

        lines.append("")

    if report.publish_result is not None:
        lines.append("## Publish Result")
        lines.append("")
        lines.append(f"- Provider: `{report.publish_result.provider}`")
        lines.append(f"- Summary posted: `{report.publish_result.summary_posted}`")
        lines.append(f"- Inline comments posted: `{report.publish_result.inline_comments_posted}`")
        if report.publish_result.warnings:
            lines.append("- Publish warnings:")
            for warn in report.publish_result.warnings[:20]:
                lines.append(f"  - {warn}")
        lines.append("")

    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Verified ratio: `{report.coverage.verified_ratio:.2%}`")
    lines.append(f"- Total candidates: `{report.coverage.total_candidates}`")
    lines.append(f"- Verified files: `{len(report.coverage.verified_files)}`")
    lines.append(f"- Stale files: `{len(report.coverage.stale_files)}`")
    lines.append(f"- Unparsed files: `{len(report.coverage.unparsed_files)}`")
    if report.coverage.warnings:
        lines.append("- Coverage warnings:")
        for warn in report.coverage.warnings[:20]:
            lines.append(f"  - {warn}")
    lines.append("")

    lines.append("## Tool Usage")
    lines.append("")
    if not report.tool_usage:
        lines.append("No tool calls recorded.")
    else:
        for row in report.tool_usage:
            state = "ok" if row.success else "error"
            note = f" ({row.note})" if row.note else ""
            lines.append(f"- `{row.skill}` -> `{row.tool}` [{state}] {row.elapsed_ms:.1f}ms{note}")
    lines.append("")
    return "\n".join(lines)
