"""Render review reports into markdown."""

from __future__ import annotations

from review_agent.models import ReviewReport


def render_markdown(report: ReviewReport) -> str:
    """Render a human-readable markdown report."""
    lines: list[str] = []
    lines.append("# CXXtract2 Semantic PR Review")
    lines.append("")
    lines.append(f"- Workspace: `{report.workspace_id}`")
    lines.append(f"- Generated at: `{report.generated_at}`")
    lines.append(f"- Decision: `{'BLOCK' if report.decision.should_block else 'PASS'}`")
    lines.append(
        f"- Threshold: `{report.decision.fail_threshold.value}` (blocking findings: {report.decision.blocking_findings})"
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

