"""Policy finalization for review results."""

from __future__ import annotations

from review_agent.models import (
    CoverageSummary,
    EvidenceRef,
    FindingCategory,
    PublishResult,
    ReviewDecision,
    ReviewExecutionStatus,
    ReviewFactSheet,
    ReviewFinding,
    ReviewReport,
    ReviewTestImpact,
    RunMetadata,
    SEVERITY_RANK,
    Severity,
    SynthesisDraft,
    ToolCallRecord,
)


def _confidence_for_fact_sheet(fact_sheet: ReviewFactSheet) -> str:
    ratio = fact_sheet.coverage.verified_ratio
    if ratio >= 0.75 and not fact_sheet.merge_analysis_degraded:
        return "high"
    if ratio >= 0.4:
        return "medium"
    return "low"


def finalize_report(
    *,
    draft: SynthesisDraft,
    fact_sheet: ReviewFactSheet,
    test_impact: ReviewTestImpact,
    fail_threshold: Severity,
    tool_usage: list[ToolCallRecord],
    workspace_id: str,
    run_id: str = "",
    run_metadata: RunMetadata | None = None,
    publish_result: PublishResult | None = None,
) -> ReviewReport:
    findings: list[ReviewFinding] = []
    for finding in draft.findings:
        if finding.severity in {Severity.HIGH, Severity.CRITICAL} and not finding.evidence:
            finding = finding.model_copy(
                update={
                    "severity": Severity.MEDIUM,
                    "tags": sorted(set(finding.tags + ["auto_downgraded_missing_evidence"])),
                }
            )
        findings.append(finding)

    low_cov = fact_sheet.coverage.verified_ratio < 0.4
    if low_cov:
        findings.append(
            ReviewFinding(
                id=f"cov-{fact_sheet.coverage.verified_ratio:.3f}",
                severity=Severity.MEDIUM,
                category=FindingCategory.CONFIDENCE_GAP,
                title="Low semantic coverage reduces confidence",
                impact="Coverage is partial; review confidence is reduced.",
                recommendation="Increase parse/tool budget and rerun review.",
                evidence=[
                    EvidenceRef(
                        tool="policy_gate",
                        description=f"verified_ratio={fact_sheet.coverage.verified_ratio:.3f}",
                    )
                ],
                confidence=0.95,
                tags=["coverage_policy"],
            )
        )

    semantic_bootstrap_failed = "semantic_bootstrap_failed" in set(fact_sheet.warnings)
    if semantic_bootstrap_failed:
        findings.append(
            ReviewFinding(
                id=f"bootstrap-failed-{workspace_id}",
                severity=Severity.HIGH,
                category=FindingCategory.CONFIDENCE_GAP,
                title="Semantic bootstrap failed for changed files",
                impact="The review could not bootstrap semantic verification from the changed files, so findings may be incomplete.",
                recommendation="Verify changed files are resolvable and parseable in the head context, then rerun the review.",
                evidence=[EvidenceRef(tool="policy_gate", description="semantic_bootstrap_failed")],
                confidence=0.98,
                tags=["semantic_bootstrap_failed"],
            )
        )

    if fact_sheet.merge_analysis_degraded:
        findings.append(
            ReviewFinding(
                id=f"merge-degraded-{workspace_id}",
                severity=Severity.MEDIUM,
                category=FindingCategory.CONFIDENCE_GAP,
                title="Merge-aware analysis is degraded",
                impact="Merge-preview comparison may be unreliable; merge-specific findings were constrained.",
                recommendation="Provide complete repo revisions for accurate merge analysis.",
                evidence=[EvidenceRef(tool="policy_gate", description="merge_analysis_degraded")],
                confidence=0.95,
                tags=["merge_degraded"],
            )
        )

    findings = sorted(findings, key=lambda f: (-SEVERITY_RANK[f.severity], f.title, f.id))
    blocking = len([f for f in findings if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[fail_threshold]])
    exec_status = ReviewExecutionStatus.BLOCK if blocking > 0 else ReviewExecutionStatus.PASS
    indeterminate_reason = ""
    should_block = blocking > 0
    if semantic_bootstrap_failed:
        exec_status = ReviewExecutionStatus.INDETERMINATE
        indeterminate_reason = "semantic_bootstrap_failed"
        should_block = True
    summary = draft.summary.strip() or f"Reviewed {len(fact_sheet.changed_files)} files; findings={len(findings)}."
    if semantic_bootstrap_failed and "semantic bootstrap" not in summary.lower():
        summary = f"{summary} Semantic bootstrap failed for the changed files, so verification coverage is incomplete."

    decision = ReviewDecision(
        fail_threshold=fail_threshold,
        blocking_findings=blocking,
        should_block=should_block,
        execution_status=exec_status,
        indeterminate_reason=indeterminate_reason,
        review_confidence=_confidence_for_fact_sheet(fact_sheet),
    )

    return ReviewReport(
        workspace_id=workspace_id,
        summary=summary,
        findings=findings,
        coverage=fact_sheet.coverage,
        decision=decision,
        tool_usage=tool_usage,
        fact_sheet=fact_sheet,
        test_impact=test_impact,
        run_metadata=run_metadata,
        publish_result=publish_result,
        run_id=run_id,
    )


def indeterminate_report(
    *,
    workspace_id: str,
    reason: str,
    summary: str,
    fail_threshold: Severity,
    should_block: bool,
    run_id: str = "",
    run_metadata: RunMetadata | None = None,
    publish_result: PublishResult | None = None,
) -> ReviewReport:
    """Build a report with INDETERMINATE execution status."""
    return ReviewReport(
        workspace_id=workspace_id,
        summary=summary,
        findings=[],
        coverage=CoverageSummary(),
        decision=ReviewDecision(
            fail_threshold=fail_threshold,
            blocking_findings=0,
            should_block=should_block,
            execution_status=ReviewExecutionStatus.INDETERMINATE,
            indeterminate_reason=reason,
            review_confidence="low",
        ),
        tool_usage=[],
        run_id=run_id,
        run_metadata=run_metadata,
        publish_result=publish_result,
    )
