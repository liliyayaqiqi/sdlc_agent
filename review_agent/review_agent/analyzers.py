"""Deterministic analyzers for risk-focused review findings."""

from __future__ import annotations

import re
from hashlib import sha1

from review_agent.models import (
    EvidenceRef,
    FindingCategory,
    PatchChange,
    ReviewFinding,
    Severity,
    SymbolImpact,
)

_SIDE_EFFECT_PATTERNS = [
    (re.compile(r"\bstatic\b"), "static_state_change"),
    (re.compile(r"\bglobal\b"), "global_state_change"),
    (re.compile(r"\bmutex\b|\block_guard\b|\bunique_lock\b"), "locking_behavior_change"),
    (re.compile(r"\bstd::atomic\b|\bmemory_order\b"), "atomic_behavior_change"),
    (re.compile(r"\bdelete\b|\bfree\s*\("), "lifetime_management_change"),
]


def analyze_hidden_side_effects(
    *,
    patch_changes: list[PatchChange],
    impacts: list[SymbolImpact],
) -> list[ReviewFinding]:
    """Flag non-obvious behavior changes from patch and semantic context."""
    findings: list[ReviewFinding] = []

    for change in patch_changes:
        for hunk in change.hunks:
            for line in hunk.lines:
                if line.kind != "add":
                    continue
                lowered = line.text.lower()
                for pattern, tag in _SIDE_EFFECT_PATTERNS:
                    if not pattern.search(lowered):
                        continue
                    fid = _finding_id("side", change.effective_path, tag, str(line.new_line))
                    findings.append(
                        ReviewFinding(
                            id=fid,
                            severity=Severity.MEDIUM,
                            category=FindingCategory.HIDDEN_SIDE_EFFECT,
                            title=f"Potential hidden side effect in {change.effective_path}",
                            impact=(
                                "Patch introduces behavior that can alter shared/runtime state in non-local call paths."
                            ),
                            recommendation=(
                                "Verify thread-safety and lifecycle invariants, and add targeted regression tests around this path."
                            ),
                            evidence=[
                                EvidenceRef(
                                    tool="patch",
                                    description=f"Matched pattern: {tag}",
                                    line=line.new_line,
                                    snippet=line.text[:240],
                                )
                            ],
                            confidence=0.68,
                            related_symbols=[],
                            related_repos=[],
                            tags=[tag],
                        )
                    )

    for impact in impacts:
        ref_count = len(impact.references)
        edge_count = len(impact.call_edges)
        if ref_count + edge_count < 80:
            continue
        findings.append(
            ReviewFinding(
                id=_finding_id("blast", impact.symbol, str(ref_count), str(edge_count)),
                severity=Severity.HIGH,
                category=FindingCategory.HIDDEN_SIDE_EFFECT,
                title=f"Large semantic blast radius for symbol `{impact.symbol}`",
                impact=(
                    f"Changes can affect many runtime paths ({ref_count} references, {edge_count} call edges), increasing hidden side-effect risk."
                ),
                recommendation="Prioritize focused integration tests around high-fan-in callers before merge.",
                evidence=[
                    EvidenceRef(
                        tool="explore.fetch_references",
                        description=f"Reference rows: {ref_count}",
                        symbol=impact.symbol,
                    ),
                    EvidenceRef(
                        tool="explore.fetch_call_edges",
                        description=f"Call-edge rows: {edge_count}",
                        symbol=impact.symbol,
                    ),
                ],
                confidence=0.81,
                related_symbols=[impact.symbol],
                related_repos=impact.repos_involved,
                tags=["blast_radius"],
            )
        )
    return findings


def analyze_cross_repo_breakage(
    *,
    impacts: list[SymbolImpact],
    cross_repo_signals: list[dict],
) -> list[ReviewFinding]:
    """Flag potential external breakages induced by changed symbols."""
    findings: list[ReviewFinding] = []
    by_symbol = {impact.symbol: impact for impact in impacts}

    for row in cross_repo_signals:
        symbol = str(row.get("symbol", ""))
        external_repos = list(row.get("external_repos", []))
        if not external_repos:
            continue
        incoming_refs = int(row.get("incoming_ref_count", 0))
        incoming_edges = int(row.get("incoming_edge_count", 0))
        severity = Severity.HIGH if (incoming_refs + incoming_edges) >= 20 else Severity.MEDIUM
        impact = by_symbol.get(symbol)
        evidence: list[EvidenceRef] = [
            EvidenceRef(
                tool="skill_cross_repo_breakage",
                description=f"External repos impacted: {', '.join(external_repos)}",
                symbol=symbol,
            )
        ]
        if impact and impact.references:
            r0 = impact.references[0]
            evidence.append(
                EvidenceRef(
                    tool="explore.fetch_references",
                    file_key=str(r0.get("file_key", "")),
                    line=int(r0.get("line", 0) or 0),
                    symbol=symbol,
                    description="Representative incoming reference.",
                )
            )
        findings.append(
            ReviewFinding(
                id=_finding_id("xrepo", symbol, ",".join(external_repos)),
                severity=severity,
                category=FindingCategory.CROSS_REPO_BREAKAGE,
                title=f"Potential cross-repo breakage for `{symbol}`",
                impact=(
                    f"Symbol usage spans external repos ({', '.join(external_repos)}); behavior/API changes may break dependent repositories."
                ),
                recommendation=(
                    "Run dependent-repo build/test matrix and verify callers against updated symbol contract."
                ),
                evidence=evidence,
                confidence=0.74,
                related_symbols=[symbol],
                related_repos=external_repos,
                tags=["cross_repo", "api_compat"],
            )
        )
    return findings


def analyze_architecture_risks(*, architecture_signals: list[dict]) -> list[ReviewFinding]:
    """Flag dependency-direction and coupling risks."""
    findings: list[ReviewFinding] = []
    for row in architecture_signals:
        symbol = str(row.get("symbol", ""))
        caller_repo = str(row.get("caller_repo", ""))
        owner_repo = str(row.get("owner_repo", ""))
        if not symbol or not caller_repo or not owner_repo:
            continue
        findings.append(
            ReviewFinding(
                id=_finding_id("arch", symbol, caller_repo, owner_repo),
                severity=Severity.HIGH,
                category=FindingCategory.ARCHITECTURE_RISK,
                title=f"Dependency-direction risk: {caller_repo} -> {owner_repo}",
                impact=(
                    "Semantic usage appears across a dependency edge not declared in workspace manifest, increasing architecture drift risk."
                ),
                recommendation=(
                    "Either enforce boundary by refactoring call sites or explicitly model dependency with architectural sign-off."
                ),
                evidence=[
                    EvidenceRef(
                        tool="skill_architecture_risk",
                        description="Dependency-direction violation inferred from semantic evidence.",
                        symbol=symbol,
                    )
                ],
                confidence=0.69,
                related_symbols=[symbol],
                related_repos=[caller_repo, owner_repo],
                tags=["dependency_direction"],
            )
        )
    return findings


def analyze_confidence_gaps(
    *,
    impacts: list[SymbolImpact],
    loop_warnings: list[str],
) -> list[ReviewFinding]:
    """Surface partial coverage and parse uncertainty explicitly."""
    findings: list[ReviewFinding] = []
    for impact in impacts:
        confidence = dict(impact.confidence or {})
        verified_ratio = float(confidence.get("verified_ratio", 0.0) or 0.0)
        total_candidates = int(confidence.get("total_candidates", 0) or 0)
        if total_candidates <= 0:
            continue
        if verified_ratio >= 0.8 and not impact.unparsed and not impact.parse_failed_file_keys:
            continue
        sev = Severity.MEDIUM if verified_ratio >= 0.4 else Severity.HIGH
        findings.append(
            ReviewFinding(
                id=_finding_id("confidence", impact.symbol, f"{verified_ratio:.3f}"),
                severity=sev,
                category=FindingCategory.CONFIDENCE_GAP,
                title=f"Low semantic verification coverage for `{impact.symbol}`",
                impact=(
                    f"Only {verified_ratio:.1%} of candidates were semantically verified; hidden breakages may remain."
                ),
                recommendation=(
                    "Increase parse coverage or run focused follow-up review on unparsed/failed candidate files."
                ),
                evidence=[
                    EvidenceRef(
                        tool="explore.get_confidence",
                        description=f"verified_ratio={verified_ratio:.3f}, total_candidates={total_candidates}",
                        symbol=impact.symbol,
                    )
                ],
                confidence=0.92,
                related_symbols=[impact.symbol],
                related_repos=impact.repos_involved,
                tags=["coverage"],
            )
        )

    for warn in loop_warnings:
        findings.append(
            ReviewFinding(
                id=_finding_id("loop_warning", warn),
                severity=Severity.INFO,
                category=FindingCategory.CONFIDENCE_GAP,
                title="Review loop warning",
                impact=f"Review execution warning: {warn}",
                recommendation="Inspect tool budget and API health before relying on this report for strict gating.",
                evidence=[EvidenceRef(tool="react_loop", description=warn)],
                confidence=1.0,
                related_symbols=[],
                related_repos=[],
                tags=["loop_warning"],
            )
        )
    return findings


def rank_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    """Sort findings by severity and title for stable output."""
    return sorted(findings, key=lambda f: (-_severity_order(f.severity), f.title, f.id))


def _severity_order(severity: Severity) -> int:
    order = {
        Severity.CRITICAL: 5,
        Severity.HIGH: 4,
        Severity.MEDIUM: 3,
        Severity.LOW: 2,
        Severity.INFO: 1,
    }
    return order[severity]


def _finding_id(*parts: str) -> str:
    raw = "|".join(parts)
    return sha1(raw.encode("utf-8")).hexdigest()[:12]

