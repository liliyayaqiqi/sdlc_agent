"""Top-level review orchestration for semantic PR analysis."""

from __future__ import annotations

from pathlib import Path

from review_agent.analyzers import (
    analyze_architecture_risks,
    analyze_confidence_gaps,
    analyze_cross_repo_breakage,
    analyze_hidden_side_effects,
    rank_findings,
)
from review_agent.manifest_resolver import dependency_map, load_workspace_manifest, resolve_file_key, repo_for_file_key
from review_agent.models import (
    CoverageSummary,
    ReviewDecision,
    ReviewFinding,
    ReviewReport,
    ReviewRequest,
    SEVERITY_RANK,
    Severity,
    ToolCallRecord,
)
from review_agent.patch_parser import parse_unified_diff
from review_agent.prompting import FindingRefiner
from review_agent.react_loop import ReActLoop
from review_agent.report_renderer import render_markdown
from review_agent.skills.registry import SkillRegistry
from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient


class ReviewOrchestrator:
    """Coordinates patch parsing, skills, analyzers, and report synthesis."""

    def __init__(
        self,
        *,
        client: CxxtractHttpClient | None = None,
        registry: SkillRegistry | None = None,
        refiner: FindingRefiner | None = None,
    ) -> None:
        self._client = client
        self._registry = registry or SkillRegistry()
        self._refiner = refiner or FindingRefiner()

    def run(self, request: ReviewRequest) -> ReviewReport:
        """Run semantic review and return structured report."""
        client = self._client or CxxtractHttpClient(
            base_url=request.cxxtract_base_url,
            workspace_id=request.workspace_id,
            timeout_s=60.0,
        )
        workspace = client.workspace_info()
        workspace_root = str(workspace.get("root_path", "")).strip()
        manifest_path = str(workspace.get("manifest_path", "")).strip()
        if not workspace_root or not manifest_path:
            raise RuntimeError("workspace info missing root_path/manifest_path")

        manifest = load_workspace_manifest(manifest_path)
        dep_map = dependency_map(manifest)

        patch_changes = parse_unified_diff(request.patch_text)
        changed_file_keys: list[str] = []
        resolution_warnings: list[str] = []
        for change in patch_changes:
            fk = resolve_file_key(
                changed_path=change.effective_path,
                workspace_root=workspace_root,
                manifest=manifest,
            )
            if fk:
                changed_file_keys.append(fk)
            else:
                resolution_warnings.append(f"unresolved_patch_path:{change.effective_path}")
        changed_file_keys = sorted(set(changed_file_keys))
        changed_repos = sorted({repo_for_file_key(fk) for fk in changed_file_keys if repo_for_file_key(fk)})

        loop = ReActLoop(registry=self._registry)
        state = loop.run(
            request=request,
            client=client,
            patch_changes=patch_changes,
            changed_file_keys=changed_file_keys,
            changed_repos=changed_repos,
            dependency_map=dep_map,
        )
        impacts = list(state.get("symbol_impacts", []))
        cross_repo_signals = list(state.get("cross_repo_signals", []))
        architecture_signals = list(state.get("architecture_signals", []))
        loop_warnings = list(state.get("loop_warnings", [])) + resolution_warnings

        findings: list[ReviewFinding] = []
        findings.extend(analyze_hidden_side_effects(patch_changes=patch_changes, impacts=impacts))
        findings.extend(analyze_cross_repo_breakage(impacts=impacts, cross_repo_signals=cross_repo_signals))
        findings.extend(analyze_architecture_risks(architecture_signals=architecture_signals))
        findings.extend(analyze_confidence_gaps(impacts=impacts, loop_warnings=loop_warnings))
        findings = _dedupe_findings(findings)
        findings = rank_findings(self._refiner.refine(findings))

        coverage = _coverage_from_impacts(impacts, extra_warnings=loop_warnings)
        decision = _decision(findings=findings, threshold=request.fail_on_severity)
        summary = _build_summary(findings=findings, changed_file_count=len(patch_changes), changed_repos=changed_repos)

        return ReviewReport(
            workspace_id=request.workspace_id,
            summary=summary,
            findings=findings,
            coverage=coverage,
            decision=decision,
            tool_usage=list(state.get("tool_usage", [])),
        )

    @staticmethod
    def write_report_files(report: ReviewReport, out_dir: str | Path) -> tuple[Path, Path]:
        """Write markdown and JSON report artifacts."""
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "review_report.md"
        json_path = out / "review_report.json"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path


def _dedupe_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:
    dedup: dict[str, ReviewFinding] = {}
    for finding in findings:
        existing = dedup.get(finding.id)
        if existing is None:
            dedup[finding.id] = finding
            continue
        if SEVERITY_RANK[finding.severity] > SEVERITY_RANK[existing.severity]:
            dedup[finding.id] = finding
    return list(dedup.values())


def _coverage_from_impacts(impacts, *, extra_warnings: list[str]) -> CoverageSummary:
    verified: set[str] = set()
    stale: set[str] = set()
    unparsed: set[str] = set()
    totals = 0
    weighted_ratio_sum = 0.0
    warnings: set[str] = set(extra_warnings)

    for impact in impacts:
        conf = dict(impact.confidence or {})
        total = int(conf.get("total_candidates", 0) or 0)
        ratio = float(conf.get("verified_ratio", 0.0) or 0.0)
        weighted_ratio_sum += ratio * max(total, 1)
        totals += max(total, 1)
        for fk in conf.get("verified_files", []) or []:
            verified.add(str(fk))
        for fk in conf.get("stale_files", []) or []:
            stale.add(str(fk))
        for fk in conf.get("unparsed_files", []) or []:
            unparsed.add(str(fk))
        for w in impact.warnings:
            warnings.add(str(w))

    verified_ratio = (weighted_ratio_sum / totals) if totals > 0 else 0.0
    return CoverageSummary(
        verified_ratio=round(verified_ratio, 4),
        total_candidates=len(verified | stale | unparsed),
        verified_files=sorted(verified),
        stale_files=sorted(stale),
        unparsed_files=sorted(unparsed),
        warnings=sorted(warnings),
    )


def _decision(*, findings: list[ReviewFinding], threshold: Severity) -> ReviewDecision:
    blocking = [f for f in findings if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[threshold]]
    return ReviewDecision(
        fail_threshold=threshold,
        blocking_findings=len(blocking),
        should_block=bool(blocking),
    )


def _build_summary(*, findings: list[ReviewFinding], changed_file_count: int, changed_repos: list[str]) -> str:
    if not findings:
        return (
            f"Analyzed {changed_file_count} changed files across {len(changed_repos)} repos. "
            "No semantic risk findings were detected above informational level."
        )
    high = len([f for f in findings if f.severity in {Severity.CRITICAL, Severity.HIGH}])
    medium = len([f for f in findings if f.severity == Severity.MEDIUM])
    low = len([f for f in findings if f.severity in {Severity.LOW, Severity.INFO}])
    return (
        f"Analyzed {changed_file_count} changed files across {len(changed_repos)} repos. "
        f"Detected {len(findings)} findings ({high} high/critical, {medium} medium, {low} low/info)."
    )

