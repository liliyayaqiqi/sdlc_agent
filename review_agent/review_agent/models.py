"""Typed models for context intake, evidence, findings, and review output."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Version constant – bump on breaking prompt/parser/model changes
# ---------------------------------------------------------------------------

AGENT_VERSION = "0.2.0"
PROMPT_VERSION = "2026-02-28"
PARSER_VERSION = "1"


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

class InputNormalizationError(RuntimeError):
    """Raised when diff/patch input cannot be normalized into ≥1 PatchChange."""


class InfrastructureError(RuntimeError):
    """Raised on backend / network / resource failures during review."""


class PublishingError(RuntimeError):
    """Raised when posting review results to a VCS host fails."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Severity(str, Enum):
    """Finding severity ordered from highest impact to lowest."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


SEVERITY_RANK: dict[Severity, int] = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 4,
    Severity.MEDIUM: 3,
    Severity.LOW: 2,
    Severity.INFO: 1,
}


class FindingCategory(str, Enum):
    """Top-level review finding buckets."""

    HIDDEN_SIDE_EFFECT = "hidden_side_effect"
    CROSS_REPO_BREAKAGE = "cross_repo_breakage"
    ARCHITECTURE_RISK = "architecture_risk"
    CONFIDENCE_GAP = "confidence_gap"


class ReviewExecutionStatus(str, Enum):
    """Three-state execution outcome for CI gating.

    * **pass** – review completed, no blocking findings.
    * **block** – review completed, blocking findings present.
    * **indeterminate** – review could not complete reliably
      (timeout, backend failure, normalization failure, cache decode
      failure, partial merge materialization, etc.).  CI policy should
      decide whether indeterminate blocks or not.
    """

    PASS = "pass"
    BLOCK = "block"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Evidence and findings
# ---------------------------------------------------------------------------

class EvidenceRef(BaseModel):
    """Concrete evidence supporting a finding."""

    tool: str = ""
    description: str = ""
    file_key: str = ""
    abs_path: str = ""
    line: int = 0
    symbol: str = ""
    snippet: str = ""


class ReviewFinding(BaseModel):
    """A single review finding with severity, evidence, and guidance."""

    id: str
    severity: Severity
    category: FindingCategory
    title: str
    impact: str
    recommendation: str
    evidence: list[EvidenceRef] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    related_symbols: list[str] = Field(default_factory=list)
    related_repos: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    # Diff position for inline publishing (optional)
    diff_path: str = ""
    diff_line: int = 0


class CoverageSummary(BaseModel):
    """Coverage and confidence context for the full review."""

    verified_ratio: float = 0.0
    total_candidates: int = 0
    verified_files: list[str] = Field(default_factory=list)
    stale_files: list[str] = Field(default_factory=list)
    unparsed_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """Tool call telemetry row for debugging and auditability."""

    skill: str
    tool: str
    success: bool
    elapsed_ms: float = 0.0
    note: str = ""


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """Final CI/blocking decision with three-state execution status."""

    fail_threshold: Severity
    blocking_findings: int
    should_block: bool
    execution_status: ReviewExecutionStatus = ReviewExecutionStatus.PASS
    indeterminate_reason: str = ""


# ---------------------------------------------------------------------------
# Diff / patch models
# ---------------------------------------------------------------------------

class HunkLine(BaseModel):
    """A line inside one unified-diff hunk."""

    kind: Literal["context", "add", "del"]
    text: str
    old_line: int = 0
    new_line: int = 0


class PatchHunk(BaseModel):
    """One @@ ... @@ patch hunk."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine] = Field(default_factory=list)


class PatchChangeType(str, Enum):
    """Canonical change type for a file-level diff."""

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


class PatchChange(BaseModel):
    """File-level patch record including all hunks."""

    old_path: str
    new_path: str
    change_type: PatchChangeType
    hunks: list[PatchHunk] = Field(default_factory=list)
    is_binary: bool = False

    @property
    def effective_path(self) -> str:
        """Return best path representing the changed file."""
        if self.change_type == PatchChangeType.DELETED:
            return self.old_path
        return self.new_path


# ---------------------------------------------------------------------------
# Pre-pass artifacts
# ---------------------------------------------------------------------------

class SeedSymbol(BaseModel):
    """A symbol candidate extracted from changed code."""

    symbol: str
    source: str
    file_key: str = ""
    score: float = 0.0


class SuspiciousAnchor(BaseModel):
    """Deterministic non-symbol anchor extracted from changed lines."""

    kind: str
    file_path: str
    line: int = 0
    reason: str = ""
    snippet: str = ""


class PrepassResult(BaseModel):
    """Deterministic pre-pass output before any LLM reasoning."""

    changed_files: list[str] = Field(default_factory=list)
    changed_hunk_count: int = 0
    seed_symbols: list[SeedSymbol] = Field(default_factory=list)
    suspicious_anchors: list[SuspiciousAnchor] = Field(default_factory=list)
    changed_methods: list[str] = Field(default_factory=list)
    added_call_sites: list[str] = Field(default_factory=list)
    removed_call_sites: list[str] = Field(default_factory=list)
    include_macro_config_changes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence collection artifacts
# ---------------------------------------------------------------------------

class SymbolImpact(BaseModel):
    """Semantic impact bundle for one symbol investigation."""

    symbol: str
    candidate_file_keys: list[str] = Field(default_factory=list)
    deleted_file_keys: list[str] = Field(default_factory=list)
    fresh: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    unparsed: list[str] = Field(default_factory=list)
    parsed_file_keys: list[str] = Field(default_factory=list)
    parse_failed_file_keys: list[str] = Field(default_factory=list)
    symbols: list[dict[str, Any]] = Field(default_factory=list)
    references: list[dict[str, Any]] = Field(default_factory=list)
    call_edges: list[dict[str, Any]] = Field(default_factory=list)
    rg_hits: list[dict[str, Any]] = Field(default_factory=list)
    read_contexts: list[dict[str, Any]] = Field(default_factory=list)
    confidence: dict[str, Any] = Field(default_factory=dict)
    macro_summary: str = ""
    warnings: list[str] = Field(default_factory=list)

    @property
    def repos_involved(self) -> list[str]:
        repos: set[str] = set()
        for file_key in self.candidate_file_keys + self.parsed_file_keys + self.unparsed:
            if ":" in file_key:
                repos.add(file_key.split(":", 1)[0])
        for row in self.references:
            repo = str(row.get("file_key", "")).split(":", 1)[0]
            if repo:
                repos.add(repo)
        for row in self.call_edges:
            repo = str(row.get("file_key", "")).split(":", 1)[0]
            if repo:
                repos.add(repo)
        return sorted(repos)


class SymbolFact(BaseModel):
    """Deterministic facts and deltas for one reviewed symbol."""

    symbol: str
    candidate_file_keys: list[str] = Field(default_factory=list)
    parsed_file_keys: list[str] = Field(default_factory=list)
    head_reference_count: int = 0
    baseline_reference_count: int = 0
    merge_preview_reference_count: int = 0
    head_call_edge_count: int = 0
    baseline_call_edge_count: int = 0
    merge_preview_call_edge_count: int = 0
    reference_delta_vs_baseline: int = 0
    call_edge_delta_vs_baseline: int = 0
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Plans and views
# ---------------------------------------------------------------------------

class ReviewPlan(BaseModel):
    """LLM-generated review plan controlling deterministic evidence collection."""

    prioritized_symbols: list[str] = Field(default_factory=list)
    lexical_files: list[str] = Field(default_factory=list)
    semantic_files: list[str] = Field(default_factory=list)
    require_merge_preview: bool = False
    budget_split: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ViewContextMaterialization(BaseModel):
    """Status of baseline/head/merge-preview context preparation."""

    baseline_context_id: str = ""
    head_context_id: str = ""
    merge_preview_context_id: str = ""
    baseline_materialized: bool = False
    head_materialized: bool = False
    merge_preview_materialized: bool = False
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Fact sheet
# ---------------------------------------------------------------------------

class ReviewFactSheet(BaseModel):
    """Deterministic fact layer that powers synthesis and policy gating."""

    changed_files: list[str] = Field(default_factory=list)
    changed_hunk_count: int = 0
    seed_symbols: list[str] = Field(default_factory=list)
    suspicious_anchors: list[SuspiciousAnchor] = Field(default_factory=list)
    changed_methods: list[str] = Field(default_factory=list)
    added_call_sites: list[str] = Field(default_factory=list)
    removed_call_sites: list[str] = Field(default_factory=list)
    include_macro_config_changes: list[str] = Field(default_factory=list)
    symbol_facts: list[SymbolFact] = Field(default_factory=list)
    evidence_anchors: list[EvidenceRef] = Field(default_factory=list)
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    view_contexts: ViewContextMaterialization = Field(default_factory=ViewContextMaterialization)
    merge_delta_signals: list[dict[str, Any]] = Field(default_factory=list)
    merge_analysis_degraded: bool = False
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Test impact (renamed to avoid pytest collection collision)
# ---------------------------------------------------------------------------

class ReviewTestImpact(BaseModel):
    """Deterministic test-impact recommendation output."""

    directly_impacted_tests: list[str] = Field(default_factory=list)
    likely_impacted_tests: list[str] = Field(default_factory=list)
    suggested_scopes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    test_dependency_edges: list[dict[str, Any]] = Field(default_factory=list)


# Backward-compat alias (deprecated)
TestImpact = ReviewTestImpact


# ---------------------------------------------------------------------------
# Run metadata and cache envelope
# ---------------------------------------------------------------------------

class RunMetadata(BaseModel):
    """Versioning envelope included in cache keys and report output."""

    agent_version: str = AGENT_VERSION
    prompt_version: str = PROMPT_VERSION
    parser_version: str = PARSER_VERSION
    input_mode: str = ""  # "patch_file", "context_bundle", "gitlab_mr"
    run_id: str = ""
    backend_base_url: str = ""


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

class ReviewReport(BaseModel):
    """Final review report in structured form."""

    workspace_id: str
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    decision: ReviewDecision
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_usage: list[ToolCallRecord] = Field(default_factory=list)
    fact_sheet: ReviewFactSheet | None = None
    test_impact: ReviewTestImpact | None = None
    run_metadata: RunMetadata | None = None
    run_id: str = ""


# ---------------------------------------------------------------------------
# Context bundle and request
# ---------------------------------------------------------------------------

class ReviewContextBundle(BaseModel):
    """PR/MR-driven review context bundle."""

    workspace_id: str
    patch_text: str = ""
    changed_files: list[str] = Field(default_factory=list)
    changed_hunks: list[dict[str, Any]] = Field(default_factory=list)
    base_sha: str = ""
    head_sha: str = ""
    target_branch_head_sha: str = ""
    merge_preview_sha: str = ""
    primary_repo_id: str = ""
    per_repo_shas: dict[str, str] = Field(default_factory=dict)
    pr_metadata: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    """Top-level request consumed by the orchestrator."""

    workspace_id: str
    patch_text: str = ""
    context_bundle: ReviewContextBundle | None = None
    llm_model: str = "openai:gpt-4o"
    cxxtract_base_url: str = "http://127.0.0.1:8000"
    fail_on_severity: Severity = Severity.HIGH
    max_symbols: int = 24
    max_symbol_slots: int = 30
    max_total_tool_calls: int = 120
    parse_timeout_s: int = 120
    parse_workers: int = 4
    max_candidates_per_symbol: int = 150
    max_fetch_limit: int = 2000
    review_timeout_s: int = 0
    enable_cache: bool = True
    cache_dir: str = ".review_agent_cache"
    infra_fail_mode: str = "block"  # "block" or "pass"

    @model_validator(mode="after")
    def _validate_request(self) -> "ReviewRequest":
        if self.context_bundle is not None and self.context_bundle.workspace_id != self.workspace_id:
            raise ValueError("context_bundle.workspace_id does not match workspace_id")
        bundle_patch = (self.context_bundle.patch_text if self.context_bundle else "").strip()
        if not self.patch_text.strip() and not bundle_patch:
            raise ValueError("patch_text is empty and context_bundle.patch_text is empty")
        return self
