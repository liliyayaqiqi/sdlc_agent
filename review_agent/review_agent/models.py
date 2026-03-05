"""Typed models for context intake, evidence, findings, and review output."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

AGENT_VERSION = "0.3.1"
PROMPT_VERSION = "2026-02-28"
PARSER_VERSION = "2"

SUPPORTED_MODEL_PROVIDERS = {"openai", "openrouter", "gateway", "openai-compatible", "fixture"}

ReviewConfidence = Literal["high", "medium", "low"]
FindingLocationSide = Literal["new", "old"]
RepoRevisionRole = Literal["primary", "dependency", "auxiliary"]
SeedRelevanceTier = Literal["declaration", "qualified", "receiver_owned", "generic_fallback"]
DeclarationKind = Literal["function", "method", "constructor", "destructor", "class", "struct", "enum"]
DiffLineKind = Literal["add", "del"]
RetrievalStatus = Literal["bootstrapped", "expanded", "empty", "failed"]


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------

class InputNormalizationError(RuntimeError):
    """Raised when diff/patch input cannot be normalized into >=1 PatchChange."""


class InfrastructureError(RuntimeError):
    """Raised on backend / network / resource failures during review."""


class PublishingError(RuntimeError):
    """Raised when posting review results to a VCS host fails."""


class ModelContractError(RuntimeError):
    """Raised when planner/exploration/synthesis responses violate schema."""


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
    """Three-state execution outcome for CI gating."""

    PASS = "pass"
    BLOCK = "block"
    INDETERMINATE = "indeterminate"


# ---------------------------------------------------------------------------
# Evidence and findings
# ---------------------------------------------------------------------------

class FindingLocation(BaseModel):
    """Structured location for diff-aware publication."""

    repo_id: str = ""
    path: str
    line: int = Field(default=0, ge=0)
    side: FindingLocationSide = "new"
    base_sha: str = ""
    head_sha: str = ""
    start_sha: str = ""


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
    location: FindingLocation | None = None
    # Deprecated compatibility fields.
    diff_path: str = ""
    diff_line: int = 0

    @model_validator(mode="after")
    def _sync_legacy_location_fields(self) -> "ReviewFinding":
        if self.location is not None:
            if not self.diff_path:
                self.diff_path = self.location.path
            if self.diff_line <= 0:
                self.diff_line = self.location.line
        elif self.diff_path and self.diff_line > 0:
            self.location = FindingLocation(path=self.diff_path, line=self.diff_line)
        return self


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
# Decision and publishing
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    """Final CI/blocking decision with three-state execution status."""

    fail_threshold: Severity
    blocking_findings: int
    should_block: bool
    execution_status: ReviewExecutionStatus = ReviewExecutionStatus.PASS
    indeterminate_reason: str = ""
    review_confidence: ReviewConfidence = "medium"


class PublishResult(BaseModel):
    """Structured publication outcome for VCS integrations."""

    provider: str = ""
    summary_posted: bool = False
    inline_comments_posted: int = 0
    warnings: list[str] = Field(default_factory=list)


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
    relevance_tier: SeedRelevanceTier = "generic_fallback"
    reasons: list[str] = Field(default_factory=list)
    receiver: str = ""
    container: str = ""
    file_paths: list[str] = Field(default_factory=list)


class ChangedDeclaration(BaseModel):
    """Declaration-like symbol added or modified in the patch."""

    symbol: str
    container: str = ""
    kind: DeclarationKind = "function"
    file_path: str
    line: int = 0


class MemberCallSite(BaseModel):
    """Member call extracted from changed lines."""

    member: str
    receiver: str
    container: str = ""
    file_path: str
    line: int = 0
    line_kind: DiffLineKind = "add"
    qualified_receiver_type: str = ""


class DiffExcerpt(BaseModel):
    """Planner-facing excerpt from one changed hunk."""

    file_path: str
    hunk_header: str
    start_line: int = 0
    end_line: int = 0
    text: str = ""
    reason: str = ""


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
    changed_declarations: list[ChangedDeclaration] = Field(default_factory=list)
    changed_containers: list[str] = Field(default_factory=list)
    member_call_sites: list[MemberCallSite] = Field(default_factory=list)
    diff_excerpts: list[DiffExcerpt] = Field(default_factory=list)
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
    candidate_provenance: list[str] = Field(default_factory=list)
    retrieval_status: RetrievalStatus = "empty"
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
    confidence: "SymbolConfidence" = Field(default_factory=lambda: SymbolConfidence())
    candidate_provenance: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SymbolConfidence(BaseModel):
    """Confidence and retrieval metadata for one investigated symbol."""

    verified_ratio: float = 0.0
    total_candidates: int = 0
    verified_files: list[str] = Field(default_factory=list)
    stale_files: list[str] = Field(default_factory=list)
    unparsed_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    candidate_sources: list[str] = Field(default_factory=list)
    retrieval_status: RetrievalStatus = "empty"


# ---------------------------------------------------------------------------
# Plans, exploration, synthesis, and views
# ---------------------------------------------------------------------------

class ReviewPlan(BaseModel):
    """LLM-generated review plan controlling deterministic evidence collection."""

    prioritized_symbols: list[str] = Field(default_factory=list)
    lexical_files: list[str] = Field(default_factory=list)
    semantic_files: list[str] = Field(default_factory=list)
    require_merge_preview: bool = False
    budget_split: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class ExplorationResult(BaseModel):
    """Typed result emitted by the exploration service."""

    new_evidence: list[EvidenceRef] = Field(default_factory=list)
    follow_up_symbols: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    summary: str = ""


class SynthesisDraft(BaseModel):
    """Model-owned synthesis output before policy gating."""

    summary: str = ""
    findings: list[ReviewFinding] = Field(default_factory=list)
    global_notes: list[str] = Field(default_factory=list)


class ViewContextMaterialization(BaseModel):
    """Status of baseline/head/merge-preview context preparation."""

    baseline_context_id: str = ""
    head_context_id: str = ""
    merge_preview_context_id: str = ""
    baseline_workspace_id: str = ""
    head_workspace_id: str = ""
    merge_preview_workspace_id: str = ""
    baseline_materialized: bool = False
    head_materialized: bool = False
    merge_preview_materialized: bool = False
    materialization_id: str = ""
    materialization_status: str = ""
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
    changed_declarations: list[ChangedDeclaration] = Field(default_factory=list)
    changed_containers: list[str] = Field(default_factory=list)
    member_call_sites: list[MemberCallSite] = Field(default_factory=list)
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
ReviewTestImpact.__test__ = False


# ---------------------------------------------------------------------------
# Run metadata and cache envelope
# ---------------------------------------------------------------------------

class PrepassDebug(BaseModel):
    """Compact debug summary for understanding pre-pass behavior."""

    ranked_seed_candidates: list[SeedSymbol] = Field(default_factory=list)
    changed_declarations: list[ChangedDeclaration] = Field(default_factory=list)
    raw_changed_declarations: list[ChangedDeclaration] = Field(default_factory=list)
    semantic_changed_declarations: list[ChangedDeclaration] = Field(default_factory=list)
    member_call_sites_top: list[MemberCallSite] = Field(default_factory=list)
    diff_excerpt_reasons: list[str] = Field(default_factory=list)
    bootstrap_file_keys: list[str] = Field(default_factory=list)
    bootstrap_seeded_symbols: list[dict[str, Any]] = Field(default_factory=list)
    zero_candidate_symbols: list[str] = Field(default_factory=list)
    retrieval_widening_events: list[dict[str, Any]] = Field(default_factory=list)

class RunMetadata(BaseModel):
    """Versioning envelope included in cache keys and report output."""

    agent_version: str = AGENT_VERSION
    prompt_version: str = PROMPT_VERSION
    parser_version: str = PARSER_VERSION
    input_mode: str = ""  # "patch_file", "context_bundle", "gitlab_mr"
    run_id: str = ""
    backend_base_url: str = ""
    prepass_debug: PrepassDebug | None = None


# ---------------------------------------------------------------------------
# Repo context
# ---------------------------------------------------------------------------

class RepoRevisionContext(BaseModel):
    """Repo-specific revision inputs for merge-aware analysis."""

    repo_id: str
    base_sha: str = ""
    head_sha: str = ""
    target_sha: str = ""
    merge_sha: str = ""
    role: RepoRevisionRole = "dependency"


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
    publish_result: PublishResult | None = None
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
    repo_revisions: list[RepoRevisionContext] = Field(default_factory=list)
    workspace_fingerprint: str = ""
    pr_metadata: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


class ReviewRequest(BaseModel):
    """Top-level request consumed by the orchestrator."""

    workspace_id: str
    patch_text: str = ""
    context_bundle: ReviewContextBundle | None = None
    llm_model: str = "openai:gpt-4o"
    llm_base_url: str = ""
    llm_api_key: str = Field(default="", repr=False)
    llm_app_url: str = ""
    llm_app_title: str = ""
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
    workspace_fingerprint: str = ""
    use_derived_workspaces: bool = False

    @model_validator(mode="after")
    def _validate_request(self) -> "ReviewRequest":
        if self.context_bundle is not None and self.context_bundle.workspace_id != self.workspace_id:
            raise ValueError("context_bundle.workspace_id does not match workspace_id")
        bundle_patch = (self.context_bundle.patch_text if self.context_bundle else "").strip()
        if not self.patch_text.strip() and not bundle_patch:
            raise ValueError("patch_text is empty and context_bundle.patch_text is empty")
        model_name = (self.llm_model or "").strip() or "openai:gpt-4o"
        provider = model_name.split(":", 1)[0]
        if provider not in SUPPORTED_MODEL_PROVIDERS:
            raise ValueError(
                f"unsupported llm provider '{provider}'; supported providers: {sorted(SUPPORTED_MODEL_PROVIDERS)}"
            )
        if ":" not in model_name:
            raise ValueError("llm_model must be in '<provider>:<model>' format")
        if provider in {"gateway", "openai-compatible"} and not self.llm_base_url.strip():
            raise ValueError("llm_base_url is required for gateway/openai-compatible llm providers")
        return self
