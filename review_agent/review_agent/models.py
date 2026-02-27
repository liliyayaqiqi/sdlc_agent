"""Typed models for patch intake, findings, and review output."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class ReviewDecision(BaseModel):
    """Final CI/blocking decision."""

    fail_threshold: Severity
    blocking_findings: int
    should_block: bool


class ReviewReport(BaseModel):
    """Final review report in structured form."""

    workspace_id: str
    summary: str
    findings: list[ReviewFinding] = Field(default_factory=list)
    coverage: CoverageSummary = Field(default_factory=CoverageSummary)
    decision: ReviewDecision
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tool_usage: list[ToolCallRecord] = Field(default_factory=list)


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


class SeedSymbol(BaseModel):
    """A symbol candidate extracted from changed code."""

    symbol: str
    source: str
    file_key: str = ""
    score: float = 0.0


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
    confidence: dict[str, Any] = Field(default_factory=dict)
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


class ReviewRequest(BaseModel):
    """Top-level request consumed by the orchestrator."""

    workspace_id: str
    patch_text: str
    cxxtract_base_url: str = "http://127.0.0.1:8000"
    fail_on_severity: Severity = Severity.HIGH
    max_symbols: int = 24
    max_tool_rounds: int = 30
    max_total_tool_calls: int = 120
    parse_timeout_s: int = 120
    parse_workers: int = 4
    max_candidates_per_symbol: int = 150
    max_fetch_limit: int = 2000

    @model_validator(mode="after")
    def _validate_patch(self) -> "ReviewRequest":
        if not self.patch_text.strip():
            raise ValueError("patch_text is empty")
        return self

