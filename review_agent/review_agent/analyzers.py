"""Deprecated deterministic analyzers.

All risk reasoning now runs inside the PydanticAI agent and macro-tool loop.
This module is retained to avoid import breakage for downstream integrations.
"""

from __future__ import annotations

from review_agent.models import ReviewFinding


def analyze_hidden_side_effects(**_kwargs) -> list[ReviewFinding]:  # pragma: no cover - compatibility only
    return []


def analyze_cross_repo_breakage(**_kwargs) -> list[ReviewFinding]:  # pragma: no cover - compatibility only
    return []


def analyze_architecture_risks(**_kwargs) -> list[ReviewFinding]:  # pragma: no cover - compatibility only
    return []


def analyze_confidence_gaps(**_kwargs) -> list[ReviewFinding]:  # pragma: no cover - compatibility only
    return []


def rank_findings(findings: list[ReviewFinding]) -> list[ReviewFinding]:  # pragma: no cover - compatibility only
    return findings
