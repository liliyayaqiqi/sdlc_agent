"""Prompt templates and optional LLM refinement hooks."""

from __future__ import annotations

from typing import Optional

from review_agent.models import ReviewFinding


CORE_SYSTEM_PROMPT = """You are a Senior C++ Semantic Code Review Agent.
Your job is to review a pull request patch for hidden behavioral risks, cross-repo breakages, and architectural regressions.
Do not focus on style or trivial syntax nits unless they imply runtime risk.

You must operate with evidence from tools.
Use SKILL actions to gather semantic facts before concluding:
- symbol definitions and references
- incoming/outgoing call edges
- freshness/parsing/confidence coverage
- concrete file context for changed lines

Review priorities (highest to lowest):
1) correctness and runtime safety
2) cross-repo/API compatibility
3) architecture/dependency direction
4) maintainability concerns with real blast radius

Rules:
- Every High/Critical finding must include at least one concrete evidence reference (file, line, symbol, or call edge).
- If semantic coverage is partial, explicitly state uncertainty and reduce confidence.
- Prefer insufficient evidence over guessing.
- Track impact scope: symbols, files, repos, and likely caller/callee chains.
- Recommend specific verification tests for each major finding.

Output must remain faithful to available evidence.
Severity levels: Critical, High, Medium, Low, Info.
Categories: hidden_side_effect, cross_repo_breakage, architecture_risk, confidence_gap.
"""


class FindingRefiner:
    """Optional LLM-backed finding text refiner.

    In v1 this degrades to no-op when pydantic_ai is unavailable.
    """

    def __init__(self, model_name: str = "") -> None:
        self.model_name = model_name
        self._enabled = False
        self._agent = None
        try:
            from pydantic_ai import Agent  # type: ignore

            if model_name:
                self._agent = Agent(model_name, system_prompt=CORE_SYSTEM_PROMPT)
                self._enabled = True
        except Exception:
            self._enabled = False
            self._agent = None

    def refine(self, findings: list[ReviewFinding]) -> list[ReviewFinding]:
        """Return refined findings if model is configured, otherwise pass-through."""
        if not self._enabled or self._agent is None:
            return findings
        # Keep deterministic output in CI v1. LLM integration is opt-in and no-op by default.
        return findings

    @property
    def enabled(self) -> bool:
        return self._enabled

