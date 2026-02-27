from __future__ import annotations

from review_agent.models import (
    CoverageSummary,
    FindingCategory,
    ReviewDecision,
    ReviewFinding,
    ReviewPlan,
    ReviewReport,
    ReviewRequest,
    Severity,
)
from review_agent.orchestrator import ReviewOrchestrator


class _FakeClient:
    def workspace_info(self):
        return {"workspace_id": "ws_main", "contexts": ["ws_main:baseline"], "repos": ["repoA"]}


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakePlanner:
    def run_sync(self, _prompt):
        return _FakeResult(ReviewPlan(prioritized_symbols=[]))


class _FakeSynth:
    def run_sync(self, _prompt):
        report = ReviewReport(
            workspace_id="wrong_ws",
            summary="",
            findings=[
                ReviewFinding(
                    id="f1",
                    severity=Severity.HIGH,
                    category=FindingCategory.CROSS_REPO_BREAKAGE,
                    title="Potential breakage",
                    impact="impact",
                    recommendation="rec",
                )
            ],
            coverage=CoverageSummary(),
            decision=ReviewDecision(
                fail_threshold=Severity.INFO,
                blocking_findings=0,
                should_block=False,
            ),
            tool_usage=[],
        )
        return _FakeResult(report)


def test_orchestrator_builds_fact_sheet_and_policy_gates():
    orchestrator = ReviewOrchestrator(
        client=_FakeClient(),
        planner_factory=lambda _model: _FakePlanner(),
        synthesis_factory=lambda _model: _FakeSynth(),
    )
    req = ReviewRequest(
        workspace_id="ws_main",
        patch_text="diff --git a/a.cpp b/a.cpp\n",
        llm_model="openai:gpt-4o",
        fail_on_severity=Severity.HIGH,
        enable_cache=False,
    )
    report = orchestrator.run(req)
    assert report.workspace_id == "ws_main"
    assert report.fact_sheet is not None
    assert report.test_impact is not None
    assert report.decision.fail_threshold == Severity.HIGH
    # High finding without evidence should be downgraded by policy gate.
    assert report.findings[0].severity != Severity.HIGH
