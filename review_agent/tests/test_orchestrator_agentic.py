from __future__ import annotations

from review_agent.models import (
    CoverageSummary,
    FindingCategory,
    ReviewDecision,
    ReviewFinding,
    ReviewReport,
    ReviewRequest,
    Severity,
    ToolCallRecord,
)
from review_agent.orchestrator import ReviewOrchestrator


class _FakeClient:
    def workspace_info(self):
        return {"workspace_id": "ws_main"}


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakeAgent:
    def __init__(self):
        self.last_prompt = ""
        self.last_deps = None

    def run_sync(self, prompt, deps):
        self.last_prompt = prompt
        self.last_deps = deps
        deps.tool_usage.append(
            ToolCallRecord(
                skill="agentic_llm",
                tool="agent.investigate_symbol",
                success=True,
                elapsed_ms=12.3,
            )
        )
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
        )
        return _FakeResult(report)


def test_orchestrator_uses_agent_and_normalizes_report():
    fake_agent = _FakeAgent()
    orchestrator = ReviewOrchestrator(
        client=_FakeClient(),
        agent_factory=lambda _model: fake_agent,
    )
    req = ReviewRequest(
        workspace_id="ws_main",
        patch_text="diff --git a/a.cpp b/a.cpp\n",
        llm_model="openai:gpt-4o",
        fail_on_severity=Severity.HIGH,
    )
    report = orchestrator.run(req)

    assert "PR patch (unified diff)" in fake_agent.last_prompt
    assert report.workspace_id == "ws_main"
    assert report.decision.fail_threshold == Severity.HIGH
    assert report.decision.blocking_findings == 1
    assert report.decision.should_block is True
    assert len(report.tool_usage) == 1
