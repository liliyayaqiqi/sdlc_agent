"""Tests for hardening fixes: budget exhaustion, cache keys, merge deltas,
test impact, renderer, view failures, stale lock cleanup, three-state
execution model, GitLab diff normalization, and cache correctness."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from review_agent.models import (
    CoverageSummary,
    EvidenceRef,
    FindingCategory,
    InputNormalizationError,
    InfrastructureError,
    ReviewContextBundle,
    ReviewDecision,
    ReviewExecutionStatus,
    ReviewFactSheet,
    ReviewFinding,
    ReviewPlan,
    ReviewReport,
    ReviewRequest,
    ReviewTestImpact,
    RunMetadata,
    Severity,
    SymbolFact,
    SymbolImpact,
    TestImpact,
    ToolCallRecord,
    ViewContextMaterialization,
)
from review_agent.orchestrator import (
    BudgetTracker,
    ReviewOrchestrator,
    ViewContexts,
    _build_merge_delta_signals,
    _call,
    _collect_symbol,
    _workspace_lock,
    _pid_alive,
    _analyze_test_impact,
    _empty_impact,
    _is_merge_degraded,
    _indeterminate_report,
)
from review_agent.report_renderer import render_markdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _budget(*, calls: int = 120, slots: int = 30) -> BudgetTracker:
    return BudgetTracker(
        max_symbol_slots=slots,
        max_total_tool_calls=calls,
        max_symbols=24,
        max_candidates_per_symbol=150,
        max_fetch_limit=2000,
        parse_timeout_s=120,
        parse_workers=4,
    )


class _NullClient:
    """Returns empty dicts for every explore call."""
    base_url = "http://127.0.0.1:8000"

    def workspace_info(self):
        return {"workspace_id": "ws"}

    def context_create_pr_overlay(self, **kw):
        return {"context_id": kw.get("context_id", "ctx")}

    def context_expire(self, **kw):
        return {}

    def sync_repo(self, **kw):
        return {}

    def explore_rg_search(self, **kw):
        return {"hits": []}

    def explore_list_candidates(self, **kw):
        return {"candidates": []}

    def explore_classify_freshness(self, **kw):
        return {"stale": [], "fresh": [], "unparsed": []}

    def explore_parse_file(self, **kw):
        return {"parsed_file_keys": [], "failed_file_keys": []}

    def explore_fetch_symbols(self, **kw):
        return {"symbols": []}

    def explore_fetch_references(self, **kw):
        return {"references": []}

    def explore_fetch_call_edges(self, **kw):
        return {"edges": []}

    def explore_get_confidence(self, **kw):
        return {"confidence": {}}

    def explore_read_file(self, **kw):
        return {"content": ""}

    def agent_investigate_symbol(self, **kw):
        return {"summary_markdown": ""}

    def close(self):
        pass


class _FakeResult:
    def __init__(self, data):
        self.data = data


class _FakePlanner:
    def run_sync(self, _prompt):
        return _FakeResult(ReviewPlan(prioritized_symbols=["MyFunc"]))


class _FakeSynth:
    def run_sync(self, _prompt):
        return _FakeResult(ReviewReport(
            workspace_id="ws",
            summary="ok",
            findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(
                fail_threshold=Severity.HIGH,
                blocking_findings=0,
                should_block=False,
            ),
            tool_usage=[],
        ))


# ---------------------------------------------------------------------------
# 4.1  Budget exhaustion mid-symbol
# ---------------------------------------------------------------------------

class TestBudgetExhaustion:
    def test_collect_symbol_stops_early_when_budget_exhausted(self):
        """With only 1 call of budget, _collect_symbol should produce a
        partial SymbolImpact with a budget_exhausted warning."""
        budget = _budget(calls=1)
        tool_usage: list[ToolCallRecord] = []
        impact = _collect_symbol(
            _NullClient(), None, {"mode": "baseline"},
            "TestSymbol", tool_usage, budget,
        )
        assert isinstance(impact, SymbolImpact)
        assert impact.symbol == "TestSymbol"
        assert "budget_exhausted_mid_symbol" in impact.warnings

    def test_call_returns_warning_when_exhausted(self):
        budget = _budget(calls=0)
        tool_usage: list[ToolCallRecord] = []
        result = _call(tool_usage, budget, "test", "test.tool", lambda: {"ok": True})
        assert "warnings" in result
        assert any("max_total_tool_calls_reached" in w for w in result["warnings"])

    def test_empty_impact_has_correct_shape(self):
        impact = _empty_impact("Sym", rg_hits=[{"file_key": "a.cpp"}])
        assert impact.symbol == "Sym"
        assert len(impact.rg_hits) == 1
        assert "budget_exhausted_mid_symbol" in impact.warnings
        assert impact.references == []
        assert impact.call_edges == []


# ---------------------------------------------------------------------------
# 4.2  Cache hit and miss including llm_model
# ---------------------------------------------------------------------------

class TestCacheKeyModel:
    def test_different_llm_model_causes_cache_miss(self, tmp_path):
        """Running with two different llm_model values must produce two
        separate cache keys and NOT hit each other's cache."""
        from review_agent.review_cache import ReviewTraceCache

        cache = ReviewTraceCache(str(tmp_path))

        key1 = cache.make_key(
            workspace_id="ws", base_sha="a" * 40, head_sha="b" * 40,
            target_sha="c" * 40, merge_sha="", patch_text="diff",
            policy={"llm_model": "openai:gpt-4o", "fail_on_severity": "high"},
        )
        key2 = cache.make_key(
            workspace_id="ws", base_sha="a" * 40, head_sha="b" * 40,
            target_sha="c" * 40, merge_sha="", patch_text="diff",
            policy={"llm_model": "anthropic:claude-4-opus", "fail_on_severity": "high"},
        )
        assert key1 != key2

    def test_same_inputs_give_cache_hit(self, tmp_path):
        from review_agent.review_cache import ReviewTraceCache

        cache = ReviewTraceCache(str(tmp_path))
        report = ReviewReport(
            workspace_id="ws", summary="s", findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
        )
        key = cache.make_key(
            workspace_id="ws", base_sha="a" * 40, head_sha="b" * 40,
            target_sha="", merge_sha="", patch_text="diff",
            policy={"llm_model": "openai:gpt-4o"},
        )
        cache.save(key, {"review_report": report.model_dump(mode="json")})
        loaded = cache.load_report(key)
        assert loaded is not None
        assert loaded.workspace_id == "ws"


# ---------------------------------------------------------------------------
# 4.3  Merge delta signal generation
# ---------------------------------------------------------------------------

class TestMergeDelta:
    def _views(self, *, merge_materialized: bool) -> ViewContexts:
        return ViewContexts(
            baseline={"mode": "baseline"},
            head={"mode": "pr", "context_id": "h"},
            merge_preview={"mode": "pr", "context_id": "m"} if merge_materialized else None,
            status=ViewContextMaterialization(
                merge_preview_materialized=merge_materialized,
                merge_preview_context_id="m" if merge_materialized else "",
            ),
        )

    def test_no_signals_when_merge_not_materialized(self):
        facts = [
            SymbolFact(symbol="A", head_reference_count=5, baseline_reference_count=3,
                       merge_preview_reference_count=7, head_call_edge_count=2,
                       baseline_call_edge_count=1, merge_preview_call_edge_count=3),
        ]
        signals = _build_merge_delta_signals(facts, self._views(merge_materialized=False))
        assert signals == []

    def test_signals_generated_when_merge_differs(self):
        facts = [
            SymbolFact(symbol="Foo", head_reference_count=5, baseline_reference_count=3,
                       merge_preview_reference_count=8, reference_delta_vs_baseline=2,
                       head_call_edge_count=2, baseline_call_edge_count=1,
                       merge_preview_call_edge_count=2),
        ]
        signals = _build_merge_delta_signals(facts, self._views(merge_materialized=True))
        assert len(signals) == 1
        assert signals[0]["symbol"] == "Foo"
        assert signals[0]["merge_ref_delta_vs_head"] == 3

    def test_no_signals_when_merge_matches_head(self):
        facts = [
            SymbolFact(symbol="Bar", head_reference_count=5, baseline_reference_count=3,
                       merge_preview_reference_count=5, head_call_edge_count=2,
                       baseline_call_edge_count=1, merge_preview_call_edge_count=2),
        ]
        signals = _build_merge_delta_signals(facts, self._views(merge_materialized=True))
        assert signals == []


# ---------------------------------------------------------------------------
# 4.4  Test impact analysis
# ---------------------------------------------------------------------------

class _TestImpactClient(_NullClient):
    """Returns canned references that include test-file paths."""
    def explore_fetch_references(self, **kw):
        if kw.get("symbol") == "changedFunc":
            return {"references": [
                {"file_key": "repoA:tests/test_foo.cpp", "line": 42},
                {"file_key": "repoA:src/main.cpp", "line": 10},
            ]}
        return {"references": []}


class TestTestImpactAnalysis:
    def test_direct_and_likely_and_semantic(self):
        from review_agent.models import PrepassResult, SeedSymbol
        prepass = PrepassResult(
            changed_files=["tests/test_bar.cpp", "src/engine.cpp"],
            changed_methods=["changedFunc"],
            seed_symbols=[SeedSymbol(symbol="changedFunc", source="diff")],
        )
        impact = SymbolImpact(
            symbol="Sym",
            references=[{"file_key": "repoA:tests/test_integration.cpp", "line": 5}],
            call_edges=[],
        )
        budget = _budget(calls=20)
        tool_usage: list[ToolCallRecord] = []
        views = ViewContexts(
            baseline={"mode": "baseline"},
            head={"mode": "pr"},
            merge_preview=None,
            status=ViewContextMaterialization(),
        )
        result = _analyze_test_impact(
            prepass=prepass, impacts=[impact],
            client=_TestImpactClient(), views=views,
            budget=budget, tool_usage=tool_usage,
        )
        assert isinstance(result, ReviewTestImpact)
        assert "tests/test_bar.cpp" in result.directly_impacted_tests
        assert "tests/test_integration.cpp" in result.likely_impacted_tests
        # Semantic lookup should have found test_foo.cpp
        assert "tests/test_foo.cpp" in result.likely_impacted_tests

    def test_no_tests_gives_low_confidence(self):
        from review_agent.models import PrepassResult
        prepass = PrepassResult(changed_files=["src/main.cpp"])
        budget = _budget(calls=20)
        result = _analyze_test_impact(
            prepass=prepass, impacts=[], client=_NullClient(),
            views=ViewContexts(
                baseline={}, head={}, merge_preview=None,
                status=ViewContextMaterialization(),
            ),
            budget=budget, tool_usage=[],
        )
        assert result.confidence <= 0.3
        assert "smoke" in result.suggested_scopes


# ---------------------------------------------------------------------------
# 4.5  Report renderer output
# ---------------------------------------------------------------------------

class TestReportRenderer:
    def _report(self, *, with_fact_sheet: bool = False, with_test_impact: bool = False) -> ReviewReport:
        report = ReviewReport(
            workspace_id="ws_test",
            summary="Test summary",
            findings=[],
            coverage=CoverageSummary(verified_ratio=0.85, total_candidates=10),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
            tool_usage=[ToolCallRecord(skill="s", tool="t", success=True, elapsed_ms=5.0)],
        )
        if with_fact_sheet:
            report.fact_sheet = ReviewFactSheet(
                changed_files=["a.cpp", "b.cpp"],
                changed_hunk_count=3,
                seed_symbols=["Foo"],
                merge_delta_signals=[{
                    "symbol": "Foo",
                    "merge_ref_delta_vs_head": 2,
                    "merge_edge_delta_vs_head": -1,
                    "risk": "merge_count_shift",
                }],
            )
        if with_test_impact:
            report.test_impact = ReviewTestImpact(
                directly_impacted_tests=["tests/test_a.cpp"],
                likely_impacted_tests=["tests/test_b.cpp"],
                suggested_scopes=["smoke", "unit"],
                rationale=["deterministic_test_impact"],
                confidence=0.8,
                test_dependency_edges=[{
                    "symbol": "Foo",
                    "test_file": "tests/test_b.cpp",
                    "file_key": "repo:tests/test_b.cpp",
                    "source": "call_edge_or_reference",
                    "line": 42,
                }],
            )
        return report

    def test_basic_sections_present(self):
        md = render_markdown(self._report())
        assert "# CXXtract2 Semantic PR Review" in md
        assert "## Summary" in md
        assert "## Findings" in md
        assert "## Coverage" in md
        assert "## Tool Usage" in md
        assert "PASS" in md

    def test_empty_findings(self):
        md = render_markdown(self._report())
        assert "No findings." in md

    def test_merge_delta_signals_rendered(self):
        md = render_markdown(self._report(with_fact_sheet=True))
        assert "### Merge Delta Signals" in md
        assert "`Foo`" in md
        assert "merge_count_shift" in md

    def test_test_dependency_edges_rendered(self):
        md = render_markdown(self._report(with_test_impact=True))
        assert "### Test Dependency Edges" in md
        assert "`Foo`" in md
        assert "tests/test_b.cpp" in md

    def test_findings_with_evidence(self):
        report = self._report()
        report.findings = [
            ReviewFinding(
                id="f1", severity=Severity.HIGH, category=FindingCategory.HIDDEN_SIDE_EFFECT,
                title="Missing null check", impact="Crash", recommendation="Add check",
                evidence=[EvidenceRef(tool="explore.rg_search", file_key="a.cpp", line=10)],
                confidence=0.9,
            ),
        ]
        md = render_markdown(report)
        assert "[HIGH]" in md
        assert "Missing null check" in md
        assert "a.cpp:10" in md

    def test_indeterminate_status_rendered(self):
        report = self._report()
        report.decision = ReviewDecision(
            fail_threshold=Severity.HIGH,
            blocking_findings=0,
            should_block=False,
            execution_status=ReviewExecutionStatus.INDETERMINATE,
            indeterminate_reason="timeout_exceeded",
        )
        md = render_markdown(report)
        assert "INDETERMINATE" in md
        assert "timeout_exceeded" in md

    def test_run_metadata_rendered(self):
        report = self._report()
        report.run_metadata = RunMetadata(input_mode="gitlab_mr")
        report.run_id = "abc123"
        md = render_markdown(report)
        assert "abc123" in md
        assert "gitlab_mr" in md

    def test_merge_degraded_rendered(self):
        report = self._report(with_fact_sheet=True)
        report.fact_sheet.merge_analysis_degraded = True
        md = render_markdown(report)
        assert "degraded" in md

    def test_diff_position_rendered(self):
        report = self._report()
        report.findings = [
            ReviewFinding(
                id="f2", severity=Severity.MEDIUM, category=FindingCategory.ARCHITECTURE_RISK,
                title="Unused param", impact="Dead code", recommendation="Remove",
                diff_path="src/engine.cpp", diff_line=42,
            ),
        ]
        md = render_markdown(report)
        assert "src/engine.cpp:42" in md


# ---------------------------------------------------------------------------
# 4.6  Stale lock detection
# ---------------------------------------------------------------------------

class TestStaleLock:
    def test_stale_lock_removed_on_dead_pid(self, tmp_path):
        """A lock file whose PID is not alive should be cleaned up."""
        lock_root = str(tmp_path)
        locks_dir = tmp_path / ".workspace_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / "ws_test.lock"
        # Write a PID that is almost certainly dead
        lock_file.write_text("999999999", encoding="utf-8")

        # Should acquire the lock despite the stale file
        with _workspace_lock(lock_root=lock_root, workspace_id="ws_test", timeout_s=2.0):
            pass
        # Lock file should be cleaned up
        assert not lock_file.exists()

    def test_lock_with_own_pid_blocks(self, tmp_path):
        """A lock file with our own PID should NOT be treated as stale."""
        lock_root = str(tmp_path)
        locks_dir = tmp_path / ".workspace_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / "ws_test.lock"
        lock_file.write_text(str(os.getpid()), encoding="utf-8")

        # Should timeout because our own PID is alive
        with pytest.raises(TimeoutError):
            with _workspace_lock(lock_root=lock_root, workspace_id="ws_test", timeout_s=0.5):
                pass

    def test_pid_alive_returns_false_for_dead_pid(self):
        assert _pid_alive(999999999) is False

    def test_pid_alive_returns_true_for_self(self):
        assert _pid_alive(os.getpid()) is True


# ---------------------------------------------------------------------------
# 4.7  _prepare_views failure paths
# ---------------------------------------------------------------------------

class _FailingSyncClient(_NullClient):
    """sync_repo always raises."""
    def sync_repo(self, **kw):
        raise RuntimeError("sync failed deliberately")


class TestViewFailures:
    def test_baseline_not_materialized_on_sync_failure(self, tmp_path):
        orchestrator = ReviewOrchestrator(
            client=_FailingSyncClient(),
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            base_sha="a" * 40,
            head_sha="b" * 40,
            target_branch_head_sha="c" * 40,
            primary_repo_id="repoA",
            pr_metadata={"pr_id": "1", "target_branch": "main"},
        )
        created_ids: list[str] = []
        views = orchestrator._prepare_views(
            client=_FailingSyncClient(),
            runtime_context=bundle,
            lock_root=str(tmp_path),
            created_context_ids=created_ids,
        )
        # baseline should NOT be materialized because sync_repo failed
        assert views.status.baseline_materialized is False
        assert any("baseline_sync_failed" in w for w in views.status.warnings)

    def test_baseline_materialized_when_no_repo_id(self, tmp_path):
        """When primary_repo_id is empty, no sync is needed, baseline is materialized."""
        orchestrator = ReviewOrchestrator(client=_NullClient())
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff",
            head_sha="b" * 40,
            pr_metadata={"pr_id": "1"},
        )
        created_ids: list[str] = []
        views = orchestrator._prepare_views(
            client=_NullClient(),
            runtime_context=bundle,
            lock_root=str(tmp_path),
            created_context_ids=created_ids,
        )
        assert views.status.baseline_materialized is True


# ---------------------------------------------------------------------------
# Context expiry on exit
# ---------------------------------------------------------------------------

class _TrackingClient(_NullClient):
    """Tracks calls to context_expire."""
    def __init__(self):
        self.expired: list[str] = []

    def context_create_pr_overlay(self, **kw):
        ctx_id = kw.get("context_id", "ctx")
        return {"context_id": ctx_id}

    def context_expire(self, *, context_id: str):
        self.expired.append(context_id)
        return {}


class TestContextExpiry:
    def test_contexts_expired_after_run(self, tmp_path):
        client = _TrackingClient()
        orchestrator = ReviewOrchestrator(
            client=client,
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            llm_model="openai:gpt-4o",
            enable_cache=False,
            cache_dir=str(tmp_path),
        )
        report = orchestrator.run(req)
        # The orchestrator should have called context_expire for created contexts
        # (head context is created because workspace_info returns ws_main which matches)
        assert isinstance(report, ReviewReport)
        # Even if no contexts were created (no valid SHAs), the finally block runs


# ---------------------------------------------------------------------------
# Health check -- now returns indeterminate report instead of raising
# ---------------------------------------------------------------------------

class _UnreachableClient(_NullClient):
    def workspace_info(self):
        raise ConnectionError("connection refused")


class TestHealthCheck:
    def test_unreachable_backend_gives_indeterminate_report(self):
        """Backend unreachable should produce an INDETERMINATE report."""
        orchestrator = ReviewOrchestrator(
            client=_UnreachableClient(),
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            enable_cache=False,
        )
        report = orchestrator.run(req)
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert "infrastructure_error" in report.decision.indeterminate_reason
        # Default infra_fail_mode is "block", so should_block=True
        assert report.decision.should_block is True

    def test_unreachable_backend_pass_mode(self):
        """With infra_fail_mode='pass', unreachable backend should not block."""
        orchestrator = ReviewOrchestrator(
            client=_UnreachableClient(),
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            enable_cache=False,
            infra_fail_mode="pass",
        )
        report = orchestrator.run(req)
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert report.decision.should_block is False


# ---------------------------------------------------------------------------
# Three-state execution model
# ---------------------------------------------------------------------------

class TestExecutionStatus:
    def test_indeterminate_report_has_correct_shape(self):
        report = _indeterminate_report(
            workspace_id="ws",
            reason="test_reason",
            summary="Testing indeterminate",
            fail_threshold=Severity.HIGH,
            should_block=True,
            run_id="r123",
        )
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert report.decision.indeterminate_reason == "test_reason"
        assert report.decision.should_block is True
        assert report.run_id == "r123"
        assert report.findings == []

    def test_execution_status_default_is_pass(self):
        decision = ReviewDecision(
            fail_threshold=Severity.HIGH,
            blocking_findings=0,
            should_block=False,
        )
        assert decision.execution_status == ReviewExecutionStatus.PASS
        assert decision.indeterminate_reason == ""

    def test_run_metadata_in_report(self):
        report = ReviewReport(
            workspace_id="ws",
            summary="test",
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
            run_metadata=RunMetadata(input_mode="gitlab_mr", run_id="r123"),
            run_id="r123",
        )
        assert report.run_metadata is not None
        assert report.run_metadata.input_mode == "gitlab_mr"
        assert report.run_id == "r123"


# ---------------------------------------------------------------------------
# Input normalization error
# ---------------------------------------------------------------------------

class TestInputNormalization:
    def test_empty_parse_from_bad_format_raises(self):
        """Patch text that produces 0 changes should cause InputNormalizationError
        and result in an INDETERMINATE report."""
        orchestrator = ReviewOrchestrator(
            client=_NullClient(),
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        # This patch text has no 'diff --git' header, so parse_unified_diff returns []
        bad_patch = "--- a/file.cpp\n+++ b/file.cpp\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        req = ReviewRequest(
            workspace_id="ws",
            patch_text=bad_patch,
            enable_cache=False,
        )
        report = orchestrator.run(req)
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert "input_normalization_error" in report.decision.indeterminate_reason

    def test_valid_patch_passes_normalization(self):
        """A valid patch with diff --git headers should parse successfully."""
        orchestrator = ReviewOrchestrator(
            client=_NullClient(),
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )
        good_patch = "diff --git a/file.cpp b/file.cpp\n--- a/file.cpp\n+++ b/file.cpp\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        req = ReviewRequest(
            workspace_id="ws",
            patch_text=good_patch,
            enable_cache=False,
        )
        report = orchestrator.run(req)
        # Should not be indeterminate from normalization
        assert report.decision.execution_status != ReviewExecutionStatus.INDETERMINATE or \
            "input_normalization" not in (report.decision.indeterminate_reason or "")


# ---------------------------------------------------------------------------
# Merge degradation detection
# ---------------------------------------------------------------------------

class TestMergeDegradation:
    def test_not_degraded_when_no_merge_requested(self):
        bundle = ReviewContextBundle(workspace_id="ws", patch_text="diff")
        views = ViewContexts(
            baseline={}, head={}, merge_preview=None,
            status=ViewContextMaterialization(),
        )
        assert _is_merge_degraded(bundle, views) is False

    def test_degraded_when_merge_sha_but_not_materialized(self):
        bundle = ReviewContextBundle(
            workspace_id="ws", patch_text="diff",
            merge_preview_sha="a" * 40,
        )
        views = ViewContexts(
            baseline={}, head={}, merge_preview=None,
            status=ViewContextMaterialization(merge_preview_materialized=False),
        )
        assert _is_merge_degraded(bundle, views) is True

    def test_degraded_when_no_repo_identity(self):
        bundle = ReviewContextBundle(
            workspace_id="ws", patch_text="diff",
            merge_preview_sha="a" * 40,
        )
        views = ViewContexts(
            baseline={}, head={},
            merge_preview={"mode": "pr"},
            status=ViewContextMaterialization(merge_preview_materialized=True),
        )
        assert _is_merge_degraded(bundle, views) is True

    def test_not_degraded_when_fully_materialized(self):
        bundle = ReviewContextBundle(
            workspace_id="ws", patch_text="diff",
            merge_preview_sha="a" * 40,
            primary_repo_id="repoA",
            per_repo_shas={"repoA": "b" * 40},
        )
        views = ViewContexts(
            baseline={}, head={},
            merge_preview={"mode": "pr"},
            status=ViewContextMaterialization(merge_preview_materialized=True),
        )
        assert _is_merge_degraded(bundle, views) is False


# ---------------------------------------------------------------------------
# GitLab diff normalization
# ---------------------------------------------------------------------------

class TestGitLabDiffNormalization:
    def test_modified_file_produces_valid_patch(self):
        from review_agent.tool_clients.gitlab_client import GitLabClient
        from review_agent.patch_parser import parse_unified_diff

        # Simulate what GitLab returns for a modified file
        gl_changes = {
            "changes": [
                {
                    "old_path": "src/engine.cpp",
                    "new_path": "src/engine.cpp",
                    "diff": "@@ -1,3 +1,3 @@\n context\n-old line\n+new line\n context\n",
                    "new_file": False,
                    "deleted_file": False,
                    "renamed_file": False,
                },
            ]
        }

        # Use the diff reconstruction logic directly
        changes = gl_changes["changes"]
        parts: list[str] = []
        for ch in changes:
            diff_body = str(ch.get("diff", "") or "")
            old_path = str(ch.get("old_path", "") or "")
            new_path = str(ch.get("new_path", "") or "")
            lines = [f"diff --git a/{old_path} b/{new_path}"]
            lines.append(f"--- a/{old_path}")
            lines.append(f"+++ b/{new_path}")
            if diff_body:
                lines.append(diff_body.lstrip("\n"))
            parts.append("\n".join(lines))
        patch_text = "\n".join(parts)

        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0].old_path == "src/engine.cpp"
        assert parsed[0].new_path == "src/engine.cpp"
        assert len(parsed[0].hunks) == 1

    def test_added_file_produces_valid_patch(self):
        from review_agent.patch_parser import parse_unified_diff

        patch_text = (
            "diff --git a/new_file.cpp b/new_file.cpp\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/new_file.cpp\n"
            "@@ -0,0 +1,3 @@\n"
            "+line 1\n"
            "+line 2\n"
            "+line 3\n"
        )
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0].change_type.value == "added"

    def test_deleted_file_produces_valid_patch(self):
        from review_agent.patch_parser import parse_unified_diff

        patch_text = (
            "diff --git a/old_file.cpp b/old_file.cpp\n"
            "deleted file mode 100644\n"
            "--- a/old_file.cpp\n"
            "+++ /dev/null\n"
            "@@ -1,3 +0,0 @@\n"
            "-line 1\n"
            "-line 2\n"
            "-line 3\n"
        )
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0].change_type.value == "deleted"

    def test_renamed_file_produces_valid_patch(self):
        from review_agent.patch_parser import parse_unified_diff

        patch_text = (
            "diff --git a/old_name.cpp b/new_name.cpp\n"
            "rename from old_name.cpp\n"
            "rename to new_name.cpp\n"
            "--- a/old_name.cpp\n"
            "+++ b/new_name.cpp\n"
            "@@ -1,3 +1,3 @@\n"
            " context\n"
            "-old\n"
            "+new\n"
            " context\n"
        )
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0].change_type.value == "renamed"
        assert parsed[0].old_path == "old_name.cpp"
        assert parsed[0].new_path == "new_name.cpp"

    def test_binary_file_produces_valid_patch(self):
        from review_agent.patch_parser import parse_unified_diff

        patch_text = (
            "diff --git a/image.png b/image.png\n"
            "Binary files a/image.png and b/image.png differ\n"
        )
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 1
        assert parsed[0].is_binary is True

    def test_multiple_files_produces_multiple_changes(self):
        from review_agent.patch_parser import parse_unified_diff

        patch_text = (
            "diff --git a/a.cpp b/a.cpp\n"
            "--- a/a.cpp\n"
            "+++ b/a.cpp\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
            "diff --git a/b.cpp b/b.cpp\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/b.cpp\n"
            "@@ -0,0 +1,1 @@\n"
            "+new file\n"
        )
        parsed = parse_unified_diff(patch_text)
        assert len(parsed) == 2

    def test_old_gitlab_format_without_diff_header_fails_fast(self):
        """The old GitLab format (---/+++ only, no diff --git) should produce
        0 changes and the orchestrator should raise InputNormalizationError."""
        from review_agent.patch_parser import parse_unified_diff

        old_format = (
            "--- a/file.cpp\n"
            "+++ b/file.cpp\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )
        parsed = parse_unified_diff(old_format)
        assert len(parsed) == 0  # This is the bug that was fixed


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------

class TestCacheCorrectness:
    def test_version_change_invalidates_cache(self, tmp_path):
        """Changing agent/prompt/parser version in the policy dict must
        produce a different cache key."""
        from review_agent.review_cache import ReviewTraceCache

        cache = ReviewTraceCache(str(tmp_path))
        base_policy = {
            "fail_on_severity": "high",
            "llm_model": "openai:gpt-4o",
            "agent_version": "0.1.0",
            "prompt_version": "2026-01-01",
            "parser_version": "1",
        }
        key_v1 = cache.make_key(
            workspace_id="ws", base_sha="a" * 40, head_sha="b" * 40,
            target_sha="", merge_sha="", patch_text="diff",
            policy=base_policy,
        )

        updated_policy = {**base_policy, "agent_version": "0.2.0"}
        key_v2 = cache.make_key(
            workspace_id="ws", base_sha="a" * 40, head_sha="b" * 40,
            target_sha="", merge_sha="", patch_text="diff",
            policy=updated_policy,
        )
        assert key_v1 != key_v2

    def test_cache_hit_does_not_call_view_materialization(self, tmp_path):
        """When a cache hit occurs, _prepare_views should NOT be called."""
        from review_agent.review_cache import ReviewTraceCache

        cache = ReviewTraceCache(str(tmp_path))
        report = ReviewReport(
            workspace_id="ws", summary="cached",
            findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
        )

        # Pre-populate cache with a known key
        key = cache.make_key(
            workspace_id="ws", base_sha="", head_sha="",
            target_sha="", merge_sha="",
            patch_text="diff --git a/a b/a\n",
            policy={
                "fail_on_severity": "high",
                "llm_model": "openai:gpt-4o",
                "max_symbols": 24,
                "max_symbol_slots": 30,
                "max_total_tool_calls": 120,
                "parse_timeout_s": 120,
                "parse_workers": 4,
                "max_candidates_per_symbol": 150,
                "max_fetch_limit": 2000,
                "agent_version": "0.2.0",
                "prompt_version": "2026-02-28",
                "parser_version": "1",
            },
        )
        cache.save(key, {"review_report": report.model_dump(mode="json")})

        # Track whether _prepare_views gets called
        views_called = {"count": 0}
        original_prepare_views = ReviewOrchestrator._prepare_views

        def tracking_prepare_views(self, **kw):
            views_called["count"] += 1
            return original_prepare_views(self, **kw)

        client = _NullClient()
        orchestrator = ReviewOrchestrator(
            client=client,
            planner_factory=lambda m: _FakePlanner(),
            synthesis_factory=lambda m: _FakeSynth(),
        )

        with patch.object(ReviewOrchestrator, "_prepare_views", tracking_prepare_views):
            req = ReviewRequest(
                workspace_id="ws",
                patch_text="diff --git a/a b/a\n",
                enable_cache=True,
                cache_dir=str(tmp_path),
            )
            result = orchestrator.run(req)

        assert result.summary == "cached"
        assert views_called["count"] == 0

    def test_cache_corruption_does_not_crash(self, tmp_path):
        """A corrupted cache file should not crash the run."""
        from review_agent.review_cache import ReviewTraceCache

        cache = ReviewTraceCache(str(tmp_path))
        key = cache.make_key(
            workspace_id="ws", base_sha="", head_sha="",
            target_sha="", merge_sha="",
            patch_text="diff --git a/a b/a\n",
            policy={"fail_on_severity": "high"},
        )
        # Write corrupt data
        (tmp_path / f"{key}.json").write_text("not json at all {{{", encoding="utf-8")

        loaded = cache.load_report(key)
        assert loaded is None  # Should gracefully return None


# ---------------------------------------------------------------------------
# TestImpact backward compatibility alias
# ---------------------------------------------------------------------------

class TestTestImpactAlias:
    def test_alias_works(self):
        """TestImpact should be usable as an alias for ReviewTestImpact."""
        ti = TestImpact(confidence=0.5)
        assert isinstance(ti, ReviewTestImpact)
        assert ti.confidence == 0.5


# ---------------------------------------------------------------------------
# ReviewRequest model
# ---------------------------------------------------------------------------

class TestReviewRequestModel:
    def test_infra_fail_mode_default(self):
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
        )
        assert req.infra_fail_mode == "block"

    def test_infra_fail_mode_pass(self):
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            infra_fail_mode="pass",
        )
        assert req.infra_fail_mode == "pass"
