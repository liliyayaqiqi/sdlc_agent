"""Hardening tests for cache safety, locking, policy, rendering, and orchestration."""

from __future__ import annotations

import json
from time import time
from unittest.mock import patch

import pytest

from review_agent.context_ingestion import ReviewContextIngestor
from review_agent.adapters.llm import build_model_services, endpoint_cache_key, resolve_llm_endpoint
from review_agent.domain.location_mapper import FindingLocationMapper
from review_agent.models import (
    CoverageSummary,
    EvidenceRef,
    FindingCategory,
    PublishResult,
    RepoRevisionContext,
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
    RetrievalStage,
    ReviewOrchestrator,
    ViewContexts,
    _analyze_test_impact,
    _build_merge_delta_signals,
    _call,
    _collect_symbol,
    _empty_impact,
    _indeterminate_report,
    _is_merge_degraded,
    _normalize_bundle_and_scope_repos,
    _policy_gate,
    _workspace_lock,
)
from review_agent.patch_parser import parse_unified_diff
from review_agent.report_renderer import render_markdown
from review_agent.review_cache import ReviewTraceCache


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
    base_url = "http://127.0.0.1:8000"

    def __init__(self):
        self.synced: list[str] = []
        self.created_contexts: list[str] = []

    def workspace_info(self):
        return {"workspace_id": "ws"}

    def context_create_pr_overlay(self, **kw):
        context_id = kw.get("context_id", "ctx")
        self.created_contexts.append(context_id)
        return {"context_id": context_id}

    def context_expire(self, **kw):
        return {}

    def sync_repo(self, **kw):
        self.synced.append(str(kw.get("repo_id", "")))
        return {}

    def explore_rg_search(self, **kw):
        return {"hits": []}

    def explore_list_candidates(self, **kw):
        return {"candidates": [], "provenance": []}

    def explore_classify_freshness(self, **kw):
        return {"stale": [], "fresh": [], "unparsed": [], "overlay_mode": "dense"}

    def explore_parse_file(self, **kw):
        return {"parsed_file_keys": [], "failed_file_keys": []}

    def explore_fetch_symbols(self, **kw):
        return {"symbols": []}

    def explore_fetch_references(self, **kw):
        return {"references": []}

    def explore_fetch_call_edges(self, **kw):
        return {"edges": []}

    def explore_get_confidence(self, **kw):
        return {"confidence": {"verified_ratio": 0.0, "total_candidates": 0}}

    def explore_read_file(self, **kw):
        return {"content": ""}

    def query_file_symbols(self, **kw):
        return {"symbols": [], "confidence": {"verified_ratio": 0.0, "total_candidates": 0}}

    def agent_investigate_symbol(self, **kw):
        return {"summary_markdown": ""}

    def close(self):
        return None


class _TestImpactClient(_NullClient):
    def explore_fetch_references(self, **kw):
        if kw.get("symbol") == "changedFunc":
            return {
                "references": [
                    {"file_key": "repoA:tests/test_foo.cpp", "line": 42},
                    {"file_key": "repoA:src/main.cpp", "line": 10},
                ]
            }
        return {"references": []}


class _FailingSyncClient(_NullClient):
    def sync_repo(self, **kw):
        raise RuntimeError("sync failed deliberately")


class _UnreachableClient(_NullClient):
    def workspace_info(self):
        raise ConnectionError("connection refused")


class _ScopedClient(_NullClient):
    def __init__(self):
        super().__init__()
        self.rg_calls: list[dict] = []
        self.list_calls: list[dict] = []

    def explore_rg_search(self, **kw):
        self.rg_calls.append(kw)
        return {"hits": []}

    def explore_list_candidates(self, **kw):
        self.list_calls.append(kw)
        return {"candidates": []}


class _BootstrapCandidateClient(_NullClient):
    def explore_list_candidates(self, **kw):
        seeded = list(kw.get("bootstrap_file_keys", []) or [])
        return {
            "candidates": seeded,
            "deleted_file_keys": [],
            "provenance": [{"file_key": file_key, "sources": ["bootstrap_seed"]} for file_key in seeded],
        }

    def explore_classify_freshness(self, **kw):
        files = list(kw.get("candidate_file_keys", []) or [])
        return {"stale": [], "fresh": files, "unparsed": [], "overlay_mode": "dense"}

    def explore_fetch_symbols(self, **kw):
        files = list(kw.get("candidate_file_keys", []) or [])
        return {"symbols": [{"file_key": files[0], "line": 10}]} if files else {"symbols": []}

    def explore_get_confidence(self, **kw):
        verified = list(kw.get("verified_files", []) or [])
        return {
            "confidence": {
                "verified_ratio": 1.0 if verified else 0.0,
                "total_candidates": len(verified),
                "verified_files": verified,
                "stale_files": [],
                "unparsed_files": [],
            }
        }


class _EmptyMacroClient(_BootstrapCandidateClient):
    def explore_fetch_symbols(self, **kw):
        return {"symbols": []}

    def explore_fetch_references(self, **kw):
        return {"references": []}

    def explore_fetch_call_edges(self, **kw):
        return {"edges": []}

    def agent_investigate_symbol(self, **kw):
        return {
            "summary_markdown": "empty",
            "metrics": {
                "total_candidates": 0,
                "definition_count": 0,
                "reference_count": 0,
                "call_edge_count": 0,
            },
            "file_paths": [],
        }


class TestBudgetExhaustion:
    def test_collect_symbol_stops_early_when_budget_exhausted(self):
        budget = _budget(calls=1)
        tool_usage: list[ToolCallRecord] = []
        impact = _collect_symbol(_NullClient(), None, {"mode": "baseline"}, "TestSymbol", tool_usage, budget)
        assert isinstance(impact, SymbolImpact)
        assert "budget_exhausted_mid_symbol" in impact.warnings

    def test_call_returns_warning_when_exhausted(self):
        budget = _budget(calls=0)
        result = _call([], budget, "test", "test.tool", lambda: {"ok": True})
        assert "warnings" in result

    def test_empty_impact_has_correct_shape(self):
        impact = _empty_impact("Sym", rg_hits=[{"file_key": "a.cpp"}])
        assert impact.symbol == "Sym"
        assert "budget_exhausted_mid_symbol" in impact.warnings

    def test_collect_symbol_passes_repo_scope(self):
        client = _ScopedClient()
        budget = _budget(calls=20)
        _collect_symbol(
            client,
            None,
            {"mode": "pr"},
            "ScopedSymbol",
            [],
            budget,
            retrieval_stages=[
                RetrievalStage(
                    name="exact_paths",
                    entry_repos=["repoA"],
                    max_repo_hops=0,
                    path_prefixes=["src/module"],
                )
            ],
        )
        assert client.rg_calls[0]["scope"] == {
            "entry_repos": ["repoA"],
            "max_repo_hops": 0,
            "path_prefixes": ["src/module"],
        }
        assert client.list_calls[0]["entry_repos"] == ["repoA"]
        assert client.list_calls[0]["max_repo_hops"] == 0
        assert client.list_calls[0]["path_prefixes"] == ["src/module"]

    def test_collect_symbol_bootstraps_candidates_without_recall(self):
        impact = _collect_symbol(
            _BootstrapCandidateClient(),
            None,
            {"mode": "pr", "context_id": "ctx"},
            "ScopedSymbol",
            [],
            _budget(calls=20),
            bootstrap_file_keys=["repoA:src/module/new_file.cpp"],
            retrieval_stages=[
                RetrievalStage(name="exact_paths", entry_repos=["repoA"], max_repo_hops=0, path_prefixes=["src/module"])
            ],
        )
        assert impact.candidate_file_keys == ["repoA:src/module/new_file.cpp"]
        assert impact.retrieval_status == "bootstrapped"
        assert "bootstrap_seed" in impact.candidate_provenance

    def test_empty_macro_fallback_is_not_treated_as_semantic_evidence(self):
        impact = _collect_symbol(
            _EmptyMacroClient(),
            None,
            {"mode": "pr", "context_id": "ctx"},
            "ScopedSymbol",
            [],
            _budget(calls=20),
            bootstrap_file_keys=["repoA:src/module/new_file.cpp"],
            retrieval_stages=[
                RetrievalStage(name="exact_paths", entry_repos=["repoA"], max_repo_hops=0, path_prefixes=["src/module"])
            ],
        )
        assert impact.macro_summary == ""
        assert "macro_fallback_empty" in impact.warnings


class TestRepoScoping:
    def test_normalize_bundle_and_scope_repos_uses_manifest_repo_id(self, tmp_path):
        workspace_root = tmp_path / "ws"
        repo_root = workspace_root / "repos" / "project_cloud" / "src"
        repo_root.mkdir(parents=True, exist_ok=True)
        (repo_root / "app.cpp").write_text("int app();", encoding="utf-8")
        manifest_path = workspace_root / "workspace.yaml"
        manifest_path.write_text(
            "\n".join(
                [
                    "workspace_id: ws",
                    "repos:",
                    "  - repo_id: project_cloud",
                    "    root: repos/project_cloud",
                    "    remote_url: https://platgit.mihoyo.com/cloud_game/nxg_cloud.git",
                    "    depends_on: [webrtc]",
                    "  - repo_id: webrtc",
                    "    root: repos/webrtc",
                    "    depends_on: []",
                ]
            ),
            encoding="utf-8",
        )
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff --git a/repos/project_cloud/src/app.cpp b/repos/project_cloud/src/app.cpp\n",
            primary_repo_id="cloud_game/nxg_cloud",
            repo_revisions=[RepoRevisionContext(repo_id="cloud_game/nxg_cloud", target_sha="a" * 40)],
        )
        normalized, entry_repos, warnings = _normalize_bundle_and_scope_repos(
            workspace={
                "workspace_id": "ws",
                "root_path": str(workspace_root),
                "manifest_path": str(manifest_path),
                "repos": ["project_cloud", "webrtc"],
            },
            bundle=bundle,
            changed_files=["repos/project_cloud/src/app.cpp"],
        )
        assert normalized.primary_repo_id == "project_cloud"
        assert normalized.repo_revisions[0].repo_id == "project_cloud"
        assert entry_repos == ["project_cloud"]
        assert warnings == []


class TestCache:
    def test_cache_key_includes_workspace_fingerprint(self, tmp_path):
        cache = ReviewTraceCache(str(tmp_path))
        key1 = cache.make_key(
            workspace_id="ws",
            base_sha="",
            head_sha="",
            target_sha="",
            merge_sha="",
            workspace_fingerprint="fp-1",
            patch_text="diff",
            policy={"llm_endpoint": "fixture|default|||"},
        )
        key2 = cache.make_key(
            workspace_id="ws",
            base_sha="",
            head_sha="",
            target_sha="",
            merge_sha="",
            workspace_fingerprint="fp-2",
            patch_text="diff",
            policy={"llm_endpoint": "fixture|default|||"},
        )
        assert key1 != key2

    def test_endpoint_cache_key_includes_gateway_base_url(self):
        req1 = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            llm_model="gateway:router-model",
            llm_base_url="https://router-a.example/v1",
        )
        req2 = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            llm_model="gateway:router-model",
            llm_base_url="https://router-b.example/v1",
        )
        assert endpoint_cache_key(req1) != endpoint_cache_key(req2)

    def test_patch_only_cache_disabled_without_fingerprint(self):
        req = ReviewRequest(workspace_id="ws", patch_text="diff --git a/a b/a\n")
        ingested = ReviewContextIngestor.ingest(req)
        assert ingested.enable_cache is False

    def test_patch_only_cache_enabled_with_fingerprint(self):
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            workspace_fingerprint="snapshot-1",
        )
        ingested = ReviewContextIngestor.ingest(req)
        assert ingested.enable_cache is True

    def test_cache_hit_does_not_call_view_materialization(self, tmp_path):
        cache = ReviewTraceCache(str(tmp_path))
        report = ReviewReport(
            workspace_id="ws",
            summary="cached",
            findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
        )
        key = cache.make_key(
            workspace_id="ws",
            base_sha="",
            head_sha="",
            target_sha="",
            merge_sha="",
            workspace_fingerprint="snapshot-1",
            patch_text="diff --git a/a b/a\n",
            policy={
                "fail_on_severity": "high",
                "llm_endpoint": "fixture|default|||",
                "max_symbols": 24,
                "max_symbol_slots": 30,
                "max_total_tool_calls": 120,
                "parse_timeout_s": 120,
                "parse_workers": 4,
                "max_candidates_per_symbol": 150,
                "max_fetch_limit": 2000,
                "agent_version": "0.3.1",
                "prompt_version": "2026-02-28",
                "parser_version": "2",
            },
        )
        cache.save(key, {"review_report": report.model_dump(mode="json")})

        views_called = {"count": 0}
        original_prepare_views = ReviewOrchestrator._prepare_views

        def tracking_prepare_views(self, **kw):
            views_called["count"] += 1
            return original_prepare_views(self, **kw)

        with patch.object(ReviewOrchestrator, "_prepare_views", tracking_prepare_views):
            orchestrator = ReviewOrchestrator(client=_NullClient())
            req = ReviewRequest(
                workspace_id="ws",
                patch_text="diff --git a/a b/a\n",
                llm_model="fixture:default",
                workspace_fingerprint="snapshot-1",
                enable_cache=True,
                cache_dir=str(tmp_path),
            )
            result = orchestrator.run(req)
        assert result.summary == "cached"
        assert views_called["count"] == 0


class TestMergeAndPolicy:
    def test_merge_delta_signal_generation(self):
        facts = [
            SymbolFact(
                symbol="Foo",
                head_reference_count=5,
                baseline_reference_count=3,
                merge_preview_reference_count=8,
                reference_delta_vs_baseline=2,
                head_call_edge_count=2,
                baseline_call_edge_count=1,
                merge_preview_call_edge_count=2,
            )
        ]
        views = ViewContexts(
            baseline={"mode": "baseline"},
            head={"mode": "pr"},
            merge_preview={"mode": "pr"},
            status=ViewContextMaterialization(merge_preview_materialized=True),
        )
        signals = _build_merge_delta_signals(facts, views)
        assert signals[0]["merge_ref_delta_vs_head"] == 3

    def test_policy_gate_low_coverage_is_not_indeterminate(self):
        report = ReviewReport(
            workspace_id="ws",
            summary="",
            findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
        )
        fact_sheet = ReviewFactSheet(
            changed_files=["src/app.cpp"],
            coverage=CoverageSummary(verified_ratio=0.1),
        )
        final = _policy_gate(
            report=report,
            fact_sheet=fact_sheet,
            test_impact=ReviewTestImpact(),
            fail_threshold=Severity.HIGH,
            tool_usage=[],
            workspace_id="ws",
        )
        assert final.decision.execution_status == ReviewExecutionStatus.PASS
        assert final.decision.review_confidence == "low"
        assert any(f.category == FindingCategory.CONFIDENCE_GAP for f in final.findings)

    def test_policy_gate_semantic_bootstrap_failure_is_indeterminate(self):
        report = ReviewReport(
            workspace_id="ws",
            summary="",
            findings=[],
            coverage=CoverageSummary(),
            decision=ReviewDecision(fail_threshold=Severity.HIGH, blocking_findings=0, should_block=False),
        )
        fact_sheet = ReviewFactSheet(
            changed_files=["src/app.cpp"],
            warnings=["semantic_bootstrap_failed"],
            coverage=CoverageSummary(verified_ratio=0.0),
        )
        final = _policy_gate(
            report=report,
            fact_sheet=fact_sheet,
            test_impact=ReviewTestImpact(),
            fail_threshold=Severity.HIGH,
            tool_usage=[],
            workspace_id="ws",
        )
        assert final.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert final.decision.indeterminate_reason == "semantic_bootstrap_failed"
        assert any("semantic_bootstrap_failed" in finding.tags for finding in final.findings)

    def test_indeterminate_report_has_low_confidence(self):
        report = _indeterminate_report(
            workspace_id="ws",
            reason="test_reason",
            summary="Testing indeterminate",
            fail_threshold=Severity.HIGH,
            should_block=True,
            run_id="r123",
        )
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert report.decision.review_confidence == "low"

    def test_merge_degraded_when_repo_revisions_missing(self):
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff",
            merge_preview_sha="a" * 40,
        )
        views = ViewContexts(
            baseline={},
            head={},
            merge_preview={"mode": "pr"},
            status=ViewContextMaterialization(merge_preview_materialized=True),
        )
        assert _is_merge_degraded(bundle, views) is True


class TestLocationAndRendering:
    def test_location_mapper_maps_to_added_line(self):
        patch = (
            "diff --git a/src/app.cpp b/src/app.cpp\n"
            "--- a/src/app.cpp\n"
            "+++ b/src/app.cpp\n"
            "@@ -1,2 +1,3 @@\n"
            " int main() {\n"
            "+  doLogin();\n"
            " }\n"
        )
        changes = parse_unified_diff(patch)
        bundle = ReviewContextBundle(workspace_id="ws", patch_text=patch)
        mapper = FindingLocationMapper(changes=changes, bundle=bundle)
        finding = ReviewFinding(
            id="f1",
            severity=Severity.HIGH,
            category=FindingCategory.ARCHITECTURE_RISK,
            title="Risk",
            impact="impact",
            recommendation="rec",
            evidence=[EvidenceRef(tool="explore.fetch_references", file_key="repoA:src/app.cpp", line=99)],
        )
        mapped = mapper.apply(finding)
        assert mapped.location is not None
        assert mapped.location.path == "src/app.cpp"
        assert mapped.location.line == 2

    def test_report_renderer_includes_location_publish_and_confidence(self):
        report = ReviewReport(
            workspace_id="ws_test",
            summary="Test summary",
            findings=[
                ReviewFinding(
                    id="f1",
                    severity=Severity.HIGH,
                    category=FindingCategory.ARCHITECTURE_RISK,
                    title="Missing null check",
                    impact="Crash",
                    recommendation="Add check",
                    evidence=[EvidenceRef(tool="explore.rg_search", file_key="a.cpp", line=10)],
                    diff_path="src/app.cpp",
                    diff_line=2,
                )
            ],
            coverage=CoverageSummary(verified_ratio=0.85, total_candidates=10),
            decision=ReviewDecision(
                fail_threshold=Severity.HIGH,
                blocking_findings=1,
                should_block=True,
                review_confidence="high",
                execution_status=ReviewExecutionStatus.BLOCK,
            ),
            tool_usage=[ToolCallRecord(skill="s", tool="t", success=True, elapsed_ms=5.0)],
            publish_result=PublishResult(provider="gitlab", summary_posted=True, inline_comments_posted=1),
        )
        md = render_markdown(report)
        assert "Review confidence" in md
        assert "Publish Result" in md
        assert "src/app.cpp:2" in md


class TestLocking:
    def test_expired_lock_removed_on_acquire(self, tmp_path):
        lock_root = str(tmp_path)
        locks_dir = tmp_path / ".workspace_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / "ws_test.lock"
        lock_file.write_text(
            json.dumps({"workspace_id": "ws_test", "run_id": "old", "expires_at": time() - 60}),
            encoding="utf-8",
        )
        with _workspace_lock(lock_root=lock_root, workspace_id="ws_test", timeout_s=2.0, run_id="new"):
            pass
        assert not lock_file.exists()

    def test_unexpired_lock_times_out(self, tmp_path):
        lock_root = str(tmp_path)
        locks_dir = tmp_path / ".workspace_locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_file = locks_dir / "ws_test.lock"
        lock_file.write_text(
            json.dumps({"workspace_id": "ws_test", "run_id": "live", "expires_at": time() + 60}),
            encoding="utf-8",
        )
        with pytest.raises(TimeoutError):
            with _workspace_lock(lock_root=lock_root, workspace_id="ws_test", timeout_s=1.0, run_id="next"):
                pass


class TestViewPreparation:
    def test_prepare_views_syncs_all_repo_revisions(self, tmp_path):
        client = _NullClient()
        orchestrator = ReviewOrchestrator(client=client)
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            head_sha="b" * 40,
            target_branch_head_sha="c" * 40,
            repo_revisions=[
                RepoRevisionContext(repo_id="repoA", target_sha="c" * 40, head_sha="b" * 40, role="primary"),
                RepoRevisionContext(repo_id="repoB", target_sha="d" * 40, head_sha="e" * 40, role="dependency"),
            ],
            pr_metadata={"pr_id": "1"},
        )
        views = orchestrator._prepare_views(
            client=client,
            runtime_context=bundle,
            lock_root=str(tmp_path),
            created_context_ids=[],
            run_id="run123",
        )
        assert views.status.baseline_materialized is True
        assert client.synced == ["repoA", "repoB"]
        assert views.status.head_context_id.endswith(":run123")

    def test_baseline_not_materialized_on_sync_failure(self, tmp_path):
        orchestrator = ReviewOrchestrator(client=_FailingSyncClient())
        bundle = ReviewContextBundle(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            head_sha="b" * 40,
            target_branch_head_sha="c" * 40,
            repo_revisions=[RepoRevisionContext(repo_id="repoA", target_sha="c" * 40, head_sha="b" * 40, role="primary")],
            pr_metadata={"pr_id": "1"},
        )
        views = orchestrator._prepare_views(
            client=_FailingSyncClient(),
            runtime_context=bundle,
            lock_root=str(tmp_path),
            created_context_ids=[],
            run_id="run123",
        )
        assert views.status.baseline_materialized is False
        assert any("baseline_sync_failed:repoA" in warn for warn in views.status.warnings)


class TestHealthCheckAndImpact:
    def test_unreachable_backend_gives_indeterminate_report(self):
        orchestrator = ReviewOrchestrator(client=_UnreachableClient())
        req = ReviewRequest(workspace_id="ws", patch_text="diff --git a/a b/a\n", enable_cache=False, llm_model="fixture:default")
        report = orchestrator.run(req)
        assert report.decision.execution_status == ReviewExecutionStatus.INDETERMINATE
        assert report.decision.should_block is True

    def test_test_impact_analysis(self):
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
        result = _analyze_test_impact(
            prepass=prepass,
            impacts=[impact],
            client=_TestImpactClient(),
            views=ViewContexts(baseline={}, head={"mode": "pr"}, merge_preview=None, status=ViewContextMaterialization()),
            budget=_budget(calls=20),
            tool_usage=[],
        )
        assert "tests/test_bar.cpp" in result.directly_impacted_tests
        assert "tests/test_foo.cpp" in result.likely_impacted_tests


class TestBackwardCompatibility:
    def test_testimpact_alias_works(self):
        ti = TestImpact(confidence=0.5)
        assert isinstance(ti, ReviewTestImpact)
        assert ti.confidence == 0.5

    def test_run_metadata_roundtrips(self):
        report = ReviewReport(
            workspace_id="ws",
            summary="test",
            decision=ReviewDecision(
                fail_threshold=Severity.HIGH,
                blocking_findings=0,
                should_block=False,
                review_confidence="medium",
            ),
            run_metadata=RunMetadata(input_mode="gitlab_mr", run_id="r123"),
            run_id="r123",
        )
        assert report.run_metadata is not None
        assert report.run_metadata.input_mode == "gitlab_mr"


class TestLlmEndpointConfig:
    def test_openrouter_request_is_valid(self):
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            llm_model="openrouter:anthropic/claude-sonnet-4-5",
            llm_api_key="token",
        )
        endpoint = resolve_llm_endpoint(req)
        assert endpoint.provider == "openrouter"
        assert endpoint.model_name == "anthropic/claude-sonnet-4-5"

    def test_gateway_request_requires_base_url(self):
        with pytest.raises(ValueError, match="llm_base_url is required"):
            ReviewRequest(
                workspace_id="ws",
                patch_text="diff --git a/a b/a\n",
                llm_model="gateway:claude-sonnet-4-5",
            )

    def test_build_model_services_accepts_gateway_request(self):
        req = ReviewRequest(
            workspace_id="ws",
            patch_text="diff --git a/a b/a\n",
            llm_model="gateway:claude-sonnet-4-5",
            llm_base_url="https://router.example/v1",
            llm_api_key="token",
        )
        planner, exploration, synthesis = build_model_services(req)
        assert planner is not None
        assert exploration is not None
        assert synthesis is not None
