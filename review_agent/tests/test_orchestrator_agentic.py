from __future__ import annotations

from review_agent.models import ReviewRequest
from review_agent.orchestrator import ReviewOrchestrator


class _FakeClient:
    base_url = "http://127.0.0.1:8000"

    def workspace_info(self):
        return {"workspace_id": "ws_main", "contexts": ["ws_main:baseline"], "repos": ["repoA"]}

    def context_create_pr_overlay(self, **kw):
        return {"context_id": kw.get("context_id", "ctx"), "partial_overlay": False}

    def context_expire(self, **kw):
        return {}

    def sync_repo(self, **kw):
        return {}

    def explore_rg_search(self, **kw):
        return {"hits": [{"file_key": "repoA:src/app.cpp", "line": 3, "line_text": "doLogin();"}]}

    def explore_list_candidates(self, **kw):
        return {"candidates": ["repoA:src/app.cpp"], "deleted_file_keys": []}

    def explore_classify_freshness(self, **kw):
        files = list(kw.get("candidate_file_keys", []) or [])
        return {"stale": [], "fresh": files, "unparsed": [], "overlay_mode": "dense"}

    def explore_parse_file(self, **kw):
        return {"parsed_file_keys": list(kw.get("file_keys", []) or []), "failed_file_keys": []}

    def explore_fetch_symbols(self, **kw):
        return {"symbols": [{"file_key": "repoA:src/app.cpp", "line": 3}]}

    def explore_fetch_references(self, **kw):
        return {"references": [{"file_key": "repoA:src/app.cpp", "line": 3}]}

    def explore_fetch_call_edges(self, **kw):
        return {"edges": [{"file_key": "repoA:src/app.cpp", "line": 3, "caller": "main", "callee": kw.get("symbol", "")}]}

    def explore_get_confidence(self, **kw):
        return {
            "confidence": {
                "verified_ratio": 0.9,
                "total_candidates": 1,
                "verified_files": list(kw.get("verified_files", []) or []),
                "stale_files": list(kw.get("stale_files", []) or []),
                "unparsed_files": list(kw.get("unparsed_files", []) or []),
            }
        }

    def explore_read_file(self, **kw):
        return {"file_key": kw.get("file_key", ""), "line_range": [1, 10], "content": "int main() {\n  doLogin();\n}\n"}

    def agent_investigate_symbol(self, **kw):
        return {"summary_markdown": ""}

    def close(self):
        pass


def test_orchestrator_builds_fact_sheet_policy_and_locations():
    orchestrator = ReviewOrchestrator(client=_FakeClient())
    req = ReviewRequest(
        workspace_id="ws_main",
        patch_text="diff --git a/src/app.cpp b/src/app.cpp\n--- a/src/app.cpp\n+++ b/src/app.cpp\n@@ -1,2 +1,3 @@\n int main() {\n+  doLogin();\n }\n",
        llm_model="fixture:blocking",
        fail_on_severity="high",
        enable_cache=False,
    )
    report = orchestrator.run(req)
    assert report.workspace_id == "ws_main"
    assert report.fact_sheet is not None
    assert report.test_impact is not None
    assert report.decision.execution_status == "block"
    assert report.decision.review_confidence == "high"
    assert report.findings[0].location is not None
    assert report.findings[0].location.path == "src/app.cpp"
    assert report.findings[0].location.line == 2
