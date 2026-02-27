from __future__ import annotations

from review_agent.context_ingestion import ReviewContextIngestor
from review_agent.models import ReviewContextBundle, ReviewRequest, Severity


def test_ingestion_supports_patch_only_request():
    req = ReviewRequest(workspace_id="ws_main", patch_text="diff --git a/a b/a\n")
    out = ReviewContextIngestor.ingest(req)
    assert out.bundle.workspace_id == "ws_main"
    assert out.bundle.patch_text
    assert out.fail_on_severity == Severity.HIGH


def test_ingestion_applies_policy_override():
    req = ReviewRequest(
        workspace_id="ws_main",
        context_bundle=ReviewContextBundle(
            workspace_id="ws_main",
            patch_text="diff --git a/a b/a\n",
            policy={"max_symbols": 5, "fail_on_severity": "medium"},
        ),
    )
    out = ReviewContextIngestor.ingest(req)
    assert out.max_symbols == 5
    assert out.fail_on_severity == Severity.MEDIUM
