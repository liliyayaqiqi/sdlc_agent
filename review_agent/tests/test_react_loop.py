from __future__ import annotations

from review_agent.models import PatchChange, PatchChangeType, ReviewRequest, SymbolImpact
from review_agent.react_loop import ReActLoop


class _Registry:
    def execute(self, name, state, **kwargs):
        if name == "skill_patch_intake":
            return {"seed_symbols": []}
        if name == "skill_cross_repo_breakage":
            return {"cross_repo_signals": []}
        if name == "skill_architecture_risk":
            return {"architecture_signals": []}
        if name == "skill_evidence_read":
            return {"file_evidence": {}}
        raise AssertionError(name)


def test_react_loop_runs_minimal_flow_without_symbols():
    req = ReviewRequest(workspace_id="ws_main", patch_text="diff --git a/a b/a\n")
    loop = ReActLoop(registry=_Registry())
    state = loop.run(
        request=req,
        client=object(),
        patch_changes=[PatchChange(old_path="a", new_path="a", change_type=PatchChangeType.MODIFIED)],
        changed_file_keys=[],
        changed_repos=[],
        dependency_map={},
    )
    assert "seed_symbols" in state
    assert state["cross_repo_signals"] == []
    assert state["architecture_signals"] == []

