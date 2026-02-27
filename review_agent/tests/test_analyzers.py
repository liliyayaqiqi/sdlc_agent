from __future__ import annotations

from review_agent.analyzers import (
    analyze_architecture_risks,
    analyze_confidence_gaps,
    analyze_cross_repo_breakage,
    analyze_hidden_side_effects,
    rank_findings,
)


def test_legacy_analyzers_are_noop():
    assert analyze_hidden_side_effects(patch_changes=[], impacts=[]) == []
    assert analyze_cross_repo_breakage(impacts=[], cross_repo_signals=[]) == []
    assert analyze_architecture_risks(architecture_signals=[]) == []
    assert analyze_confidence_gaps(impacts=[], loop_warnings=[]) == []
    assert rank_findings([]) == []
