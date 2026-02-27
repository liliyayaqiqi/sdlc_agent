from __future__ import annotations

from review_agent.analyzers import (
    analyze_architecture_risks,
    analyze_confidence_gaps,
    analyze_cross_repo_breakage,
    analyze_hidden_side_effects,
)
from review_agent.models import PatchChange, PatchChangeType, PatchHunk, HunkLine, SymbolImpact


def _patch_with_static_side_effect() -> list[PatchChange]:
    return [
        PatchChange(
            old_path="repos/repoA/src/a.cpp",
            new_path="repos/repoA/src/a.cpp",
            change_type=PatchChangeType.MODIFIED,
            hunks=[
                PatchHunk(
                    old_start=1,
                    old_count=1,
                    new_start=1,
                    new_count=2,
                    lines=[
                        HunkLine(kind="context", text="int x = 0;", old_line=1, new_line=1),
                        HunkLine(kind="add", text="static int g_state = 1;", old_line=0, new_line=2),
                    ],
                )
            ],
        )
    ]


def test_hidden_side_effect_finding_created():
    impacts = [
        SymbolImpact(
            symbol="foo",
            references=[{"file_key": "repoA:src/a.cpp"}] * 50,
            call_edges=[{"file_key": "repoA:src/a.cpp"}] * 50,
        )
    ]
    findings = analyze_hidden_side_effects(patch_changes=_patch_with_static_side_effect(), impacts=impacts)
    assert len(findings) >= 2
    assert any(f.category.value == "hidden_side_effect" for f in findings)


def test_cross_repo_and_architecture_findings_created():
    impacts = [SymbolImpact(symbol="auth::Session::Start", references=[{"file_key": "repoB:src/b.cpp"}])]
    cross = analyze_cross_repo_breakage(
        impacts=impacts,
        cross_repo_signals=[
            {
                "symbol": "auth::Session::Start",
                "external_repos": ["repoB"],
                "incoming_ref_count": 12,
                "incoming_edge_count": 2,
            }
        ],
    )
    arch = analyze_architecture_risks(
        architecture_signals=[
            {
                "symbol": "auth::Session::Start",
                "caller_repo": "repoB",
                "owner_repo": "repoA",
            }
        ]
    )
    assert cross
    assert arch


def test_confidence_gap_finding_created():
    impacts = [
        SymbolImpact(
            symbol="net::Dial",
            confidence={"verified_ratio": 0.25, "total_candidates": 20},
            unparsed=["repoA:src/net.cpp"],
        )
    ]
    findings = analyze_confidence_gaps(impacts=impacts, loop_warnings=["max_tool_rounds_reached"])
    assert findings
    assert any(f.category.value == "confidence_gap" for f in findings)

