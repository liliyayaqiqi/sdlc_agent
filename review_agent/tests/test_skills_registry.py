from __future__ import annotations

from review_agent.patch_parser import parse_unified_diff
from review_agent.skills.registry import SkillRegistry


class _Req:
    max_symbols = 8


def test_registry_loads_catalog_names():
    reg = SkillRegistry()
    assert "skill_patch_intake" in reg.names
    assert "skill_symbol_impact" in reg.names
    assert "skill_architecture_risk" in reg.names


def test_patch_intake_skill_extracts_seed_symbols():
    reg = SkillRegistry()
    patch = """diff --git a/a.cpp b/a.cpp
--- a/a.cpp
+++ b/a.cpp
@@ -1 +1,2 @@
-foo();
+foo();
+auth::Session::Start();
"""
    changes = parse_unified_diff(patch)
    updates = reg.execute("skill_patch_intake", {"patch_changes": changes, "request": _Req()})
    assert updates["seed_symbols"] == []
    assert updates["patch_signals"]["changed_file_count"] == 1
