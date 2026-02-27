"""Patch-intake skill: derive high-level patch heuristics."""

from __future__ import annotations

from typing import Any

def run_patch_intake(state: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
    """Extract patch-level signals from parsed changes."""
    patch_changes = state.get("patch_changes", [])
    deleted_paths = []
    renamed_paths = []
    for change in patch_changes:
        change_type = getattr(getattr(change, "change_type", None), "value", str(getattr(change, "change_type", "")))
        if change_type == "deleted":
            deleted_paths.append(change.old_path)
        if change_type == "renamed":
            renamed_paths.append(f"{change.old_path}->{change.new_path}")
    return {
        "seed_symbols": [],
        "patch_signals": {
            "changed_file_count": len(patch_changes),
            "deleted_paths": deleted_paths,
            "renamed_paths": renamed_paths,
        },
    }
