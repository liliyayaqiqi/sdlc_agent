"""Unified diff parsing helpers."""

from __future__ import annotations

import re

from review_agent.models import HunkLine, PatchChange, PatchChangeType, PatchHunk

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_unified_diff(patch_text: str) -> list[PatchChange]:
    """Parse a unified git diff string into file changes and hunks."""
    changes: list[PatchChange] = []
    current: PatchChange | None = None
    current_hunk: PatchHunk | None = None

    def _finalize_current() -> None:
        nonlocal current, current_hunk
        if current is None:
            return
        if current_hunk is not None:
            current.hunks.append(current_hunk)
            current_hunk = None
        if current.old_path == "/dev/null":
            current.change_type = PatchChangeType.ADDED
        elif current.new_path == "/dev/null":
            current.change_type = PatchChangeType.DELETED
        changes.append(current)
        current = None

    for line in patch_text.splitlines():
        diff_match = _DIFF_HEADER_RE.match(line)
        if diff_match:
            _finalize_current()
            current = PatchChange(
                old_path=diff_match.group(1).strip(),
                new_path=diff_match.group(2).strip(),
                change_type=PatchChangeType.MODIFIED,
                hunks=[],
                is_binary=False,
            )
            current_hunk = None
            continue

        if current is None:
            continue

        if line.startswith("new file mode "):
            current.change_type = PatchChangeType.ADDED
            continue
        if line.startswith("deleted file mode "):
            current.change_type = PatchChangeType.DELETED
            continue
        if line.startswith("rename from "):
            current.change_type = PatchChangeType.RENAMED
            current.old_path = line[len("rename from ") :].strip()
            continue
        if line.startswith("rename to "):
            current.change_type = PatchChangeType.RENAMED
            current.new_path = line[len("rename to ") :].strip()
            continue
        if line.startswith("Binary files "):
            current.is_binary = True
            continue
        if line.startswith("--- "):
            path = line[4:].strip()
            current.old_path = path[2:] if path.startswith("a/") else path
            continue
        if line.startswith("+++ "):
            path = line[4:].strip()
            current.new_path = path[2:] if path.startswith("b/") else path
            continue

        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match:
            if current_hunk is not None:
                current.hunks.append(current_hunk)
            current_hunk = PatchHunk(
                old_start=int(hunk_match.group(1)),
                old_count=int(hunk_match.group(2) or "1"),
                new_start=int(hunk_match.group(3)),
                new_count=int(hunk_match.group(4) or "1"),
                lines=[],
            )
            continue

        if current_hunk is None or line.startswith("\\ No newline at end of file"):
            continue

        if line.startswith("+"):
            current_hunk.lines.append(
                HunkLine(
                    kind="add",
                    text=line[1:],
                    old_line=0,
                    new_line=_next_new_line(current_hunk),
                )
            )
            continue
        if line.startswith("-"):
            current_hunk.lines.append(
                HunkLine(
                    kind="del",
                    text=line[1:],
                    old_line=_next_old_line(current_hunk),
                    new_line=0,
                )
            )
            continue
        if line.startswith(" "):
            current_hunk.lines.append(
                HunkLine(
                    kind="context",
                    text=line[1:],
                    old_line=_next_old_line(current_hunk),
                    new_line=_next_new_line(current_hunk),
                )
            )

    _finalize_current()
    return changes


def _next_old_line(hunk: PatchHunk) -> int:
    old_consumed = len([line for line in hunk.lines if line.kind in {"context", "del"}])
    return hunk.old_start + old_consumed


def _next_new_line(hunk: PatchHunk) -> int:
    new_consumed = len([line for line in hunk.lines if line.kind in {"context", "add"}])
    return hunk.new_start + new_consumed
