"""Unified diff parsing and seed-symbol extraction."""

from __future__ import annotations

import re
from collections import defaultdict

from review_agent.models import HunkLine, PatchChange, PatchChangeType, PatchHunk, SeedSymbol

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_QUALIFIED_RE = re.compile(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\b")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w+)\s*\(")

_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
    "new",
    "delete",
    "static_cast",
    "dynamic_cast",
    "reinterpret_cast",
    "const_cast",
}


def parse_unified_diff(patch_text: str) -> list[PatchChange]:
    """Parse a unified git diff string into typed file changes."""
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

    lines = patch_text.splitlines()
    for line in lines:
        diff_match = _DIFF_HEADER_RE.match(line)
        if diff_match:
            _finalize_current()
            old_path = diff_match.group(1).strip()
            new_path = diff_match.group(2).strip()
            current = PatchChange(
                old_path=old_path,
                new_path=new_path,
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
            old_start = int(hunk_match.group(1))
            old_count = int(hunk_match.group(2) or "1")
            new_start = int(hunk_match.group(3))
            new_count = int(hunk_match.group(4) or "1")
            current_hunk = PatchHunk(
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                lines=[],
            )
            continue

        if current_hunk is None:
            continue

        if line.startswith("\\ No newline at end of file"):
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
            old_ln = _next_old_line(current_hunk)
            new_ln = _next_new_line(current_hunk)
            current_hunk.lines.append(
                HunkLine(
                    kind="context",
                    text=line[1:],
                    old_line=old_ln,
                    new_line=new_ln,
                )
            )
            continue

    _finalize_current()
    return changes


def _next_old_line(hunk: PatchHunk) -> int:
    old_consumed = len([l for l in hunk.lines if l.kind in {"context", "del"}])
    return hunk.old_start + old_consumed


def _next_new_line(hunk: PatchHunk) -> int:
    new_consumed = len([l for l in hunk.lines if l.kind in {"context", "add"}])
    return hunk.new_start + new_consumed


def extract_seed_symbols(changes: list[PatchChange], max_symbols: int = 24) -> list[SeedSymbol]:
    """Extract likely symbols from modified lines in the patch."""
    scores: dict[str, float] = defaultdict(float)
    source: dict[str, str] = {}
    symbol_file_key: dict[str, str] = {}

    for change in changes:
        for hunk in change.hunks:
            for line in hunk.lines:
                if line.kind not in {"add", "del"}:
                    continue
                for sym in _QUALIFIED_RE.findall(line.text):
                    scores[sym] += 2.0
                    source[sym] = "qualified"
                for sym in _CALL_RE.findall(line.text):
                    if sym in _KEYWORDS:
                        continue
                    if len(sym) < 3:
                        continue
                    scores[sym] += 1.0
                    source.setdefault(sym, "call_token")
                if "(" in line.text and ")" in line.text:
                    sig = _extract_signature_name(line.text)
                    if sig:
                        scores[sig] += 1.5
                        source[sig] = "signature"
                for sym in list(scores.keys()):
                    if symbol_file_key.get(sym):
                        continue
                    symbol_file_key[sym] = ""

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:max_symbols]
    return [
        SeedSymbol(
            symbol=symbol,
            source=source.get(symbol, "patch"),
            file_key=symbol_file_key.get(symbol, ""),
            score=round(score, 3),
        )
        for symbol, score in ranked
    ]


def _extract_signature_name(line_text: str) -> str:
    text = line_text.strip()
    if "(" not in text:
        return ""
    head = text.split("(", 1)[0].strip()
    if not head:
        return ""
    tokens = re.split(r"\s+", head)
    if not tokens:
        return ""
    candidate = tokens[-1].strip("*&")
    if not candidate or candidate in _KEYWORDS:
        return ""
    return candidate

