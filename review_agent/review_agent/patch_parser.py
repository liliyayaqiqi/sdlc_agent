"""Unified diff parsing and deterministic pre-pass extraction."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import PurePosixPath

logger = logging.getLogger("review_agent.patch_parser")

from review_agent.models import (
    HunkLine,
    PatchChange,
    PatchChangeType,
    PatchHunk,
    PrepassResult,
    SeedSymbol,
    SuspiciousAnchor,
)

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_QUALIFIED_RE = re.compile(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\b")
_CALL_RE = re.compile(r"\b([A-Za-z_]\w+)\s*\(")
_METHOD_SIG_RE = re.compile(
    r"^\s*(?:template\s*<[^>]+>\s*)?(?:[\w:\<\>\*&\s]+\s+)?([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*\([^;]*\)\s*(?:const\b)?\s*(?:noexcept\b)?"
)
_INCLUDE_RE = re.compile(r"^\s*#\s*include\b")
_MACRO_RE = re.compile(r"^\s*#\s*(?:define|undef|ifdef|ifndef|if|elif|endif|pragma)\b")

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

_SUSPICIOUS_ANCHORS = [
    (re.compile(r"\bmutex\b|\block_guard\b|\bunique_lock\b"), "concurrency"),
    (re.compile(r"\bstd::atomic\b|\bmemory_order\b"), "atomic"),
    (re.compile(r"\bdelete\b|\bfree\s*\(|\bunique_ptr\b|\bshared_ptr\b"), "lifetime"),
    (re.compile(r"\bthrow\b|\btry\b|\bcatch\b"), "exception_flow"),
    (re.compile(r"\bvirtual\b|\boverride\b|\bfinal\b"), "abi_or_dispatch"),
]

_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".cmake"}


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


def extract_seed_symbols(changes: list[PatchChange], *, max_symbols: int = 24) -> list[SeedSymbol]:
    """Extract ranked seed symbols from add/del lines."""
    scores: dict[str, float] = defaultdict(float)
    sources: dict[str, str] = {}

    for change in changes:
        for hunk in change.hunks:
            for line in hunk.lines:
                if line.kind not in {"add", "del"}:
                    continue
                for symbol in _QUALIFIED_RE.findall(line.text):
                    scores[symbol] += 2.4 if line.kind == "add" else 1.8
                    sources[symbol] = "qualified"
                for symbol in _CALL_RE.findall(line.text):
                    if symbol in _KEYWORDS or len(symbol) < 3:
                        continue
                    scores[symbol] += 1.2 if line.kind == "add" else 1.0
                    sources.setdefault(symbol, "call_site")
                maybe_sig = _extract_signature_name(line.text)
                if maybe_sig:
                    scores[maybe_sig] += 1.6
                    sources[maybe_sig] = "signature"

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[: max(1, max_symbols)]
    return [
        SeedSymbol(symbol=symbol, source=sources.get(symbol, "patch"), score=round(score, 4))
        for symbol, score in ranked
    ]


def build_prepass_result(changes: list[PatchChange], *, max_symbols: int = 24) -> PrepassResult:
    """Build deterministic patch pre-pass output."""
    changed_files = sorted({change.effective_path for change in changes if change.effective_path})
    changed_hunk_count = sum(len(change.hunks) for change in changes)
    seed_symbols = extract_seed_symbols(changes, max_symbols=max_symbols)
    changed_methods: set[str] = set()
    added_call_sites: set[str] = set()
    removed_call_sites: set[str] = set()
    include_macro_config_changes: set[str] = set()
    suspicious_anchors: list[SuspiciousAnchor] = []
    warnings: list[str] = []

    for change in changes:
        path = change.effective_path
        if _is_config_file(path):
            include_macro_config_changes.add(f"config:{path}")
        for hunk in change.hunks:
            for line in hunk.lines:
                if line.kind not in {"add", "del"}:
                    continue
                text = line.text
                lowered = text.lower()
                location_line = line.new_line if line.kind == "add" else line.old_line

                signature_name = _extract_signature_name(text)
                if signature_name:
                    changed_methods.add(signature_name)

                if _INCLUDE_RE.search(text):
                    include_macro_config_changes.add(f"include:{path}")
                if _MACRO_RE.search(text):
                    include_macro_config_changes.add(f"macro:{path}")

                for call_name in _CALL_RE.findall(text):
                    if call_name in _KEYWORDS or len(call_name) < 3:
                        continue
                    if line.kind == "add":
                        added_call_sites.add(call_name)
                    else:
                        removed_call_sites.add(call_name)

                for pattern, kind in _SUSPICIOUS_ANCHORS:
                    if pattern.search(lowered):
                        suspicious_anchors.append(
                            SuspiciousAnchor(
                                kind=kind,
                                file_path=path,
                                line=location_line,
                                reason=f"matched:{kind}",
                                snippet=text[:240],
                            )
                        )

                if "TODO" in text or "FIXME" in text:
                    warnings.append(f"todo_marker:{path}:{location_line}")

    return PrepassResult(
        changed_files=changed_files,
        changed_hunk_count=changed_hunk_count,
        seed_symbols=seed_symbols,
        suspicious_anchors=suspicious_anchors[:500],
        changed_methods=sorted(changed_methods),
        added_call_sites=sorted(added_call_sites),
        removed_call_sites=sorted(removed_call_sites),
        include_macro_config_changes=sorted(include_macro_config_changes),
        warnings=sorted(set(warnings)),
    )


def _next_old_line(hunk: PatchHunk) -> int:
    old_consumed = len([line for line in hunk.lines if line.kind in {"context", "del"}])
    return hunk.old_start + old_consumed


def _next_new_line(hunk: PatchHunk) -> int:
    new_consumed = len([line for line in hunk.lines if line.kind in {"context", "add"}])
    return hunk.new_start + new_consumed


def _extract_signature_name(text: str) -> str:
    stripped = text.strip()
    if not stripped or stripped.startswith("//"):
        return ""
    match = _METHOD_SIG_RE.match(stripped)
    if not match:
        return ""
    symbol = match.group(1).strip()
    if not symbol or symbol in _KEYWORDS:
        return ""
    return symbol


def _is_config_file(path: str) -> bool:
    p = PurePosixPath(path)
    if p.name in {"CMakeLists.txt", "BUILD", "BUILD.bazel"}:
        return True
    return p.suffix.lower() in _CONFIG_EXTENSIONS
