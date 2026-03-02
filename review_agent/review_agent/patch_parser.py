"""Unified diff parsing and deterministic pre-pass extraction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import logging
import re
from pathlib import PurePosixPath

from review_agent.models import (
    ChangedDeclaration,
    DiffExcerpt,
    HunkLine,
    MemberCallSite,
    PatchChange,
    PatchChangeType,
    PatchHunk,
    PrepassResult,
    SeedRelevanceTier,
    SeedSymbol,
    SuspiciousAnchor,
)

logger = logging.getLogger("review_agent.patch_parser")

_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

_NAME_TOKEN = r"~?[A-Za-z_]\w*"
_QUALIFIED_TOKEN = rf"{_NAME_TOKEN}(?:::{_NAME_TOKEN})*"
_QUALIFIED_RE = re.compile(rf"\b{_NAME_TOKEN}(?:::{_NAME_TOKEN})+\b")
_CALL_RE = re.compile(rf"\b(?P<name>{_NAME_TOKEN})\s*\(")
_MEMBER_CALL_RE = re.compile(rf"\b(?P<receiver>{_QUALIFIED_TOKEN})\s*(?P<op>\.|->)\s*(?P<member>{_NAME_TOKEN})\s*\(")
_QUALIFIED_CALL_RE = re.compile(rf"\b(?P<receiver>{_NAME_TOKEN}(?:::{_NAME_TOKEN})*)::(?P<member>{_NAME_TOKEN})\s*\(")
_TYPE_DECL_RE = re.compile(rf"^\s*(?P<kind>class|struct|enum(?:\s+class)?)\s+(?P<name>{_NAME_TOKEN})\b")
_METHOD_SIG_RE = re.compile(
    rf"^\s*"
    rf"(?:template\s*<[^>]+>\s*)?"
    rf"(?:(?:inline|constexpr|virtual|static|explicit|friend|extern|mutable|typename)\s+)*"
    rf"(?:[\w:\<\>\*&,\s]+\s+)?"
    rf"(?P<name>{_QUALIFIED_TOKEN})"
    rf"\s*\([^;{{}}]*\)"
    rf"\s*(?:const\b)?"
    rf"\s*(?:noexcept\b)?"
    rf"(?:\s*=\s*(?:0|default|delete))?"
    rf"\s*(?:\{{|;)?\s*$"
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
    "this",
}

_INVALID_DECLARATION_NAMES = {
    "void",
    "int",
    "char",
    "bool",
    "float",
    "double",
    "short",
    "long",
    "signed",
    "unsigned",
    "size_t",
    "ssize_t",
    "auto",
}

_SYSTEM_FUNCTION_HINTS = {
    "CreateEvent",
    "CreateEventA",
    "CreateEventW",
    "SetEvent",
    "ResetEvent",
    "CloseHandle",
    "GetLastError",
    "memcpy",
    "memmove",
    "memcmp",
    "strlen",
    "strcpy",
}

_SUSPICIOUS_ANCHORS = [
    (re.compile(r"\bmutex\b|\block_guard\b|\bunique_lock\b"), "concurrency"),
    (re.compile(r"\bstd::atomic\b|\bmemory_order\b"), "atomic"),
    (re.compile(r"\bdelete\b|\bfree\s*\(|\bunique_ptr\b|\bshared_ptr\b"), "lifetime"),
    (re.compile(r"\bthrow\b|\btry\b|\bcatch\b"), "exception_flow"),
    (re.compile(r"\bvirtual\b|\boverride\b|\bfinal\b"), "abi_or_dispatch"),
]

_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".cmake"}
_TIER_PRIORITY: dict[SeedRelevanceTier, int] = {
    "declaration": 4,
    "qualified": 3,
    "receiver_owned": 2,
    "generic_fallback": 1,
}
_MAX_EXCERPT_LINES = 20
_MAX_EXCERPTS = 30


@dataclass
class _SeedCandidate:
    symbol: str
    source: str
    tier: SeedRelevanceTier
    score: float = 0.0
    reasons: set[str] = field(default_factory=set)
    receiver: str = ""
    container: str = ""
    file_paths: set[str] = field(default_factory=set)
    declared_locally: bool = False
    qualified_occurrence: bool = False
    low_entropy: bool = False
    standard_like: bool = False
    owner_anchors: set[str] = field(default_factory=set)
    hunk_keys: set[str] = field(default_factory=set)

    def register(
        self,
        *,
        source: str,
        tier: SeedRelevanceTier,
        score_delta: float,
        reason: str,
        file_path: str,
        receiver: str = "",
        container: str = "",
        declared_locally: bool = False,
        qualified_occurrence: bool = False,
        hunk_key: str = "",
    ) -> None:
        if _TIER_PRIORITY[tier] > _TIER_PRIORITY[self.tier]:
            self.tier = tier
            self.source = source
        elif _TIER_PRIORITY[tier] == _TIER_PRIORITY[self.tier] and self.source == "call_site":
            self.source = source
        self.score += score_delta
        self.reasons.add(reason)
        if receiver and not self.receiver:
            self.receiver = receiver
        if container and not self.container:
            self.container = container
        if file_path:
            self.file_paths.add(file_path)
        if hunk_key:
            self.hunk_keys.add(hunk_key)
            if container:
                self.owner_anchors.add(f"{container}|{hunk_key}")
        self.declared_locally = self.declared_locally or declared_locally
        self.qualified_occurrence = self.qualified_occurrence or qualified_occurrence
        self.low_entropy = self.low_entropy or _is_low_entropy_symbol(self.symbol)
        self.standard_like = self.standard_like or _is_standard_or_system_symbol(self.symbol)

    def repeated_owner_context(self) -> bool:
        grouped: dict[str, set[str]] = defaultdict(set)
        for raw in self.owner_anchors:
            owner, hunk_key = raw.split("|", 1)
            grouped[owner].add(hunk_key)
        return any(len(hunks) >= 2 for hunks in grouped.values())


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
                HunkLine(kind="add", text=line[1:], old_line=0, new_line=_next_new_line(current_hunk))
            )
            continue
        if line.startswith("-"):
            current_hunk.lines.append(
                HunkLine(kind="del", text=line[1:], old_line=_next_old_line(current_hunk), new_line=0)
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
    return build_prepass_result(changes, max_symbols=max_symbols).seed_symbols


def build_prepass_result(changes: list[PatchChange], *, max_symbols: int = 24) -> PrepassResult:
    """Build deterministic patch pre-pass output."""
    changed_files = sorted({change.effective_path for change in changes if change.effective_path})
    changed_hunk_count = sum(len(change.hunks) for change in changes)

    candidates: dict[str, _SeedCandidate] = {}
    changed_declarations: list[ChangedDeclaration] = []
    member_call_sites: list[MemberCallSite] = []
    suspicious_anchors: list[SuspiciousAnchor] = []
    include_macro_config_changes: set[str] = set()
    added_call_sites: set[str] = set()
    removed_call_sites: set[str] = set()
    warnings: list[str] = []
    changed_containers: set[str] = set()
    file_coverage: dict[str, set[str]] = defaultdict(set)
    excerpt_candidates: list[tuple[int, int, DiffExcerpt]] = []

    for change in changes:
        path = change.effective_path
        if _is_config_file(path):
            include_macro_config_changes.add(f"config:{path}")
        active_container = ""
        active_symbol = ""

        for hunk_index, hunk in enumerate(change.hunks):
            hunk_key = f"{path}:{hunk_index}"
            hunk_decls: list[str] = []
            hunk_member_calls: list[MemberCallSite] = []
            hunk_anchor_kinds: set[str] = set()

            for line in hunk.lines:
                if line.kind == "context":
                    if line.text.strip().startswith(("};", "}")):
                        active_container = ""
                        active_symbol = ""
                    continue
                text = line.text
                lowered = text.lower()
                location_line = line.new_line if line.kind == "add" else line.old_line

                line_anchor_kinds: set[str] = set()
                for pattern, kind in _SUSPICIOUS_ANCHORS:
                    if pattern.search(lowered):
                        line_anchor_kinds.add(kind)
                        suspicious_anchors.append(
                            SuspiciousAnchor(
                                kind=kind,
                                file_path=path,
                                line=location_line,
                                reason=f"matched:{kind}",
                                snippet=text[:240],
                            )
                        )
                hunk_anchor_kinds.update(line_anchor_kinds)

                if "TODO" in text or "FIXME" in text:
                    warnings.append(f"todo_marker:{path}:{location_line}")
                if _INCLUDE_RE.search(text):
                    include_macro_config_changes.add(f"include:{path}")
                if _MACRO_RE.search(text):
                    include_macro_config_changes.add(f"macro:{path}")

                declaration = _extract_declaration(
                    text=text,
                    file_path=path,
                    line=location_line,
                    active_container=active_container,
                    active_symbol=active_symbol,
                )
                if declaration is not None:
                    changed_declarations.append(declaration)
                    if declaration.container:
                        active_container = declaration.container
                        changed_containers.add(declaration.container)
                    elif declaration.kind in {"class", "struct", "enum"}:
                        active_container = declaration.symbol
                        changed_containers.add(declaration.symbol)
                    active_symbol = declaration.symbol
                    hunk_decls.append(declaration.symbol)
                    file_coverage[path].add(declaration.symbol)
                    _record_candidate(
                        candidates,
                        symbol=declaration.symbol,
                        source="declaration",
                        tier="declaration",
                        score_delta=4.0,
                        reason=f"changed_declaration:{declaration.kind}",
                        file_path=path,
                        container=declaration.container,
                        declared_locally=True,
                        hunk_key=hunk_key,
                    )
                    if line_anchor_kinds:
                        _boost_candidate(candidates[declaration.symbol], line_anchor_kinds)

                qualified_hits = {match.group(0) for match in _QUALIFIED_RE.finditer(text) if _looks_symbolic(match.group(0))}
                for symbol in sorted(qualified_hits):
                    if symbol.startswith("std::") and not declaration:
                        score_delta = 1.0
                    else:
                        score_delta = 3.0
                    _record_candidate(
                        candidates,
                        symbol=symbol,
                        source="qualified",
                        tier="qualified",
                        score_delta=score_delta,
                        reason="qualified_occurrence",
                        file_path=path,
                        container=_container_from_symbol(symbol),
                        qualified_occurrence=True,
                        hunk_key=hunk_key,
                    )
                    file_coverage[path].add(symbol)
                    if line_anchor_kinds:
                        _boost_candidate(candidates[symbol], line_anchor_kinds)

                line_member_calls: list[MemberCallSite] = []
                if declaration is None:
                    line_member_calls = _extract_member_calls(
                        text=text,
                        file_path=path,
                        line=location_line,
                        line_kind=line.kind,
                        active_symbol=active_symbol,
                        active_container=active_container,
                    )
                for call in line_member_calls:
                    member_call_sites.append(call)
                    hunk_member_calls.append(call)
                    compound = _compound_call_name(call)
                    if line.kind == "add":
                        added_call_sites.add(compound)
                    else:
                        removed_call_sites.add(compound)

                    owner_symbol = call.container or call.qualified_receiver_type or call.receiver
                    if owner_symbol and owner_symbol not in _KEYWORDS:
                        _record_candidate(
                            candidates,
                            symbol=owner_symbol,
                            source="member_owner",
                            tier="receiver_owned",
                            score_delta=2.5 if call.container else 2.0,
                            reason=f"member_call:{call.member}",
                            file_path=path,
                            receiver=call.receiver,
                            container=call.container,
                            hunk_key=hunk_key,
                        )
                        file_coverage[path].add(owner_symbol)
                        if line_anchor_kinds:
                            _boost_candidate(candidates[owner_symbol], line_anchor_kinds)

                bare_calls = [] if declaration is not None else _extract_bare_calls(text)
                for call_name in bare_calls:
                    compound = call_name
                    if line.kind == "add":
                        added_call_sites.add(compound)
                    else:
                        removed_call_sites.add(compound)
                    _record_candidate(
                        candidates,
                        symbol=call_name,
                        source="call_site",
                        tier="generic_fallback",
                        score_delta=1.2 if line.kind == "add" else 1.0,
                        reason="bare_call_site",
                        file_path=path,
                        container=active_container,
                        hunk_key=hunk_key,
                    )
                    if line_anchor_kinds:
                        _boost_candidate(candidates[call_name], line_anchor_kinds)

            excerpt = _build_diff_excerpt(
                change=change,
                hunk=hunk,
                file_path=path,
                declaration_symbols=hunk_decls,
                member_calls=hunk_member_calls,
                anchor_kinds=hunk_anchor_kinds,
            )
            if excerpt is not None:
                priority = 0
                if hunk_decls:
                    priority = 3
                elif hunk_member_calls:
                    priority = 2
                elif hunk_anchor_kinds:
                    priority = 1
                excerpt_candidates.append((priority, -hunk_index, excerpt))

    seed_symbols = _select_seed_symbols(
        candidates=candidates,
        changed_files=changed_files,
        file_coverage=file_coverage,
        max_symbols=max_symbols,
    )
    diff_excerpts = [item[2] for item in sorted(excerpt_candidates, key=lambda row: (-row[0], row[1], row[2].file_path))[:_MAX_EXCERPTS]]
    changed_methods = [
        decl.symbol
        for decl in changed_declarations
        if decl.kind in {"function", "method", "constructor", "destructor"}
    ]

    return PrepassResult(
        changed_files=changed_files,
        changed_hunk_count=changed_hunk_count,
        seed_symbols=seed_symbols,
        suspicious_anchors=suspicious_anchors[:500],
        changed_declarations=changed_declarations[:200],
        changed_containers=sorted(changed_containers),
        member_call_sites=member_call_sites[:200],
        diff_excerpts=diff_excerpts,
        changed_methods=changed_methods[:100],
        added_call_sites=sorted(added_call_sites)[:200],
        removed_call_sites=sorted(removed_call_sites)[:200],
        include_macro_config_changes=sorted(include_macro_config_changes),
        warnings=sorted(set(warnings)),
    )


def _next_old_line(hunk: PatchHunk) -> int:
    old_consumed = len([line for line in hunk.lines if line.kind in {"context", "del"}])
    return hunk.old_start + old_consumed


def _next_new_line(hunk: PatchHunk) -> int:
    new_consumed = len([line for line in hunk.lines if line.kind in {"context", "add"}])
    return hunk.new_start + new_consumed


def _extract_declaration(
    *,
    text: str,
    file_path: str,
    line: int,
    active_container: str,
    active_symbol: str = "",
) -> ChangedDeclaration | None:
    stripped = text.strip()
    if not stripped or stripped.startswith("//") or _INCLUDE_RE.match(stripped) or _MACRO_RE.match(stripped):
        return None

    type_match = _TYPE_DECL_RE.match(stripped)
    if type_match:
        kind = type_match.group("kind")
        name = type_match.group("name")
        decl_kind = "enum" if kind.startswith("enum") else kind
        return ChangedDeclaration(symbol=name, container="", kind=decl_kind, file_path=file_path, line=line)

    open_paren = stripped.find("(")
    if open_paren <= 0:
        return None
    prefix = stripped[:open_paren]
    if "." in prefix or "->" in prefix:
        return None

    match = _METHOD_SIG_RE.match(stripped)
    if not match:
        return None
    symbol = match.group("name").strip()
    if not symbol or symbol in _KEYWORDS or symbol.startswith("std::"):
        return None
    base_name = symbol.split("::")[-1].lstrip("~")
    if base_name in _INVALID_DECLARATION_NAMES:
        return None
    prefix_text = prefix.strip()
    container = _container_from_symbol(symbol)
    if (
        _is_implementation_file(file_path)
        and stripped.endswith(";")
        and "::" not in symbol
    ):
        in_type_body = bool(active_container and (not active_symbol or active_symbol == active_container))
        if not in_type_body:
            return None
    base_name = symbol.split("::")[-1]
    if " " not in prefix_text:
        active_tail = active_container.split("::")[-1] if active_container else ""
        if not active_tail or base_name.lstrip("~") != active_tail:
            return None
    if not container and active_container and "::" not in symbol:
        container = active_container
        symbol = f"{active_container}::{symbol}"
        base_name = symbol.split("::")[-1]

    if base_name.startswith("~"):
        kind = "destructor"
    elif container and base_name == container.split("::")[-1]:
        kind = "constructor"
    elif container:
        kind = "method"
    else:
        kind = "function"
    return ChangedDeclaration(symbol=symbol, container=container, kind=kind, file_path=file_path, line=line)


def _extract_member_calls(
    *,
    text: str,
    file_path: str,
    line: int,
    line_kind: str,
    active_symbol: str,
    active_container: str,
) -> list[MemberCallSite]:
    calls: list[MemberCallSite] = []
    seen: set[tuple[str, str, int]] = set()

    for match in _MEMBER_CALL_RE.finditer(text):
        receiver = match.group("receiver")
        member = match.group("member")
        if member in _KEYWORDS or receiver in _KEYWORDS:
            continue
        call = MemberCallSite(
            member=member,
            receiver=receiver,
            container=active_symbol or active_container,
            file_path=file_path,
            line=line,
            line_kind="add" if line_kind == "add" else "del",
            qualified_receiver_type=receiver if "::" in receiver else "",
        )
        key = (call.receiver, call.member, call.line)
        if key not in seen:
            seen.add(key)
            calls.append(call)

    for match in _QUALIFIED_CALL_RE.finditer(text):
        receiver = match.group("receiver")
        member = match.group("member")
        if member in _KEYWORDS:
            continue
        if receiver == "std" or receiver.startswith("std::"):
            continue
        call = MemberCallSite(
            member=member,
            receiver=receiver,
            container=active_symbol or active_container or receiver,
            file_path=file_path,
            line=line,
            line_kind="add" if line_kind == "add" else "del",
            qualified_receiver_type=receiver,
        )
        key = (call.receiver, call.member, call.line)
        if key not in seen:
            seen.add(key)
            calls.append(call)
    return calls


def _extract_bare_calls(text: str) -> list[str]:
    calls: list[str] = []
    seen: set[str] = set()
    for match in _CALL_RE.finditer(text):
        name = match.group("name")
        if name in _KEYWORDS or len(name) < 3:
            continue
        prefix = text[: match.start("name")]
        trimmed = prefix.rstrip()
        if trimmed.endswith(".") or trimmed.endswith("->") or trimmed.endswith("::"):
            continue
        if name.startswith("std::"):
            continue
        if name.upper() == name and "_" in name:
            continue
        if name not in seen:
            seen.add(name)
            calls.append(name)
    return calls


def _record_candidate(
    candidates: dict[str, _SeedCandidate],
    *,
    symbol: str,
    source: str,
    tier: SeedRelevanceTier,
    score_delta: float,
    reason: str,
    file_path: str,
    receiver: str = "",
    container: str = "",
    declared_locally: bool = False,
    qualified_occurrence: bool = False,
    hunk_key: str = "",
) -> None:
    clean = symbol.strip()
    if not clean or clean in _KEYWORDS:
        return
    candidate = candidates.get(clean)
    if candidate is None:
        candidate = _SeedCandidate(symbol=clean, source=source, tier=tier)
        candidates[clean] = candidate
    candidate.register(
        source=source,
        tier=tier,
        score_delta=score_delta,
        reason=reason,
        file_path=file_path,
        receiver=receiver,
        container=container,
        declared_locally=declared_locally,
        qualified_occurrence=qualified_occurrence,
        hunk_key=hunk_key,
    )


def _boost_candidate(candidate: _SeedCandidate, anchor_kinds: set[str]) -> None:
    for kind in sorted(anchor_kinds):
        candidate.score += 1.5
        candidate.reasons.add(f"suspicious_anchor:{kind}")


def _select_seed_symbols(
    *,
    candidates: dict[str, _SeedCandidate],
    changed_files: list[str],
    file_coverage: dict[str, set[str]],
    max_symbols: int,
) -> list[SeedSymbol]:
    prepared = []
    for candidate in candidates.values():
        if len(candidate.file_paths) > 1:
            candidate.score += 1.0
            candidate.reasons.add("repeated_across_changed_files")
        if candidate.tier == "receiver_owned" and len(candidate.hunk_keys) >= 2:
            candidate.score += 2.0
            candidate.reasons.add("receiver_repeated_across_hunks")
        repeated_owner = candidate.repeated_owner_context()
        if repeated_owner:
            candidate.score += 1.5
            candidate.reasons.add("repeated_same_container_multi_hunk")

        promotable = True
        if candidate.standard_like and not candidate.declared_locally and candidate.tier != "declaration":
            promotable = False
            candidate.reasons.add("demoted_standard_or_system_symbol")
        elif candidate.tier == "generic_fallback":
            if candidate.low_entropy and not (
                candidate.declared_locally
                or candidate.qualified_occurrence
                or repeated_owner
                or candidate.container
            ):
                promotable = False
                candidate.reasons.add("demoted_unowned_low_entropy_call")
        prepared.append((candidate, promotable))

    sorted_candidates = []
    for candidate, promotable in sorted(
        prepared,
        key=lambda row: (
            not row[1],
            -row[0].score,
            -_TIER_PRIORITY[row[0].tier],
            row[0].symbol,
        ),
    ):
        if promotable or row_promotable_for_fallback(candidate):
            sorted_candidates.append(candidate)

    tier_limits = {"declaration": 8, "generic_fallback": 4}
    owner_limit = 8
    chosen: list[_SeedCandidate] = []
    chosen_symbols: set[str] = set()
    tier_counts: dict[str, int] = defaultdict(int)
    owner_count = 0

    def can_take(candidate: _SeedCandidate) -> bool:
        nonlocal owner_count
        if candidate.symbol in chosen_symbols:
            return False
        limit = tier_limits.get(candidate.tier)
        if limit is not None and tier_counts[candidate.tier] >= limit:
            return False
        if candidate.tier in {"qualified", "receiver_owned"} and owner_count >= owner_limit:
            return False
        return True

    def take(candidate: _SeedCandidate) -> None:
        nonlocal owner_count
        chosen.append(candidate)
        chosen_symbols.add(candidate.symbol)
        tier_counts[candidate.tier] += 1
        if candidate.tier in {"qualified", "receiver_owned"}:
            owner_count += 1

    # Each changed file should contribute at least one seed when possible.
    for file_path in changed_files:
        if len(chosen) >= max_symbols:
            break
        for candidate in sorted_candidates:
            if file_path in candidate.file_paths and can_take(candidate):
                take(candidate)
                break

    for candidate in sorted_candidates:
        if len(chosen) >= max_symbols:
            break
        if can_take(candidate):
            take(candidate)

    ranked = sorted(chosen, key=lambda cand: (-cand.score, -_TIER_PRIORITY[cand.tier], cand.symbol))
    return [
        SeedSymbol(
            symbol=candidate.symbol,
            source=candidate.source,
            score=round(candidate.score, 4),
            relevance_tier=candidate.tier,
            reasons=sorted(candidate.reasons),
            receiver=candidate.receiver,
            container=candidate.container,
            file_paths=sorted(candidate.file_paths),
        )
        for candidate in ranked[: max(1, max_symbols)]
    ]


def row_promotable_for_fallback(candidate: _SeedCandidate) -> bool:
    return candidate.tier == "declaration"


def _build_diff_excerpt(
    *,
    change: PatchChange,
    hunk: PatchHunk,
    file_path: str,
    declaration_symbols: list[str],
    member_calls: list[MemberCallSite],
    anchor_kinds: set[str],
) -> DiffExcerpt | None:
    reasons: list[str] = []
    if declaration_symbols:
        reasons.append(f"changed_declaration:{declaration_symbols[0]}")
    elif member_calls:
        reasons.append(f"member_call:{_compound_call_name(member_calls[0])}")
    elif anchor_kinds:
        reasons.append(f"suspicious_anchor:{sorted(anchor_kinds)[0]}")
    if not reasons:
        return None

    rendered_lines = []
    for line in hunk.lines[:_MAX_EXCERPT_LINES]:
        prefix = " "
        if line.kind == "add":
            prefix = "+"
        elif line.kind == "del":
            prefix = "-"
        rendered_lines.append(f"{prefix}{line.text}")
    start_line = _excerpt_start_line(hunk)
    end_line = _excerpt_end_line(hunk)
    return DiffExcerpt(
        file_path=file_path,
        hunk_header=f"@@ -{hunk.old_start},{hunk.old_count} +{hunk.new_start},{hunk.new_count} @@",
        start_line=start_line,
        end_line=end_line,
        text="\n".join(rendered_lines),
        reason="; ".join(reasons),
    )


def _excerpt_start_line(hunk: PatchHunk) -> int:
    lines = [line.new_line or line.old_line for line in hunk.lines if (line.new_line or line.old_line) > 0]
    return min(lines) if lines else 0


def _excerpt_end_line(hunk: PatchHunk) -> int:
    lines = [line.new_line or line.old_line for line in hunk.lines if (line.new_line or line.old_line) > 0]
    return max(lines) if lines else 0


def _compound_call_name(call: MemberCallSite) -> str:
    if call.qualified_receiver_type:
        return f"{call.qualified_receiver_type}::{call.member}"
    return f"{call.receiver}.{call.member}"


def _container_from_symbol(symbol: str) -> str:
    if "::" not in symbol:
        return ""
    return symbol.rsplit("::", 1)[0]


def _looks_symbolic(symbol: str) -> bool:
    text = symbol.strip()
    if not text:
        return False
    if text.upper() == text and "_" in text:
        return False
    return True


def _is_config_file(path: str) -> bool:
    p = PurePosixPath(path)
    if p.name in {"CMakeLists.txt", "BUILD", "BUILD.bazel"}:
        return True
    return p.suffix.lower() in _CONFIG_EXTENSIONS


def _is_implementation_file(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in {".c", ".cc", ".cpp", ".cxx", ".m", ".mm"}


def _is_low_entropy_symbol(symbol: str) -> bool:
    base = symbol.split("::")[-1].split(".")[-1].lstrip("~")
    return bool(base) and base.islower() and "_" not in base and len(base) <= 10


def _is_standard_or_system_symbol(symbol: str) -> bool:
    text = symbol.strip()
    if text.startswith("std::") or text.lower().startswith("std::"):
        return True
    base = text.split("::")[-1].split(".")[-1].lstrip("~")
    return base in _SYSTEM_FUNCTION_HINTS
