"""Map evidence to changed diff locations."""

from __future__ import annotations

from dataclasses import dataclass

from review_agent.models import FindingLocation, ReviewContextBundle, ReviewFinding


@dataclass
class _ChangedFileLines:
    path: str
    changed_new_lines: list[int]
    first_new_line: int


class FindingLocationMapper:
    """Derive deterministic finding locations from patch hunks and evidence."""

    def __init__(self, *, changes, bundle: ReviewContextBundle) -> None:
        self._bundle = bundle
        self._by_path: dict[str, _ChangedFileLines] = {}
        for change in changes:
            path = change.effective_path.replace("\\", "/")
            new_lines: list[int] = []
            for hunk in change.hunks:
                for row in hunk.lines:
                    if row.kind == "add" and row.new_line > 0:
                        new_lines.append(row.new_line)
            if not new_lines:
                fallback = [max(1, hunk.new_start) for hunk in change.hunks if hunk.new_start > 0]
                new_lines = fallback or [1]
            self._by_path[path] = _ChangedFileLines(
                path=path,
                changed_new_lines=sorted(set(new_lines)),
                first_new_line=min(new_lines),
            )

    def apply(self, finding: ReviewFinding) -> ReviewFinding:
        if finding.location is not None:
            return finding

        candidates: list[tuple[str, int, str]] = []
        for evidence in finding.evidence:
            if evidence.file_key:
                repo_id, _, rel = evidence.file_key.partition(":")
                path = rel or evidence.file_key
                candidates.append((path.replace("\\", "/"), evidence.line, repo_id))
            elif evidence.abs_path:
                candidates.append((evidence.abs_path.replace("\\", "/"), evidence.line, ""))

        if finding.diff_path:
            candidates.append((finding.diff_path.replace("\\", "/"), finding.diff_line, ""))

        for raw_path, line, repo_id in candidates:
            if raw_path in self._by_path:
                return self._with_location(finding, raw_path, line, repo_id)
            normalized = raw_path.split(":", 1)[-1]
            if normalized in self._by_path:
                return self._with_location(finding, normalized, line, repo_id)

        return finding

    def _with_location(self, finding: ReviewFinding, path: str, line: int, repo_id: str) -> ReviewFinding:
        changed = self._by_path[path]
        if line <= 0 or line not in changed.changed_new_lines:
            resolved_line = changed.first_new_line
        else:
            resolved_line = line
        location = FindingLocation(
            repo_id=repo_id,
            path=changed.path,
            line=resolved_line,
            side="new",
            base_sha=self._bundle.base_sha,
            head_sha=self._bundle.head_sha,
            start_sha=self._bundle.target_branch_head_sha,
        )
        return finding.model_copy(update={"location": location, "diff_path": location.path, "diff_line": location.line})
