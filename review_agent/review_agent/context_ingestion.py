"""Context bundle ingestion and legacy patch-only normalization."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from review_agent.models import RepoRevisionContext, ReviewContextBundle, ReviewRequest, Severity

logger = logging.getLogger("review_agent.context_ingestion")

_BOUNDED_INT_POLICY: dict[str, tuple[int, int]] = {
    "max_symbols": (1, 100),
    "max_symbol_slots": (1, 200),
    "max_total_tool_calls": (1, 1000),
    "parse_timeout_s": (1, 600),
    "parse_workers": (1, 64),
    "max_candidates_per_symbol": (1, 2000),
    "max_fetch_limit": (10, 20000),
}


class IngestedReviewContext(BaseModel):
    """Normalized execution context and effective policy knobs."""

    bundle: ReviewContextBundle
    fail_on_severity: Severity
    max_symbols: int
    max_symbol_slots: int
    max_total_tool_calls: int
    parse_timeout_s: int
    parse_workers: int
    max_candidates_per_symbol: int
    max_fetch_limit: int
    enable_cache: bool
    cache_dir: str

    @property
    def workspace_id(self) -> str:
        return self.bundle.workspace_id

    @property
    def patch_text(self) -> str:
        return self.bundle.patch_text

    @property
    def changed_files(self) -> list[str]:
        return self.bundle.changed_files

    @property
    def workspace_fingerprint(self) -> str:
        return self.bundle.workspace_fingerprint

    @property
    def primary_repo_id(self) -> str:
        return self.bundle.primary_repo_id

    @property
    def per_repo_shas(self) -> dict[str, str]:
        return self.bundle.per_repo_shas

    @property
    def repo_revisions(self) -> list[RepoRevisionContext]:
        return self.bundle.repo_revisions

    @property
    def base_sha(self) -> str:
        return self.bundle.base_sha

    @property
    def head_sha(self) -> str:
        return self.bundle.head_sha

    @property
    def target_branch_head_sha(self) -> str:
        return self.bundle.target_branch_head_sha

    @property
    def merge_preview_sha(self) -> str:
        return self.bundle.merge_preview_sha

    @property
    def pr_metadata(self) -> dict[str, object]:
        return self.bundle.pr_metadata


class ReviewContextIngestor:
    """Transforms request payloads into a single context-driven runtime model."""

    _INT_POLICY_KEYS = set(_BOUNDED_INT_POLICY)

    @classmethod
    def ingest(cls, request: ReviewRequest) -> IngestedReviewContext:
        bundle = (
            request.context_bundle.model_copy(deep=True)
            if request.context_bundle
            else ReviewContextBundle(
                workspace_id=request.workspace_id,
                patch_text=request.patch_text,
                workspace_fingerprint=request.workspace_fingerprint,
            )
        )
        if not bundle.workspace_id:
            bundle.workspace_id = request.workspace_id
        if bundle.workspace_id != request.workspace_id:
            raise ValueError("context bundle workspace_id mismatch")
        if not bundle.patch_text.strip():
            bundle.patch_text = request.patch_text
        if not bundle.patch_text.strip():
            raise ValueError("context bundle patch_text is empty")
        if not bundle.workspace_fingerprint.strip():
            bundle.workspace_fingerprint = request.workspace_fingerprint.strip()

        bundle = bundle.model_copy(update={"repo_revisions": cls._normalize_repo_revisions(bundle)})

        policy = dict(bundle.policy or {})
        fail = request.fail_on_severity
        fail_raw = str(policy.get("fail_on_severity", "")).strip().lower()
        if fail_raw in {s.value for s in Severity}:
            fail = Severity(fail_raw)

        values = {
            "max_symbols": request.max_symbols,
            "max_symbol_slots": request.max_symbol_slots,
            "max_total_tool_calls": request.max_total_tool_calls,
            "parse_timeout_s": request.parse_timeout_s,
            "parse_workers": request.parse_workers,
            "max_candidates_per_symbol": request.max_candidates_per_symbol,
            "max_fetch_limit": request.max_fetch_limit,
        }
        if "max_tool_rounds" in policy and "max_symbol_slots" not in policy:
            policy["max_symbol_slots"] = policy["max_tool_rounds"]
        for key in cls._INT_POLICY_KEYS:
            if key in policy:
                try:
                    lower, upper = _BOUNDED_INT_POLICY[key]
                    values[key] = max(lower, min(int(policy[key]), upper))
                except Exception:
                    logger.warning("ignoring invalid policy override for %s", key)
                    continue

        enable_cache = bool(request.enable_cache) and cls._cache_is_safe(bundle)

        return IngestedReviewContext(
            bundle=bundle,
            fail_on_severity=fail,
            max_symbols=values["max_symbols"],
            max_symbol_slots=values["max_symbol_slots"],
            max_total_tool_calls=values["max_total_tool_calls"],
            parse_timeout_s=values["parse_timeout_s"],
            parse_workers=values["parse_workers"],
            max_candidates_per_symbol=values["max_candidates_per_symbol"],
            max_fetch_limit=values["max_fetch_limit"],
            enable_cache=enable_cache,
            cache_dir=request.cache_dir,
        )

    @classmethod
    def _normalize_repo_revisions(cls, bundle: ReviewContextBundle) -> list[RepoRevisionContext]:
        out: list[RepoRevisionContext] = []
        seen: set[str] = set()

        def _add(row: RepoRevisionContext) -> None:
            repo_id = row.repo_id.strip()
            if not repo_id or repo_id in seen:
                return
            seen.add(repo_id)
            out.append(row)

        for row in bundle.repo_revisions:
            _add(row)

        if bundle.primary_repo_id and bundle.primary_repo_id not in seen:
            _add(
                RepoRevisionContext(
                    repo_id=bundle.primary_repo_id,
                    base_sha=bundle.base_sha,
                    head_sha=bundle.head_sha,
                    target_sha=bundle.target_branch_head_sha,
                    merge_sha=bundle.merge_preview_sha,
                    role="primary",
                )
            )

        for repo_id, sha in sorted((bundle.per_repo_shas or {}).items()):
            if repo_id in seen:
                continue
            _add(
                RepoRevisionContext(
                    repo_id=repo_id,
                    head_sha=str(sha or ""),
                    target_sha=bundle.target_branch_head_sha,
                    role="dependency",
                )
            )

        return out

    @staticmethod
    def _cache_is_safe(bundle: ReviewContextBundle) -> bool:
        if bundle.workspace_fingerprint.strip():
            return True
        top_level_revisions = [
            bundle.base_sha,
            bundle.head_sha,
            bundle.target_branch_head_sha,
            bundle.merge_preview_sha,
        ]
        if any(str(value).strip() for value in top_level_revisions):
            return True
        for repo in bundle.repo_revisions:
            if any(
                str(value).strip()
                for value in (repo.base_sha, repo.head_sha, repo.target_sha, repo.merge_sha)
            ):
                return True
        return False
