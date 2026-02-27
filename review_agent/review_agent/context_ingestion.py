"""Context bundle ingestion and legacy patch-only normalization."""

from __future__ import annotations

from pydantic import BaseModel

from review_agent.models import ReviewContextBundle, ReviewRequest, Severity


class IngestedReviewContext(BaseModel):
    """Normalized execution context and effective policy knobs."""

    bundle: ReviewContextBundle
    fail_on_severity: Severity
    max_symbols: int
    max_tool_rounds: int
    max_total_tool_calls: int
    parse_timeout_s: int
    parse_workers: int
    max_candidates_per_symbol: int
    max_fetch_limit: int
    enable_cache: bool
    cache_dir: str


class ReviewContextIngestor:
    """Transforms request payloads into a single context-driven runtime model."""

    _INT_POLICY_KEYS = {
        "max_symbols",
        "max_tool_rounds",
        "max_total_tool_calls",
        "parse_timeout_s",
        "parse_workers",
        "max_candidates_per_symbol",
        "max_fetch_limit",
    }

    @classmethod
    def ingest(cls, request: ReviewRequest) -> IngestedReviewContext:
        bundle = request.context_bundle.model_copy(deep=True) if request.context_bundle else ReviewContextBundle(
            workspace_id=request.workspace_id,
            patch_text=request.patch_text,
        )
        if not bundle.workspace_id:
            bundle.workspace_id = request.workspace_id
        if bundle.workspace_id != request.workspace_id:
            raise ValueError("context bundle workspace_id mismatch")
        if not bundle.patch_text.strip():
            bundle.patch_text = request.patch_text
        if not bundle.patch_text.strip():
            raise ValueError("context bundle patch_text is empty")

        policy = dict(bundle.policy or {})
        fail = request.fail_on_severity
        fail_raw = str(policy.get("fail_on_severity", "")).strip().lower()
        if fail_raw in {s.value for s in Severity}:
            fail = Severity(fail_raw)

        values = {
            "max_symbols": request.max_symbols,
            "max_tool_rounds": request.max_tool_rounds,
            "max_total_tool_calls": request.max_total_tool_calls,
            "parse_timeout_s": request.parse_timeout_s,
            "parse_workers": request.parse_workers,
            "max_candidates_per_symbol": request.max_candidates_per_symbol,
            "max_fetch_limit": request.max_fetch_limit,
        }
        for key in cls._INT_POLICY_KEYS:
            if key in policy:
                try:
                    values[key] = int(policy[key])
                except Exception:
                    continue

        return IngestedReviewContext(
            bundle=bundle,
            fail_on_severity=fail,
            max_symbols=max(1, values["max_symbols"]),
            max_tool_rounds=max(1, values["max_tool_rounds"]),
            max_total_tool_calls=max(1, values["max_total_tool_calls"]),
            parse_timeout_s=max(1, values["parse_timeout_s"]),
            parse_workers=max(1, values["parse_workers"]),
            max_candidates_per_symbol=max(1, values["max_candidates_per_symbol"]),
            max_fetch_limit=max(10, values["max_fetch_limit"]),
            enable_cache=bool(request.enable_cache),
            cache_dir=request.cache_dir,
        )
