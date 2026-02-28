"""Lightweight file-backed cache for review traces and reports."""

from __future__ import annotations

from hashlib import sha256
import json
import logging
from pathlib import Path
from typing import Any

from review_agent.models import ReviewReport

logger = logging.getLogger("review_agent.review_cache")


class ReviewTraceCache:
    """Small JSON artifact cache keyed by review context and policy hashes."""

    def __init__(self, cache_dir: str | Path) -> None:
        self._root = Path(cache_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def make_key(
        self,
        *,
        workspace_id: str,
        base_sha: str,
        head_sha: str,
        target_sha: str,
        merge_sha: str,
        workspace_fingerprint: str,
        patch_text: str,
        policy: dict[str, Any],
    ) -> str:
        payload = {
            "workspace_id": workspace_id,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "target_sha": target_sha,
            "merge_sha": merge_sha,
            "workspace_fingerprint": workspace_fingerprint,
            "patch_hash": sha256(patch_text.encode("utf-8")).hexdigest(),
            "policy_hash": sha256(json.dumps(policy, sort_keys=True).encode("utf-8")).hexdigest(),
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def load_report(self, key: str) -> ReviewReport | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        report_raw = data.get("review_report")
        if not isinstance(report_raw, dict):
            return None
        try:
            return ReviewReport.model_validate(report_raw)
        except Exception:
            return None

    def save(self, key: str, payload: dict[str, Any]) -> None:
        p = self._path(key)
        tmp = p.with_suffix(f"{p.suffix}.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        tmp.replace(p)

    def _path(self, key: str) -> Path:
        return self._root / f"{key}.json"
