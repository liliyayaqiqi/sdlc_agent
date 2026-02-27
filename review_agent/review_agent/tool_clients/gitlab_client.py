"""Thin GitLab API client for MR-driven review integration."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

logger = logging.getLogger("review_agent.tool_clients.gitlab_client")


class GitLabApiError(RuntimeError):
    """Raised on HTTP failure from the GitLab API."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"GitLab API error status={status_code} detail={detail}")


class GitLabClient:
    """Minimal GitLab REST client for review-agent use cases.

    Supports:
    - Fetching MR metadata (source/target branches, SHAs, diff_refs)
    - Fetching the full MR diff as unified patch text
    - Posting a note (comment) on an MR
    - Posting an inline discussion on an MR
    """

    def __init__(
        self,
        *,
        base_url: str,
        private_token: str,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = private_token
        self._timeout = timeout_s
        self._http: httpx.Client | None = None

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self) -> "GitLabClient":
        self._ensure_http()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        if self._http is not None:
            try:
                self._http.close()
            except Exception:
                pass
            self._http = None

    def _ensure_http(self) -> httpx.Client:
        if self._http is None or self._http.is_closed:
            self._http = httpx.Client(
                timeout=self._timeout,
                headers={
                    "PRIVATE-TOKEN": self._token,
                    "Accept": "application/json",
                },
            )
        return self._http

    # -- public API --------------------------------------------------------

    def get_mr_metadata(self, *, project_id: str, mr_iid: int) -> dict[str, Any]:
        """Fetch MR metadata including diff_refs, branches, SHAs."""
        path = f"/api/v4/projects/{quote(str(project_id), safe='')}/merge_requests/{mr_iid}"
        return self._get(path)

    def get_mr_diff(self, *, project_id: str, mr_iid: int) -> str:
        """Fetch the full MR diff as a single unified patch string."""
        path = f"/api/v4/projects/{quote(str(project_id), safe='')}/merge_requests/{mr_iid}/changes"
        data = self._get(path)
        changes = data.get("changes", [])
        if not changes:
            return ""
        parts: list[str] = []
        for ch in changes:
            diff = ch.get("diff", "")
            if diff:
                old_path = ch.get("old_path", "")
                new_path = ch.get("new_path", "")
                header = f"--- a/{old_path}\n+++ b/{new_path}\n"
                parts.append(header + diff)
        return "\n".join(parts)

    def post_mr_note(self, *, project_id: str, mr_iid: int, body: str) -> dict[str, Any]:
        """Post a note (general comment) on an MR."""
        path = f"/api/v4/projects/{quote(str(project_id), safe='')}/merge_requests/{mr_iid}/notes"
        return self._post(path, {"body": body})

    def post_mr_inline_discussion(
        self,
        *,
        project_id: str,
        mr_iid: int,
        body: str,
        new_path: str,
        new_line: int,
        base_sha: str = "",
        head_sha: str = "",
        start_sha: str = "",
    ) -> dict[str, Any]:
        """Post an inline discussion thread on a specific file/line of an MR diff."""
        path = f"/api/v4/projects/{quote(str(project_id), safe='')}/merge_requests/{mr_iid}/discussions"
        position: dict[str, Any] = {
            "position_type": "text",
            "new_path": new_path,
            "new_line": new_line,
        }
        if base_sha:
            position["base_sha"] = base_sha
        if head_sha:
            position["head_sha"] = head_sha
        if start_sha:
            position["start_sha"] = start_sha
        return self._post(path, {"body": body, "position": position})

    # -- transport ---------------------------------------------------------

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        http = self._ensure_http()
        try:
            response = http.get(url)
        except Exception as exc:
            raise GitLabApiError(0, str(exc)) from exc
        return self._decode(response)

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        http = self._ensure_http()
        try:
            response = http.post(url, json=body)
        except Exception as exc:
            raise GitLabApiError(0, str(exc)) from exc
        return self._decode(response)

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}
        if response.status_code < 200 or response.status_code >= 300:
            raise GitLabApiError(int(response.status_code), payload)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"items": payload}
        return {"result": payload}
