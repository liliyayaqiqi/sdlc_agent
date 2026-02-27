"""Thin HTTP client over CXXtract2 endpoints used by review orchestration."""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx


class CxxtractHttpError(RuntimeError):
    """Raised on HTTP or transport failure from CXXtract2."""

    def __init__(self, status_code: int, detail: Any) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"cxxtract request failed status={status_code} detail={detail}")


class CxxtractHttpClient:
    """HTTP wrapper used by agent orchestration and evidence collection."""

    def __init__(self, base_url: str, workspace_id: str, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.workspace_id = workspace_id
        self.timeout_s = timeout_s

    def workspace_info(self) -> dict[str, Any]:
        return self._get(f"/workspace/{quote(self.workspace_id, safe='')}")

    def context_create_pr_overlay(
        self,
        *,
        pr_id: str,
        base_ref: str = "",
        head_ref: str = "",
        context_id: str = "",
    ) -> dict[str, Any]:
        return self._post(
            "/context/create-pr-overlay",
            {
                "workspace_id": self.workspace_id,
                "pr_id": pr_id or "review",
                "base_ref": base_ref,
                "head_ref": head_ref,
                "context_id": context_id,
            },
        )

    def context_expire(self, *, context_id: str) -> dict[str, Any]:
        return self._post(f"/context/{quote(context_id, safe='')}/expire", {})

    def sync_repo(
        self,
        *,
        repo_id: str,
        commit_sha: str,
        branch: str = "",
        force_clean: bool = True,
    ) -> dict[str, Any]:
        return self._post(
            f"/workspace/{quote(self.workspace_id, safe='')}/sync-repo",
            {
                "repo_id": repo_id,
                "commit_sha": commit_sha,
                "branch": branch,
                "force_clean": bool(force_clean),
            },
        )

    def agent_investigate_symbol(self, *, symbol: str) -> dict[str, Any]:
        return self._post(
            "/agent/investigate-symbol",
            {"workspace_id": self.workspace_id, "symbol": symbol},
            headers={"x-cxxtract-workspace-id": self.workspace_id},
        )

    def agent_search_analyze_recent_commits(self, *, query: str, limit: int = 5) -> dict[str, Any]:
        return self._post(
            "/agent/search-analyze-recent-commits",
            {"workspace_id": self.workspace_id, "query": query, "limit": limit},
            headers={"x-cxxtract-workspace-id": self.workspace_id},
        )

    def agent_read_file_context(self, *, file_path: str, start_line: int = 1, end_line: int = 0) -> dict[str, Any]:
        return self._post(
            "/agent/read-file-context",
            {
                "workspace_id": self.workspace_id,
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
            },
            headers={"x-cxxtract-workspace-id": self.workspace_id},
        )

    def explore_rg_search(
        self,
        *,
        query: str,
        mode: str = "symbol",
        analysis_context: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
        max_hits: int = 200,
        max_files: int = 200,
        timeout_s: int = 30,
        context_lines: int = 0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "query": query,
            "mode": mode,
            "scope": scope or {"entry_repos": [], "max_repo_hops": 2},
            "max_hits": max_hits,
            "max_files": max_files,
            "timeout_s": timeout_s,
            "context_lines": context_lines,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/rg-search", body)

    def explore_list_candidates(
        self,
        *,
        symbol: str,
        max_files: int,
        include_rg: bool = True,
        entry_repos: list[str] | None = None,
        max_repo_hops: int = 2,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "symbol": symbol,
            "scope": {"entry_repos": entry_repos or [], "max_repo_hops": max_repo_hops},
            "max_files": max_files,
            "include_rg": include_rg,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/list-candidates", body)

    def explore_classify_freshness(
        self,
        *,
        candidate_file_keys: list[str],
        max_files: int,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "candidate_file_keys": candidate_file_keys,
            "max_files": max_files,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/classify-freshness", body)

    def explore_parse_file(
        self,
        *,
        file_keys: list[str],
        max_parse_workers: int,
        timeout_s: int,
        skip_if_fresh: bool = True,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "file_keys": file_keys,
            "max_parse_workers": max_parse_workers,
            "timeout_s": timeout_s,
            "skip_if_fresh": skip_if_fresh,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/parse-file", body)

    def explore_fetch_symbols(
        self,
        *,
        symbol: str,
        candidate_file_keys: list[str],
        excluded_file_keys: list[str],
        limit: int,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "symbol": symbol,
            "candidate_file_keys": candidate_file_keys,
            "excluded_file_keys": excluded_file_keys,
            "limit": limit,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/fetch-symbols", body)

    def explore_fetch_references(
        self,
        *,
        symbol: str,
        candidate_file_keys: list[str],
        excluded_file_keys: list[str],
        limit: int,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "symbol": symbol,
            "candidate_file_keys": candidate_file_keys,
            "excluded_file_keys": excluded_file_keys,
            "limit": limit,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/fetch-references", body)

    def explore_fetch_call_edges(
        self,
        *,
        symbol: str,
        direction: str,
        candidate_file_keys: list[str],
        excluded_file_keys: list[str],
        limit: int,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "workspace_id": self.workspace_id,
            "symbol": symbol,
            "direction": direction,
            "candidate_file_keys": candidate_file_keys,
            "excluded_file_keys": excluded_file_keys,
            "limit": limit,
        }
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/fetch-call-edges", body)

    def explore_get_confidence(
        self,
        *,
        verified_files: list[str],
        stale_files: list[str],
        unparsed_files: list[str],
        warnings: list[str],
        overlay_mode: str,
    ) -> dict[str, Any]:
        return self._post(
            "/explore/get-confidence",
            {
                "verified_files": verified_files,
                "stale_files": stale_files,
                "unparsed_files": unparsed_files,
                "warnings": warnings,
                "overlay_mode": overlay_mode,
            },
        )

    def explore_read_file(
        self,
        *,
        file_key: str,
        start_line: int = 1,
        end_line: int = 220,
        max_bytes: int = 120_000,
    ) -> dict[str, Any]:
        return self._post(
            "/explore/read-file",
            {
                "workspace_id": self.workspace_id,
                "file_key": file_key,
                "start_line": start_line,
                "end_line": end_line,
                "max_bytes": max_bytes,
            },
        )

    def explore_get_compile_command(
        self,
        *,
        file_key: str,
        analysis_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"workspace_id": self.workspace_id, "file_key": file_key}
        if analysis_context:
            body["analysis_context"] = analysis_context
        return self._post("/explore/get-compile-command", body)

    def _get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout_s) as client:
            try:
                response = client.get(url, params=params, headers=headers)
            except Exception as exc:
                raise CxxtractHttpError(0, str(exc)) from exc
        return self._decode(response)

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout_s) as client:
            try:
                response = client.post(url, json=body, headers=headers)
            except Exception as exc:
                raise CxxtractHttpError(0, str(exc)) from exc
        return self._decode(response)

    @staticmethod
    def _decode(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            payload = {"raw": response.text}
        if response.status_code < 200 or response.status_code >= 300:
            raise CxxtractHttpError(int(response.status_code), payload)
        if isinstance(payload, dict):
            return payload
        return {"result": payload}
