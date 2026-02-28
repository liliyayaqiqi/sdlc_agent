"""Deterministic fake servers for end-to-end validation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Any


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


@dataclass
class _CxxtractState:
    requests: list[dict[str, Any]] = field(default_factory=list)


class CxxtractFixtureServer(AbstractContextManager):
    """Fake CXXtract backend for subprocess E2E tests."""

    def __init__(self) -> None:
        self.state = _CxxtractState()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None
        self.url = ""

    def __enter__(self) -> "CxxtractFixtureServer":
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                state.requests.append({"method": "GET", "path": self.path})
                if self.path == "/workspace/ws_main":
                    _json_response(self, 200, {"workspace_id": "ws_main"})
                    return
                _json_response(self, 404, {"detail": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
                state.requests.append({"method": "POST", "path": self.path, "body": body})
                if self.path == "/workspace/ws_main/sync-repo":
                    _json_response(self, 200, {"ok": True})
                    return
                if self.path == "/context/create-pr-overlay":
                    _json_response(self, 200, {"context_id": body.get("context_id", "ctx"), "partial_overlay": False})
                    return
                if self.path.startswith("/context/") and self.path.endswith("/expire"):
                    _json_response(self, 200, {"ok": True})
                    return
                if self.path == "/explore/rg-search":
                    query = str(body.get("query", ""))
                    hits = []
                    if query:
                        hits.append(
                            {
                                "file_key": "repoA:src/app.cpp",
                                "line": 3,
                                "line_text": f"{query}();",
                            }
                        )
                    _json_response(self, 200, {"hits": hits})
                    return
                if self.path == "/explore/list-candidates":
                    _json_response(self, 200, {"candidates": ["repoA:src/app.cpp"], "deleted_file_keys": []})
                    return
                if self.path == "/explore/classify-freshness":
                    _json_response(
                        self,
                        200,
                        {
                            "fresh": list(body.get("candidate_file_keys", []) or []),
                            "stale": [],
                            "unparsed": [],
                            "overlay_mode": "dense",
                        },
                    )
                    return
                if self.path == "/explore/parse-file":
                    _json_response(self, 200, {"parsed_file_keys": list(body.get("file_keys", []) or []), "failed_file_keys": []})
                    return
                if self.path == "/explore/fetch-symbols":
                    symbol = str(body.get("symbol", ""))
                    _json_response(self, 200, {"symbols": [{"file_key": "repoA:src/app.cpp", "symbol": symbol, "line": 3}]})
                    return
                if self.path == "/explore/fetch-references":
                    symbol = str(body.get("symbol", ""))
                    refs = [
                        {"file_key": "repoA:src/app.cpp", "line": 3},
                        {"file_key": "repoA:tests/test_app.cpp", "line": 12},
                    ] if symbol else []
                    _json_response(self, 200, {"references": refs})
                    return
                if self.path == "/explore/fetch-call-edges":
                    symbol = str(body.get("symbol", ""))
                    edges = [{"file_key": "repoA:src/app.cpp", "line": 3, "caller": "main", "callee": symbol}] if symbol else []
                    _json_response(self, 200, {"edges": edges})
                    return
                if self.path == "/explore/get-confidence":
                    verified_files = list(body.get("verified_files", []) or [])
                    _json_response(
                        self,
                        200,
                        {
                            "confidence": {
                                "verified_ratio": 0.9,
                                "total_candidates": max(1, len(verified_files)),
                                "verified_files": verified_files,
                                "stale_files": list(body.get("stale_files", []) or []),
                                "unparsed_files": list(body.get("unparsed_files", []) or []),
                            }
                        },
                    )
                    return
                if self.path == "/explore/read-file":
                    start_line = int(body.get("start_line", 1) or 1)
                    end_line = int(body.get("end_line", start_line) or start_line)
                    _json_response(
                        self,
                        200,
                        {
                            "file_key": body.get("file_key", ""),
                            "line_range": [start_line, end_line],
                            "content": "int main() {\n  doLogin();\n}\n",
                        },
                    )
                    return
                if self.path == "/agent/investigate-symbol":
                    _json_response(self, 200, {"summary_markdown": "fallback macro summary"})
                    return
                _json_response(self, 404, {"detail": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False


@dataclass
class _GitLabState:
    notes: list[dict[str, Any]] = field(default_factory=list)
    discussions: list[dict[str, Any]] = field(default_factory=list)


class GitLabFixtureServer(AbstractContextManager):
    """Fake GitLab API for end-to-end publishing tests."""

    def __init__(self) -> None:
        self.state = _GitLabState()
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None
        self.url = ""

    def __enter__(self) -> "GitLabFixtureServer":
        state = self.state

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                return

            def do_GET(self) -> None:
                if self.path.endswith("/merge_requests/7"):
                    _json_response(
                        self,
                        200,
                        {
                            "iid": 7,
                            "title": "Fixture MR",
                            "source_branch": "feature/login",
                            "target_branch": "main",
                            "web_url": "https://example.test/group/project/-/merge_requests/7",
                            "path_with_namespace": "group/project",
                            "merge_commit_sha": "d" * 40,
                            "diff_refs": {
                                "base_sha": "a" * 40,
                                "head_sha": "b" * 40,
                                "start_sha": "c" * 40,
                            },
                        },
                    )
                    return
                if self.path.endswith("/merge_requests/7/changes"):
                    _json_response(
                        self,
                        200,
                        {
                            "changes": [
                                {
                                    "old_path": "src/app.cpp",
                                    "new_path": "src/app.cpp",
                                    "diff": "@@ -1,2 +1,3 @@\n int main() {\n+  doLogin();\n }\n",
                                    "new_file": False,
                                    "deleted_file": False,
                                    "renamed_file": False,
                                }
                            ]
                        },
                    )
                    return
                _json_response(self, 404, {"detail": "not found"})

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(length) if length > 0 else b"{}"
                body = json.loads(raw.decode("utf-8") or "{}")
                if self.path.endswith("/merge_requests/7/notes"):
                    state.notes.append(body)
                    _json_response(self, 201, {"id": len(state.notes)})
                    return
                if self.path.endswith("/merge_requests/7/discussions"):
                    state.discussions.append(body)
                    _json_response(self, 201, {"id": len(state.discussions)})
                    return
                _json_response(self, 404, {"detail": "not found"})

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        return False
