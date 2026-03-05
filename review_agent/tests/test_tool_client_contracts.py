from __future__ import annotations

import copy
from unittest.mock import patch

import pytest

from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient, CxxtractHttpError
from review_agent.tool_clients.gitlab_client import GitLabClient


def test_workspace_info_uses_expected_path():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"workspace_id": "ws_main"}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        body = client.workspace_info()
    assert body["workspace_id"] == "ws_main"
    assert captured["url"].endswith("/workspace/ws_main")


def test_workspace_info_accepts_workspace_override():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"workspace_id": "ws_head"}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, params=None, headers=None):
            captured["url"] = url
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        body = client.workspace_info(workspace_id="ws_head")
    assert body["workspace_id"] == "ws_head"
    assert captured["url"].endswith("/workspace/ws_head")


def test_non_2xx_raises_structured_error():
    class _Resp:
        status_code = 422
        text = "bad"

        @staticmethod
        def json():
            return {"detail": "invalid"}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        with pytest.raises(CxxtractHttpError) as exc:
            client.explore_list_candidates(symbol="foo", max_files=10)
    assert exc.value.status_code == 422


def test_macro_tool_paths_include_workspace_binding():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"ok": True}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.agent_investigate_symbol(symbol="auth::Session::Start")

    assert captured["url"].endswith("/agent/investigate-symbol")
    assert captured["json"]["workspace_id"] == "ws_main"
    assert captured["headers"]["x-cxxtract-workspace-id"] == "ws_main"


def test_macro_tool_accepts_contextual_candidate_file_keys():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"ok": True}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.agent_investigate_symbol(
            symbol="auth::Session::Start",
            analysis_context={"mode": "pr", "context_id": "ctx1"},
            candidate_file_keys=["repoA:src/main.cpp"],
        )

    assert captured["json"]["analysis_context"]["context_id"] == "ctx1"
    assert captured["json"]["candidate_file_keys"] == ["repoA:src/main.cpp"]


def test_explore_candidates_accepts_analysis_context():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"candidates": []}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.explore_list_candidates(
            symbol="foo",
            max_files=10,
            path_prefixes=["src/module"],
            analysis_context={"mode": "pr", "context_id": "ctx1"},
        )

    assert captured["url"].endswith("/explore/list-candidates")
    assert captured["json"]["analysis_context"]["context_id"] == "ctx1"
    assert captured["json"]["scope"]["path_prefixes"] == ["src/module"]


def test_explore_candidates_uses_analysis_context_workspace_override():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"candidates": []}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["json"] = json
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.explore_list_candidates(
            symbol="foo",
            max_files=10,
            analysis_context={"mode": "baseline", "context_id": "ws_head:baseline", "workspace_id": "ws_head"},
        )

    assert captured["json"]["workspace_id"] == "ws_head"


def test_explore_candidates_accepts_bootstrap_file_keys():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"candidates": []}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.explore_list_candidates(
            symbol="foo",
            max_files=10,
            bootstrap_file_keys=["repoA:src/module/a.cpp"],
        )

    assert captured["json"]["bootstrap_file_keys"] == ["repoA:src/module/a.cpp"]


def test_explore_candidates_retries_without_unsupported_bootstrap_field():
    captured = {"requests": []}

    class _Resp:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self.text = ""
            self._payload = payload

        def json(self):
            return self._payload

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["requests"].append(copy.deepcopy(json))
            if len(captured["requests"]) == 1:
                return _Resp(
                    422,
                    {
                        "detail": [
                            {
                                "type": "extra_forbidden",
                                "loc": ["body", "bootstrap_file_keys"],
                                "msg": "Extra inputs are not permitted",
                            }
                        ]
                    },
                )
            return _Resp(200, {"candidates": ["repoA:src/module/a.cpp"], "warnings": []})

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        body = client.explore_list_candidates(
            symbol="foo",
            max_files=10,
            bootstrap_file_keys=["repoA:src/module/a.cpp"],
        )

    assert len(captured["requests"]) == 2
    assert "bootstrap_file_keys" in captured["requests"][0]
    assert "bootstrap_file_keys" not in captured["requests"][1]
    assert body["candidates"] == ["repoA:src/module/a.cpp"]
    assert "server_unsupported_optional_field:bootstrap_file_keys" in body["warnings"]


def test_query_file_symbols_uses_expected_path():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"file_key": "repoA:src/main.cpp", "symbols": []}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.query_file_symbols(file_key="repoA:src/main.cpp", analysis_context={"mode": "pr"})

    assert captured["url"].endswith("/query/file-symbols")
    assert captured["json"]["file_key"] == "repoA:src/main.cpp"
    assert captured["json"]["analysis_context"]["mode"] == "pr"


def test_materialize_review_workspaces_posts_workspace_scoped_request():
    captured = {}

    class _Resp:
        status_code = 200
        text = ""

        @staticmethod
        def json():
            return {"materialization_id": "mat:123", "workspace_id": "ws_main", "review_key": "mr18"}

    class _Client:
        def __init__(self, timeout):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        client.materialize_review_workspaces(
            review_key="mr18",
            repo_revisions=[{"repo_id": "repoA", "target_sha": "a" * 40}],
            views=["target", "head"],
        )

    assert captured["url"].endswith("/workspace/ws_main/materialize-review")
    assert captured["json"]["workspace_id"] == "ws_main"
    assert captured["json"]["views"] == ["target", "head"]


def test_gitlab_get_retries_on_transient_error():
    attempts = {"count": 0}

    class _Resp:
        def __init__(self, status_code):
            self.status_code = status_code
            self.text = ""

        def json(self):
            if self.status_code >= 500:
                return {"detail": "temporary"}
            return {"iid": 7}

    class _Client:
        def __init__(self, timeout, headers=None):
            self.timeout = timeout
            self.headers = headers
            self.is_closed = False

        def close(self):
            self.is_closed = True

        def get(self, url):
            attempts["count"] += 1
            if attempts["count"] == 1:
                return _Resp(502)
            return _Resp(200)

    with patch("review_agent.tool_clients.gitlab_client.httpx.Client", _Client):
        client = GitLabClient(base_url="https://gitlab.example", private_token="token")
        body = client.get_mr_metadata(project_id="1", mr_iid=7)
    assert attempts["count"] == 2
    assert body["iid"] == 7


def test_gitlab_note_posts_with_idempotency_key():
    captured = {}

    class _Resp:
        status_code = 201
        text = ""

        @staticmethod
        def json():
            return {"id": 1}

    class _Client:
        def __init__(self, timeout, headers=None):
            self.timeout = timeout
            self.headers = headers
            self.is_closed = False

        def close(self):
            self.is_closed = True

        def post(self, url, json=None, headers=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return _Resp()

    with patch("review_agent.tool_clients.gitlab_client.httpx.Client", _Client):
        client = GitLabClient(base_url="https://gitlab.example", private_token="token")
        client.post_mr_note(project_id="1", mr_iid=7, body="hello")

    assert captured["url"].endswith("/merge_requests/7/notes")
    assert captured["json"]["body"] == "hello"
    assert "Idempotency-Key" in captured["headers"]
