from __future__ import annotations

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
            analysis_context={"mode": "pr", "context_id": "ctx1"},
        )

    assert captured["url"].endswith("/explore/list-candidates")
    assert captured["json"]["analysis_context"]["context_id"] == "ctx1"


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
