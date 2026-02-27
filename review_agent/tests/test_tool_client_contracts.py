from __future__ import annotations

from unittest.mock import patch

import pytest

from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient, CxxtractHttpError


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

        def get(self, url, params=None):
            captured["url"] = url
            captured["params"] = params
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

        def post(self, url, json=None):
            return _Resp()

    with patch("review_agent.tool_clients.cxxtract_http_client.httpx.Client", _Client):
        client = CxxtractHttpClient("http://127.0.0.1:8000", "ws_main")
        with pytest.raises(CxxtractHttpError) as exc:
            client.explore_list_candidates(symbol="foo", max_files=10)
    assert exc.value.status_code == 422

