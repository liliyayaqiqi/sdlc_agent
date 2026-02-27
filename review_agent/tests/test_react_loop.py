from __future__ import annotations

import pytest

from review_agent.react_loop import ReActLoop


def test_react_loop_is_removed():
    with pytest.raises(RuntimeError):
        ReActLoop(registry=object())
