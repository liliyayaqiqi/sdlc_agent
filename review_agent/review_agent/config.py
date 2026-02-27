"""Configuration loading for the review agent."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from review_agent.models import Severity


class AgentSettings(BaseModel):
    """Configurable runtime knobs for review execution."""

    cxxtract_base_url: str = "http://127.0.0.1:8000"
    fail_on_severity: Severity = Severity.HIGH
    max_symbols: int = Field(default=24, ge=1, le=100)
    max_tool_rounds: int = Field(default=30, ge=1, le=200)
    max_total_tool_calls: int = Field(default=120, ge=1, le=1000)
    parse_timeout_s: int = Field(default=120, ge=1, le=600)
    parse_workers: int = Field(default=4, ge=1, le=64)
    max_candidates_per_symbol: int = Field(default=150, ge=1, le=2000)
    max_fetch_limit: int = Field(default=2000, ge=10, le=20000)

    @classmethod
    def from_env(cls) -> "AgentSettings":
        """Load settings from REVIEW_AGENT_* environment variables."""
        fail_raw = os.getenv("REVIEW_AGENT_FAIL_ON_SEVERITY", "high").strip().lower()
        fail = Severity(fail_raw) if fail_raw in {s.value for s in Severity} else Severity.HIGH
        return cls(
            cxxtract_base_url=os.getenv("REVIEW_AGENT_CXXTRACT_BASE_URL", "http://127.0.0.1:8000").strip(),
            fail_on_severity=fail,
            max_symbols=int(os.getenv("REVIEW_AGENT_MAX_SYMBOLS", "24")),
            max_tool_rounds=int(os.getenv("REVIEW_AGENT_MAX_TOOL_ROUNDS", "30")),
            max_total_tool_calls=int(os.getenv("REVIEW_AGENT_MAX_TOTAL_TOOL_CALLS", "120")),
            parse_timeout_s=int(os.getenv("REVIEW_AGENT_PARSE_TIMEOUT_S", "120")),
            parse_workers=int(os.getenv("REVIEW_AGENT_PARSE_WORKERS", "4")),
            max_candidates_per_symbol=int(os.getenv("REVIEW_AGENT_MAX_CANDIDATES_PER_SYMBOL", "150")),
            max_fetch_limit=int(os.getenv("REVIEW_AGENT_MAX_FETCH_LIMIT", "2000")),
        )

