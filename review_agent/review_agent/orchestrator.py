"""Top-level agentic orchestration for semantic PR analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from review_agent.models import (
    SEVERITY_RANK,
    ReviewDecision,
    ReviewReport,
    ReviewRequest,
    ToolCallRecord,
)
from review_agent.prompting import CORE_SYSTEM_PROMPT, build_review_prompt
from review_agent.report_renderer import render_markdown
from review_agent.tool_clients.cxxtract_http_client import CxxtractHttpClient


@dataclass
class AgentDependencies:
    """Shared runtime objects visible to PydanticAI tools."""

    client: CxxtractHttpClient
    workspace_id: str
    max_total_tool_calls: int
    tool_usage: list[ToolCallRecord] = field(default_factory=list)


def build_review_agent(model_name: str):
    """Build and return the PydanticAI review agent with bound macro-tools."""
    try:
        from pydantic_ai import Agent, RunContext  # type: ignore
    except Exception as exc:  # pragma: no cover - import path depends on env
        raise RuntimeError("pydantic-ai is required; install dependencies for the review agent") from exc

    agent = Agent(
        model_name,
        system_prompt=CORE_SYSTEM_PROMPT,
        deps_type=AgentDependencies,
        result_type=ReviewReport,
    )

    def _record(ctx: RunContext[AgentDependencies], *, tool: str, ok: bool, elapsed_ms: float, note: str = "") -> None:
        ctx.deps.tool_usage.append(
            ToolCallRecord(
                skill="agentic_llm",
                tool=tool,
                success=ok,
                elapsed_ms=round(elapsed_ms, 3),
                note=note[:600],
            )
        )

    def _budget_guard(ctx: RunContext[AgentDependencies]) -> None:
        if len(ctx.deps.tool_usage) >= ctx.deps.max_total_tool_calls:
            raise RuntimeError(f"max_total_tool_calls_reached:{ctx.deps.max_total_tool_calls}")

    @agent.tool
    def investigate_symbol(ctx: RunContext[AgentDependencies], symbol: str) -> dict[str, Any]:
        """Run CXXtract aggregated symbol investigation for a suspicious symbol."""
        _budget_guard(ctx)
        t0 = perf_counter()
        try:
            payload = ctx.deps.client.agent_investigate_symbol(symbol=symbol)
            _record(
                ctx,
                tool="agent.investigate_symbol",
                ok=True,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
            )
            return {
                "symbol": str(payload.get("symbol", symbol)),
                "summary_markdown": str(payload.get("summary_markdown", "")),
                "metrics": dict(payload.get("metrics", {}) or {}),
                "file_paths": list(payload.get("file_paths", []) or []),
                "confidence": dict(payload.get("confidence", {}) or {}),
                "definitions": list(payload.get("definitions", []) or []),
                "references": list(payload.get("references", []) or []),
                "call_edges": list(payload.get("call_edges", []) or []),
            }
        except Exception as exc:
            _record(
                ctx,
                tool="agent.investigate_symbol",
                ok=False,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
                note=str(exc),
            )
            raise

    @agent.tool
    def search_and_analyze_recent_commits(
        ctx: RunContext[AgentDependencies],
        query: str,
        limit: int = 5,
    ) -> dict[str, Any]:
        """Search semantically similar commits and inspect symbol-level patterns."""
        _budget_guard(ctx)
        t0 = perf_counter()
        safe_limit = max(1, min(int(limit), 10))
        try:
            payload = ctx.deps.client.agent_search_analyze_recent_commits(query=query, limit=safe_limit)
            _record(
                ctx,
                tool="agent.search_analyze_recent_commits",
                ok=True,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
            )
            return {
                "summary_markdown": str(payload.get("summary_markdown", "")),
                "query": str(payload.get("query", query)),
                "limit": int(payload.get("limit", safe_limit) or safe_limit),
                "commit_hits": list(payload.get("commit_hits", []) or []),
                "investigated_symbols": list(payload.get("investigated_symbols", []) or []),
                "warnings": list(payload.get("warnings", []) or []),
            }
        except Exception as exc:
            _record(
                ctx,
                tool="agent.search_analyze_recent_commits",
                ok=False,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
                note=str(exc),
            )
            raise

    @agent.tool
    def read_file_context(
        ctx: RunContext[AgentDependencies],
        file_path: str,
        start_line: int = 1,
        end_line: int = 0,
    ) -> dict[str, Any]:
        """Read concrete source context for evidence-backed findings."""
        _budget_guard(ctx)
        t0 = perf_counter()
        safe_start = max(1, int(start_line))
        safe_end = max(0, int(end_line))
        try:
            payload = ctx.deps.client.agent_read_file_context(
                file_path=file_path,
                start_line=safe_start,
                end_line=safe_end,
            )
            _record(
                ctx,
                tool="agent.read_file_context",
                ok=True,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
            )
            content = str(payload.get("content", ""))
            if len(content) > 16_000:
                content = content[:16_000] + "\n...[truncated by review agent]"
            return {
                "summary_markdown": str(payload.get("summary_markdown", "")),
                "file_path": str(payload.get("file_path", file_path)),
                "file_key": str(payload.get("file_key", "")),
                "abs_path": str(payload.get("abs_path", "")),
                "line_range": list(payload.get("line_range", []) or []),
                "content": content,
                "truncated": bool(payload.get("truncated", False)),
                "warnings": list(payload.get("warnings", []) or []),
            }
        except Exception as exc:
            _record(
                ctx,
                tool="agent.read_file_context",
                ok=False,
                elapsed_ms=(perf_counter() - t0) * 1000.0,
                note=str(exc),
            )
            raise

    return agent


class ReviewOrchestrator:
    """Runs end-to-end semantic review with a PydanticAI-controlled tool loop."""

    def __init__(
        self,
        *,
        client: CxxtractHttpClient | None = None,
        agent_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._client = client
        self._agent_factory = agent_factory or build_review_agent
        self._agent_cache: dict[str, Any] = {}

    def run(self, request: ReviewRequest) -> ReviewReport:
        """Run semantic review and return structured report."""
        client = self._client or CxxtractHttpClient(
            base_url=request.cxxtract_base_url,
            workspace_id=request.workspace_id,
            timeout_s=60.0,
        )

        # Validate workspace binding before model execution.
        workspace = client.workspace_info()
        resolved_workspace_id = str(workspace.get("workspace_id", "")).strip()
        if resolved_workspace_id and resolved_workspace_id != request.workspace_id:
            raise RuntimeError("workspace info mismatch for requested workspace_id")

        agent = self._agent_for(request.llm_model)
        deps = AgentDependencies(
            client=client,
            workspace_id=request.workspace_id,
            max_total_tool_calls=request.max_total_tool_calls,
        )
        prompt = build_review_prompt(request)
        result = agent.run_sync(prompt, deps=deps)
        report = result.data
        return _normalize_report(report=report, request=request, tool_usage=deps.tool_usage)

    def _agent_for(self, model_name: str):
        model = (model_name or "").strip() or "openai:gpt-4o"
        if model not in self._agent_cache:
            self._agent_cache[model] = self._agent_factory(model)
        return self._agent_cache[model]

    @staticmethod
    def write_report_files(report: ReviewReport, out_dir: str | Path) -> tuple[Path, Path]:
        """Write markdown and JSON report artifacts."""
        out = Path(out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / "review_report.md"
        json_path = out / "review_report.json"
        md_path.write_text(render_markdown(report), encoding="utf-8")
        json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        return md_path, json_path


def _normalize_report(*, report: ReviewReport, request: ReviewRequest, tool_usage: list[ToolCallRecord]) -> ReviewReport:
    threshold = request.fail_on_severity
    blocking = len([f for f in report.findings if SEVERITY_RANK[f.severity] >= SEVERITY_RANK[threshold]])
    summary = report.summary.strip()
    if not summary:
        summary = f"Reviewed PR patch with {len(report.findings)} findings."
    return report.model_copy(
        update={
            "workspace_id": request.workspace_id,
            "summary": summary,
            "decision": ReviewDecision(
                fail_threshold=threshold,
                blocking_findings=blocking,
                should_block=blocking > 0,
            ),
            "tool_usage": tool_usage,
        }
    )
