"""CLI entrypoint for running the semantic review agent in CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from review_agent.config import AgentSettings
from review_agent.models import ReviewRequest, Severity
from review_agent.orchestrator import ReviewOrchestrator


def main(argv: list[str] | None = None) -> int:
    """Run CLI command and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 1

    patch_text = _load_patch_text(args)
    settings = AgentSettings.from_env()
    fail_on = Severity(args.fail_on.lower()) if args.fail_on else settings.fail_on_severity

    request = ReviewRequest(
        workspace_id=args.workspace_id,
        patch_text=patch_text,
        llm_model=args.llm_model or settings.llm_model,
        cxxtract_base_url=args.cxxtract_base_url or settings.cxxtract_base_url,
        fail_on_severity=fail_on,
        max_symbols=args.max_symbols or settings.max_symbols,
        max_tool_rounds=args.max_tool_rounds or settings.max_tool_rounds,
        max_total_tool_calls=args.max_total_tool_calls or settings.max_total_tool_calls,
        parse_timeout_s=args.parse_timeout_s or settings.parse_timeout_s,
        parse_workers=args.parse_workers or settings.parse_workers,
        max_candidates_per_symbol=args.max_candidates_per_symbol or settings.max_candidates_per_symbol,
        max_fetch_limit=args.max_fetch_limit or settings.max_fetch_limit,
    )

    orchestrator = ReviewOrchestrator()
    try:
        report = orchestrator.run(request)
    except Exception as exc:
        print(f"[ERROR] review execution failed: {exc}", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir).resolve()
    md_path, json_path = orchestrator.write_report_files(report, out_dir)
    print(f"[INFO] markdown report: {md_path}")
    print(f"[INFO] json report: {json_path}")
    print(
        f"[INFO] findings={len(report.findings)} "
        f"blocking={report.decision.blocking_findings} "
        f"threshold={report.decision.fail_threshold.value}"
    )
    return 2 if report.decision.should_block else 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CXXtract2 semantic PR review agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run one semantic review from a PR patch")
    run.add_argument("--workspace-id", required=True, help="Registered CXXtract workspace id")
    run.add_argument("--patch-file", default="", help="Path to unified git patch")
    run.add_argument("--patch-stdin", action="store_true", default=False, help="Read patch text from stdin")
    run.add_argument("--llm-model", default="", help="PydanticAI model id (e.g. openai:gpt-4o)")
    run.add_argument("--cxxtract-base-url", default="", help="CXXtract API base URL")
    run.add_argument("--fail-on", default="", choices=[s.value for s in Severity], help="CI fail threshold")
    run.add_argument("--out-dir", default="./review_agent_output", help="Output directory for report files")
    run.add_argument("--max-symbols", type=int, default=0)
    run.add_argument("--max-tool-rounds", type=int, default=0)
    run.add_argument("--max-total-tool-calls", type=int, default=0)
    run.add_argument("--parse-timeout-s", type=int, default=0)
    run.add_argument("--parse-workers", type=int, default=0)
    run.add_argument("--max-candidates-per-symbol", type=int, default=0)
    run.add_argument("--max-fetch-limit", type=int, default=0)
    return parser


def _load_patch_text(args: argparse.Namespace) -> str:
    if args.patch_stdin:
        data = sys.stdin.read()
        if not data.strip():
            raise ValueError("stdin patch is empty")
        return data
    if args.patch_file:
        p = Path(args.patch_file).resolve()
        if not p.exists():
            raise ValueError(f"patch file does not exist: {p}")
        data = p.read_text(encoding="utf-8", errors="replace")
        if not data.strip():
            raise ValueError(f"patch file is empty: {p}")
        return data
    raise ValueError("one of --patch-file or --patch-stdin is required")


if __name__ == "__main__":
    raise SystemExit(main())
