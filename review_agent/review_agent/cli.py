"""CLI entrypoint for running the semantic review agent in CI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from review_agent.config import AgentSettings
from review_agent.models import (
    ReviewContextBundle,
    ReviewExecutionStatus,
    ReviewRequest,
    Severity,
)
from review_agent.orchestrator import ReviewOrchestrator

logger = logging.getLogger("review_agent.cli")

# Exit codes
EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_BLOCK = 2
EXIT_INDETERMINATE = 3


def main(argv: list[str] | None = None) -> int:
    """Run CLI command and return process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return 1

    # --- Logging bootstrap ---
    settings = AgentSettings.from_env()
    log_level_name = (args.log_level or settings.log_level or "WARNING").upper()
    numeric_level = getattr(logging, log_level_name, logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    patch_text, context_bundle = _load_inputs(args)
    fail_on = Severity(args.fail_on.lower()) if args.fail_on else settings.fail_on_severity

    # Resolve max_symbol_slots: prefer new flag, fall back to deprecated --max-tool-rounds, then env
    max_symbol_slots = args.max_symbol_slots or args.max_tool_rounds or settings.max_symbol_slots

    # Resolve infra fail mode
    infra_fail_mode = (args.infra_fail_mode or "block").lower()
    if infra_fail_mode not in {"block", "pass"}:
        infra_fail_mode = "block"

    request = ReviewRequest(
        workspace_id=args.workspace_id,
        patch_text=patch_text,
        context_bundle=context_bundle,
        llm_model=args.llm_model or settings.llm_model,
        cxxtract_base_url=args.cxxtract_base_url or settings.cxxtract_base_url,
        fail_on_severity=fail_on,
        max_symbols=args.max_symbols or settings.max_symbols,
        max_symbol_slots=max_symbol_slots,
        max_total_tool_calls=args.max_total_tool_calls or settings.max_total_tool_calls,
        parse_timeout_s=args.parse_timeout_s or settings.parse_timeout_s,
        parse_workers=args.parse_workers or settings.parse_workers,
        max_candidates_per_symbol=args.max_candidates_per_symbol or settings.max_candidates_per_symbol,
        max_fetch_limit=args.max_fetch_limit or settings.max_fetch_limit,
        review_timeout_s=args.review_timeout_s or settings.review_timeout_s,
        enable_cache=not args.no_cache if args.no_cache else settings.enable_cache,
        cache_dir=args.cache_dir or settings.cache_dir,
        infra_fail_mode=infra_fail_mode,
    )

    orchestrator = ReviewOrchestrator()
    try:
        report = orchestrator.run(request)
    except Exception as exc:
        logger.error("review execution failed: %s", exc)
        print(f"[ERROR] review execution failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    out_dir = Path(args.out_dir).resolve()
    md_path, json_path = orchestrator.write_report_files(report, out_dir)
    print(f"[INFO] markdown report: {md_path}")
    print(f"[INFO] json report: {json_path}")

    exec_status = report.decision.execution_status
    print(
        f"[INFO] findings={len(report.findings)} "
        f"blocking={report.decision.blocking_findings} "
        f"threshold={report.decision.fail_threshold.value} "
        f"status={exec_status.value}"
    )
    if report.run_id:
        print(f"[INFO] run_id={report.run_id}")

    # Post review results to GitLab MR if flags provided
    if args.gitlab_url and args.project_id and args.mr_iid:
        gitlab_token = args.gitlab_token or os.getenv("REVIEW_AGENT_GITLAB_TOKEN", "")
        if not gitlab_token:
            logger.warning("--gitlab-token or REVIEW_AGENT_GITLAB_TOKEN not set; skipping MR comment")
        else:
            _publish_to_gitlab(
                args=args,
                token=gitlab_token,
                report=report,
                publish_inline=bool(args.publish_inline_comments),
            )

    # Map execution status to exit code
    if exec_status == ReviewExecutionStatus.INDETERMINATE:
        return EXIT_INDETERMINATE if report.decision.should_block else EXIT_PASS
    return EXIT_BLOCK if report.decision.should_block else EXIT_PASS


def _publish_to_gitlab(
    *,
    args: argparse.Namespace,
    token: str,
    report,
    publish_inline: bool,
) -> None:
    """Post review summary (and optionally inline comments) to GitLab MR."""
    try:
        from review_agent.tool_clients.gitlab_client import GitLabClient
        from review_agent.report_renderer import render_markdown as _render_md

        gl = GitLabClient(
            base_url=args.gitlab_url,
            private_token=token,
        )
        mr_iid = int(args.mr_iid)
        project_id = args.project_id

        # Always post the summary note
        body = _render_md(report)
        gl.post_mr_note(
            project_id=project_id,
            mr_iid=mr_iid,
            body=body,
        )
        logger.info("posted review summary to MR !%s", mr_iid)

        # Optionally post inline discussions for positioned findings
        if publish_inline and report.findings:
            diff_refs = {}
            if report.run_metadata and hasattr(report, '_context_bundle_cache'):
                pass  # We'll fetch from MR metadata
            # Try to get diff_refs from context bundle
            bundle = getattr(report, '_context_bundle', None)
            base_sha = ""
            head_sha = ""
            start_sha = ""
            if hasattr(args, 'project_id'):
                try:
                    meta = gl.get_mr_metadata(project_id=project_id, mr_iid=mr_iid)
                    diff_refs = meta.get("diff_refs", {})
                    base_sha = str(diff_refs.get("base_sha", ""))
                    head_sha = str(diff_refs.get("head_sha", ""))
                    start_sha = str(diff_refs.get("start_sha", ""))
                except Exception as exc:
                    logger.warning("failed to fetch MR diff_refs for inline comments: %s", exc)

            for finding in report.findings:
                if not finding.diff_path or finding.diff_line <= 0:
                    continue
                try:
                    inline_body = (
                        f"**[{finding.severity.value.upper()}]** {finding.title}\n\n"
                        f"{finding.impact}\n\n"
                        f"**Recommendation:** {finding.recommendation}"
                    )
                    gl.post_mr_inline_discussion(
                        project_id=project_id,
                        mr_iid=mr_iid,
                        body=inline_body,
                        new_path=finding.diff_path,
                        new_line=finding.diff_line,
                        base_sha=base_sha,
                        head_sha=head_sha,
                        start_sha=start_sha,
                    )
                    logger.info("posted inline comment on %s:%d", finding.diff_path, finding.diff_line)
                except Exception as exc:
                    logger.warning("failed to post inline comment for finding %s: %s", finding.id, exc)

        gl.close()
    except Exception as exc:
        logger.error("failed to publish to GitLab: %s", exc)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CXXtract2 semantic PR review agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run one semantic review from a PR patch")
    run.add_argument("--workspace-id", required=True, help="Registered CXXtract workspace id")
    run.add_argument("--patch-file", default="", help="Path to unified git patch")
    run.add_argument("--patch-stdin", action="store_true", default=False, help="Read patch text from stdin")
    run.add_argument("--context-file", default="", help="Path to review context bundle JSON")
    run.add_argument("--context-stdin", action="store_true", default=False, help="Read review context bundle JSON from stdin")
    run.add_argument("--llm-model", default="", help="PydanticAI model id (e.g. openai:gpt-4o)")
    run.add_argument("--cxxtract-base-url", default="", help="CXXtract API base URL")
    run.add_argument("--fail-on", default="", choices=[s.value for s in Severity], help="CI fail threshold")
    run.add_argument("--out-dir", default="./review_agent_output", help="Output directory for report files")
    run.add_argument("--max-symbols", type=int, default=0)
    # New canonical name
    run.add_argument("--max-symbol-slots", type=int, default=0, help="Max symbol investigation slots per review")
    # Deprecated alias for backward compatibility
    run.add_argument("--max-tool-rounds", type=int, default=0, help="(deprecated, use --max-symbol-slots)")
    run.add_argument("--max-total-tool-calls", type=int, default=0)
    run.add_argument("--parse-timeout-s", type=int, default=0)
    run.add_argument("--parse-workers", type=int, default=0)
    run.add_argument("--max-candidates-per-symbol", type=int, default=0)
    run.add_argument("--max-fetch-limit", type=int, default=0)
    run.add_argument("--review-timeout-s", type=int, default=0, help="Wall-clock timeout for the entire review run (0=disabled)")
    run.add_argument("--log-level", default="", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    run.add_argument("--no-cache", action="store_true", default=False, help="Disable local review trace cache")
    run.add_argument("--cache-dir", default="", help="Directory for review trace cache")

    # Infrastructure policy
    run.add_argument(
        "--infra-fail-mode", default="block",
        choices=["block", "pass"],
        help="CI behavior on indeterminate results: 'block' (default) or 'pass'",
    )

    # GitLab integration flags
    run.add_argument("--gitlab-url", default="", help="GitLab instance base URL (e.g. https://gitlab.com)")
    run.add_argument("--gitlab-token", default="", help="GitLab private/project access token (prefer REVIEW_AGENT_GITLAB_TOKEN env var)")
    run.add_argument("--project-id", default="", help="GitLab project ID (numeric or URL-encoded path)")
    run.add_argument("--mr-iid", default="", help="GitLab Merge Request IID")
    run.add_argument(
        "--publish-inline-comments", action="store_true", default=False,
        help="Post inline discussion threads on positioned findings (requires --gitlab-url, --project-id, --mr-iid)",
    )
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[str, ReviewContextBundle | None]:
    # If GitLab flags are provided, fetch MR diff and context from GitLab API
    if getattr(args, "gitlab_url", "") and getattr(args, "project_id", "") and getattr(args, "mr_iid", ""):
        gitlab_token = getattr(args, "gitlab_token", "") or os.getenv("REVIEW_AGENT_GITLAB_TOKEN", "")
        if not gitlab_token:
            raise ValueError("--gitlab-token or REVIEW_AGENT_GITLAB_TOKEN environment variable required for GitLab mode")
        return _load_from_gitlab(args, gitlab_token)

    if args.context_stdin and args.patch_stdin:
        raise ValueError("cannot read both context and patch from stdin in one run")
    context = _load_context_bundle(args)
    patch = _load_patch_text(args)
    if not patch and context is None:
        raise ValueError("one of patch input (--patch-file/--patch-stdin) or context input (--context-file/--context-stdin) is required")
    return patch, context


def _load_from_gitlab(args: argparse.Namespace, token: str) -> tuple[str, ReviewContextBundle | None]:
    """Fetch MR diff and metadata from GitLab API and produce inputs."""
    from review_agent.tool_clients.gitlab_client import GitLabClient

    gl = GitLabClient(base_url=args.gitlab_url, private_token=token)
    mr_iid = int(args.mr_iid)
    project_id = args.project_id

    metadata = gl.get_mr_metadata(project_id=project_id, mr_iid=mr_iid)
    diff_text = gl.get_mr_diff(project_id=project_id, mr_iid=mr_iid)

    # Derive primary_repo_id from project path or ID
    primary_repo_id = str(metadata.get("path_with_namespace", "") or project_id)

    bundle = ReviewContextBundle(
        workspace_id=args.workspace_id,
        patch_text=diff_text,
        base_sha=str(metadata.get("diff_refs", {}).get("base_sha", "")),
        head_sha=str(metadata.get("diff_refs", {}).get("head_sha", "")),
        target_branch_head_sha=str(metadata.get("diff_refs", {}).get("start_sha", "")),
        merge_preview_sha=str(metadata.get("merge_commit_sha", "") or ""),
        primary_repo_id=primary_repo_id,
        per_repo_shas={
            primary_repo_id: str(metadata.get("diff_refs", {}).get("head_sha", "")),
        } if primary_repo_id else {},
        pr_metadata={
            "mr_id": str(metadata.get("iid", "")),
            "pr_id": str(metadata.get("iid", "")),
            "title": str(metadata.get("title", "")),
            "source_branch": str(metadata.get("source_branch", "")),
            "target_branch": str(metadata.get("target_branch", "")),
            "web_url": str(metadata.get("web_url", "")),
        },
    )
    gl.close()
    return diff_text, bundle


def _load_context_bundle(args: argparse.Namespace) -> ReviewContextBundle | None:
    raw = ""
    if args.context_stdin:
        raw = sys.stdin.read()
    elif args.context_file:
        p = Path(args.context_file).resolve()
        if not p.exists():
            raise ValueError(f"context file does not exist: {p}")
        raw = p.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"context bundle is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("context bundle JSON must be an object")
    return ReviewContextBundle.model_validate(payload)


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
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
