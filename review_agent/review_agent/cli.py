"""CLI entrypoint for running the semantic review agent in CI."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from review_agent.adapters.gitlab import GitLabPublisher
from review_agent.config import AgentSettings
from review_agent.models import (
    PublishingError,
    RepoRevisionContext,
    ReviewContextBundle,
    ReviewExecutionStatus,
    ReviewRequest,
    Severity,
)
from review_agent.orchestrator import ReviewOrchestrator

logger = logging.getLogger("review_agent.cli")

EXIT_PASS = 0
EXIT_ERROR = 1
EXIT_BLOCK = 2
EXIT_INDETERMINATE = 3


def main(argv: list[str] | None = None) -> int:
    """Run CLI command and return process exit code."""
    _load_runtime_dotenv()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":
        parser.print_help()
        return EXIT_ERROR

    settings = AgentSettings.from_env()
    log_level_name = (args.log_level or settings.log_level or "WARNING").upper()
    numeric_level = getattr(logging, log_level_name, logging.WARNING)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )

    patch_text, context_bundle = _load_inputs(args)
    fail_on = Severity(args.fail_on.lower()) if args.fail_on else settings.fail_on_severity
    max_symbol_slots = args.max_symbol_slots or args.max_tool_rounds or settings.max_symbol_slots
    infra_fail_mode = (args.infra_fail_mode or "block").lower()
    if infra_fail_mode not in {"block", "pass"}:
        infra_fail_mode = "block"

    request = ReviewRequest(
        workspace_id=args.workspace_id,
        patch_text=patch_text,
        context_bundle=context_bundle,
        llm_model=args.llm_model or settings.llm_model,
        llm_base_url=args.llm_base_url or settings.llm_base_url,
        llm_api_key=args.llm_api_key or settings.llm_api_key,
        llm_app_url=args.llm_app_url or settings.llm_app_url,
        llm_app_title=args.llm_app_title or settings.llm_app_title,
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
        workspace_fingerprint=args.workspace_fingerprint or "",
        use_derived_workspaces=bool(args.use_derived_workspaces) if args.use_derived_workspaces else settings.use_derived_workspaces,
    )

    logger.info(
        "starting review workspace=%s model=%s cache=%s gitlab_mr=%s out_dir=%s",
        request.workspace_id,
        request.llm_model,
        "on" if request.enable_cache else "off",
        bool(args.gitlab_url and args.project_id and args.mr_iid),
        args.out_dir,
    )

    orchestrator = ReviewOrchestrator()
    try:
        report = orchestrator.run(request)
    except Exception as exc:
        logger.error("review execution failed: %s", exc)
        print(f"[ERROR] review execution failed: {exc}", file=sys.stderr)
        return EXIT_ERROR

    out_dir = Path(args.out_dir).resolve()

    if args.gitlab_url and args.project_id and args.mr_iid:
        token = args.gitlab_token or settings.gitlab_token
        try:
            publisher = GitLabPublisher(
                base_url=args.gitlab_url,
                private_token=token,
                project_id=args.project_id,
                mr_iid=int(args.mr_iid),
            )
            publish_result = publisher.publish(report, publish_inline=bool(args.publish_inline_comments))
            report = report.model_copy(update={"publish_result": publish_result})
        except PublishingError as exc:
            logger.error("publishing failed: %s", exc)
            print(f"[ERROR] publishing failed: {exc}", file=sys.stderr)
            return EXIT_ERROR

    md_path, json_path = orchestrator.write_report_files(report, out_dir)
    print(f"[INFO] markdown report: {md_path}")
    print(f"[INFO] json report: {json_path}")

    exec_status = report.decision.execution_status
    print(
        f"[INFO] findings={len(report.findings)} "
        f"blocking={report.decision.blocking_findings} "
        f"threshold={report.decision.fail_threshold.value} "
        f"status={exec_status.value} "
        f"confidence={report.decision.review_confidence}"
    )
    if report.run_id:
        print(f"[INFO] run_id={report.run_id}")
    if report.publish_result is not None:
        print(
            f"[INFO] publish provider={report.publish_result.provider} "
            f"summary={report.publish_result.summary_posted} "
            f"inline_comments={report.publish_result.inline_comments_posted}"
        )

    if exec_status == ReviewExecutionStatus.INDETERMINATE:
        return EXIT_INDETERMINATE if report.decision.should_block else EXIT_PASS
    return EXIT_BLOCK if report.decision.should_block else EXIT_PASS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CXXtract2 semantic PR review agent")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run one semantic review from a PR patch")
    run.add_argument("--workspace-id", required=True, help="Registered CXXtract workspace id")
    run.add_argument("--patch-file", default="", help="Path to unified git patch")
    run.add_argument("--patch-stdin", action="store_true", default=False, help="Read patch text from stdin")
    run.add_argument("--context-file", default="", help="Path to review context bundle JSON")
    run.add_argument("--context-stdin", action="store_true", default=False, help="Read review context bundle JSON from stdin")
    run.add_argument("--workspace-fingerprint", default="", help="Explicit fingerprint for cache-safe patch-only reviews")
    run.add_argument(
        "--llm-model",
        default="",
        help="Supported model ids: openai:*, openrouter:*, gateway:* (OpenAI-compatible), or fixture:*",
    )
    run.add_argument("--llm-base-url", default="", help="Base URL for gateway/openai-compatible LLM endpoints")
    run.add_argument("--llm-api-key", default="", help="Explicit API key for gateway/openrouter LLM endpoints")
    run.add_argument("--llm-app-url", default="", help="Optional OpenRouter app URL")
    run.add_argument("--llm-app-title", default="", help="Optional OpenRouter app title")
    run.add_argument("--cxxtract-base-url", default="", help="CXXtract API base URL")
    run.add_argument("--fail-on", default="", choices=[s.value for s in Severity], help="CI fail threshold")
    run.add_argument("--out-dir", default="./review_agent_output", help="Output directory for report files")
    run.add_argument("--max-symbols", type=int, default=0)
    run.add_argument("--max-symbol-slots", type=int, default=0, help="Max symbol investigation slots per review")
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
    run.add_argument(
        "--use-derived-workspaces",
        action="store_true",
        default=False,
        help="Use target/head/merge derived CXXtract workspaces instead of PR overlays",
    )
    run.add_argument(
        "--infra-fail-mode",
        default="block",
        choices=["block", "pass"],
        help="CI behavior on infrastructure failures: 'block' (default) or 'pass'",
    )
    run.add_argument("--gitlab-url", default="", help="GitLab instance base URL (e.g. https://gitlab.com)")
    run.add_argument("--gitlab-token", default="", help="GitLab private/project access token (prefer REVIEW_AGENT_GITLAB_TOKEN env var)")
    run.add_argument("--project-id", default="", help="GitLab project ID (numeric or URL-encoded path)")
    run.add_argument("--mr-iid", default="", help="GitLab Merge Request IID")
    run.add_argument(
        "--publish-inline-comments",
        action="store_true",
        default=False,
        help="Post inline discussion threads on findings with deterministic locations",
    )
    return parser


def _load_inputs(args: argparse.Namespace) -> tuple[str, ReviewContextBundle | None]:
    if getattr(args, "gitlab_url", "") and getattr(args, "project_id", "") and getattr(args, "mr_iid", ""):
        settings = AgentSettings.from_env()
        gitlab_token = getattr(args, "gitlab_token", "") or settings.gitlab_token
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

    primary_repo_id = str(metadata.get("path_with_namespace", "") or project_id)
    head_sha = str(metadata.get("diff_refs", {}).get("head_sha", ""))
    target_sha = str(metadata.get("diff_refs", {}).get("start_sha", ""))
    bundle = ReviewContextBundle(
        workspace_id=args.workspace_id,
        patch_text=diff_text,
        base_sha=str(metadata.get("diff_refs", {}).get("base_sha", "")),
        head_sha=head_sha,
        target_branch_head_sha=target_sha,
        merge_preview_sha=str(metadata.get("merge_commit_sha", "") or ""),
        primary_repo_id=primary_repo_id,
        per_repo_shas={primary_repo_id: head_sha} if primary_repo_id else {},
        repo_revisions=(
            [
                RepoRevisionContext(
                    repo_id=primary_repo_id,
                    base_sha=str(metadata.get("diff_refs", {}).get("base_sha", "")),
                    head_sha=head_sha,
                    target_sha=target_sha,
                    merge_sha=str(metadata.get("merge_commit_sha", "") or ""),
                    role="primary",
                )
            ]
            if primary_repo_id
            else []
        ),
        workspace_fingerprint=":".join(filter(None, [primary_repo_id, head_sha, target_sha])),
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


def _load_runtime_dotenv() -> None:
    """Load .env via python-dotenv unless explicitly disabled."""
    if os.getenv("REVIEW_AGENT_DISABLE_DOTENV", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    explicit = os.getenv("REVIEW_AGENT_ENV_FILE", "").strip()
    if explicit:
        load_dotenv(dotenv_path=explicit, override=False)
        return
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


if __name__ == "__main__":
    raise SystemExit(main())
