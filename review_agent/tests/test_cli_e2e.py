from __future__ import annotations

import json
from pathlib import Path

from review_agent import cli
from review_agent.models import (
    CoverageSummary,
    ReviewDecision,
    ReviewExecutionStatus,
    ReviewReport,
    Severity,
)


def _make_report(should_block: bool, *, exec_status: ReviewExecutionStatus = ReviewExecutionStatus.PASS) -> ReviewReport:
    return ReviewReport(
        workspace_id="ws_main",
        summary="summary",
        findings=[],
        coverage=CoverageSummary(),
        decision=ReviewDecision(
            fail_threshold=Severity.HIGH,
            blocking_findings=1 if should_block else 0,
            should_block=should_block,
            execution_status=exec_status,
        ),
        tool_usage=[],
    )


def test_cli_returns_pass_exit_code(monkeypatch, tmp_path: Path):
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")

    class _FakeOrchestrator:
        def run(self, _request):
            return _make_report(False)

        @staticmethod
        def write_report_files(report, out_dir):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            md = out / "review_report.md"
            js = out / "review_report.json"
            md.write_text("ok", encoding="utf-8")
            js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return md, js

    monkeypatch.setattr(cli, "ReviewOrchestrator", _FakeOrchestrator)
    code = cli.main(
        [
            "run",
            "--workspace-id",
            "ws_main",
            "--patch-file",
            str(patch_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0


def test_cli_returns_block_exit_code(monkeypatch, tmp_path: Path):
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")

    class _FakeOrchestrator:
        def run(self, _request):
            return _make_report(True, exec_status=ReviewExecutionStatus.BLOCK)

        @staticmethod
        def write_report_files(report, out_dir):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            md = out / "review_report.md"
            js = out / "review_report.json"
            md.write_text("block", encoding="utf-8")
            js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return md, js

    monkeypatch.setattr(cli, "ReviewOrchestrator", _FakeOrchestrator)
    code = cli.main(
        [
            "run",
            "--workspace-id",
            "ws_main",
            "--patch-file",
            str(patch_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 2


def test_cli_returns_indeterminate_exit_code(monkeypatch, tmp_path: Path):
    """INDETERMINATE + should_block → exit code 3."""
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")

    class _FakeOrchestrator:
        def run(self, _request):
            return _make_report(True, exec_status=ReviewExecutionStatus.INDETERMINATE)

        @staticmethod
        def write_report_files(report, out_dir):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            md = out / "review_report.md"
            js = out / "review_report.json"
            md.write_text("indet", encoding="utf-8")
            js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return md, js

    monkeypatch.setattr(cli, "ReviewOrchestrator", _FakeOrchestrator)
    code = cli.main(
        [
            "run",
            "--workspace-id",
            "ws_main",
            "--patch-file",
            str(patch_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 3


def test_cli_indeterminate_pass_mode(monkeypatch, tmp_path: Path):
    """INDETERMINATE + should_block=False → exit code 0."""
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text("diff --git a/a b/a\n", encoding="utf-8")

    class _FakeOrchestrator:
        def run(self, _request):
            return _make_report(False, exec_status=ReviewExecutionStatus.INDETERMINATE)

        @staticmethod
        def write_report_files(report, out_dir):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            md = out / "review_report.md"
            js = out / "review_report.json"
            md.write_text("indet-pass", encoding="utf-8")
            js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return md, js

    monkeypatch.setattr(cli, "ReviewOrchestrator", _FakeOrchestrator)
    code = cli.main(
        [
            "run",
            "--workspace-id",
            "ws_main",
            "--patch-file",
            str(patch_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--infra-fail-mode",
            "pass",
        ]
    )
    assert code == 0


def test_cli_accepts_context_bundle(monkeypatch, tmp_path: Path):
    context_path = tmp_path / "context.json"
    context_path.write_text(
        json.dumps(
            {
                "workspace_id": "ws_main",
                "patch_text": "diff --git a/a b/a\n",
                "base_sha": "a" * 40,
                "head_sha": "b" * 40,
                "target_branch_head_sha": "c" * 40,
            }
        ),
        encoding="utf-8",
    )

    class _FakeOrchestrator:
        def run(self, request):
            assert request.context_bundle is not None
            return _make_report(False)

        @staticmethod
        def write_report_files(report, out_dir):
            out = Path(out_dir)
            out.mkdir(parents=True, exist_ok=True)
            md = out / "review_report.md"
            js = out / "review_report.json"
            md.write_text("ok", encoding="utf-8")
            js.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            return md, js

    monkeypatch.setattr(cli, "ReviewOrchestrator", _FakeOrchestrator)
    code = cli.main(
        [
            "run",
            "--workspace-id",
            "ws_main",
            "--context-file",
            str(context_path),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert code == 0
