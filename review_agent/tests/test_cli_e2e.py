from __future__ import annotations

from pathlib import Path

from review_agent import cli
from review_agent.models import CoverageSummary, ReviewDecision, ReviewReport, Severity


def _make_report(should_block: bool) -> ReviewReport:
    return ReviewReport(
        workspace_id="ws_main",
        summary="summary",
        findings=[],
        coverage=CoverageSummary(),
        decision=ReviewDecision(
            fail_threshold=Severity.HIGH,
            blocking_findings=1 if should_block else 0,
            should_block=should_block,
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
            return _make_report(True)

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

