from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from review_agent.testing.fakes import CxxtractFixtureServer, GitLabFixtureServer


ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env["REVIEW_AGENT_DISABLE_DOTENV"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "review_agent.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _run_cli_with_env_file(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["PYTHONPATH"] = str(ROOT)
    env.pop("REVIEW_AGENT_DISABLE_DOTENV", None)
    return subprocess.run(
        [sys.executable, "-m", "review_agent.cli", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_patch_file_end_to_end(tmp_path: Path):
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/src/app.cpp b/src/app.cpp",
                "--- a/src/app.cpp",
                "+++ b/src/app.cpp",
                "@@ -1,2 +1,3 @@",
                " int main() {",
                "+  doLogin();",
                " }",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    with CxxtractFixtureServer() as cxxtract:
        result = _run_cli(
            [
                "run",
                "--workspace-id",
                "ws_main",
                "--patch-file",
                str(patch_path),
                "--workspace-fingerprint",
                "ws_main:test-snapshot",
                "--llm-model",
                "fixture:blocking",
                "--cxxtract-base-url",
                cxxtract.url,
                "--out-dir",
                str(out_dir),
            ],
            cwd=ROOT,
        )

    assert result.returncode == 2, result.stderr
    report = json.loads((out_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["execution_status"] == "block"
    assert report["decision"]["review_confidence"] == "high"
    assert report["findings"][0]["location"]["path"] == "src/app.cpp"
    assert report["findings"][0]["location"]["line"] == 2
    assert "publish_result" in report
    assert report["publish_result"] is None
    assert "confidence=high" in result.stdout


def test_cli_loads_runtime_settings_from_dotenv(tmp_path: Path):
    patch_path = tmp_path / "pr.patch"
    patch_path.write_text(
        "\n".join(
            [
                "diff --git a/src/app.cpp b/src/app.cpp",
                "--- a/src/app.cpp",
                "+++ b/src/app.cpp",
                "@@ -1,2 +1,3 @@",
                " int main() {",
                "+  doLogin();",
                " }",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    with CxxtractFixtureServer() as cxxtract:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "REVIEW_AGENT_LLM_MODEL=fixture:blocking",
                    f"REVIEW_AGENT_CXXTRACT_BASE_URL={cxxtract.url}",
                    "REVIEW_AGENT_LOG_LEVEL=INFO",
                ]
            ),
            encoding="utf-8",
        )
        result = _run_cli_with_env_file(
            [
                "run",
                "--workspace-id",
                "ws_main",
                "--patch-file",
                str(patch_path),
                "--workspace-fingerprint",
                "ws_main:dotenv-snapshot",
                "--out-dir",
                str(out_dir),
            ],
            cwd=tmp_path,
        )

    assert result.returncode == 2, result.stderr
    report = json.loads((out_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["decision"]["execution_status"] == "block"
    assert report["findings"][0]["location"]["path"] == "src/app.cpp"


def test_cli_gitlab_publish_end_to_end(tmp_path: Path):
    out_dir = tmp_path / "out"

    with CxxtractFixtureServer() as cxxtract, GitLabFixtureServer() as gitlab:
        result = _run_cli(
            [
                "run",
                "--workspace-id",
                "ws_main",
                "--gitlab-url",
                gitlab.url,
                "--gitlab-token",
                "token",
                "--project-id",
                "group%2Fproject",
                "--mr-iid",
                "7",
                "--publish-inline-comments",
                "--llm-model",
                "fixture:gitlab-inline",
                "--cxxtract-base-url",
                cxxtract.url,
                "--out-dir",
                str(out_dir),
            ],
            cwd=ROOT,
        )

        assert len(gitlab.state.notes) == 1
        assert len(gitlab.state.discussions) == 1

    assert result.returncode == 2, result.stderr
    report = json.loads((out_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["run_metadata"]["input_mode"] == "gitlab_mr"
    assert report["publish_result"]["provider"] == "gitlab"
    assert report["publish_result"]["summary_posted"] is True
    assert report["publish_result"]["inline_comments_posted"] == 1
    assert report["findings"][0]["location"]["path"] == "src/app.cpp"


def test_cli_loads_gitlab_token_from_dotenv(tmp_path: Path):
    out_dir = tmp_path / "out"

    with CxxtractFixtureServer() as cxxtract, GitLabFixtureServer() as gitlab:
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "REVIEW_AGENT_LLM_MODEL=fixture:gitlab-inline",
                    f"REVIEW_AGENT_CXXTRACT_BASE_URL={cxxtract.url}",
                    "REVIEW_AGENT_GITLAB_TOKEN=token",
                ]
            ),
            encoding="utf-8",
        )
        result = _run_cli_with_env_file(
            [
                "run",
                "--workspace-id",
                "ws_main",
                "--gitlab-url",
                gitlab.url,
                "--project-id",
                "group%2Fproject",
                "--mr-iid",
                "7",
                "--publish-inline-comments",
                "--out-dir",
                str(out_dir),
            ],
            cwd=tmp_path,
        )

        assert len(gitlab.state.notes) == 1
        assert len(gitlab.state.discussions) == 1

    assert result.returncode == 2, result.stderr
    report = json.loads((out_dir / "review_report.json").read_text(encoding="utf-8"))
    assert report["publish_result"]["summary_posted"] is True
    assert report["publish_result"]["inline_comments_posted"] == 1
