from __future__ import annotations

from pathlib import Path

from review_agent.manifest_resolver import (
    WorkspaceManifest,
    dependency_map,
    load_workspace_manifest,
    repo_for_file_key,
    resolve_file_key,
)


def test_load_manifest_and_dependency_map(tmp_path: Path):
    p = tmp_path / "workspace.yaml"
    p.write_text(
        "\n".join(
            [
                "workspace_id: ws_main",
                "repos:",
                "  - repo_id: repoA",
                "    root: repos/repoA",
                "    depends_on: [repoB]",
                "  - repo_id: repoB",
                "    root: repos/repoB",
                "    depends_on: []",
            ]
        ),
        encoding="utf-8",
    )
    mf = load_workspace_manifest(p)
    deps = dependency_map(mf)
    assert mf.workspace_id == "ws_main"
    assert deps["repoA"] == {"repoB"}


def test_resolve_file_key_from_prefixed_patch_path(tmp_path: Path):
    workspace_root = tmp_path / "ws"
    (workspace_root / "repos/repoA/src").mkdir(parents=True, exist_ok=True)
    (workspace_root / "repos/repoA/src/a.cpp").write_text("int a;", encoding="utf-8")

    mf = WorkspaceManifest.model_validate(
        {
            "workspace_id": "ws_main",
            "repos": [{"repo_id": "repoA", "root": "repos/repoA"}],
        }
    )
    fk = resolve_file_key(
        changed_path="repos/repoA/src/a.cpp",
        workspace_root=workspace_root,
        manifest=mf,
    )
    assert fk == "repoA:src/a.cpp"
    assert repo_for_file_key(fk) == "repoA"


def test_resolve_file_key_from_repo_relative_path(tmp_path: Path):
    workspace_root = tmp_path / "ws"
    (workspace_root / "repos/repoA/src").mkdir(parents=True, exist_ok=True)
    (workspace_root / "repos/repoA/src/a.cpp").write_text("int a;", encoding="utf-8")
    (workspace_root / "repos/repoB/src").mkdir(parents=True, exist_ok=True)

    mf = WorkspaceManifest.model_validate(
        {
            "workspace_id": "ws_main",
            "repos": [
                {"repo_id": "repoA", "root": "repos/repoA"},
                {"repo_id": "repoB", "root": "repos/repoB"},
            ],
        }
    )
    fk = resolve_file_key(
        changed_path="src/a.cpp",
        workspace_root=workspace_root,
        manifest=mf,
    )
    assert fk == "repoA:src/a.cpp"

