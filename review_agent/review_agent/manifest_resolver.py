"""Workspace manifest loading and file-key resolution helpers."""

from __future__ import annotations

import logging
from pathlib import Path, PurePosixPath
from typing import Optional

logger = logging.getLogger("review_agent.manifest_resolver")

import yaml
from pydantic import BaseModel, Field, model_validator


class ManifestRepo(BaseModel):
    """Minimal repo block required for review-time dependency checks."""

    repo_id: str
    root: str
    depends_on: list[str] = Field(default_factory=list)


class WorkspaceManifest(BaseModel):
    """Minimal manifest model used by the review agent."""

    workspace_id: str
    repos: list[ManifestRepo] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_repo(self) -> "WorkspaceManifest":
        seen: set[str] = set()
        for repo in self.repos:
            if repo.repo_id in seen:
                raise ValueError(f"duplicate repo_id: {repo.repo_id}")
            seen.add(repo.repo_id)
        return self

    def repo_map(self) -> dict[str, ManifestRepo]:
        return {r.repo_id: r for r in self.repos}


def load_workspace_manifest(path: str | Path) -> WorkspaceManifest:
    """Load workspace manifest from YAML."""
    p = Path(path).resolve()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid workspace manifest structure: {p}")
    return WorkspaceManifest.model_validate(raw)


def resolve_file_key(
    *,
    changed_path: str,
    workspace_root: str | Path,
    manifest: WorkspaceManifest,
) -> Optional[str]:
    """Resolve a patch path to canonical repo_id:rel/path key."""
    clean = _normalize_rel(changed_path)
    if not clean:
        return None

    # Case 1: patch path already includes repo.root prefix.
    for repo in manifest.repos:
        root = _normalize_rel(repo.root).rstrip("/")
        if not root:
            continue
        if clean == root:
            return f"{repo.repo_id}:."
        if clean.startswith(root + "/"):
            rel = clean[len(root) + 1 :]
            rel_clean = _normalize_rel(rel)
            if rel_clean:
                return f"{repo.repo_id}:{rel_clean}"

    # Case 2: path is repo-relative inside one repo root.
    workspace = Path(workspace_root).resolve()
    candidates: list[str] = []
    for repo in manifest.repos:
        repo_root = (workspace / repo.root).resolve()
        probe = (repo_root / PurePosixPath(clean)).resolve()
        if probe.exists() and probe.is_relative_to(repo_root):
            rel = probe.relative_to(repo_root).as_posix()
            candidates.append(f"{repo.repo_id}:{rel}")
    if len(candidates) == 1:
        return candidates[0]
    return None


def repo_for_file_key(file_key: str) -> str:
    """Return repo_id prefix from canonical file key."""
    if ":" not in file_key:
        return ""
    return file_key.split(":", 1)[0]


def dependency_map(manifest: WorkspaceManifest) -> dict[str, set[str]]:
    """Return dependency adjacency map keyed by repo_id."""
    return {repo.repo_id: set(repo.depends_on) for repo in manifest.repos}


def _normalize_rel(path_text: str) -> str:
    text = path_text.strip().replace("\\", "/")
    if text.startswith("a/") or text.startswith("b/"):
        text = text[2:]
    if text.startswith("/"):
        text = text[1:]
    parts = [p for p in PurePosixPath(text).parts if p]
    if not parts or any(p in {".", ".."} for p in parts):
        return ""
    return "/".join(parts)

