"""Native skill catalog loader and executor dispatch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml

from review_agent.skills.skill_architecture_risk import run_architecture_risk
from review_agent.skills.skill_cross_repo_breakage import run_cross_repo_breakage
from review_agent.skills.skill_evidence_read import run_evidence_read
from review_agent.skills.skill_patch_intake import run_patch_intake
from review_agent.skills.skill_symbol_impact import run_symbol_impact

SkillExecutor = Callable[[dict[str, Any]], dict[str, Any]]


EXECUTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "skill_patch_intake": run_patch_intake,
    "skill_symbol_impact": run_symbol_impact,
    "skill_evidence_read": run_evidence_read,
    "skill_cross_repo_breakage": run_cross_repo_breakage,
    "skill_architecture_risk": run_architecture_risk,
}


class SkillRegistry:
    """Load and execute native skill functions from catalog."""

    def __init__(self, catalog_path: str | Path | None = None) -> None:
        base = Path(__file__).resolve().parent
        self.catalog_path = Path(catalog_path) if catalog_path else (base / "catalog.yaml")
        raw = yaml.safe_load(self.catalog_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"invalid skills catalog format: {self.catalog_path}")
        rows = raw.get("skills", [])
        if not isinstance(rows, list):
            raise ValueError(f"invalid skills list in catalog: {self.catalog_path}")
        self._catalog = rows
        self._names = [str(item.get("name", "")).strip() for item in rows if str(item.get("name", "")).strip()]

    @property
    def names(self) -> list[str]:
        return list(self._names)

    def execute(self, name: str, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Run a skill and return state updates."""
        if name not in self._names:
            raise ValueError(f"skill not declared in catalog: {name}")
        fn = EXECUTORS.get(name)
        if fn is None:
            raise ValueError(f"no executor bound for skill: {name}")
        return fn(state, **kwargs)

