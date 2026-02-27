"""Prompt templates for planner and synthesis agent stages."""

from __future__ import annotations

import json

from review_agent.models import PrepassResult, ReviewContextBundle, ReviewFactSheet

PLANNER_SYSTEM_PROMPT = """You are a C++ semantic review planning agent.
You do not write findings. You only plan investigation order and budget usage.
Use deterministic pre-pass evidence as the primary signal.
Prefer high-risk symbols first and keep the plan bounded by provided budgets.
Return only data matching the ReviewPlan schema.
"""

SYNTHESIS_SYSTEM_PROMPT = """You are a senior C++ semantic code reviewer.
You receive a deterministic fact sheet and must synthesize actionable findings.
Do not invent evidence. If coverage is partial, reflect uncertainty in confidence.
Every high/critical finding must cite concrete evidence rows.
Return only data matching the ReviewReport schema.
"""


def build_planner_prompt(
    *,
    context: ReviewContextBundle,
    prepass: PrepassResult,
    budgets: dict[str, int],
) -> str:
    """Build planning-stage prompt from deterministic context and pre-pass facts."""
    payload = {
        "workspace_id": context.workspace_id,
        "base_sha": context.base_sha,
        "head_sha": context.head_sha,
        "target_branch_head_sha": context.target_branch_head_sha,
        "merge_preview_sha": context.merge_preview_sha,
        "changed_files_hint": context.changed_files[:300],
        "changed_hunks_hint": context.changed_hunks[:80],
        "changed_files": prepass.changed_files[:300],
        "seed_symbols": [s.model_dump() for s in prepass.seed_symbols[:200]],
        "suspicious_anchors": [a.model_dump() for a in prepass.suspicious_anchors[:300]],
        "changed_methods": prepass.changed_methods[:200],
        "added_call_sites": prepass.added_call_sites[:300],
        "removed_call_sites": prepass.removed_call_sites[:300],
        "include_macro_config_changes": prepass.include_macro_config_changes[:200],
        "budgets": budgets,
    }
    return (
        "Build a bounded review execution plan.\n"
        "Output a ReviewPlan.\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=True)}"
    )


def build_synthesis_prompt(*, fact_sheet: ReviewFactSheet, fail_threshold: str) -> str:
    """Build synthesis prompt from deterministic fact-sheet output."""
    payload = {
        "fail_threshold": fail_threshold,
        "fact_sheet": fact_sheet.model_dump(mode="json"),
    }
    return (
        "Generate final review findings and summary from this fact sheet.\n"
        "Use categories: hidden_side_effect, cross_repo_breakage, architecture_risk, confidence_gap.\n"
        "Output ReviewReport only.\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=True)}"
    )
