"""Prompt templates for planner, exploration, and synthesis agent stages."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("review_agent.prompting")

from review_agent.models import PrepassResult, ReviewContextBundle, ReviewFactSheet


PLANNER_SYSTEM_PROMPT = """\
You are a C++ semantic review planning agent.

Your job is to produce a bounded investigation plan (ReviewPlan) that will
guide deterministic evidence collection.  You do NOT write findings.

## How to use the pre-pass evidence

The user message contains deterministic signals extracted from the patch
before any LLM reasoning:

* **seed_symbols** -- ranked symbols extracted from added/deleted lines.
  Each symbol includes a relevance tier and reasons explaining why it was
  promoted.
* **suspicious_anchors** -- lines that touch concurrency primitives
  (mutex, atomic), lifetime management (unique_ptr, delete),
  exception flow (throw/catch), or ABI/dispatch (virtual/override).
  These are high-priority review targets.
* **changed_declarations** -- declaration-like symbols that were added or
  modified in the diff, including classes/structs/enums.
* **member_call_sites** -- member-style calls observed in changed code.
  When these reference generic APIs like `.load()`, prefer the owning
  declaration/container rather than the bare member name.
* **added_call_sites / removed_call_sites** -- planner-facing call-site
  hints that were introduced or removed. A removed call-site that still has
  references elsewhere is a potential hidden side-effect.
* **include_macro_config_changes** -- files where #include, #define,
  or build config changed.  These can have wide blast radius.
* **diff_context** -- curated raw diff context. Use this to disambiguate
  generic APIs that are only meaningful inside the changed container.

Use these as the PRIMARY signal for prioritizing symbols.
Prefer declaration/container-owned symbols over bare generic call names.
Keep the plan bounded by the provided budgets.

## Merge-awareness

If `merge_preview_sha` is present the review targets a pull-request with
a merge-preview commit.  Flag symbols that may interact differently in the
merge result compared to head-only.  Set `require_merge_preview = true`
when the diff touches shared interfaces or config that could conflict.

Return only data matching the ReviewPlan schema.
"""

SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior C++ semantic code reviewer.

You receive a deterministic fact sheet assembled from pre-pass extraction
and bounded evidence collection.  Your job is to synthesize actionable,
evidence-backed findings.

## Rules

1. Do NOT invent evidence.  Every high/critical finding MUST cite at
   least one concrete evidence anchor (file_key + line or symbol).
2. If coverage is partial (verified_ratio < 0.5), reflect that
   uncertainty in finding confidence scores.
3. Use categories: hidden_side_effect, cross_repo_breakage,
   architecture_risk, confidence_gap.

## Merge-aware reasoning

When the fact sheet includes `view_contexts` with a materialized
merge-preview, reason about three-way differences:

* **Baseline impact** -- what changed from the target branch baseline.
* **Change intent** -- what the PR head introduces.
* **Merge collision risk** -- symbols where merge-preview reference or
  call-edge counts differ from BOTH head and baseline, indicating the
  merge introduces new interactions not present in either alone.

If `merge_delta_signals` are present, treat them as high-priority
review targets that may indicate semantic merge conflicts.

## Suspicious anchors

The fact sheet may include suspicious_anchors flagging concurrency,
lifetime, exception-flow, or ABI changes.  These should be investigated
for hidden side-effects even when coverage is high.

Return only data matching the SynthesisDraft schema.
"""

EXPLORATION_SYSTEM_PROMPT = """\
You are a C++ code exploration agent performing targeted follow-up
investigation for a semantic PR review.

You have already received deterministic base evidence.  Your job is to
fill evidence gaps by calling explore tools for specific symbols, files,
or queries.

## Available tools

- **explore_list_candidates** -- find candidate files for a symbol
- **explore_fetch_symbols** -- look up symbol definitions
- **explore_fetch_references** -- trace who references a symbol
- **explore_fetch_call_edges** -- trace callers/callees of a symbol
- **explore_read_file** -- read source code context
- **explore_rg_search** -- text/regex search across the workspace

## Strategy

1. Prioritize suspicious anchors and low-confidence symbols.
2. Do NOT re-investigate symbols that already have good evidence.
3. Respect your remaining call budget -- stop when exhausted.
4. Return data matching the ExplorationResult schema.
"""


def build_planner_prompt(
    *,
    context: ReviewContextBundle,
    prepass: PrepassResult,
    budgets: dict[str, int],
) -> str:
    """Build planning-stage prompt from deterministic context and pre-pass facts."""
    # Summarize suspicious anchors by kind for compactness
    anchor_summary: dict[str, int] = {}
    for a in prepass.suspicious_anchors:
        anchor_summary[a.kind] = anchor_summary.get(a.kind, 0) + 1

    payload: dict[str, Any] = {
        "workspace_id": context.workspace_id,
        "base_sha": context.base_sha[:12] if context.base_sha else "",
        "head_sha": context.head_sha[:12] if context.head_sha else "",
        "target_branch_head_sha": context.target_branch_head_sha[:12] if context.target_branch_head_sha else "",
        "merge_preview_sha": context.merge_preview_sha[:12] if context.merge_preview_sha else "",
        "changed_files": prepass.changed_files[:300],
        "changed_hunk_count": prepass.changed_hunk_count,
        "seed_symbols": [
            {
                "symbol": s.symbol,
                "source": s.source,
                "score": s.score,
                "tier": s.relevance_tier,
                "reasons": s.reasons[:6],
                "receiver": s.receiver,
                "container": s.container,
                "file_paths": s.file_paths[:6],
            }
            for s in prepass.seed_symbols[:60]
        ],
        "suspicious_anchor_summary": anchor_summary,
        "suspicious_anchors_top": [
            {"kind": a.kind, "file": a.file_path, "line": a.line, "snippet": a.snippet[:120]}
            for a in prepass.suspicious_anchors[:30]
        ],
        "changed_declarations": [
            {
                "symbol": decl.symbol,
                "container": decl.container,
                "kind": decl.kind,
                "file": decl.file_path,
                "line": decl.line,
            }
            for decl in prepass.changed_declarations[:100]
        ],
        "changed_containers": prepass.changed_containers[:80],
        "member_call_sites": [
            {
                "receiver": call.receiver,
                "member": call.member,
                "container": call.container,
                "qualified_receiver_type": call.qualified_receiver_type,
                "file": call.file_path,
                "line": call.line,
            }
            for call in prepass.member_call_sites[:80]
        ],
        "changed_methods": prepass.changed_methods[:100],
        "added_call_sites": prepass.added_call_sites[:100],
        "removed_call_sites": prepass.removed_call_sites[:100],
        "include_macro_config_changes": prepass.include_macro_config_changes[:80],
        "diff_context": _planner_diff_payload(context=context, prepass=prepass),
        "budgets": budgets,
    }

    sections = ["Build a bounded review execution plan.", "Output a ReviewPlan."]

    if context.merge_preview_sha:
        sections.append(
            "MERGE CONTEXT: A merge-preview commit is available. "
            "Consider setting require_merge_preview=true if the diff "
            "touches shared interfaces, configs, or ABI-sensitive symbols."
        )

    if anchor_summary:
        anchors_desc = ", ".join(f"{k}={v}" for k, v in sorted(anchor_summary.items(), key=lambda kv: -kv[1]))
        sections.append(f"SUSPICIOUS ANCHORS detected: {anchors_desc}. Prioritize these.")

    sections.append(json.dumps(payload, indent=2, ensure_ascii=True))
    return "\n\n".join(sections)


def _planner_diff_payload(*, context: ReviewContextBundle, prepass: PrepassResult) -> dict[str, Any]:
    patch_text = context.patch_text or ""
    changed_line_count = _count_changed_lines(patch_text)
    if patch_text and len(patch_text.encode("utf-8")) <= 8 * 1024 and changed_line_count <= 200:
        return {
            "mode": "full_diff",
            "changed_line_count": changed_line_count,
            "patch_text": patch_text,
        }
    return {
        "mode": "excerpts",
        "changed_line_count": changed_line_count,
        "excerpts": [
            {
                "file": excerpt.file_path,
                "hunk_header": excerpt.hunk_header,
                "start_line": excerpt.start_line,
                "end_line": excerpt.end_line,
                "reason": excerpt.reason,
                "text": excerpt.text,
            }
            for excerpt in prepass.diff_excerpts[:30]
        ],
    }


def _count_changed_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if line.startswith(("+++", "---", "@@")):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def build_synthesis_prompt(*, fact_sheet: ReviewFactSheet, fail_threshold: str) -> str:
    """Build synthesis prompt from deterministic fact-sheet output."""
    # Build compact view-context summary
    vc = fact_sheet.view_contexts
    view_summary = {
        "baseline_materialized": vc.baseline_materialized,
        "head_materialized": vc.head_materialized,
        "merge_preview_materialized": vc.merge_preview_materialized,
    }
    if vc.warnings:
        view_summary["view_warnings"] = vc.warnings[:10]

    # Highlight merge delta signals
    merge_signals: list[dict[str, Any]] = []
    for sf in fact_sheet.symbol_facts:
        if sf.merge_preview_reference_count or sf.merge_preview_call_edge_count:
            merge_ref_delta = sf.merge_preview_reference_count - sf.head_reference_count
            merge_edge_delta = sf.merge_preview_call_edge_count - sf.head_call_edge_count
            if merge_ref_delta != 0 or merge_edge_delta != 0:
                merge_signals.append({
                    "symbol": sf.symbol,
                    "head_refs": sf.head_reference_count,
                    "merge_refs": sf.merge_preview_reference_count,
                    "merge_ref_delta_vs_head": merge_ref_delta,
                    "head_edges": sf.head_call_edge_count,
                    "merge_edges": sf.merge_preview_call_edge_count,
                    "merge_edge_delta_vs_head": merge_edge_delta,
                    "baseline_ref_delta": sf.reference_delta_vs_baseline,
                    "baseline_edge_delta": sf.call_edge_delta_vs_baseline,
                })

    payload: dict[str, Any] = {
        "fail_threshold": fail_threshold,
        "view_contexts": view_summary,
        "fact_sheet": fact_sheet.model_dump(mode="json"),
        "symbol_confidence": [
            {
                "symbol": sf.symbol,
                "verified_ratio": sf.confidence.verified_ratio,
                "total_candidates": sf.confidence.total_candidates,
                "retrieval_status": sf.confidence.retrieval_status,
                "candidate_provenance": sf.candidate_provenance,
                "warnings": sf.warnings[:6],
            }
            for sf in fact_sheet.symbol_facts[:40]
        ],
    }

    sections = [
        "Generate final review findings and summary from this fact sheet.",
        "Use categories: hidden_side_effect, cross_repo_breakage, architecture_risk, confidence_gap.",
        "Treat empty candidate retrieval, failed retrieval, and macro-only evidence as confidence gaps unless direct changed-line evidence is strong and local.",
        "Use confidence=0.0 only for purely speculative concerns. Evidence anchored in changed local code should usually be >= 0.3 even under partial semantic coverage.",
    ]

    if merge_signals:
        payload["merge_delta_signals"] = merge_signals[:20]
        sections.append(
            "MERGE DELTA SIGNALS are present. Symbols listed in merge_delta_signals "
            "show different reference/edge counts between head and merge-preview. "
            "These may indicate semantic merge conflicts. Evaluate carefully."
        )

    anchor_kinds = set()
    for a in fact_sheet.suspicious_anchors:
        anchor_kinds.add(a.kind)
    if anchor_kinds:
        sections.append(
            f"SUSPICIOUS ANCHORS detected: {', '.join(sorted(anchor_kinds))}. "
            "Check these for hidden side-effects even when coverage is high."
        )

    sections.append("Output SynthesisDraft only.")
    sections.append(json.dumps(payload, indent=2, ensure_ascii=True))
    return "\n\n".join(sections)


def build_exploration_prompt(
    *,
    fact_sheet: ReviewFactSheet,
    prepass: PrepassResult,
    remaining_calls: int,
    remaining_rounds: int,
) -> str:
    """Build exploration prompt for agent-driven follow-up investigation."""
    # Identify gaps: symbols with low confidence or macro_fallback_used
    low_confidence_symbols: list[str] = []
    macro_fallback_symbols: list[str] = []
    high_delta_symbols: list[str] = []

    for sf in fact_sheet.symbol_facts:
        if sf.confidence.verified_ratio < 0.4 or sf.confidence.retrieval_status in {"empty", "failed"}:
            low_confidence_symbols.append(sf.symbol)
        if "macro_fallback_used" in sf.warnings:
            macro_fallback_symbols.append(sf.symbol)
        if abs(sf.reference_delta_vs_baseline) >= 3 or abs(sf.call_edge_delta_vs_baseline) >= 3:
            high_delta_symbols.append(sf.symbol)

    # Anchors not yet covered by symbol_facts
    covered_symbols = {sf.symbol for sf in fact_sheet.symbol_facts}
    uncovered_anchors: list[dict[str, Any]] = []
    for anchor in prepass.suspicious_anchors[:50]:
        # Check if any covered symbol appears in the anchor snippet
        covered = any(sym in anchor.snippet for sym in covered_symbols) if covered_symbols else False
        if not covered:
            uncovered_anchors.append({
                "kind": anchor.kind,
                "file": anchor.file_path,
                "line": anchor.line,
                "snippet": anchor.snippet[:100],
            })

    payload: dict[str, Any] = {
        "remaining_tool_calls": remaining_calls,
        "remaining_rounds": remaining_rounds,
        "low_confidence_symbols": low_confidence_symbols[:10],
        "macro_fallback_symbols": macro_fallback_symbols[:10],
        "high_delta_symbols": high_delta_symbols[:10],
        "uncovered_suspicious_anchors": uncovered_anchors[:15],
        "seed_symbols_not_investigated": [
            s.symbol for s in prepass.seed_symbols
            if s.symbol not in covered_symbols
        ][:15],
    }

    sections = [
        "Perform targeted follow-up investigation to fill evidence gaps.",
        f"You have {remaining_calls} tool calls and {remaining_rounds} rounds remaining.",
        "Focus on: (1) symbols that fell back to macro investigation, "
        "(2) low-confidence or empty-retrieval symbols, "
        "(3) high-delta symbols with large reference/edge changes, "
        "(4) uncovered suspicious anchors (concurrency, lifetime, exception, ABI).",
        "Do NOT re-investigate symbols that already have good evidence.",
        "Return an ExplorationResult only.",
    ]

    sections.append(json.dumps(payload, indent=2, ensure_ascii=True))
    return "\n\n".join(sections)
