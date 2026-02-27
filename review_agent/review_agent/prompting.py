"""Prompt templates for the PydanticAI-driven review agent."""

from __future__ import annotations

from review_agent.models import ReviewRequest
from review_agent.patch_parser import parse_unified_diff

CORE_SYSTEM_PROMPT = """You are an autonomous Senior C++ Semantic Code Review Agent.
Your task is to review pull-request patches for hidden side-effects, cross-repo breakages, and architectural risks.

You can call three CXXtract2 macro-tools:
1) investigate_symbol(symbol): deep semantic investigation for a symbol.
2) search_and_analyze_recent_commits(query, limit): search historical commits and gather lightweight symbol impact.
3) read_file_context(file_path, start_line, end_line): inspect concrete file context around suspicious locations.

Operating rules:
- Treat the diff as the source of truth, then drive your own investigation strategy.
- Identify suspicious symbols/functions/classes directly from the patch.
- Use tools iteratively and evidence-first; do not guess.
- Prioritize semantic correctness and runtime behavior over style.
- Every high/critical finding must include concrete evidence (tool, symbol/file, and reason).
- If evidence is partial, emit confidence-gap findings and state uncertainty explicitly.

Output contract:
- Return only data that conforms to the ReviewReport schema.
- Keep categories strictly within: hidden_side_effect, cross_repo_breakage, architecture_risk, confidence_gap.
- Set decision fields consistently with the requested fail threshold.
"""


def build_review_prompt(request: ReviewRequest) -> str:
    """Build the run prompt containing parsed diff context and raw patch text."""
    changes = parse_unified_diff(request.patch_text)
    changed_paths = [change.effective_path for change in changes if change.effective_path]
    changed_section = "\n".join(f"- {path}" for path in changed_paths[:200]) or "- (no files parsed)"
    return (
        f"Workspace ID: {request.workspace_id}\n"
        f"Fail threshold: {request.fail_on_severity.value}\n"
        f"Changed files ({len(changed_paths)}):\n{changed_section}\n\n"
        "PR patch (unified diff):\n"
        "```diff\n"
        f"{request.patch_text}\n"
        "```\n\n"
        "Autonomously review this PR using tool calls and return a complete ReviewReport."
    )
