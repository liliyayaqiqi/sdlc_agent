# CXXtract2 Review Agent

Standalone, CI-first AI code review agent for C++ pull requests.

Supported production provider in v1: `openai:*`

Deterministic test provider: `fixture:*`

## What it does

1. Accepts either a legacy patch input or a PR/MR context bundle.
2. Runs deterministic pre-pass extraction (seed symbols, suspicious anchors,
   changed methods, call-site deltas) before any LLM reasoning.
3. Uses three LLM stages on top of deterministic evidence:
   - **Planner** -- produces a bounded review plan from pre-pass signals
   - **Exploration** -- agent-driven follow-up with 6 explore tools
   - **Synthesis** -- produces findings from the fact sheet
4. Calls CXXtract2 `/explore/*` primitives for bounded evidence collection:
   - `/explore/rg-search`
   - `/explore/list-candidates`
   - `/explore/classify-freshness`
   - `/explore/parse-file`
   - `/explore/fetch-symbols`
   - `/explore/fetch-references`
   - `/explore/fetch-call-edges`
   - `/explore/read-file`
   - `/explore/get-confidence`
5. Falls back to macro tools when explore primitives yield no results:
   - `/agent/investigate-symbol`
6. Performs merge-aware review when baseline/head/merge-preview contexts
   are available.
7. Runs semantic test impact analysis using call-edge traversal.
8. Enforces budget knobs (max_symbols, max_symbol_slots,
   max_total_tool_calls, etc.) across all stages.
9. Produces:
   - `review_report.md`
   - `review_report.json`
10. Returns a CI-friendly exit code based on configured severity threshold.

## Quick start

```powershell
cd sdlc_agent/review_agent
py -m pip install -e .[dev]
review-agent run `
  --workspace-id ws_main `
  --patch-file F:/path/to/pr.patch `
  --llm-model openai:gpt-4o `
  --cxxtract-base-url http://127.0.0.1:8000 `
  --fail-on high `
  --out-dir ./out
```

Context-driven mode:

```powershell
review-agent run `
  --workspace-id ws_main `
  --context-file F:/path/to/review_context.json `
  --llm-model openai:gpt-4o `
  --out-dir ./out
```

Patch-only reviews should provide a stable cache fingerprint if SHAs are not available:

```powershell
review-agent run `
  --workspace-id ws_main `
  --patch-file F:/path/to/pr.patch `
  --workspace-fingerprint ws_main:2026-02-28:snapshot `
  --llm-model fixture:default `
  --out-dir ./out
```

## Environment

See `.env.example` for defaults.
