# CXXtract2 Review Agent

Standalone, CI-first AI code review agent for C++ pull requests.

## What it does

1. Accepts either a legacy patch input or a PR/MR context bundle.
2. Runs deterministic pre-pass extraction before LLM synthesis.
3. Uses two LLM stages (planning + synthesis) on top of deterministic evidence.
4. Calls CXXtract2 macro tools and `/explore/*` primitives for bounded evidence collection:
   - `/agent/investigate-symbol`
   - `/agent/search-analyze-recent-commits`
   - `/agent/read-file-context`
   - `/explore/rg-search`
   - `/explore/list-candidates`
   - `/explore/classify-freshness`
   - `/explore/parse-file`
   - `/explore/fetch-symbols`
   - `/explore/fetch-references`
   - `/explore/fetch-call-edges`
5. Produces:
   - `review_report.md`
   - `review_report.json`
6. Returns a CI-friendly exit code based on configured severity threshold.

## Quick start

```powershell
cd sdlc_agent/review_agent
pip install -e .[dev]
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

## Environment

See `.env.example` for defaults.
