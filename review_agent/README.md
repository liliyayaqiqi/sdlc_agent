# CXXtract2 Review Agent

Standalone, CI-first AI code review agent for C++ pull requests.

## What it does

1. Takes a git patch as input.
2. Uses CXXtract2 semantic APIs to analyze definitions, references, call edges, and freshness confidence.
3. Produces:
   - `review_report.md`
   - `review_report.json`
4. Returns a CI-friendly exit code based on configured severity threshold.

## Quick start

```powershell
cd sdlc_agent/review_agent
pip install -e .[dev]
review-agent run `
  --workspace-id ws_main `
  --patch-file F:/path/to/pr.patch `
  --cxxtract-base-url http://127.0.0.1:8000 `
  --fail-on high `
  --out-dir ./out
```

## Environment

See `.env.example` for defaults.

