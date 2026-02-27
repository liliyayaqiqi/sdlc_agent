# CXXtract2 Review Agent

Standalone, CI-first AI code review agent for C++ pull requests.

## What it does

1. Takes a git patch as input.
2. Uses a PydanticAI agent as the controller (ReAct via tool-calling), not procedural loops.
3. Calls CXXtract2 macro-tools directly:
   - `/agent/investigate-symbol`
   - `/agent/search-analyze-recent-commits`
   - `/agent/read-file-context`
4. Produces:
   - `review_report.md`
   - `review_report.json`
5. Returns a CI-friendly exit code based on configured severity threshold.

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

## Environment

See `.env.example` for defaults.
