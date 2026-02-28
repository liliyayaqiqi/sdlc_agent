"""Application pipeline entry point."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReviewPipeline:
    """Thin pipeline container used by the orchestrator facade."""

    orchestrator: object

    def run(self, request):
        return self.orchestrator._run_request(request)
