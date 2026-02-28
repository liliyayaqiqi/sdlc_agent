"""External adapter layer for backend, LLM, and VCS integrations."""

from review_agent.adapters.cxxtract import CxxtractHttpClient, CxxtractHttpError
from review_agent.adapters.gitlab import GitLabPublisher
from review_agent.adapters.llm import (
    ExplorationService,
    PlannerService,
    PydanticAiExplorationService,
    PydanticAiPlannerService,
    PydanticAiSynthesisService,
    SynthesisService,
    build_model_services,
)

__all__ = [
    "CxxtractHttpClient",
    "CxxtractHttpError",
    "ExplorationService",
    "GitLabPublisher",
    "PlannerService",
    "PydanticAiExplorationService",
    "PydanticAiPlannerService",
    "PydanticAiSynthesisService",
    "SynthesisService",
    "build_model_services",
]
