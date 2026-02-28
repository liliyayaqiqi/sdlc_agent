"""GitLab publishing adapter."""

from __future__ import annotations

import logging

from review_agent.models import PublishResult, PublishingError, ReviewReport
from review_agent.report_renderer import render_markdown
from review_agent.tool_clients.gitlab_client import GitLabClient

logger = logging.getLogger("review_agent.adapters.gitlab")


class GitLabPublisher:
    """Publish review results to a GitLab merge request."""

    def __init__(
        self,
        *,
        base_url: str,
        private_token: str,
        project_id: str,
        mr_iid: int,
        client: GitLabClient | None = None,
    ) -> None:
        self._base_url = base_url
        self._private_token = private_token
        self._project_id = project_id
        self._mr_iid = mr_iid
        self._client = client

    def publish(self, report: ReviewReport, *, publish_inline: bool) -> PublishResult:
        owns_client = self._client is None
        client = self._client or GitLabClient(
            base_url=self._base_url,
            private_token=self._private_token,
        )
        warnings: list[str] = []
        try:
            body = render_markdown(report)
            try:
                client.post_mr_note(
                    project_id=self._project_id,
                    mr_iid=self._mr_iid,
                    body=body,
                )
            except Exception as exc:
                raise PublishingError(f"failed to publish GitLab summary note: {exc}") from exc

            inline_comments_posted = 0
            if publish_inline and report.findings:
                base_sha = ""
                head_sha = ""
                start_sha = ""
                try:
                    meta = client.get_mr_metadata(project_id=self._project_id, mr_iid=self._mr_iid)
                    diff_refs = meta.get("diff_refs", {})
                    base_sha = str(diff_refs.get("base_sha", ""))
                    head_sha = str(diff_refs.get("head_sha", ""))
                    start_sha = str(diff_refs.get("start_sha", ""))
                except Exception as exc:
                    warnings.append(f"diff_refs_fetch_failed:{exc}")

                for finding in report.findings:
                    if finding.location is None:
                        continue
                    if finding.location.side != "new" or finding.location.line <= 0:
                        warnings.append(f"finding_not_inlineable:{finding.id}")
                        continue
                    try:
                        inline_body = (
                            f"**[{finding.severity.value.upper()}]** {finding.title}\n\n"
                            f"{finding.impact}\n\n"
                            f"**Recommendation:** {finding.recommendation}"
                        )
                        client.post_mr_inline_discussion(
                            project_id=self._project_id,
                            mr_iid=self._mr_iid,
                            body=inline_body,
                            new_path=finding.location.path,
                            new_line=finding.location.line,
                            base_sha=finding.location.base_sha or base_sha,
                            head_sha=finding.location.head_sha or head_sha,
                            start_sha=finding.location.start_sha or start_sha,
                        )
                        inline_comments_posted += 1
                    except Exception as exc:
                        warnings.append(f"inline_publish_failed:{finding.id}:{exc}")

            return PublishResult(
                provider="gitlab",
                summary_posted=True,
                inline_comments_posted=inline_comments_posted,
                warnings=warnings,
            )
        finally:
            if owns_client:
                client.close()
