"""Overview page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.formatting import format_health_status
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Overview")
    health = _safe_call(client.health)
    ready = _safe_call(client.ready)
    version = _safe_call(client.version)
    st_module.json(
        format_health_status(
            health if isinstance(health, dict) else None, ready if isinstance(ready, dict) else None
        )
    )
    st_module.write(f"Configured API endpoint: `{config.api_base_url}`")
    api_version = "unavailable"
    if isinstance(version, dict):
        api_version = str(version.get("api_version", "unavailable"))
    st_module.write(f"Service version: `{api_version}`")
    st_module.write(
        {
            "segmentation": config.enable_segmentation_page,
            "classification": config.enable_classification_page,
            "longitudinal": config.enable_longitudinal_page,
            "evidence": config.enable_evidence_page,
            "governance": config.enable_governance_page,
        }
    )
    if not isinstance(ready, dict) or ready.get("status") != "ready":
        st_module.error("API readiness is not passing. Do not treat the UI as operational.")
    st_module.info(
        "Human reviewers remain responsible for engineering review decisions. "
        "The UI does not provide clinical advice."
    )


def _safe_call(function: object) -> dict[str, object] | ReviewerAPIError:
    try:
        return function()  # type: ignore[operator,no-any-return]
    except ReviewerAPIError as exc:
        return exc
