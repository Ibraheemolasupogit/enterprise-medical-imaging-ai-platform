"""Evidence review page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.formatting import format_evidence_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import (
    ReviewerUISecurityError,
    validate_evidence_id,
)
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Existing Evidence Inspection")
    evidence_type = st_module.selectbox(
        "Evidence type", ["segmentation", "classification", "longitudinal"]
    )
    evidence_id = st_module.text_input("Experiment or analysis ID")
    if st_module.button("Inspect evidence"):
        try:
            safe_id = validate_evidence_id(evidence_id)
            if evidence_type == "segmentation":
                response = client.review_segmentation(safe_id)
            elif evidence_type == "classification":
                response = client.review_classification(safe_id)
            else:
                response = client.review_longitudinal(safe_id)
            remember_response(
                st_module.session_state, {"page": "evidence", "type": evidence_type}, response
            )
            st_module.json(format_evidence_response(response))
        except (ReviewerAPIError, ReviewerUISecurityError) as exc:
            st_module.error(str(exc))
    if config.maximum_review_items:
        st_module.caption(f"Maximum review items per session: {config.maximum_review_items}")
