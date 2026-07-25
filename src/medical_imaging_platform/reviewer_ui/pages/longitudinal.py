"""Longitudinal comparison review page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.formatting import format_longitudinal_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import array_payload, load_uploaded_npy
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Longitudinal Comparison Review")
    previous = st_module.file_uploader("Previous mask .npy", type=["npy"], key="previous_mask")
    current = st_module.file_uploader("Current mask .npy", type=["npy"], key="current_mask")
    case_id = st_module.text_input("Research-safe case ID", value="case-review")
    subject_id = st_module.text_input("Research-safe subject ID", value="research-subject-review")
    side = st_module.selectbox("Side", ["left", "right"])
    previous_spacing = st_module.text_input("Previous spacing z,y,x mm", value="1.0,1.0,1.0")
    current_spacing = st_module.text_input("Current spacing z,y,x mm", value="1.0,1.0,1.0")
    if (
        st_module.button("Submit longitudinal request")
        and previous is not None
        and current is not None
    ):
        try:
            previous_array = load_uploaded_npy(previous.name, previous.getvalue(), config)
            current_array = load_uploaded_npy(current.name, current.getvalue(), config)
            payload = {
                "previous_array": array_payload(previous_array),
                "current_array": array_payload(current_array),
                "previous_spacing_mm": [
                    float(item.strip()) for item in previous_spacing.split(",")
                ],
                "current_spacing_mm": [float(item.strip()) for item in current_spacing.split(",")],
                "case_id": case_id,
                "research_subject_id": subject_id,
                "side": side,
            }
            response = client.analyse_longitudinal(payload)
            remember_response(st_module.session_state, {"page": "longitudinal"}, response)
            formatted = format_longitudinal_response(response)
            if formatted["indeterminate"]:
                st_module.warning("Indeterminate longitudinal engineering label.")
            st_module.json(formatted)
        except (ReviewerAPIError, ValueError) as exc:
            st_module.error(str(exc))
