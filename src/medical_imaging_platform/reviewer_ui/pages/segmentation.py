"""Segmentation review page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.formatting import format_segmentation_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import array_payload, load_uploaded_npy
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Synthetic Segmentation Review")
    st_module.caption("Synthetic lesion segmentation only. Not diagnostic.")
    uploaded = st_module.file_uploader("Upload bounded .npy volume", type=["npy"])
    spacing_text = st_module.text_input("Spacing z,y,x mm", value="1.0,1.0,1.0")
    if st_module.button("Submit segmentation request") and uploaded is not None:
        try:
            content = uploaded.getvalue()
            array = load_uploaded_npy(uploaded.name, content, config)
            spacing = [float(item.strip()) for item in spacing_text.split(",")]
            payload = {"array": array_payload(array), "spacing_mm": spacing}
            response = client.predict_segmentation(payload)
            remember_response(st_module.session_state, {"page": "segmentation"}, response)
            st_module.json(format_segmentation_response(response))
        except (ReviewerAPIError, ValueError) as exc:
            st_module.error(str(exc))
