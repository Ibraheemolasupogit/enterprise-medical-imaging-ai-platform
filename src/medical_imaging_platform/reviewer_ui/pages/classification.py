"""Classification review page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.formatting import format_classification_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import array_payload, load_uploaded_npy
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Synthetic Classification Review")
    st_module.caption("Binary synthetic lesion-presence classification only.")
    uploaded = st_module.file_uploader("Upload bounded ROI .npy array", type=["npy"])
    if st_module.button("Submit classification request") and uploaded is not None:
        try:
            array = load_uploaded_npy(uploaded.name, uploaded.getvalue(), config)
            response = client.predict_classification({"array": array_payload(array)})
            remember_response(st_module.session_state, {"page": "classification"}, response)
            formatted = format_classification_response(response)
            if formatted["abstained"]:
                st_module.warning("Indeterminate classification: abstention is active.")
            st_module.json(formatted)
        except (ReviewerAPIError, ValueError) as exc:
            st_module.error(str(exc))
