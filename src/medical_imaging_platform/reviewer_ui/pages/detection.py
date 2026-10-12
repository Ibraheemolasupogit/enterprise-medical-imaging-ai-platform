"""Detection review page."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components.visualisation import render_slice_with_boxes
from medical_imaging_platform.reviewer_ui.formatting import format_detection_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import ReviewerUISecurityError, load_uploaded_npy
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Synthetic Detection Review")
    st_module.caption("Synthetic 2D lesion boxes only. Not diagnostic.")
    uploaded = st_module.file_uploader("Upload bounded .npy slice", type=["npy"])
    if st_module.button("Submit detection request") and uploaded is not None:
        try:
            content = uploaded.getvalue()
            try:
                array = load_uploaded_npy(uploaded.name, content, config, expected_ndim=2)
            except ReviewerUISecurityError:
                array = load_uploaded_npy(uploaded.name, content, config, expected_ndim=3)
            image = np.asarray(
                array[array.shape[0] // 2] if array.ndim == 3 else array, dtype=np.float32
            )
            payload = {"array": {"shape": list(image.shape), "values": image.ravel().tolist()}}
            response = client.predict_detection(payload)
            remember_response(st_module.session_state, {"page": "detection"}, response)
            formatted = format_detection_response(response)
            st_module.json(formatted)
            render_slice_with_boxes(
                st_module, image, cast(list[list[float]], formatted["boxes"]), "Detection overlay"
            )
        except (ReviewerAPIError, ValueError) as exc:
            st_module.error(str(exc))
