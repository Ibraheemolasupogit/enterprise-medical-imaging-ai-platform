"""Segmentation review page."""

from __future__ import annotations

from typing import Any

import numpy as np

from medical_imaging_platform.evidence import array_checksum
from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components.provenance import render_provenance
from medical_imaging_platform.reviewer_ui.components.quality import render_quality
from medical_imaging_platform.reviewer_ui.components.visualisation import (
    render_segmentation_overlay,
)
from medical_imaging_platform.reviewer_ui.formatting import format_segmentation_response
from medical_imaging_platform.reviewer_ui.models import ReviewerAPIError, ReviewerUIConfig
from medical_imaging_platform.reviewer_ui.security import array_payload, load_uploaded_npy
from medical_imaging_platform.reviewer_ui.state import remember_response


def render(st_module: Any, config: ReviewerUIConfig, client: ReviewerAPIClient) -> None:
    st_module.header("Synthetic Segmentation Review")
    st_module.caption("Synthetic lesion segmentation only. Not diagnostic.")
    uploaded = st_module.file_uploader("Upload bounded .npy volume", type=["npy"])
    spacing_text = st_module.text_input("Spacing z,y,x mm", value="1.0,1.0,1.0")
    try:
        if uploaded is None:
            st_module.info("Upload a prepared segmentation volume to review its prediction.")
            return
        array = load_uploaded_npy(uploaded.name, uploaded.getvalue(), config)
        selected = int(
            st_module.number_input(
                "Axial slice index",
                min_value=0,
                max_value=array.shape[0] - 1,
                value=array.shape[0] // 2,
                step=1,
            )
        )
        if st_module.button("Submit segmentation request"):
            spacing = [float(item.strip()) for item in spacing_text.split(",")]
            payload = {
                "array": array_payload(array),
                "spacing_mm": spacing,
                "preview_slice_index": selected,
            }
            response = client.predict_segmentation(payload)
            remember_response(st_module.session_state, {"page": "segmentation"}, response)
            st_module.session_state["segmentation_response"] = response
        response = st_module.session_state.get("segmentation_response")
        if isinstance(response, dict):
            st_module.json(format_segmentation_response(response))
            render_provenance(
                st_module,
                {
                    key: response.get(key)
                    for key in (
                        "artifact",
                        "source_checksum",
                        "checkpoint_checksum",
                        "source_provenance",
                        "evidence_reference",
                    )
                },
            )
            render_quality(st_module, response.get("quality_findings", []))
            preview = response.get("mask_preview")
            if preview is not None:
                if response.get("source_checksum") != array_checksum(array.astype(np.float32)):
                    raise ValueError("Source checksum differs from the reviewed prediction.")
                if preview["slice_index"] != selected:
                    st_module.info("Submit the selected slice to refresh its mask preview.")
                    return
                mask = np.asarray(preview["values"]).reshape(preview["shape"])
                render_segmentation_overlay(st_module, array, mask, slice_index=selected)
    except (ReviewerAPIError, ValueError) as exc:
        st_module.error(str(exc))
