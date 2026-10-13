"""Governance and human-review page."""

from __future__ import annotations

from typing import Any

from medical_imaging_platform.evidence import ArtifactBinding
from medical_imaging_platform.reviewer_ui.export import export_review_session
from medical_imaging_platform.reviewer_ui.models import (
    ReviewerDecision,
    ReviewerUIConfig,
)
from medical_imaging_platform.reviewer_ui.review import create_review_decision
from medical_imaging_platform.reviewer_ui.state import reset_state


def render(st_module: Any, config: ReviewerUIConfig) -> None:
    st_module.header("Governance And Human Review")
    st_module.write(
        "Reviewer decisions are engineering workflow artefacts and do not overwrite model output."
    )
    evidence_type = st_module.selectbox(
        "Evidence type",
        ["segmentation", "classification", "detection", "longitudinal", "embeddings"],
    )
    summary = st_module.session_state.get("last_response") or {}
    artifact = (
        ArtifactBinding.model_validate(summary["artifact"]) if summary.get("artifact") else None
    )
    if artifact is not None:
        st_module.subheader("Exact reviewed artifact")
        st_module.json(artifact.model_dump(mode="json"))
        evidence_type = {"segmentation_mask": "segmentation", "embedding_index": "embeddings"}.get(
            artifact.artifact_type, evidence_type
        )
        st_module.caption(f"Bound evidence type: {evidence_type}")
    else:
        st_module.info(
            "This legacy evidence has no exact artifact binding; "
            "the decision remains an unbound engineering review."
        )
    evidence_id = st_module.text_input("Evidence ID", value="unresolved-evidence")
    request_id = st_module.text_input("Request ID", value="unresolved-request")
    model_label = st_module.text_input("Model engineering label", value="indeterminate")
    quality_status = st_module.text_input("Quality status", value="UNKNOWN")
    decision = st_module.selectbox(
        "Reviewer decision",
        [
            "accepted_for_engineering_review",
            "needs_secondary_review",
            "rejected_due_to_quality",
            "insufficient_information",
        ],
    )
    notes = st_module.text_area("Optional review notes", max_chars=500)
    if st_module.button("Create review decision"):
        review = create_review_decision(
            request_id=str(summary.get("request_id", request_id)),
            evidence_type=evidence_type,
            artifact=artifact,
            evidence_id=evidence_id,
            model_engineering_label=model_label,
            quality_status=quality_status,
            reviewer_decision=decision,
            review_notes=notes,
        )
        st_module.session_state["review_decision"] = review.model_dump(mode="json")
        st_module.json(review.model_dump(mode="json"))
    if config.allow_evidence_export and st_module.button("Export review decision"):
        stored = st_module.session_state.get("review_decision")
        if isinstance(stored, dict):
            review = ReviewerDecision.model_validate(stored)
            result = export_review_session(
                decision=review,
                evidence_summary=st_module.session_state.get("last_response") or {},
                output_root=config.review_output_directory,
            )
            st_module.session_state["export_status"] = result.model_dump(mode="json")
            st_module.success("Review session exported.")
            st_module.json(result.model_dump(mode="json"))
    if st_module.button("Clear reviewer session"):
        reset_state(st_module.session_state)
        st_module.info("Reviewer session state cleared.")
