"""Human-review decision helpers."""

from __future__ import annotations

import hashlib

from medical_imaging_platform import __version__
from medical_imaging_platform.reviewer_ui.models import (
    EvidenceType,
    ReviewerDecision,
    ReviewerDecisionValue,
    create_timestamp,
)


def create_review_decision(
    *,
    request_id: str,
    evidence_type: EvidenceType,
    evidence_id: str,
    model_engineering_label: str,
    quality_status: str,
    reviewer_decision: ReviewerDecisionValue,
    review_notes: str = "",
) -> ReviewerDecision:
    seed = "|".join(
        [
            request_id,
            evidence_type,
            evidence_id,
            model_engineering_label,
            quality_status,
            reviewer_decision,
        ]
    )
    review_id = f"review-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"
    return ReviewerDecision(
        review_id=review_id,
        request_id=request_id,
        evidence_type=evidence_type,
        evidence_id=evidence_id,
        model_engineering_label=model_engineering_label,
        quality_status=quality_status,
        reviewer_decision=reviewer_decision,
        review_notes=review_notes,
        review_timestamp=create_timestamp(),
        application_version=__version__,
    )
