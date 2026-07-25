# Reviewer Workflow

Milestone 12 separates model-generated engineering output from human reviewer decisions.

Reviewer workflow:

1. Confirm API health and readiness on the Overview page.
2. Submit synthetic segmentation, classification, or longitudinal inputs through the governed API.
3. Review model-generated engineering output, quality findings, provenance, checksums, abstention
   state, and disclaimers.
4. Inspect existing evidence through read-only API review endpoints when needed.
5. Record a reviewer decision separately from the model output.
6. Optionally export the review session to a local ignored directory.

Supported reviewer decisions:

- `accepted_for_engineering_review`
- `needs_secondary_review`
- `rejected_due_to_quality`
- `insufficient_information`

Reviewer decisions capture `review_id`, `request_id`, `evidence_type`, `evidence_id`,
`model_engineering_label`, `quality_status`, `reviewer_decision`, optional bounded notes,
`review_timestamp`, and `application_version`.

Review notes must not include personal identifiers, patient identifiers, credentials, or clinical
claims. Reviewer decisions do not overwrite model output and do not represent clinical approval.

Exports are written under `reports/generated/reviewer-sessions/<review_id>/`:

- `review_decision.json`
- `reviewed_evidence_summary.json`
- `review_report.md`

Exports use stable JSON, atomic writes, overwrite protection, and SHA-256 checksums. They do not
include raw arrays, model weights, unrestricted filesystem paths, or clinical patient evidence.
