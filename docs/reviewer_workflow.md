# Reviewer workflow

The implemented Streamlit UI calls the governed API for all inference. It presents segmentation,
classification, detection, longitudinal, embeddings/retrieval, evidence and governance pages.
[Local startup](local_startup.md), [API contracts](api_contracts.md).

1. Check Overview liveness/readiness and capability availability.
2. Submit a prepared input or select an explicitly configured embedding index/query and integer K.
3. Inspect warnings, provenance, hashes, model output and research limitations. Segmentation displays
   a source axial slice and predicted-mask overlay at opacity 0.35; changing slice requires resubmission.
   Detection displays ranked boxes/scores; retrieval displays ranked neighbours/similarity scores.
4. Inspect validated experiment evidence when needed.
5. Record a separate engineering decision and optional bounded reason.
6. Export the decision and bounded evidence summary without overwriting predictions.

Decisions are `accepted_for_engineering_review`, `needs_secondary_review`, `rejected_due_to_quality`
or `insufficient_information`. Existing request/evidence IDs, timestamps and application version
remain recorded. Available artifact bindings add exact type/reference/checksum, run/source/model
identity. Export rejects a binding different from the decision; legacy unbound evidence is labelled.

Exports contain `review_decision.json`, `reviewed_evidence_summary.json` and `review_report.md`, with
atomic writes, overwrite protection and hashes. Preview pixels are excluded. Transient API predictions
are not a permanent audit store; persist model output separately when later inspection is required.
Do not put identifiers, credentials or clinical claims in notes. No mask editing or prompt correction
is implemented. Embedding-space similarity is not diagnostic similarity.
