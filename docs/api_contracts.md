# API Contracts

All endpoints return JSON and include the standard research-only disclaimer in successful prediction
or analysis responses.

## Health

- `GET /health`: returns service liveness, request ID, and disclaimer.
- `GET /version`: returns package and API versions.
- `GET /ready`: returns readiness quality findings or `API-NOTREADY-503`.

## Segmentation

`POST /v1/segmentation/predict`

Request fields:

- `input_path` or `array`: exactly one input.
- `spacing_mm`: three positive values in `[z, y, x]` order.
- `threshold`: optional and blocked unless policy allows threshold override.
- `persist_output`: optional local ignored-output persistence.

Response fields include probability summary, predicted voxel count, predicted volume, quality
findings, checkpoint checksum, optional non-absolute output paths, request ID, duration, and
disclaimer.

## Classification

`POST /v1/classification/predict`

Request fields:

- `input_path` or `array`: exactly one input.

Response fields include raw probability, calibrated probability, threshold, engineering label,
abstention state, artefact checksums, request ID, duration, and disclaimer.

## Longitudinal

`POST /v1/longitudinal/analyse`

Request fields:

- `previous_mask_path` and `current_mask_path`, or `previous_array` and `current_array`.
- Previous and current spacing in `[z, y, x]` order.
- Research-only `case_id`, `research_subject_id`, anatomical `side`, and timepoint labels.
- Upstream quality and abstention statuses.

Response fields include measurements, matching summary, change metrics, engineering labels, upstream
quality propagation, quality findings, optional non-absolute evidence path, request ID, duration, and
disclaimer.

## Review

- `GET /v1/review/segmentation/{experiment_id}`
- `GET /v1/review/classification/{experiment_id}`
- `GET /v1/review/longitudinal/{analysis_id}`

Review endpoints validate local evidence checksums before returning a bounded summary. They do not
return local absolute paths, arrays, model weights, or raw unrestricted manifests.

Milestone 12 reviewer UI consumes these contracts through a typed local API client. The UI preserves
API error codes, quality statuses, abstention state, and evidence-integrity failures for display. It
does not call modelling code directly.

## Error Codes

- `API-REQ-400`: malformed request metadata.
- `API-AUTH-403`: path, symlink, policy, or root-allowlist rejection.
- `API-NOTFOUND-404`: configured local input or evidence is missing.
- `API-SIZE-413`: request body or array exceeds configured limits.
- `API-VALID-422`: contract, spacing, array, shape, or domain validation failure.
- `API-INTEGRITY-409`: evidence checksum or model/checkpoint compatibility failure.
- `API-QUALITY-409`: reserved for quality-gate conflicts.
- `API-INTERNAL-500`: sanitized unexpected internal error.
- `API-NOTREADY-503`: required local artefacts are not configured or not ready.
