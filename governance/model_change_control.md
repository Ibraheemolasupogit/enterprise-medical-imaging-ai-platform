# Model Change Control

No models are implemented in Milestone 1.

Future model changes must record:

- Dataset version.
- Training configuration.
- Model version.
- Evaluation results.
- Quality-gate outcome.
- Approval status.
- Reviewer or owner.
- Deployment status.

Candidate models must not be treated as approved merely because training completed.

Milestone 11 API configuration records local checkpoint, calibration, and threshold-policy paths and
readiness checksums. Changing any model artefact served by the API requires corresponding evidence
review, checksum traceability, quality-gate review, and explicit documentation. API readiness is not
model approval.

Milestone 12 reviewer UI configuration records the API endpoint and local review-export policy but
does not approve model artefacts. UI changes that alter label, quality, abstention, or reviewer
decision presentation require governance review because display changes can affect reviewer
interpretation.
