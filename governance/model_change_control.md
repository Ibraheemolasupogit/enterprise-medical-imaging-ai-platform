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
## Container Release Evidence

Milestone 13 records image names, dependency versions, Dockerfile checksums, configuration
checksums, scanner status and smoke-test status. Generated release evidence does not approve model
changes, publish images, or replace model governance review.

## Governed Model Registry

Milestone 14 records local registry manifests for synthetic segmentation and classification model
versions. Each record includes checkpoint checksum, model type, framework versions, config checksum,
training-data reference, evaluation metrics, thresholds, calibration metadata, lifecycle state, and
approval metadata when approved.

Allowed lifecycle states are `candidate`, `approved`, `rejected`, and `retired`. Registration
creates candidates only. Approval requires explicit human approval metadata, including reviewer,
ticket, timestamp, and rationale. The registry does not implement automatic promotion, automated
rollback, automated retraining, deployment, or clinical release.

Monitoring `WARN` or `ALERT` evidence must open an investigation, verify data and evidence
integrity, review recent model/config/container changes, and document any rollback or retirement
decision through change control. No automated action is permitted.
