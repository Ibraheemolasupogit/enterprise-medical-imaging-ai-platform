# Human Oversight

Human review is mandatory for any future output.

Future interfaces must present model and quality information as technical review aids, not as final clinical determinations. Reviewers must be able to reject, correct, or mark outputs as indeterminate.

Milestone 6 registration reports and review arrays require human engineering review before any
downstream research use. Reviewers must treat optimiser convergence and metric improvement as
technical signals only; they do not prove anatomical correctness or diagnostic suitability.

Milestone 7 localisation reports and overlay arrays require human engineering review before any
downstream research use. Reviewers must treat atlas centres, confidence heuristics, and synthetic
metrics as technical signals only; they do not prove adrenal localisation accuracy or diagnostic
suitability.

Milestone 8 segmentation reports, checkpoints, and inference masks require human engineering review
before any downstream research use. Reviewers must treat synthetic Dice, recall, thresholded masks,
and post-processing outputs as technical evidence only; empty predictions, false negatives, false
positives, and failed quality gates must not be ignored.

Milestone 9 classification reports, checkpoints, calibration artefacts, threshold policies, and
inference outputs require human engineering review before any downstream research use. Reviewers
must treat synthetic AUROC, AUPRC, recall, calibration diagnostics, threshold policy, and
`indeterminate` abstention as technical evidence only. False negatives, false positives,
miscalibration, class imbalance, broad or narrow abstention intervals, and failed quality gates must
block downstream research use until reviewed.

Milestone 10 longitudinal reports and review arrays require human engineering review before any
downstream research use. Reviewers must treat `new`, `increased`, `stable`, `reduced`, `resolved`,
and `indeterminate` as synthetic engineering labels only. They are not progression, treatment
response, RECIST, diagnosis, or clinical decision support. Ambiguous matches, failed upstream
quality, spacing concerns, small denominators, and classification abstention must not be bypassed.

Milestone 11 API responses require the same human engineering review as the underlying artefacts.
API consumers must treat summaries, probabilities, masks, change labels, checksums, and readiness
findings as technical evidence only. The API must not be presented as autonomous review, triage,
diagnosis, or patient-management support.

Milestone 12 reviewer UI decisions remain separate from model output. Reviewers must keep
abstention, failed quality gates, readiness failures, and evidence-integrity failures visible in
their engineering decision. A reviewer decision is not clinical approval.
## Containerised Review

Milestone 13 does not change the human-review boundary. The containerised reviewer UI remains an
engineering review interface, the model output remains separate from the reviewer decision, and no
container smoke-test result may be interpreted as clinical approval.

## Registry And Monitoring Oversight

Milestone 14 requires human governance approval for any model version to enter the `approved`
lifecycle state. Candidate registration, passing synthetic metrics, clean monitoring evidence, or
container release evidence must not be treated as deployment approval.

Monitoring `WARN` and `ALERT` results require human investigation and documented change-control
review. Reviewers must treat drift output as synthetic engineering evidence only, not as diagnosis,
triage, clinical performance deterioration, or medical-device safety evidence.

Milestone 15 Kubernetes evidence also requires human interpretation. Static policy checks and local
smoke evidence can support a deployment review, but they do not authorize clinical use, public
inference exposure, model promotion, or production release.

Milestone 16 AWS evidence also requires human interpretation. Terraform validation, policy checks,
scanner output, CloudWatch alarm definitions, and CloudTrail architecture can support infrastructure
review, but they do not approve clinical use, deploy AWS resources, promote models, publish images,
or replace reviewer accountability.
