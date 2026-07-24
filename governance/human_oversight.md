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
