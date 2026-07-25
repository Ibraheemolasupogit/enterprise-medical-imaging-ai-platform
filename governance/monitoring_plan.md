# Monitoring Plan

Monitoring is Planned - not yet implemented.

Future monitoring should cover:

- Operational health.
- Imaging metadata distributions.
- Data drift.
- Model performance.
- Calibration.
- Human disagreement and correction rates.
- Audit completeness.

Milestone 4 quality reports introduce technical data-quality signals that may feed future monitoring, including corrupt-file rates, private-tag presence, burned-in annotation status, and quality-control failure rates.

Milestone 5 preprocessing reports introduce additional future monitoring candidates: preprocessing
success/failure rate, quality-override use, rejected/blocked series counts, spacing fallback rates,
irregular spacing flags, clipping percentages, crop/padding frequency, checksum failures, and output
validation failures. These are engineering signals only and not clinical performance metrics.

Milestone 6 registration reports introduce future monitoring candidates: registration status,
fixed/moving role reversal attempts, optimiser stop conditions, transform magnitudes, affine scale
and shear, metric degradation, centre-of-mass distance changes, padding-fraction failures, and output
checksum failures. These remain engineering signals and are not clinical alignment performance
metrics.

Milestone 7 localisation reports introduce future monitoring candidates: localisation status,
left/right swap findings, centre-distance failures on synthetic labels, target-coverage failures,
bounding-box IoU failures, padding fraction, upstream override propagation, missing ground-truth
rates, and checksum failures. These remain engineering signals and are not clinical localisation
performance metrics.

Milestone 8 segmentation reports introduce future monitoring candidates: dataset leakage findings,
training loss finiteness, validation Dice, test recall, false-positive voxel burden, relative volume
error, checkpoint checksum failures, inference empty-mask warnings, and failed model-quality gates.
These remain synthetic engineering signals and are not clinical segmentation performance metrics.

Milestone 9 classification reports introduce future monitoring candidates: derived dataset leakage
findings, class balance by split, finite loss status, validation AUROC and AUPRC, validation recall,
Brier score, expected calibration error, calibration fallback rate, selected threshold, false
positive and false negative counts, abstention rate, checkpoint and calibration checksum failures,
and failed model-quality gates. These remain synthetic engineering signals and are not clinical
classification performance metrics.

Milestone 10 longitudinal reports introduce future monitoring candidates: pair validity failures,
temporal-order reversals, side mismatches, geometry mismatches, registration and segmentation
failure propagation, ambiguous match rate, new/resolved label counts, small-denominator cases,
indeterminate rate, checksum failures, and failed longitudinal quality gates. These remain synthetic
engineering signals and are not clinical progression or response metrics.

Milestone 11 API monitoring should cover readiness failures, request validation failures,
model/checkpoint integrity failures, quality-gate blocks, abstention or degraded responses, request
latency, error rates, and filesystem/path security violations. API monitoring must not log sensitive
payload values, raw arrays, image contents, model weights, credentials, direct identifiers, or
unredacted local paths. These remain local research engineering signals and are not clinical safety,
diagnostic performance, or deployment-approval metrics.

Milestone 12 reviewer UI monitoring should cover UI readiness, API dependency failures, upload
rejection counts, oversized upload attempts, invalid NumPy uploads, non-finite array rejections,
remote API configuration attempts, evidence-integrity errors, reviewer decision distribution,
abstention/degraded-response display frequency, export failures, export path violations, latency,
and UI error rates. Monitoring must not log raw arrays, model weights, sensitive payloads,
unredacted local paths, credentials, reviewer notes, or identifiers.
