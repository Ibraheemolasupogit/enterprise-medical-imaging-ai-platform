# Monitoring Plan

Milestone 14 implements deterministic local synthetic monitoring evidence. It is not a production
monitoring service and does not connect to clinical systems.

Implemented synthetic monitoring covers:

- Operational health.
- Request volume.
- Latency.
- Inference failures.
- Preprocessing failures.
- Image-quality failures.
- Segmentation output volume and confidence.
- Classification probability, confidence, and abstention.
- Calibration drift when synthetic labels are available.
- Audit completeness.

Current windows are compared with a stored synthetic baseline using simple deterministic threshold
checks. Results are `PASS`, `WARN`, or `ALERT`; `WARN` and `ALERT` require human investigation,
rollback/change-control review where appropriate, and explicit governance disposition. They must not
be described as clinical performance deterioration.

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
## Container Release Signals

Local release assurance should record container health failures, readiness failures, API/UI version
mismatches, scanner unavailable statuses, high or critical vulnerability findings, missing SBOMs,
release manifest checksum mismatches, sensitive-log redaction failures, and unexpected writable
filesystem or root-user findings. Release evidence must not log secrets, raw arrays, model weights,
reviewer notes with identifiers, or sensitive payloads.

## Kubernetes Deployment Signals

Milestone 15 Kubernetes evidence should record Helm lint availability, values-schema validation,
rendered-manifest checksums, static policy checks, workload inventory, security-control summaries,
runtime smoke availability, cleanup status, and overall deployment-evidence status. Missing local
runtime evidence is `UNAVAILABLE` or `INCOMPLETE` and must not be treated as a pass.

## AWS Infrastructure Monitoring Boundaries

Milestone 16 maps engineering signals to a target AWS observability design using CloudWatch log
groups, metrics and alarms for readiness failures, API error rate, pod restarts, latency, and node
pressure. CloudWatch evidence is infrastructure and application-operations evidence only; it must
not be described as clinical-performance monitoring, diagnostic safety surveillance, or medical
device post-market monitoring. CloudTrail records AWS control-plane activity and remains separate
from application audit evidence, reviewer actions, registry lifecycle events, and monitoring drift
evidence.

## Operations Observability Requirements

Milestone 17 records production-style engineering observability controls while remaining a local
demonstrator. Required signals include:

- API liveness, startup and readiness status, including degraded readiness when model artefacts or
  quality gates require investigation.
- Request volume, latency, status codes, API error rates, inference failures, preprocessing
  failures, image-quality failures, abstentions and degraded responses.
- Model/checkpoint integrity failures, configuration checksum mismatches, path-security violations
  and filesystem containment blocks.
- Reviewer UI dependency failures, retries, circuit-breaker openings, upload validation failures and
  export failures.
- Kubernetes readiness, probe, resource, NetworkPolicy and security-context evidence when local
  runtime checks are executed.
- AWS target-state mappings for CloudWatch logs, metric filters, alarms and dashboards without
  deploying AWS resources.

Metrics must use bounded engineering labels only. Structured logs and evidence must not include raw
arrays, DICOM pixel data, model weights, sensitive payloads, reviewer notes, credentials, direct
identifiers, unredacted local secret paths, or free-text PHI. `WARN` and `ALERT` statuses require
human investigation and documented disposition; they must not be described as clinical performance
deterioration or diagnostic safety surveillance.
