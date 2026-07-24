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
