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
