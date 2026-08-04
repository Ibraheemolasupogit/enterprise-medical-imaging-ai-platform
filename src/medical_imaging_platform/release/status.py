"""Release status aggregation."""

from __future__ import annotations

from medical_imaging_platform.release.models import (
    ReleaseCheckResult,
    SmokeTestResult,
    Status,
    ToolResult,
)

BLOCKING_STATUSES = {"FAIL", "ERROR"}
INCOMPLETE_STATUSES = {"INCOMPLETE", "UNAVAILABLE", "SKIPPED", "WARN"}


def aggregate_release_status(
    checks: list[ReleaseCheckResult],
    scan_results: list[ToolResult],
    smoke_result: SmokeTestResult,
) -> Status:
    """Aggregate mandatory local release evidence without allowing a false PASS."""
    required_scan_statuses = [
        scan.status
        for scan in scan_results
        if bool(scan.details.get("mandatory")) or scan.status in {"FAIL", "ERROR"}
    ]
    statuses: list[Status] = [
        *(check.status for check in checks),
        *required_scan_statuses,
        smoke_result.status,
    ]
    if any(status in BLOCKING_STATUSES for status in statuses):
        return "FAIL"
    if any(status in INCOMPLETE_STATUSES for status in statuses):
        return "INCOMPLETE"
    return "PASS"
