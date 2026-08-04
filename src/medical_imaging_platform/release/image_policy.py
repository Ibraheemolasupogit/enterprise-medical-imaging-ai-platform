"""Image and build-context content policy checks."""

from __future__ import annotations

from pathlib import Path

from medical_imaging_platform.release.models import ReleaseCheckResult

PROHIBITED_PATTERNS = (
    ".git",
    ".env",
    "id_rsa",
    "aws_access_key_id",
    "secret_access_key",
    "data/synthetic/generated",
    "ml/datasets",
    "ml/experiments",
    "reports/generated",
    ".coverage",
    "coverage.xml",
    ".pytest_cache",
)


def inspect_dockerignore(path: Path = Path(".dockerignore")) -> list[ReleaseCheckResult]:
    """Confirm prohibited build-context paths are excluded."""
    if not path.exists():
        return [_result("IMAGE-POLICY-DOCKERIGNORE", False, ".dockerignore exists.")]
    text = path.read_text(encoding="utf-8")
    checks = []
    for pattern in PROHIBITED_PATTERNS:
        checks.append(
            _result(
                f"IMAGE-POLICY-IGNORE-{_slug(pattern)}",
                pattern in text,
                f".dockerignore excludes {pattern}.",
            )
        )
    return checks


def detect_prohibited_files(paths: list[str]) -> list[ReleaseCheckResult]:
    """Check a supplied image-file list for prohibited paths."""
    joined = "\n".join(paths).lower()
    checks = []
    for pattern in PROHIBITED_PATTERNS:
        checks.append(
            _result(
                f"IMAGE-CONTENT-NO-{_slug(pattern)}",
                pattern.lower() not in joined,
                f"Image content does not contain {pattern}.",
            )
        )
    return checks


def _slug(value: str) -> str:
    return value.replace("/", "-").replace(".", "dot").replace("_", "-").upper()


def _result(check_id: str, passed: bool, message: str) -> ReleaseCheckResult:
    return ReleaseCheckResult(
        check_id=check_id, status="PASS" if passed else "FAIL", message=message
    )
