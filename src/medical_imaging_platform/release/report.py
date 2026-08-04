"""Markdown report rendering for release evidence."""

from __future__ import annotations

from medical_imaging_platform.release.models import ReleaseManifest


def release_report_markdown(manifest: ReleaseManifest) -> str:
    """Render a concise local release evidence report."""
    lines = [
        f"# Local Release Evidence {manifest.release_id}",
        "",
        f"Overall status: `{manifest.release_status}`",
        "",
        manifest.disclaimer,
        "",
        f"- Git revision: `{manifest.git_revision}`",
        f"- Dirty tree: `{manifest.git_dirty}`",
        f"- Build timestamp: `{manifest.build_timestamp}`",
        f"- Python: `{manifest.python_version}`",
        f"- API image: `{manifest.images['api']['name']}:{manifest.images['api']['tag']}`",
        f"- API image size: `{manifest.images['api']['size_bytes']}` bytes",
        "- Reviewer UI image: "
        f"`{manifest.images['reviewer_ui']['name']}:{manifest.images['reviewer_ui']['tag']}`",
        f"- Reviewer UI image size: `{manifest.images['reviewer_ui']['size_bytes']}` bytes",
        f"- Smoke status: `{manifest.smoke_test_results.status}`",
        "",
        "## Container Dependencies",
        "",
        f"- Requirements file: `{manifest.dependency_strategy['requirements_file']}`",
        f"- PyTorch wheel: `{manifest.dependency_strategy['pytorch_version']}`",
        f"- PyTorch index: `{manifest.dependency_strategy['pytorch_index_url']}`",
        f"- MONAI pin: `{manifest.dependency_strategy['monai_version']}`",
        f"- Minimum safe MONAI: `{manifest.dependency_strategy['minimum_safe_monai_version']}`",
        f"- Policy: {manifest.dependency_strategy['cuda_nvidia_policy']}",
        "",
        "## Scan Results",
        "",
    ]
    for scan_result in manifest.scan_results:
        lines.append(f"- `{scan_result.tool}`: `{scan_result.status}`")
    lines.extend(["", "## Security Checks", ""])
    for check_result in manifest.test_results:
        lines.append(
            f"- `{check_result.check_id}`: `{check_result.status}` - {check_result.message}"
        )
    lines.append("")
    return "\n".join(lines)
