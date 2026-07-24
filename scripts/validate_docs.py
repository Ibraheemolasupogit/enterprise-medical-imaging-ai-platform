"""Lightweight repository documentation validation."""

from __future__ import annotations

from pathlib import Path

REQUIRED_MARKDOWN_FILES = (
    Path("README.md"),
    Path("NOTICE.md"),
    Path("CONTRIBUTING.md"),
    Path("SECURITY.md"),
    Path("docs/architecture.md"),
    Path("docs/clinical_workflow.md"),
    Path("docs/data_flow.md"),
    Path("docs/roadmap.md"),
    Path("docs/security.md"),
    Path("docs/attribution.md"),
    Path("docs/limitations.md"),
    Path("governance/intended_use.md"),
    Path("governance/excluded_use.md"),
    Path("governance/human_oversight.md"),
    Path("governance/clinical_safety_approach.md"),
    Path("governance/data_protection_design.md"),
    Path("governance/model_change_control.md"),
    Path("governance/monitoring_plan.md"),
    Path("governance/limitations.md"),
)

DISCLAIMER = (
    "This platform is a research and engineering demonstrator. Outputs are intended "
    "for technical evaluation and human review only and must not be used for clinical "
    "diagnosis or patient-management decisions."
)


def validate_markdown_file(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing required documentation file: {path}"]

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        errors.append(f"Documentation file is empty: {path}")
    if "\t" in content:
        errors.append(f"Documentation file contains tab characters: {path}")
    if not content.endswith("\n"):
        errors.append(f"Documentation file must end with a newline: {path}")
    if path.suffix == ".md" and not content.lstrip().startswith("#"):
        errors.append(f"Markdown file must start with a heading: {path}")
    return errors


def main() -> int:
    errors: list[str] = []
    for path in REQUIRED_MARKDOWN_FILES:
        errors.extend(validate_markdown_file(path))

    readme = Path("README.md").read_text(encoding="utf-8")
    if DISCLAIMER not in readme:
        errors.append("README.md is missing the required research-only disclaimer.")

    if errors:
        for error in errors:
            print(error)
        return 1

    print(f"Validated {len(REQUIRED_MARKDOWN_FILES)} documentation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
