"""Tests for final portfolio evidence and demo readiness."""

from __future__ import annotations

import json
from pathlib import Path

from medical_imaging_platform.cli import main
from medical_imaging_platform.portfolio import evidence as portfolio
from medical_imaging_platform.portfolio.models import PortfolioCheck


def _redirect_portfolio_evidence(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(portfolio, "PORTFOLIO_EVIDENCE_DIR", tmp_path)
    for name, value in vars(portfolio).copy().items():
        if name.endswith("_PATH") and isinstance(value, Path):
            monkeypatch.setattr(portfolio, name, tmp_path / value.name)


def test_milestone_matrix_represents_all_18_and_claims_are_bounded() -> None:
    matrix = portfolio.milestone_completion_matrix()

    assert [entry.milestone for entry in matrix] == list(range(1, 19))
    assert {entry.claim_classification for entry in matrix} >= {
        "implemented",
        "locally executed",
        "statically validated",
        "simulated",
        "target-state only",
    }
    aws = next(entry for entry in matrix if entry.milestone == 16)
    assert aws.claim_classification == "target-state only"
    assert "not deployed" in aws.deployment_status.lower()


def test_portfolio_evidence_is_deterministic_and_checksummed(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_portfolio_evidence(monkeypatch, tmp_path)

    first = portfolio.build_portfolio_evidence()
    second = portfolio.build_portfolio_evidence()
    checks = portfolio.validate_portfolio_evidence()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.overall_status == "PASS"
    assert all(check.status == "PASS" for check in checks)
    assert len(first.milestone_completion_matrix) == 18
    assert portfolio.CHECKSUMS_PATH.exists()
    checksums = json.loads(portfolio.CHECKSUMS_PATH.read_text(encoding="utf-8"))
    assert portfolio.READINESS_PATH.as_posix() in checksums


def test_portfolio_validation_fails_when_evidence_missing(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_portfolio_evidence(monkeypatch, tmp_path)

    checks = portfolio.validate_portfolio_evidence()

    assert any(check.status == "FAIL" for check in checks)
    assert any(check.check_id.startswith("PORTFOLIO-EVIDENCE") for check in checks)


def test_no_false_pass_when_manifest_contains_failed_mandatory_check(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    _redirect_portfolio_evidence(monkeypatch, tmp_path)
    manifest = portfolio.build_portfolio_evidence()
    payload = manifest.model_dump(mode="json")
    payload["overall_status"] = "PASS"
    payload["validation_summary"][0]["status"] = "FAIL"
    portfolio.READINESS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    checks = portfolio.validate_portfolio_evidence()

    false_pass = next(
        check for check in checks if check.check_id == "PORTFOLIO-OVERALL-NO-FALSE-PASS"
    )
    assert false_pass.status == "FAIL"


def test_readme_sections_and_portfolio_documents_present() -> None:
    section_checks = portfolio.validate_readme_sections()
    doc_checks = portfolio.validate_portfolio_documents()
    claim_checks = portfolio.validate_claim_boundaries()

    assert all(check.status == "PASS" for check in section_checks)
    assert all(check.status == "PASS" for check in doc_checks)
    assert all(check.status == "PASS" for check in claim_checks)


def test_demo_fast_and_cleanup_behaviour(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_portfolio_evidence(monkeypatch, tmp_path)

    demo = portfolio.run_demo_fast()
    assert demo["executed"] is True
    assert demo["overall_status"] == "PASS"
    assert portfolio.DEMO_FAST_PATH.exists()

    cleanup = portfolio.clean_demo()
    assert cleanup["status"] == "PASS"
    assert portfolio.DEMO_FAST_PATH.as_posix() in cleanup["removed_paths"]
    assert not portfolio.DEMO_FAST_PATH.exists()
    assert cleanup["preserved_evidence_pack"] is True


def test_cli_portfolio_commands(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    _redirect_portfolio_evidence(monkeypatch, tmp_path)

    for command in [
        "build-portfolio-evidence",
        "validate-portfolio-evidence",
        "run-demo-fast",
        "clean-demo",
    ]:
        assert main([command]) == 0

    output = capsys.readouterr().out
    assert "portfolio evidence" in output.lower() or "demo-fast" in output.lower()


def test_makefile_exposes_final_portfolio_targets() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for target in [
        "demo-fast:",
        "demo:",
        "build-portfolio-evidence:",
        "validate-portfolio-evidence:",
        "portfolio-readiness:",
        "clean-demo:",
    ]:
        assert target in makefile


def test_aggregate_status_does_not_pass_incomplete_or_unavailable() -> None:
    assert (
        portfolio._aggregate_status(  # noqa: SLF001
            [
                PortfolioCheck(
                    check_id="P",
                    status="UNAVAILABLE",
                    message="Unavailable mandatory evidence.",
                )
            ]
        )
        == "INCOMPLETE"
    )
    assert (
        portfolio._aggregate_status(  # noqa: SLF001
            [PortfolioCheck(check_id="F", status="FAIL", message="Failed mandatory evidence.")]
        )
        == "FAIL"
    )
