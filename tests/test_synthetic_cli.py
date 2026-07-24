from pathlib import Path

from medical_imaging_platform.cli import main


def test_cli_generation_and_dataset_validation(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output_dir = tmp_path / "synthetic"

    exit_code = main(
        [
            "generate-synthetic-data",
            "--output-dir",
            str(output_dir),
            "--cases",
            "3",
            "--seed",
            "55",
        ]
    )

    assert exit_code == 0
    assert "Generated 3 synthetic cases" in capsys.readouterr().out
    assert (output_dir / "manifest.json").exists()

    validate_exit_code = main(["validate-dataset", str(output_dir)])

    assert validate_exit_code == 0
    assert "Validated synthetic dataset with 3 cases." in capsys.readouterr().out


def test_cli_generation_overwrite_protection(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output_dir = tmp_path / "synthetic"

    assert main(["generate-synthetic-data", "--output-dir", str(output_dir), "--cases", "1"]) == 0
    assert main(["generate-synthetic-data", "--output-dir", str(output_dir), "--cases", "1"]) == 1
    assert "Output directory is not empty" in capsys.readouterr().out


def test_cli_dataset_summary(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output_dir = tmp_path / "synthetic"
    assert main(["generate-synthetic-data", "--output-dir", str(output_dir), "--cases", "2"]) == 0

    exit_code = main(["summarise-dataset", str(output_dir)])

    assert exit_code == 0
    assert "case_count" in capsys.readouterr().out


def test_cli_dataset_validation_failure(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["validate-dataset", "missing-dataset"])

    assert exit_code == 1
    assert "Dataset validation failed" in capsys.readouterr().out
