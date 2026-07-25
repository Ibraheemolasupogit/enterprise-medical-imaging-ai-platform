from medical_imaging_platform import __version__
from medical_imaging_platform.cli import main


def test_cli_version_command(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["version"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == __version__


def test_cli_configuration_validation_command(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["validate-config"])

    assert exit_code == 0
    assert "Validated 13 configuration files." in capsys.readouterr().out


def test_cli_configuration_validation_failure(capsys) -> None:  # type: ignore[no-untyped-def]
    exit_code = main(["validate-config", "--config-dir", "missing-config"])

    assert exit_code == 1
    assert "Configuration validation failed" in capsys.readouterr().out
