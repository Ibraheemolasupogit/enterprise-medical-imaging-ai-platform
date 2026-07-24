from pathlib import Path

import pytest

from medical_imaging_platform.utils.config import (
    ConfigError,
    load_config,
    validate_repository_configs,
)


def test_valid_configuration_loading(tmp_path: Path) -> None:
    config_path = tmp_path / "platform.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config_name: platform",
                "milestone: 1",
                "status: foundation",
                "purpose: Test configuration.",
                "owner: tests",
                "settings:",
                "  example: value",
                "safeguards:",
                "  - Test safeguard.",
                "future_capabilities:",
                "  - Test future capability.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.config_name == "platform"
    assert config.settings["example"] == "value"


def test_missing_configuration_file() -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(Path("missing.yaml"))


def test_invalid_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("config_name: [unterminated\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid YAML"):
        load_config(config_path)


def test_configuration_file_must_be_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must contain a YAML mapping"):
        load_config(config_path)


def test_missing_required_configuration_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "platform.yaml"
    config_path.write_text("config_name: platform\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(config_path)


def test_invalid_configuration_status(tmp_path: Path) -> None:
    config_path = tmp_path / "platform.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config_name: platform",
                "milestone: 1",
                "status: implemented_medical_device",
                "purpose: Test configuration.",
                "owner: tests",
                "settings:",
                "  example: value",
                "safeguards:",
                "  - Test safeguard.",
                "future_capabilities:",
                "  - Test future capability.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="status must be one of"):
        load_config(config_path)


def test_empty_configuration_settings_are_invalid(tmp_path: Path) -> None:
    config_path = tmp_path / "platform.yaml"
    config_path.write_text(
        "\n".join(
            [
                "config_name: platform",
                "milestone: 1",
                "status: foundation",
                "purpose: Test configuration.",
                "owner: tests",
                "settings: {}",
                "safeguards:",
                "  - Test safeguard.",
                "future_capabilities:",
                "  - Test future capability.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="settings must document"):
        load_config(config_path)


def test_validate_repository_configs_requires_directory(tmp_path: Path) -> None:
    config_file = tmp_path / "config"
    config_file.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ConfigError, match="not a directory"):
        validate_repository_configs(config_file)


def test_validate_all_repository_yaml_configuration_files() -> None:
    configs = validate_repository_configs(Path("config"))

    assert len(configs.configs) == 10
    assert "platform.yaml" in configs.configs
    assert configs.configs["governance.yaml"].settings["human_review"] == "mandatory"
