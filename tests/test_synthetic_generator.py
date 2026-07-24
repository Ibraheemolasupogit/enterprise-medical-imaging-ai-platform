from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.synthetic.generator import (
    SyntheticDataConfig,
    generate_case,
    load_synthetic_config,
)
from medical_imaging_platform.synthetic.validation import DatasetValidationError, validate_case
from medical_imaging_platform.utils.config import ConfigError


def config(tmp_path: Path) -> SyntheticDataConfig:
    return SyntheticDataConfig(
        dataset_id="test-synthetic",
        dataset_version="0.2.0",
        manifest_version="1.0",
        output_root=tmp_path,
        dataset_size=5,
        random_seed=42,
        volume_shape=(32, 32, 32),
        voxel_spacing_mm=(2.5, 2.5, 2.5),
        noise_std_hu=2.0,
        adrenal_radius_voxels=4,
        lesion_radius_voxels=2,
        lesion_intensity_hu=95.0,
        scenarios=["stable", "increased", "reduced", "new", "resolved"],
        split_ratios={"train": 0.6, "validation": 0.2, "test": 0.2},
        generator_version="test-generator",
    )


def test_deterministic_generation_from_seed(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    first = generate_case(cfg, 0, "stable")
    second = generate_case(cfg, 0, "stable")

    assert np.array_equal(first.previous_volume, second.previous_volume)
    assert np.array_equal(first.current_lesion_mask, second.current_lesion_mask)


def test_different_seeds_produce_different_outputs(tmp_path: Path) -> None:
    first_cfg = config(tmp_path)
    second_cfg = first_cfg.model_copy(update={"random_seed": 43})

    first = generate_case(first_cfg, 0, "stable")
    second = generate_case(second_cfg, 0, "stable")

    assert not np.array_equal(first.previous_volume, second.previous_volume)


def test_correct_volume_and_mask_shapes_and_binary_masks(tmp_path: Path) -> None:
    case = generate_case(config(tmp_path), 0, "stable")

    validate_case(case)
    assert case.previous_volume.shape == (32, 32, 32)
    assert set(np.unique(case.previous_lesion_mask).tolist()).issubset({False, True})


def test_lesion_containment(tmp_path: Path) -> None:
    case = generate_case(config(tmp_path), 0, "stable")
    roi = case.previous_left_adrenal_mask

    assert np.all(case.previous_lesion_mask <= roi)
    assert np.all(case.previous_lesion_mask <= case.previous_body_mask)


@pytest.mark.parametrize(
    ("scenario", "expected_relation"),
    [
        ("stable", "equal"),
        ("increased", "greater"),
        ("reduced", "less"),
        ("new", "new"),
        ("resolved", "resolved"),
    ],
)
def test_supported_lesion_scenarios(
    tmp_path: Path,
    scenario: str,
    expected_relation: str,
) -> None:
    case = generate_case(config(tmp_path), 0, scenario)  # type: ignore[arg-type]
    previous_volume = int(np.sum(case.previous_lesion_mask))
    current_volume = int(np.sum(case.current_lesion_mask))

    if expected_relation == "equal":
        assert previous_volume == current_volume > 0
    elif expected_relation == "greater":
        assert current_volume > previous_volume > 0
    elif expected_relation == "less":
        assert 0 < current_volume < previous_volume
    elif expected_relation == "new":
        assert previous_volume == 0 < current_volume
    elif expected_relation == "resolved":
        assert previous_volume > 0 == current_volume


def test_translated_scenario_records_metadata(tmp_path: Path) -> None:
    case = generate_case(config(tmp_path), 0, "translated")

    validate_case(case)
    assert case.change_metadata["translation_voxels_z"] == 2


def test_case_validation_catches_shape_mismatch(tmp_path: Path) -> None:
    case = generate_case(config(tmp_path), 0, "stable")
    broken_case = case.__class__(
        **{
            **case.__dict__,
            "current_lesion_mask": case.current_lesion_mask[1:],
        }
    )

    with pytest.raises(DatasetValidationError, match="shape does not match"):
        validate_case(broken_case)


def test_invalid_synthetic_configuration_handling(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text(
        "\n".join(
            [
                "config_name: data",
                "milestone: 2",
                "status: foundation",
                "purpose: bad",
                "owner: tests",
                "settings:",
                "  not_synthetic: true",
                "safeguards:",
                "  - safeguard",
                "future_capabilities:",
                "  - future",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Missing settings.synthetic_data"):
        load_synthetic_config(path)
