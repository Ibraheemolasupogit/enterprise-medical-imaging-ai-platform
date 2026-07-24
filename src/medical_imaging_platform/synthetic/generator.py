"""Configuration-driven synthetic CT-like volume generation.

The generated arrays are engineering fixtures only. They are not clinically
realistic CT scans and must not be used for clinical-performance claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from medical_imaging_platform.utils.config import ConfigError, load_yaml_file

Scenario = Literal["stable", "increased", "reduced", "new", "resolved", "translated"]
LesionSide = Literal["left", "right"]


class SyntheticDataConfig(BaseModel):
    """Typed settings for synthetic longitudinal data generation."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    manifest_version: str = Field(min_length=1)
    output_root: Path
    dataset_size: int = Field(ge=1)
    random_seed: int = Field(ge=0)
    volume_shape: tuple[int, int, int]
    voxel_spacing_mm: tuple[float, float, float]
    noise_std_hu: float = Field(ge=0)
    adrenal_radius_voxels: int = Field(ge=2)
    lesion_radius_voxels: int = Field(ge=1)
    lesion_intensity_hu: float
    scenarios: list[Scenario] = Field(min_length=1)
    split_ratios: dict[str, float]
    generator_version: str = Field(min_length=1)

    @field_validator("volume_shape")
    @classmethod
    def volume_shape_must_be_3d(cls, value: tuple[int, int, int]) -> tuple[int, int, int]:
        if len(value) != 3 or any(axis < 16 for axis in value):
            raise ValueError("volume_shape must contain three dimensions of at least 16 voxels")
        return value

    @field_validator("voxel_spacing_mm")
    @classmethod
    def spacing_must_be_3d(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if len(value) != 3 or any(axis <= 0 for axis in value):
            raise ValueError("voxel_spacing_mm must contain three positive values")
        return value

    @field_validator("split_ratios")
    @classmethod
    def split_ratios_must_be_valid(cls, value: dict[str, float]) -> dict[str, float]:
        required = {"train", "validation", "test"}
        if set(value) != required:
            raise ValueError("split_ratios must contain train, validation, and test")
        if any(ratio < 0 for ratio in value.values()):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(value.values()) - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1.0")
        return value


@dataclass(frozen=True)
class SyntheticCase:
    """In-memory generated longitudinal synthetic case."""

    case_id: str
    research_subject_id: str
    scenario: Scenario
    lesion_side: LesionSide
    random_seed: int
    previous_volume: np.ndarray
    current_volume: np.ndarray
    previous_body_mask: np.ndarray
    current_body_mask: np.ndarray
    previous_left_adrenal_mask: np.ndarray
    previous_right_adrenal_mask: np.ndarray
    current_left_adrenal_mask: np.ndarray
    current_right_adrenal_mask: np.ndarray
    previous_lesion_mask: np.ndarray
    current_lesion_mask: np.ndarray
    change_metadata: dict[str, object]


def load_synthetic_config(path: Path) -> SyntheticDataConfig:
    """Load synthetic settings from the repository data configuration."""
    data = load_yaml_file(path)
    settings = data.get("settings")
    if not isinstance(settings, dict) or "synthetic_data" not in settings:
        raise ConfigError(f"Missing settings.synthetic_data in {path}")
    try:
        return SyntheticDataConfig.model_validate(settings["synthetic_data"])
    except ValueError as exc:
        raise ConfigError(f"Invalid synthetic data configuration in {path}: {exc}") from exc


def generate_case(
    config: SyntheticDataConfig, index: int, scenario: Scenario | None = None
) -> SyntheticCase:
    """Generate one deterministic longitudinal synthetic case."""
    case_seed = config.random_seed + index
    rng = np.random.default_rng(case_seed)
    selected_scenario = scenario or config.scenarios[index % len(config.scenarios)]
    lesion_side: LesionSide = "left" if index % 2 == 0 else "right"

    previous_radius, current_radius = _scenario_radii(
        selected_scenario, config.lesion_radius_voxels
    )
    translated = selected_scenario == "translated"

    previous = _generate_single_timepoint(config, rng, lesion_side, previous_radius, translation=0)
    current_shift = 2 if translated else 0
    current = _generate_single_timepoint(
        config, rng, lesion_side, current_radius, translation=current_shift
    )

    return SyntheticCase(
        case_id=f"synthetic-case-{index + 1:04d}",
        research_subject_id=f"research-subject-{index + 1:04d}",
        scenario=selected_scenario,
        lesion_side=lesion_side,
        random_seed=case_seed,
        previous_volume=previous["volume"],
        current_volume=current["volume"],
        previous_body_mask=previous["body_mask"],
        current_body_mask=current["body_mask"],
        previous_left_adrenal_mask=previous["left_adrenal_mask"],
        previous_right_adrenal_mask=previous["right_adrenal_mask"],
        current_left_adrenal_mask=current["left_adrenal_mask"],
        current_right_adrenal_mask=current["right_adrenal_mask"],
        previous_lesion_mask=previous["lesion_mask"],
        current_lesion_mask=current["lesion_mask"],
        change_metadata={
            "scenario": selected_scenario,
            "translation_voxels_z": current_shift,
            "previous_lesion_radius_voxels": previous_radius,
            "current_lesion_radius_voxels": current_radius,
            "description": _scenario_description(selected_scenario),
        },
    )


def generate_cases(config: SyntheticDataConfig, count: int | None = None) -> list[SyntheticCase]:
    """Generate a deterministic list of synthetic cases."""
    case_count = count or config.dataset_size
    return [generate_case(config, index) for index in range(case_count)]


def _generate_single_timepoint(
    config: SyntheticDataConfig,
    rng: np.random.Generator,
    lesion_side: LesionSide,
    lesion_radius: int,
    translation: int,
) -> dict[str, np.ndarray]:
    shape = config.volume_shape
    zz, yy, xx = np.indices(shape)
    center = np.array(shape, dtype=float) / 2
    center[0] += translation

    body_mask = _ellipsoid_mask(
        shape,
        center=tuple(center),
        radii=(shape[0] * 0.42, shape[1] * 0.38, shape[2] * 0.42),
    )
    volume = np.full(shape, -1000.0, dtype=np.float32)
    volume[body_mask] = 45.0

    # Simplified engineering-only structures.
    spine = ((yy - center[1]) ** 2 + (xx - (center[2] + shape[2] * 0.22)) ** 2) < 9
    volume[spine & body_mask] = 250.0
    vessel = ((yy - (center[1] - 3)) ** 2 + (xx - center[2]) ** 2) < 4
    volume[vessel & body_mask] = 120.0

    left_center = (center[0] - 2, center[1] - 5, center[2] - shape[2] * 0.22)
    right_center = (center[0] - 2, center[1] - 5, center[2] + shape[2] * 0.22)
    adrenal_radii = (
        config.adrenal_radius_voxels,
        config.adrenal_radius_voxels,
        config.adrenal_radius_voxels,
    )
    left_adrenal_mask = _ellipsoid_mask(shape, left_center, adrenal_radii) & body_mask
    right_adrenal_mask = _ellipsoid_mask(shape, right_center, adrenal_radii) & body_mask
    volume[left_adrenal_mask | right_adrenal_mask] = 75.0

    lesion_mask = np.zeros(shape, dtype=bool)
    if lesion_radius > 0:
        roi_center = left_center if lesion_side == "left" else right_center
        lesion_mask = _ellipsoid_mask(
            shape,
            roi_center,
            (lesion_radius, lesion_radius, lesion_radius),
        )
        roi_mask = left_adrenal_mask if lesion_side == "left" else right_adrenal_mask
        lesion_mask &= roi_mask
        volume[lesion_mask] = config.lesion_intensity_hu

    noise = rng.normal(0.0, config.noise_std_hu, size=shape).astype(np.float32)
    volume[body_mask] += noise[body_mask]
    volume[~body_mask] += noise[~body_mask] * 0.2

    return {
        "volume": volume.astype(np.float32),
        "body_mask": body_mask,
        "left_adrenal_mask": left_adrenal_mask,
        "right_adrenal_mask": right_adrenal_mask,
        "lesion_mask": lesion_mask,
    }


def _ellipsoid_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    zz, yy, xx = np.indices(shape)
    distance = (
        ((zz - center[0]) / radii[0]) ** 2
        + ((yy - center[1]) / radii[1]) ** 2
        + ((xx - center[2]) / radii[2]) ** 2
    )
    return np.asarray(distance <= 1.0, dtype=bool)


def _scenario_radii(scenario: Scenario, base_radius: int) -> tuple[int, int]:
    if scenario == "stable":
        return base_radius, base_radius
    if scenario == "increased":
        return base_radius, base_radius + 1
    if scenario == "reduced":
        return base_radius + 1, base_radius
    if scenario == "new":
        return 0, base_radius
    if scenario == "resolved":
        return base_radius, 0
    if scenario == "translated":
        return base_radius, base_radius
    raise ValueError(f"Unsupported scenario: {scenario}")


def _scenario_description(scenario: Scenario) -> str:
    descriptions = {
        "stable": "Lesion radius is unchanged between previous and current volumes.",
        "increased": "Current lesion radius is larger than previous lesion radius.",
        "reduced": "Current lesion radius is smaller than previous lesion radius.",
        "new": "Previous volume has no lesion and current volume has a lesion.",
        "resolved": "Previous volume has a lesion and current volume has no lesion.",
        "translated": "Current anatomy is shifted without implementing registration.",
    }
    return descriptions[scenario]
