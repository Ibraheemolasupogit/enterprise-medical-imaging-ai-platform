import json
from pathlib import Path

import numpy as np
import pydicom
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series
from medical_imaging_platform.preprocessing.cropping import apply_crop_and_padding
from medical_imaging_platform.preprocessing.errors import (
    PreprocessingError,
    PreprocessingOutputError,
    PreprocessingQualityError,
    PreprocessingRejectedError,
)
from medical_imaging_platform.preprocessing.export import (
    build_output_paths,
    validate_preprocessed_volume,
)
from medical_imaging_platform.preprocessing.geometry import summarise_geometry
from medical_imaging_platform.preprocessing.intensity import apply_intensity_transform
from medical_imaging_platform.preprocessing.models import (
    FixedCropBounds,
    IntensityProfile,
    PreprocessingConfig,
)
from medical_imaging_platform.preprocessing.pipeline import preprocess_dicom_series
from medical_imaging_platform.quality_control.models import QualityControlConfig


def preprocessing_config(tmp_path: Path, **updates: object) -> PreprocessingConfig:
    data = {
        "policy_version": "test-preprocessing-v0.5",
        "axis_order": "zyx",
        "require_quality_report": True,
        "allow_quality_override": True,
        "pixel_dtype": "float32",
        "missing_rescale_slope_default": 1.0,
        "missing_rescale_intercept_default": 0.0,
        "intensity_profiles": {
            "none": IntensityProfile(lower=None, upper=None),
            "soft": IntensityProfile(lower=-1024.0, upper=-1020.0),
            "wide": IntensityProfile(lower=-2048.0, upper=2048.0),
        },
        "default_intensity_profile": "none",
        "normalisation_mode": "none",
        "crop_mode": "none",
        "centre_crop_shape": (2, 6, 6),
        "fixed_crop_bounds": FixedCropBounds(z=(0, 2), y=(1, 7), x=(1, 7)),
        "minimum_output_shape": None,
        "padding_value": 0.0,
        "output_format": "npy",
        "output_directory": tmp_path / "processed",
        "overwrite": False,
        "background_threshold": 0.0,
        "spacing_irregularity_tolerance_mm": 0.01,
    }
    data.update(updates)
    return PreprocessingConfig.model_validate(data)


def qc_config(tmp_path: Path) -> QualityControlConfig:
    return QualityControlConfig(
        policy_version="test-qc",
        output_dir=tmp_path / "quality",
        accepted_modalities=["CT"],
        body_region_allowlist=["ABDOMEN"],
        require_body_part_examined=True,
        pixel_spacing_tolerance=0.001,
        slice_thickness_tolerance=0.001,
        orientation_tolerance=0.001,
        slice_gap_tolerance=0.01,
        minimum_slice_count=2,
        maximum_slice_count=100,
        maximum_corrupt_file_ratio=0.0,
        burned_in_annotation_policy="reject_yes_warn_missing",
        private_tag_policy="warn_before_deidentification_error_after",
        supported_transfer_syntaxes=["1.2.840.10008.1.2.1"],
        pixel_value_bounds=(-4096, 4096),
        scoring_weights={"INFO": 0, "WARNING": 5, "ERROR": 25, "CRITICAL": 100},
        status_thresholds={"pass_minimum": 95, "fail_below": 70},
        critical_rule_ids=["DICOM-QC-PHI-001"],
    )


def preprocess_fixture(tmp_path: Path, **config_updates: object):
    source = tmp_path / "dicom"
    generate_dicom_fixture_series(source, slice_count=3)
    config = preprocessing_config(tmp_path, **config_updates)
    return preprocess_dicom_series(
        source,
        output_root=tmp_path / "processed",
        preprocessing_config=config,
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        overwrite=True,
    )


def test_preprocess_assembles_volume_in_zyx_order_and_records_source_qc(tmp_path: Path) -> None:
    result = preprocess_fixture(tmp_path)
    volume = np.load(Path(result.output_paths.volume))

    assert result.axis_order == "z,y,x"
    assert result.volume_shape == (3, 8, 8)
    assert result.spacing_mm == (2.5, 2.5, 2.5)
    assert result.orientation_classification == "axial_like"
    assert result.source_quality_report_id == "dicom-qc-report-0001"
    assert result.source_quality_status == "PASS_WITH_WARNINGS"
    assert volume[:, 0, 0].tolist() == [-1023.0, -1022.0, -1021.0]
    assert list(result.sop_instance_uid_to_index.values()) == [0, 1, 2]


def test_per_slice_slope_intercept_and_missing_defaults_are_recorded(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    paths = generate_dicom_fixture_series(source, slice_count=2)
    first = pydicom.dcmread(paths[0])
    first.RescaleSlope = "2"
    del first.RescaleIntercept
    first.save_as(paths[0])

    result = preprocess_dicom_series(
        source,
        output_root=tmp_path / "processed",
        preprocessing_config=preprocessing_config(tmp_path),
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        overwrite=True,
    )
    volume = np.load(Path(result.output_paths.volume))

    assert volume[0, 0, 0] == 2.0
    assert result.pixel_conversion.slope_values[0] == 2.0
    assert result.pixel_conversion.defaulted_intercept_sop_uids
    assert result.pixel_conversion.terminology == "CT-like rescaled intensity"


def test_geometry_oblique_and_spacing_fallback_are_preserved(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    paths = generate_dicom_fixture_series(source, slice_count=2)
    for path in paths:
        dataset = pydicom.dcmread(path)
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 0, 1]
        del dataset.ImagePositionPatient
        dataset.SliceThickness = "3.5"
        dataset.save_as(path)

    result = preprocess_dicom_series(
        source,
        output_root=tmp_path / "processed",
        preprocessing_config=preprocessing_config(tmp_path),
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        overwrite=True,
        quality_override=True,
        override_reason="test geometry fallback",
    )

    assert result.orientation_classification == "oblique"
    assert result.geometry.spacing_source == "slice_thickness_fallback"
    assert result.geometry.slice_thickness_fallback_used is True
    assert result.spacing_mm[0] == 3.5


def test_intensity_profiles_minmax_zscore_none_and_constant_fallback(tmp_path: Path) -> None:
    volume = np.array([[[0, 1], [2, 3]]], dtype=np.float32)
    config = preprocessing_config(
        tmp_path,
        default_intensity_profile="soft",
        normalisation_mode="minmax",
    )
    transformed, summary, warnings = apply_intensity_transform(volume, config=config)

    assert transformed.min() == 0
    assert transformed.max() == 0
    assert summary.fallback_used is True
    assert warnings

    zscore, z_summary, _ = apply_intensity_transform(
        volume,
        config=preprocessing_config(tmp_path, normalisation_mode="zscore"),
        profile_name="none",
    )
    assert np.isclose(float(np.mean(zscore)), 0.0)
    assert z_summary.normalisation_mode == "zscore"

    unchanged, none_summary, _ = apply_intensity_transform(
        volume,
        config=preprocessing_config(tmp_path, normalisation_mode="none"),
        profile_name="none",
    )
    assert unchanged.tolist() == volume.tolist()
    assert none_summary.clipped_voxel_count == 0

    scaled, scaled_summary, _ = apply_intensity_transform(
        volume,
        config=preprocessing_config(
            tmp_path,
            default_intensity_profile="wide",
            normalisation_mode="minmax",
        ),
    )
    assert np.allclose(scaled, np.array([[[0.0, 1 / 3], [2 / 3, 1.0]]], dtype=np.float32))
    assert scaled_summary.fallback_used is False

    with pytest.raises(ValueError):
        apply_intensity_transform(
            volume,
            config=preprocessing_config(tmp_path),
            profile_name="missing",
        )


def test_crop_modes_and_padding_metadata(tmp_path: Path) -> None:
    volume = np.zeros((3, 8, 8), dtype=np.float32)
    volume[1, 2:5, 3:7] = 1.0

    cropped, summary, warnings = apply_crop_and_padding(
        volume,
        config=preprocessing_config(
            tmp_path,
            crop_mode="non_background",
            minimum_output_shape=(4, 6, 6),
            padding_value=-1.0,
        ),
    )
    assert cropped.shape == (4, 6, 6)
    assert summary.crop_bounds_zyx["z"] == (1, 2)
    assert summary.pad_widths_zyx[0] == (1, 2)
    assert warnings

    centre, centre_summary, _ = apply_crop_and_padding(
        volume, config=preprocessing_config(tmp_path, crop_mode="centre")
    )
    assert centre.shape == (2, 6, 6)
    assert centre_summary.crop_offsets_zyx == (0, 1, 1)

    with pytest.raises(PreprocessingError):
        apply_crop_and_padding(
            volume,
            config=preprocessing_config(
                tmp_path,
                crop_mode="fixed",
                fixed_crop_bounds=FixedCropBounds(z=(0, 4), y=(0, 8), x=(0, 8)),
            ),
        )


def test_export_validation_checksums_overwrite_and_identifier_exclusion(tmp_path: Path) -> None:
    result = preprocess_fixture(tmp_path)
    output_dir = Path(result.output_paths.output_dir)

    validated = validate_preprocessed_volume(output_dir)
    assert validated.run_id == result.run_id
    assert set(validated.checksums) == {"volume", "report"}
    assert "Synthetic^Fixture" not in (output_dir / "metadata.json").read_text(encoding="utf-8")

    with pytest.raises(PreprocessingOutputError):
        preprocess_dicom_series(
            tmp_path / "dicom",
            output_root=tmp_path / "processed",
            preprocessing_config=preprocessing_config(tmp_path),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
            overwrite=False,
        )

    (output_dir / "preprocessing_report.md").write_text("changed", encoding="utf-8")
    with pytest.raises(PreprocessingOutputError):
        validate_preprocessed_volume(output_dir)

    missing = tmp_path / "missing-output"
    missing.mkdir()
    with pytest.raises(PreprocessingOutputError):
        validate_preprocessed_volume(missing)

    with pytest.raises(PreprocessingOutputError):
        build_output_paths(tmp_path / "root", "../escape")


def test_preprocessing_rejects_empty_missing_pixel_mixed_dimension_and_bad_rescale(
    tmp_path: Path,
) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(PreprocessingError, match="No DICOM series"):
        preprocess_dicom_series(
            tmp_path / "empty",
            output_root=tmp_path / "processed-empty",
            preprocessing_config=preprocessing_config(tmp_path),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
        )

    missing_pixel = tmp_path / "missing-pixel"
    generate_dicom_fixture_series(missing_pixel, slice_count=2, malformed="missing_pixel_data")
    with pytest.raises(PreprocessingError, match="PixelData"):
        preprocess_dicom_series(
            missing_pixel,
            output_root=tmp_path / "processed-missing",
            preprocessing_config=preprocessing_config(
                tmp_path, require_quality_report=False, allow_quality_override=True
            ),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
            quality_override=True,
        )

    mixed = tmp_path / "mixed"
    mixed_paths = generate_dicom_fixture_series(mixed, slice_count=2)
    mixed_dataset = pydicom.dcmread(mixed_paths[1])
    mixed_dataset.Rows = 9
    mixed_dataset.PixelData = np.full((9, 8), fill_value=2, dtype=np.int16).tobytes()
    mixed_dataset.save_as(mixed_paths[1])
    with pytest.raises(PreprocessingError, match="Rows and Columns"):
        preprocess_dicom_series(
            mixed,
            output_root=tmp_path / "processed-mixed",
            preprocessing_config=preprocessing_config(tmp_path),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
            quality_override=True,
        )

    bad_rescale = tmp_path / "bad-rescale"
    paths = generate_dicom_fixture_series(bad_rescale, slice_count=2)
    dataset = pydicom.dcmread(paths[0])
    dataset.RescaleSlope = "nan"
    dataset.save_as(paths[0])
    with pytest.raises(PreprocessingError, match="rescale"):
        preprocess_dicom_series(
            bad_rescale,
            output_root=tmp_path / "processed-bad-rescale",
            preprocessing_config=preprocessing_config(tmp_path),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
            quality_override=True,
        )


def test_ambiguous_ordering_requires_override_and_records_warning(tmp_path: Path) -> None:
    source = tmp_path / "duplicate-order"
    paths = generate_dicom_fixture_series(source, slice_count=2)
    duplicate = pydicom.dcmread(paths[1])
    duplicate.InstanceNumber = 1
    duplicate.ImagePositionPatient = [0.0, 0.0, 0.0]
    duplicate.save_as(paths[1])

    with pytest.raises(PreprocessingQualityError):
        preprocess_dicom_series(
            source,
            output_root=tmp_path / "processed-ambiguous",
            preprocessing_config=preprocessing_config(tmp_path),
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
        )

    result = preprocess_dicom_series(
        source,
        output_root=tmp_path / "processed-ambiguous",
        preprocessing_config=preprocessing_config(tmp_path),
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        quality_override=True,
        overwrite=True,
    )
    assert any("Ordering override accepted" in warning for warning in result.warnings)


def test_geometry_indeterminate_defaults_are_reported(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    paths = generate_dicom_fixture_series(source, slice_count=1)
    metadata = pydicom.dcmread(paths[0])
    from medical_imaging_platform.ingestion.metadata import extract_metadata

    safe_metadata = extract_metadata(metadata, paths[0]).model_copy(
        update={
            "pixel_spacing": None,
            "image_orientation_patient": None,
            "image_position_patient": None,
            "slice_thickness": None,
        }
    )

    geometry, warnings = summarise_geometry([safe_metadata], irregularity_tolerance_mm=0.01)
    assert geometry.spacing_mm == (1.0, 1.0, 1.0)
    assert geometry.orientation_classification == "indeterminate"
    assert warnings


def test_quality_fail_override_and_rejected_critical_stop(tmp_path: Path) -> None:
    wrong_modality = tmp_path / "wrong"
    generate_dicom_fixture_series(wrong_modality, slice_count=2, malformed="wrong_modality")
    config = preprocessing_config(tmp_path)

    with pytest.raises(PreprocessingQualityError):
        preprocess_dicom_series(
            wrong_modality,
            output_root=tmp_path / "blocked",
            preprocessing_config=config,
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
        )

    overridden = preprocess_dicom_series(
        wrong_modality,
        output_root=tmp_path / "overridden",
        preprocessing_config=config,
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        quality_override=True,
        override_reason="engineering test",
    )
    assert overridden.override_used is True
    assert overridden.source_quality_status == "FAIL"

    burned = tmp_path / "burned"
    generate_dicom_fixture_series(burned, slice_count=2, malformed="burned_in")
    with pytest.raises(PreprocessingRejectedError):
        preprocess_dicom_series(
            burned,
            output_root=tmp_path / "rejected",
            preprocessing_config=config,
            qc_config=qc_config(tmp_path),
            dicom_max_files=20,
            max_file_size_bytes=1_000_000,
            quality_override=True,
        )


def test_multiple_series_require_explicit_selection_and_cli_roundtrip(
    tmp_path: Path, capsys
) -> None:
    source = tmp_path / "dicom"
    study_uid = "1.2.826.0.1.3680043.10.54321.3.1.50"
    series_a = "1.2.826.0.1.3680043.10.54321.3.2.50"
    series_b = "1.2.826.0.1.3680043.10.54321.3.2.51"
    generate_dicom_fixture_series(
        source / "a", slice_count=2, study_uid=study_uid, series_uid=series_a
    )
    generate_dicom_fixture_series(
        source / "b", slice_count=2, study_uid=study_uid, series_uid=series_b
    )

    assert (
        main(
            [
                "preprocess-dicom",
                str(source),
                "--output-dir",
                str(tmp_path / "processed"),
                "--config",
                "config/preprocessing.yaml",
                "--data-config",
                "config/data.yaml",
            ]
        )
        == 1
    )
    capsys.readouterr()

    assert (
        main(
            [
                "preprocess-dicom",
                str(source),
                "--study-uid",
                study_uid,
                "--series-uid",
                series_a,
                "--output-dir",
                str(tmp_path / "processed"),
                "--config",
                "config/preprocessing.yaml",
                "--data-config",
                "config/data.yaml",
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    output_dir = Path(result["output_paths"]["output_dir"])

    assert main(["validate-preprocessed-volume", str(output_dir)]) == 0
    capsys.readouterr()
    assert main(["inspect-preprocessed-volume", str(output_dir), "--json"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["shape"] == [4, 8, 8]
