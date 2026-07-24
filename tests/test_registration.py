import json
from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.preprocessing.models import PreprocessingResult
from medical_imaging_platform.registration.centre_of_mass import (
    centre_of_mass_transform,
    foreground_mask,
)
from medical_imaging_platform.registration.conversion import (
    numpy_to_sitk,
    sitk_to_numpy,
    validate_round_trip,
)
from medical_imaging_platform.registration.export import (
    RegistrationOutputError,
    build_output_paths,
    inspect_registration_output,
    validate_registration_output,
)
from medical_imaging_platform.registration.fixtures import generate_registration_fixture_pair
from medical_imaging_platform.registration.metrics import (
    compute_metrics,
    dice_coefficient,
    foreground_overlap,
    mutual_information,
    normalised_cross_correlation,
    sitk_metric_value,
)
from medical_imaging_platform.registration.models import (
    RegistrationConfig,
    RegistrationFinding,
    TransformSummary,
)
from medical_imaging_platform.registration.pipeline import (
    register_preprocessed_volumes,
    summarise_transform,
)
from medical_imaging_platform.registration.preconditions import (
    load_preprocessed_input,
    validate_registration_inputs,
)
from medical_imaging_platform.registration.quality import evaluate_registration_quality
from medical_imaging_platform.registration.resampling import resample_moving_to_fixed
from medical_imaging_platform.registration.rigid import _registration_method
from medical_imaging_platform.registration.visualisation import mid_axial_slices


def reg_config(tmp_path: Path, **updates: object) -> RegistrationConfig:
    data = {
        "policy_version": "test-registration-v0.6",
        "default_mode": "rigid",
        "initialisation_method": "centre_of_mass",
        "allow_constant_volume": False,
        "foreground_threshold": 0.01,
        "metric": "mattes_mutual_information",
        "metric_bins": 16,
        "metric_sampling_strategy": "none",
        "metric_sampling_percentage": 1.0,
        "random_seed": 7,
        "optimiser": "gradient_descent",
        "learning_rate": 0.5,
        "minimum_step": 0.0001,
        "maximum_iterations": 5,
        "convergence_window_size": 3,
        "shrink_factors": [1],
        "smoothing_sigmas": [0.0],
        "interpolator": "linear",
        "default_pixel_value": 0.0,
        "maximum_translation_mm": 50.0,
        "maximum_rotation_degrees": 25.0,
        "affine_scale_bounds": (0.85, 1.2),
        "maximum_shear": 0.2,
        "minimum_metric_improvement": -0.000001,
        "maximum_padding_fraction": 0.99,
        "quality_status_policy": {},
        "output_directory": tmp_path / "registration",
        "overwrite": False,
    }
    data.update(updates)
    return RegistrationConfig.model_validate(data)


def fixture_pair(tmp_path: Path) -> tuple[Path, Path]:
    return generate_registration_fixture_pair(tmp_path / "fixtures", overwrite=True)


def test_registration_fixture_outputs_validate_as_preprocessed(tmp_path: Path) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    fixed_volume, fixed_meta = load_preprocessed_input(fixed_dir)
    moving_volume, moving_meta = load_preprocessed_input(moving_dir)

    assert fixed_volume.shape == (24, 24, 24)
    assert moving_volume.shape == fixed_volume.shape
    assert fixed_meta.axis_order == "z,y,x"
    assert moving_meta.source_quality_status == "PASS"


def test_numpy_simpleitk_conversion_preserves_shape_spacing_origin(tmp_path: Path) -> None:
    fixed_dir, _ = fixture_pair(tmp_path)
    volume, metadata = load_preprocessed_input(fixed_dir)

    image = numpy_to_sitk(volume, metadata)
    validate_round_trip(volume, metadata)
    round_trip = sitk_to_numpy(image)

    assert round_trip.shape == volume.shape
    assert image.GetSpacing() == (1.0, 1.0, 1.0)
    assert image.GetOrigin() == (0.0, 0.0, 0.0)

    no_direction = metadata.model_copy(
        update={
            "geometry": metadata.geometry.model_copy(
                update={
                    "row_direction_cosines": None,
                    "column_direction_cosines": None,
                    "slice_normal": None,
                    "image_positions_patient": [None],
                }
            )
        }
    )
    defaulted = numpy_to_sitk(volume, no_direction)
    assert defaulted.GetOrigin() == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        numpy_to_sitk(volume[0], metadata)


def test_centre_of_mass_translation_recovers_known_shift(tmp_path: Path) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    fixed, fixed_meta = load_preprocessed_input(fixed_dir)
    moving, _ = load_preprocessed_input(moving_dir)

    transform, translation_mm, translation_voxels = centre_of_mass_transform(
        fixed, moving, spacing_zyx=fixed_meta.spacing_mm, threshold=0.01
    )

    assert transform.GetParameters() == pytest.approx((3.0, -1.0, 2.0))
    assert translation_mm == pytest.approx((-3.0, 1.0, -2.0))
    assert translation_voxels == pytest.approx((-2.0, 1.0, -3.0))
    assert foreground_mask(fixed, 0.01).dtype == bool


def test_empty_foreground_fails(tmp_path: Path) -> None:
    empty = np.zeros((4, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        centre_of_mass_transform(empty, empty, spacing_zyx=(1.0, 1.0, 1.0), threshold=0.01)


def test_metrics_cover_similarity_overlap_and_empty_masks() -> None:
    first = np.array([[[0, 1], [1, 0]]], dtype=np.float32)
    second = first.copy()
    empty = np.zeros_like(first, dtype=bool)

    metrics = compute_metrics(first, second, spacing_zyx=(1.0, 1.0, 1.0), foreground_threshold=0.1)

    assert metrics.mean_squared_error == 0.0
    assert metrics.normalised_cross_correlation == pytest.approx(1.0)
    assert metrics.dice_coefficient == pytest.approx(1.0)
    assert metrics.foreground_overlap == pytest.approx(1.0)
    assert normalised_cross_correlation(first, second) == pytest.approx(1.0)
    assert mutual_information(first, second) > 0
    assert dice_coefficient(empty, empty) is None
    assert foreground_overlap(empty, empty) is None


def test_preconditions_reject_same_input_constant_and_propagate_override(tmp_path: Path) -> None:
    fixed_dir, _ = fixture_pair(tmp_path)
    _, metadata = load_preprocessed_input(fixed_dir)
    constant = np.zeros((24, 24, 24), dtype=np.float32)
    constant_dir = tmp_path / "constant"
    _copy_preprocessed_with_volume(constant_dir, metadata, constant, run_id="constant-run")
    _, _, _, _, findings = validate_registration_inputs(
        fixed_dir,
        constant_dir,
        allow_constant_volume=False,
    )

    assert any("constant" in finding.message for finding in findings)

    _, _, _, _, same_findings = validate_registration_inputs(
        fixed_dir,
        fixed_dir,
        allow_constant_volume=True,
    )
    assert any("same run" in finding.message for finding in same_findings)


def test_preconditions_flag_nonfinite_spacing_axis_qc_and_override(tmp_path: Path) -> None:
    fixed_dir, _ = fixture_pair(tmp_path)
    _, metadata = load_preprocessed_input(fixed_dir)
    bad = metadata.model_copy(
        update={
            "run_id": "bad-run",
            "spacing_mm": (0.0, 1.0, 1.0),
            "source_quality_status": "REJECTED",
            "override_used": True,
        }
    )
    bad_dir = tmp_path / "bad"
    _write_metadata_and_volume(bad_dir, bad, np.full((24, 24, 24), np.nan, dtype=np.float32))

    _, _, _, _, findings = validate_registration_inputs(
        fixed_dir,
        bad_dir,
        allow_constant_volume=True,
    )

    messages = " ".join(finding.message for finding in findings)
    assert "non-finite" in messages
    assert "spacing" in messages
    assert "REJECTED" in messages
    assert "override" in messages


def test_register_modes_export_and_improve_metrics(tmp_path: Path) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    for mode in ("centre_of_mass", "rigid", "rigid_then_affine"):
        result = register_preprocessed_volumes(
            fixed_dir,
            moving_dir,
            output_root=tmp_path / "registration",
            config=reg_config(tmp_path),
            mode=mode,
            fixed_temporal_label="reference",
            moving_temporal_label="followup",
            overwrite=True,
        )
        assert result.status == "PASS"
        assert result.metrics_after.mean_squared_error <= result.metrics_before.mean_squared_error
        assert Path(result.output_paths.registered_moving_volume).exists()
        assert validate_registration_output(Path(result.output_paths.output_dir)).status == "PASS"
        assert inspect_registration_output(Path(result.output_paths.output_dir))["shape"] == [
            24,
            24,
            24,
        ]


def test_resampling_uses_fixed_geometry_and_visual_arrays(tmp_path: Path) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    fixed, fixed_meta = load_preprocessed_input(fixed_dir)
    moving, moving_meta = load_preprocessed_input(moving_dir)
    fixed_image = numpy_to_sitk(fixed, fixed_meta)
    moving_image = numpy_to_sitk(moving, moving_meta)
    transform, _, _ = centre_of_mass_transform(
        fixed, moving, spacing_zyx=fixed_meta.spacing_mm, threshold=0.01
    )

    registered = resample_moving_to_fixed(
        moving_image,
        fixed_image,
        transform,
        interpolator="nearest",
        default_pixel_value=0.0,
    )
    arrays = mid_axial_slices(fixed, moving, registered)

    assert registered.shape == fixed.shape
    assert set(arrays) == {
        "fixed_mid_axial",
        "moving_mid_axial",
        "registered_mid_axial",
        "overlay_mid_axial",
        "difference_mid_axial",
    }

    with pytest.raises(ValueError):
        resample_moving_to_fixed(
            moving_image,
            fixed_image,
            transform,
            interpolator="linear",
            default_pixel_value=float("nan"),
        )


def test_quality_gates_fail_implausible_transform_and_padding(tmp_path: Path) -> None:
    config = reg_config(tmp_path, maximum_translation_mm=1.0, maximum_padding_fraction=0.1)
    before = compute_metrics(
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        spacing_zyx=(1.0, 1.0, 1.0),
        foreground_threshold=0.1,
    )
    after = compute_metrics(
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        spacing_zyx=(1.0, 1.0, 1.0),
        foreground_threshold=0.1,
    )
    transform = TransformSummary(
        transform_type="TranslationTransform",
        parameters=[10.0, 0.0, 0.0],
        fixed_parameters=[],
        translation_mm_xyz=(10.0, 0.0, 0.0),
        translation_voxels_zyx=(0.0, 0.0, 10.0),
        rotation_degrees=(0.0, 0.0, 0.0),
    )

    status, findings = evaluate_registration_quality(
        transform,
        before,
        after,
        np.zeros((2, 2, 2), dtype=np.float32),
        default_pixel_value=0.0,
        config=config,
        precondition_findings=[],
    )

    assert status == "FAIL"
    assert {"REG-QC-TRN-001", "REG-QC-PAD-001"} <= {finding.rule_id for finding in findings}


def test_quality_gates_cover_rotation_affine_metric_and_warning(tmp_path: Path) -> None:
    config = reg_config(tmp_path, maximum_rotation_degrees=1.0, minimum_metric_improvement=1.0)
    before = compute_metrics(
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        spacing_zyx=(1.0, 1.0, 1.0),
        foreground_threshold=0.1,
    )
    after = compute_metrics(
        np.ones((2, 2, 2)),
        np.zeros((2, 2, 2)),
        spacing_zyx=(1.0, 1.0, 1.0),
        foreground_threshold=0.1,
    )
    transform = TransformSummary(
        transform_type="AffineTransform",
        parameters=[float("nan")],
        fixed_parameters=[],
        translation_mm_xyz=(0.0, 0.0, 0.0),
        translation_voxels_zyx=(0.0, 0.0, 0.0),
        rotation_degrees=(2.0, 0.0, 0.0),
        affine_matrix=[2.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        affine_scale=(2.0, 0.5, 1.0),
        affine_shear=1.0,
    )
    status, findings = evaluate_registration_quality(
        transform,
        before,
        after,
        np.ones((2, 2, 2), dtype=np.float32),
        default_pixel_value=0.0,
        config=config,
        precondition_findings=[],
    )
    assert status == "REJECTED"
    assert {"REG-QC-CONV-001", "REG-QC-ROT-001", "REG-QC-AFF-001", "REG-QC-MET-001"} <= {
        finding.rule_id for finding in findings
    }

    warning_status, _ = evaluate_registration_quality(
        TransformSummary(
            transform_type="TranslationTransform",
            parameters=[0.0, 0.0, 0.0],
            fixed_parameters=[],
            translation_mm_xyz=(0.0, 0.0, 0.0),
            translation_voxels_zyx=(0.0, 0.0, 0.0),
            rotation_degrees=(0.0, 0.0, 0.0),
        ),
        after,
        before,
        np.ones((2, 2, 2), dtype=np.float32),
        default_pixel_value=0.0,
        config=reg_config(tmp_path),
        precondition_findings=[
            RegistrationFinding(
                rule_id="REG-QC-INP-001",
                severity="WARNING",
                status="FAIL",
                message="warning",
                remediation="review",
            )
        ],
    )
    assert warning_status == "PASS_WITH_WARNINGS"


def test_quality_rejects_critical_precondition(tmp_path: Path) -> None:
    config = reg_config(tmp_path)
    metrics = compute_metrics(
        np.ones((2, 2, 2)),
        np.ones((2, 2, 2)),
        spacing_zyx=(1.0, 1.0, 1.0),
        foreground_threshold=0.1,
    )
    status, _ = evaluate_registration_quality(
        TransformSummary(
            transform_type="TranslationTransform",
            parameters=[0.0, 0.0, 0.0],
            fixed_parameters=[],
            translation_mm_xyz=(0.0, 0.0, 0.0),
            translation_voxels_zyx=(0.0, 0.0, 0.0),
            rotation_degrees=(0.0, 0.0, 0.0),
        ),
        metrics,
        metrics,
        np.ones((2, 2, 2), dtype=np.float32),
        default_pixel_value=0.0,
        config=config,
        precondition_findings=[
            RegistrationFinding(
                rule_id="REG-QC-INP-001",
                severity="CRITICAL",
                status="FAIL",
                message="bad input",
                remediation="fix",
            )
        ],
    )
    assert status == "REJECTED"


def test_registration_cli_json_validation_and_overwrite(tmp_path: Path, capsys) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    output = tmp_path / "registration"
    args = [
        "register-volumes",
        "--fixed",
        str(fixed_dir),
        "--moving",
        str(moving_dir),
        "--fixed-temporal-label",
        "reference",
        "--moving-temporal-label",
        "followup",
        "--mode",
        "centre_of_mass",
        "--output-dir",
        str(output),
        "--config",
        "config/registration.yaml",
        "--overwrite",
        "--json",
    ]

    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    output_dir = Path(result["output_paths"]["output_dir"])
    assert main(["validate-registration", str(output_dir), "--json"]) == 0
    capsys.readouterr()
    assert main(["inspect-registration", str(output_dir), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"

    no_overwrite_args = [item for item in args if item not in {"--overwrite", "--json"}]
    assert main(no_overwrite_args) == 4

    assert main(["inspect-registration", str(tmp_path / "missing")]) == 4
    assert main(["validate-registration", str(tmp_path / "missing")]) == 4


def test_registration_fixture_cli_and_overwrite_protection(tmp_path: Path) -> None:
    output = tmp_path / "fixtures-cli"
    assert main(["generate-registration-fixtures", "--output-dir", str(output)]) == 0
    assert main(["generate-registration-fixtures", "--output-dir", str(output)]) == 1


def test_registration_output_path_and_checksum_validation(tmp_path: Path) -> None:
    fixed_dir, moving_dir = fixture_pair(tmp_path)
    result = register_preprocessed_volumes(
        fixed_dir,
        moving_dir,
        output_root=tmp_path / "registration",
        config=reg_config(tmp_path),
        mode="centre_of_mass",
        fixed_temporal_label="reference",
        moving_temporal_label="followup",
        overwrite=True,
    )
    output_dir = Path(result.output_paths.output_dir)

    with pytest.raises(RegistrationOutputError):
        build_output_paths(tmp_path / "root", "../escape")

    (output_dir / "transform.json").write_text("changed", encoding="utf-8")
    with pytest.raises(RegistrationOutputError):
        validate_registration_output(output_dir)


def test_summarise_transform_affine_fields() -> None:
    import SimpleITK as sitk  # noqa: N813

    transform = sitk.AffineTransform(3)
    transform.Scale((1.05, 0.95, 1.0))
    transform.Translate((1.0, 2.0, 3.0))
    summary = summarise_transform(
        transform,
        spacing_zyx=(1.0, 1.0, 1.0),
        fallback_translation_mm=(0.0, 0.0, 0.0),
        fallback_translation_voxels=(0.0, 0.0, 0.0),
    )

    assert summary.affine_matrix is not None
    assert summary.affine_scale == pytest.approx((1.05, 0.95, 1.0))
    assert summary.translation_mm_xyz == pytest.approx((-1.0, -2.0, -3.0))


def test_sitk_metric_and_registration_method_sampling(tmp_path: Path) -> None:
    fixed_dir, _ = fixture_pair(tmp_path)
    volume, metadata = load_preprocessed_input(fixed_dir)
    image = numpy_to_sitk(volume, metadata)
    import SimpleITK as sitk  # noqa: N813

    metric = sitk_metric_value(
        image,
        image,
        sitk.TranslationTransform(3),
        "normalised_correlation",
        bins=8,
    )
    regular_method = _registration_method(
        reg_config(
            tmp_path,
            metric="normalised_correlation",
            metric_sampling_strategy="regular",
        )
    )
    random_method = _registration_method(reg_config(tmp_path, metric_sampling_strategy="random"))

    assert np.isfinite(metric)
    assert regular_method is not None
    assert random_method is not None


def _copy_preprocessed_with_volume(
    output_dir: Path,
    metadata: PreprocessingResult,
    volume: np.ndarray,
    *,
    run_id: str,
) -> None:
    from medical_imaging_platform.registration.fixtures import _write_preprocessed_fixture

    _write_preprocessed_fixture(output_dir, volume, run_id=run_id)


def _write_metadata_and_volume(
    output_dir: Path, metadata: PreprocessingResult, volume: np.ndarray
) -> None:
    from medical_imaging_platform.registration.fixtures import _write_preprocessed_fixture
    from medical_imaging_platform.synthetic.manifest import sha256_file

    _write_preprocessed_fixture(output_dir, np.nan_to_num(volume), run_id=metadata.run_id)
    volume_path = output_dir / "volume.npy"
    with volume_path.open("wb") as handle:
        np.save(handle, volume)
    metadata_path = output_dir / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    data.update(metadata.model_dump(mode="json"))
    data["output_paths"]["output_dir"] = str(output_dir)
    data["output_paths"]["volume"] = str(volume_path)
    data["output_paths"]["metadata"] = str(metadata_path)
    data["output_paths"]["report"] = str(output_dir / "preprocessing_report.md")
    data["checksums"]["volume"] = sha256_file(volume_path)
    metadata_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
