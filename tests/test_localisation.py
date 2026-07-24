import json
from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.localisation.atlas import centre_mm, predict_centre_voxel
from medical_imaging_platform.localisation.export import (
    LocalisationOutputError,
    inspect_localisation_output,
    validate_localisation_output,
)
from medical_imaging_platform.localisation.fixtures import generate_localisation_fixture
from medical_imaging_platform.localisation.metrics import (
    evaluate_side_metrics,
    mask_centre_voxel,
)
from medical_imaging_platform.localisation.models import BoundingBox, LocalisationConfig
from medical_imaging_platform.localisation.pipeline import (
    load_source_volume,
    localise_adrenal_regions,
)
from medical_imaging_platform.localisation.quality import evaluate_quality
from medical_imaging_platform.localisation.roi import (
    bounding_box_for_centre,
    extract_roi,
    overlay_mid_slice,
    roi_size_voxels,
)
from medical_imaging_platform.registration.fixtures import _write_preprocessed_fixture
from medical_imaging_platform.utils.config import load_localisation_config


def loc_config(tmp_path: Path, **updates: object) -> LocalisationConfig:
    data = {
        "policy_version": "test-localisation-v0.7",
        "default_mode": "atlas",
        "axis_order": "z,y,x",
        "left_relative_centre": (16 / 31, 12 / 31, 10 / 31),
        "right_relative_centre": (16 / 31, 12 / 31, 22 / 31),
        "left_physical_offset_mm": (0.0, 0.0, 0.0),
        "right_physical_offset_mm": (0.0, 0.0, 0.0),
        "roi_size_voxels": (10, 10, 10),
        "roi_size_mm": None,
        "padding_value": -1.0,
        "maximum_padding_fraction": 0.25,
        "maximum_centre_distance_mm": 4.0,
        "minimum_target_coverage": 0.95,
        "minimum_box_iou": 0.05,
        "left_right_minimum_separation_mm": 8.0,
        "confidence_weights": {"padding_fraction": 1.0},
        "quality_status_policy": {},
        "output_directory": tmp_path / "localisation",
        "overwrite": False,
    }
    data.update(updates)
    return LocalisationConfig.model_validate(data)


def fixture(tmp_path: Path) -> Path:
    return generate_localisation_fixture(tmp_path / "fixture", overwrite=True)


def test_fixture_is_deterministic_preprocessed_and_labelled(tmp_path: Path) -> None:
    first = fixture(tmp_path)
    second = generate_localisation_fixture(tmp_path / "fixture2", overwrite=True)

    left = np.load(first / "left_adrenal_mask.npy")
    right = np.load(first / "right_adrenal_mask.npy")
    volume = np.load(first / "volume.npy")
    metadata = json.loads((first / "localisation_fixture_metadata.json").read_text())

    assert np.array_equal(left, np.load(second / "left_adrenal_mask.npy"))
    assert np.array_equal(right, np.load(second / "right_adrenal_mask.npy"))
    assert left.shape == volume.shape == (32, 32, 32)
    assert set(np.unique(left)) <= {0, 1}
    assert set(np.unique(right)) <= {0, 1}
    assert np.count_nonzero(left) > 0
    assert np.count_nonzero(right) > 0
    assert not np.any(left.astype(bool) & right.astype(bool))
    assert metadata["labels_are_clinical"] is False


def test_fixture_translation_and_overwrite_protection(tmp_path: Path) -> None:
    output = generate_localisation_fixture(
        tmp_path / "shifted", translation=(1, 2, -1), overwrite=True
    )
    metadata = json.loads((output / "localisation_fixture_metadata.json").read_text())

    assert metadata["left_centre_voxel"] == [17, 14, 9]
    with pytest.raises(FileExistsError):
        generate_localisation_fixture(output)


def test_config_loader_and_validation(tmp_path: Path) -> None:
    config = load_localisation_config(Path("config/localisation.yaml"))

    assert config.default_mode == "atlas"
    assert config.axis_order == "z,y,x"
    with pytest.raises(ValueError):
        loc_config(tmp_path, roi_size_voxels=(1, 2, 3), roi_size_mm=(5.0, 5.0, 5.0))
    with pytest.raises(ValueError):
        loc_config(
            tmp_path,
            left_relative_centre=(0.1, 0.1, 0.1),
            right_relative_centre=(0.1, 0.1, 0.1),
        )


def test_atlas_prediction_is_spacing_aware(tmp_path: Path) -> None:
    config = loc_config(tmp_path, left_physical_offset_mm=(2.0, 0.0, -2.0))
    centre = predict_centre_voxel(
        "left", volume_shape=(32, 32, 32), spacing_mm_zyx=(2.0, 1.0, 1.0), config=config
    )

    assert centre == (17, 12, 8)
    assert centre_mm(centre, (2.0, 1.0, 1.0)) == (34.0, 12.0, 8.0)


def test_roi_extraction_preserves_dtype_and_reports_padding(tmp_path: Path) -> None:
    volume = np.ones((6, 6, 6), dtype=np.float32)
    config = loc_config(tmp_path, roi_size_voxels=(5, 5, 5))

    box, pads = bounding_box_for_centre((0, 0, 0), (5, 5, 5), (6, 6, 6))
    roi, extraction = extract_roi(
        volume, centre=(0, 0, 0), spacing_mm_zyx=(1.0, 1.0, 1.0), config=config
    )

    assert box == BoundingBox(z=(0, 3), y=(0, 3), x=(0, 3))
    assert pads == ((2, 0), (2, 0), (2, 0))
    assert roi.dtype == volume.dtype
    assert roi.shape == (5, 5, 5)
    assert extraction.padding_fraction > 0
    physical_config = loc_config(tmp_path, roi_size_voxels=None, roi_size_mm=(6.0, 5.0, 4.0))
    assert roi_size_voxels((2.0, 1.0, 1.0), physical_config) == (3, 5, 4)
    with pytest.raises(ValueError):
        extract_roi(volume, centre=(20, 20, 20), spacing_mm_zyx=(1.0, 1.0, 1.0), config=config)


def test_metrics_cover_perfect_missing_empty_partial_and_swap(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    left = np.load(source / "left_adrenal_mask.npy").astype(bool)
    right = np.load(source / "right_adrenal_mask.npy").astype(bool)
    left_centre = mask_centre_voxel(left)
    assert left_centre is not None
    perfect_box = BoundingBox(z=(11, 22), y=(7, 18), x=(5, 16))

    metrics = evaluate_side_metrics(
        "left",
        (16, 12, 10),
        perfect_box,
        (1.0, 1.0, 1.0),
        left_mask=left,
        right_mask=right,
    )
    missing = evaluate_side_metrics(
        "left",
        (16, 12, 10),
        perfect_box,
        (1.0, 1.0, 1.0),
        left_mask=None,
        right_mask=None,
    )
    empty = evaluate_side_metrics(
        "left",
        (16, 12, 10),
        perfect_box,
        (1.0, 1.0, 1.0),
        left_mask=np.zeros_like(left),
        right_mask=right,
    )
    swapped = evaluate_side_metrics(
        "left",
        (16, 12, 22),
        BoundingBox(z=(11, 22), y=(7, 18), x=(17, 28)),
        (1.0, 1.0, 1.0),
        left_mask=left,
        right_mask=right,
    )

    assert metrics.centre_distance_mm == pytest.approx(0.0)
    assert metrics.target_coverage == pytest.approx(1.0)
    assert missing.evaluation_status == "NOT_EVALUATED"
    assert empty.ground_truth_volume_voxels == 0
    assert empty.missed_target is True
    assert swapped.left_right_swap is True


def test_pipeline_exports_reports_and_validates(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    result = localise_adrenal_regions(
        source,
        output_root=tmp_path / "localisation",
        config=loc_config(tmp_path),
        left_mask_path=source / "left_adrenal_mask.npy",
        right_mask_path=source / "right_adrenal_mask.npy",
        overwrite=True,
    )
    output_dir = Path(result.output_paths.output_dir)

    assert result.overall_status == "LOCALISED"
    assert result.ground_truth_available is True
    assert Path(result.output_paths.left_roi).exists()
    assert Path(result.output_paths.right_overlay).exists()
    validated = validate_localisation_output(output_dir)
    assert validated.localisation_run_id == result.localisation_run_id
    assert inspect_localisation_output(output_dir)["status"] == "LOCALISED"
    assert "diagnostic" not in (output_dir / "localisation.json").read_text().lower()

    left_roi = np.load(result.output_paths.left_roi)
    overlay = overlay_mid_slice(
        np.load(source / "volume.npy"), result.left.bounding_box_voxel, (16, 12, 10)
    )
    assert left_roi.shape == (10, 10, 10)
    assert overlay.shape == (32, 32)


def test_pipeline_without_ground_truth_records_not_evaluated(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    result = localise_adrenal_regions(
        source,
        output_root=tmp_path / "localisation",
        config=loc_config(tmp_path),
        overwrite=True,
    )

    assert result.overall_status == "LOCALISED"
    assert result.ground_truth_available is False
    assert result.metrics["left"].evaluation_status == "NOT_EVALUATED"
    assert any(finding.status == "NOT_EVALUATED" for finding in result.quality_findings)


def test_quality_flags_swap_padding_distance_coverage_and_input_rejection(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    config = loc_config(tmp_path, maximum_centre_distance_mm=1.0, minimum_target_coverage=0.9)
    result = localise_adrenal_regions(
        source,
        output_root=tmp_path / "localisation",
        config=config,
        left_mask_path=source / "right_adrenal_mask.npy",
        right_mask_path=source / "left_adrenal_mask.npy",
        overwrite=True,
    )

    assert result.overall_status == "FAILED"
    rule_ids = {finding.rule_id for finding in result.quality_findings}
    assert {"LOC-QC-SWP-001", "LOC-QC-CEN-001", "LOC-QC-COV-001"} <= rule_ids

    padding_result = localise_adrenal_regions(
        source,
        output_root=tmp_path / "padding",
        config=loc_config(
            tmp_path,
            left_relative_centre=(0.0, 0.0, 0.0),
            right_relative_centre=(0.0, 0.1, 0.9),
            maximum_padding_fraction=0.01,
        ),
        overwrite=True,
    )
    assert padding_result.overall_status in {"FAILED", "LOCALISED_WITH_WARNINGS"}
    assert any(finding.rule_id == "LOC-QC-PAD-001" for finding in padding_result.quality_findings)

    left = padding_result.left
    right = padding_result.right
    rejected_status, findings = evaluate_quality(
        left,
        right,
        {"left": padding_result.metrics["left"], "right": padding_result.metrics["right"]},
        spacing_mm_zyx=(1.0, 1.0, 1.0),
        config=loc_config(tmp_path),
        input_findings=[
            padding_result.quality_findings[0].model_copy(
                update={"severity": "CRITICAL", "rule_id": "LOC-QC-GEO-001"}
            )
        ],
    )
    assert rejected_status == "REJECTED"
    assert any(finding.rule_id == "LOC-QC-GEO-001" for finding in findings)


def test_source_validation_rejects_invalid_geometry_and_mask_shape(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    volume, _, _ = load_source_volume(source)
    bad_dir = tmp_path / "bad"
    _write_preprocessed_fixture(bad_dir, volume, run_id="bad-localisation")
    raw = json.loads((bad_dir / "metadata.json").read_text())
    raw["spacing_mm"] = [0.0, 1.0, 1.0]
    (bad_dir / "metadata.json").write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")

    result = localise_adrenal_regions(
        bad_dir,
        output_root=tmp_path / "bad-output",
        config=loc_config(tmp_path),
        overwrite=True,
    )
    assert result.overall_status == "REJECTED"

    wrong_mask = tmp_path / "wrong.npy"
    with wrong_mask.open("wb") as handle:
        np.save(handle, np.ones((2, 2, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        localise_adrenal_regions(
            source,
            output_root=tmp_path / "wrong-output",
            config=loc_config(tmp_path),
            left_mask_path=wrong_mask,
            overwrite=True,
        )


def test_output_validation_detects_checksum_changes(tmp_path: Path) -> None:
    source = fixture(tmp_path)
    result = localise_adrenal_regions(
        source,
        output_root=tmp_path / "localisation",
        config=loc_config(tmp_path),
        overwrite=True,
    )
    with Path(result.output_paths.left_roi).open("wb") as handle:
        np.save(handle, np.zeros((10, 10, 10), dtype=np.float32))

    with pytest.raises(LocalisationOutputError):
        validate_localisation_output(Path(result.output_paths.output_dir))


def test_cli_commands_generate_localise_inspect_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "cli-fixture"
    output_root = tmp_path / "cli-output"

    assert (
        main(["generate-localisation-fixtures", "--output-dir", str(fixture_dir), "--overwrite"])
        == 0
    )
    assert (
        main(
            [
                "localise-adrenal-regions",
                str(fixture_dir),
                "--left-mask",
                str(fixture_dir / "left_adrenal_mask.npy"),
                "--right-mask",
                str(fixture_dir / "right_adrenal_mask.npy"),
                "--output-dir",
                str(output_root),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    run_dir = next(output_root.iterdir())
    assert main(["inspect-localisation", str(run_dir), "--json"]) == 0
    assert main(["validate-localisation", str(run_dir)]) == 0
    assert "Validated localisation" in capsys.readouterr().out


def test_cli_localisation_failure_exit_codes(tmp_path: Path) -> None:
    fixture_dir = fixture(tmp_path)

    assert (
        main(
            [
                "localise-adrenal-regions",
                str(fixture_dir),
                "--left-mask",
                str(fixture_dir / "right_adrenal_mask.npy"),
                "--right-mask",
                str(fixture_dir / "left_adrenal_mask.npy"),
                "--output-dir",
                str(tmp_path / "swapped"),
                "--overwrite",
            ]
        )
        == 2
    )
    assert main(["validate-localisation", str(tmp_path / "missing")]) == 4
    assert main(["generate-localisation-fixtures", "--output-dir", str(fixture_dir)]) == 1
