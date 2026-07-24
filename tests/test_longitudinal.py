import json
from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.longitudinal.change import calculate_changes
from medical_imaging_platform.longitudinal.export import (
    LongitudinalOutputError,
    inspect_longitudinal_analysis,
    validate_longitudinal_analysis,
)
from medical_imaging_platform.longitudinal.matching import match_lesions
from medical_imaging_platform.longitudinal.measurements import (
    LongitudinalMeasurementError,
    axial_maximum_diameter,
    extract_components,
    maximum_3d_diameter,
    measure_components,
    validate_binary_mask,
)
from medical_imaging_platform.longitudinal.models import LongitudinalConfig
from medical_imaging_platform.longitudinal.pipeline import (
    LongitudinalAnalysisError,
    analyse_longitudinal_pair,
    load_upstream_statuses,
)
from medical_imaging_platform.longitudinal.quality import (
    evaluate_quality,
    forced_indeterminate_reasons,
)
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset
from medical_imaging_platform.utils.config import load_longitudinal_config


def lng_config(tmp_path: Path, **updates: object) -> LongitudinalConfig:
    data = {
        "policy_version": "test-longitudinal-v1",
        "minimum_component_voxels": 1,
        "maximum_match_distance_mm": 8.0,
        "minimum_match_iou": 0.0,
        "minimum_match_dice": 0.0,
        "centroid_weight": 1.0,
        "overlap_weight": 1.0,
        "volume_increase_threshold_percent": 25.0,
        "volume_reduction_threshold_percent": 25.0,
        "diameter_increase_threshold_percent": 10.0,
        "diameter_reduction_threshold_percent": 10.0,
        "small_denominator_mm3": 1.0,
        "require_registration_pass": True,
        "require_segmentation_pass": True,
        "classification_abstention_forces_indeterminate": True,
        "allow_multiple_components": True,
        "output_directory": tmp_path / "longitudinal",
        "overwrite": False,
    }
    data.update(updates)
    return LongitudinalConfig.model_validate(data)


def mask(points: list[tuple[int, int, int]], shape: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    array = np.zeros(shape, dtype=np.uint8)
    for point in points:
        array[point] = 1
    return array


def cube(
    start: tuple[int, int, int], size: int, shape: tuple[int, int, int] = (12, 12, 12)
) -> np.ndarray:
    array = np.zeros(shape, dtype=np.uint8)
    z, y, x = start
    array[z : z + size, y : y + size, x : x + size] = 1
    return array


def write_mask(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, array.astype(np.uint8))
    return path


def synthetic_dataset(tmp_path: Path) -> Path:
    config = load_synthetic_config(Path("config/data.yaml")).model_copy(
        update={"output_root": tmp_path / "synthetic", "dataset_size": 6}
    )
    generate_dataset(config, tmp_path / "synthetic", overwrite=True)
    return tmp_path / "synthetic"


def run_pair(
    tmp_path: Path,
    previous: np.ndarray,
    current: np.ndarray,
    **updates: object,
) -> dict[str, object]:
    config = lng_config(tmp_path, **updates.pop("config_updates", {}))
    return analyse_longitudinal_pair(
        previous_mask_path=write_mask(tmp_path / "prev.npy", previous),
        current_mask_path=write_mask(tmp_path / "curr.npy", current),
        previous_spacing_mm=updates.pop("previous_spacing_mm", (2.0, 1.0, 1.0)),
        current_spacing_mm=updates.pop("current_spacing_mm", (2.0, 1.0, 1.0)),
        case_id=str(updates.pop("case_id", "case-001")),
        research_subject_id=str(updates.pop("research_subject_id", "subject-001")),
        side=str(updates.pop("side", "left")),
        previous_timepoint=str(updates.pop("previous_timepoint", "previous")),
        current_timepoint=str(updates.pop("current_timepoint", "current")),
        output_root=tmp_path / "out",
        config=config,
        registration_run_id=updates.pop("registration_run_id", "registration-pass"),  # type: ignore[arg-type]
        upstream_quality_statuses=updates.pop(
            "upstream_quality_statuses", {"registration": "PASS", "segmentation": "PASS"}
        ),  # type: ignore[arg-type]
        overwrite=True,
    )


def test_config_loader_and_validation(tmp_path: Path) -> None:
    config = load_longitudinal_config(Path("config/longitudinal.yaml"))

    assert config.policy_version == "m10-longitudinal-v1"
    with pytest.raises(ValueError):
        lng_config(tmp_path, centroid_weight=0.0, overlap_weight=0.0)
    with pytest.raises(ValueError):
        lng_config(tmp_path, minimum_match_iou=2.0)


def test_measurements_are_spacing_aware_and_binary_validated() -> None:
    array = mask([(1, 1, 1), (1, 1, 2), (1, 2, 1)])
    measurement = measure_components(
        array, (2.0, 1.0, 3.0), timepoint="previous", minimum_component_voxels=1
    )[0]

    assert measurement.voxel_count == 3
    assert measurement.physical_volume_mm3 == pytest.approx(18.0)
    assert measurement.physical_volume_ml == pytest.approx(0.018)
    assert measurement.bounding_box_dimensions_mm == pytest.approx((2.0, 2.0, 6.0))
    assert measurement.centroid_voxel == pytest.approx((1.0, 4 / 3, 4 / 3))
    assert measurement.centroid_physical_mm == pytest.approx((2.0, 4 / 3, 4.0))
    assert measurement.maximum_3d_diameter_mm == pytest.approx(np.sqrt(10.0))
    assert measurement.axial_maximum_diameter_mm == pytest.approx(np.sqrt(10.0))
    with pytest.raises(LongitudinalMeasurementError):
        validate_binary_mask(np.asarray([0, 2], dtype=np.uint8))
    with pytest.raises(LongitudinalMeasurementError):
        measure_components(array, (0.0, 1.0, 1.0), timepoint="current", minimum_component_voxels=1)


def test_component_extraction_multiple_components_and_empty_mask(tmp_path: Path) -> None:
    array = mask([(1, 1, 1), (6, 6, 6)])
    components = extract_components(array, minimum_component_voxels=1)
    empty = measure_components(
        np.zeros((4, 4, 4), dtype=np.uint8),
        (1.0, 1.0, 1.0),
        timepoint="current",
        minimum_component_voxels=1,
    )[0]

    assert len(components) == 2
    assert empty.empty_mask is True
    assert empty.connected_component_count == 0
    assert maximum_3d_diameter(np.asarray([[0, 0, 0]]), (1.0, 1.0, 1.0)) == 0.0
    assert axial_maximum_diameter(np.asarray([[0, 0, 0]]), (1.0, 1.0, 1.0)) == 0.0


def test_matching_is_deterministic_and_detects_ambiguity(tmp_path: Path) -> None:
    previous = mask([(3, 3, 3)])
    current = mask([(3, 2, 3), (3, 4, 3)])
    config = lng_config(tmp_path, maximum_match_distance_mm=3.0, overlap_weight=0.0)
    prev_meas = measure_components(
        previous, (1.0, 1.0, 1.0), timepoint="previous", minimum_component_voxels=1
    )
    curr_meas = measure_components(
        current, (1.0, 1.0, 1.0), timepoint="current", minimum_component_voxels=1
    )
    matches = match_lesions(previous, current, prev_meas, curr_meas, config)

    assert matches[0].status == "ambiguous"
    assert matches[0].ambiguous is True
    assert [match.match_id for match in matches] == sorted(match.match_id for match in matches)


def test_change_labels_cover_stable_increased_reduced_new_resolved_and_small_denominator(
    tmp_path: Path,
) -> None:
    assert run_pair(tmp_path / "stable", cube((3, 3, 3), 2), cube((3, 3, 3), 2))["summary"][
        "engineering_labels"
    ] == ["stable"]
    assert run_pair(tmp_path / "increased", cube((3, 3, 3), 2), cube((2, 2, 2), 4))["summary"][
        "engineering_labels"
    ] == ["increased"]
    assert run_pair(tmp_path / "reduced", cube((2, 2, 2), 4), cube((3, 3, 3), 2))["summary"][
        "engineering_labels"
    ] == ["reduced"]
    assert run_pair(tmp_path / "new", np.zeros((12, 12, 12), dtype=np.uint8), cube((3, 3, 3), 2))[
        "summary"
    ]["engineering_labels"] == ["new"]
    assert run_pair(
        tmp_path / "resolved", cube((3, 3, 3), 2), np.zeros((12, 12, 12), dtype=np.uint8)
    )["summary"]["engineering_labels"] == ["resolved"]

    config = lng_config(tmp_path, small_denominator_mm3=100.0)
    prev = measure_components(
        mask([(1, 1, 1)]), (1.0, 1.0, 1.0), timepoint="previous", minimum_component_voxels=1
    )
    curr = measure_components(
        mask([(1, 1, 1)]), (1.0, 1.0, 1.0), timepoint="current", minimum_component_voxels=1
    )
    matches = match_lesions(mask([(1, 1, 1)]), mask([(1, 1, 1)]), prev, curr, config)
    changes = calculate_changes(matches, prev, curr, config)
    assert changes[0].percentage_volume_change is None


def test_upstream_quality_and_abstention_force_indeterminate(tmp_path: Path) -> None:
    payload = run_pair(
        tmp_path,
        cube((3, 3, 3), 2),
        cube((3, 3, 3), 2),
        upstream_quality_statuses={
            "registration": "FAIL",
            "segmentation": "PASS",
            "classification_abstention": "ABSTAINED",
        },
    )

    assert payload["status"] == "FAIL"
    assert payload["summary"]["engineering_labels"] == ["indeterminate"]
    pair = validate_longitudinal_analysis(
        tmp_path / "out" / "longitudinal-case-001-left-test-longitudinal-v1"
    )["pair_manifest"]
    reasons = forced_indeterminate_reasons(
        type("Pair", (), {"upstream_quality_statuses": pair["upstream_quality_statuses"]})(),
        lng_config(tmp_path),
        True,
    )
    assert "registration quality did not pass" in reasons
    assert "classification abstention propagated" in reasons


def test_forced_indeterminate_without_matches_and_upstream_status_loading(tmp_path: Path) -> None:
    config = lng_config(tmp_path)
    changes = calculate_changes([], [], [], config, force_indeterminate_reasons=["blocked"])
    status_path = tmp_path / "upstream.json"
    status_path.write_text('{"registration": "PASS_WITH_WARNINGS"}\n', encoding="utf-8")
    bad_status_path = tmp_path / "bad-upstream.json"
    bad_status_path.write_text('["not", "an", "object"]\n', encoding="utf-8")

    assert changes[0].label == "indeterminate"
    assert load_upstream_statuses(status_path) == {"registration": "PASS_WITH_WARNINGS"}
    assert load_upstream_statuses(None) is None
    with pytest.raises(LongitudinalAnalysisError):
        load_upstream_statuses(bad_status_path)


def test_geometry_mismatch_with_registration_is_explicit_indeterminate(tmp_path: Path) -> None:
    payload = run_pair(
        tmp_path,
        cube((2, 2, 2), 2, shape=(8, 8, 8)),
        cube((2, 2, 2), 2, shape=(9, 9, 9)),
        registration_run_id="explicit-registered-space",
    )

    assert payload["status"] == "FAIL"
    assert payload["summary"]["engineering_labels"] == ["indeterminate"]
    assert any(
        finding["rule_id"] == "LNG-QC-GEO-001" and finding["status"] == "FAIL"
        for finding in payload["quality_findings"]
    )


def test_rejections_for_side_time_geometry_and_identifier(tmp_path: Path) -> None:
    with pytest.raises(LongitudinalAnalysisError):
        run_pair(tmp_path / "side", cube((2, 2, 2), 2), cube((2, 2, 2), 2), side="middle")
    with pytest.raises(LongitudinalAnalysisError):
        run_pair(
            tmp_path / "time",
            cube((2, 2, 2), 2),
            cube((2, 2, 2), 2),
            previous_timepoint="current",
            current_timepoint="previous",
        )
    with pytest.raises(LongitudinalAnalysisError):
        run_pair(
            tmp_path / "geo",
            cube((2, 2, 2), 2, shape=(8, 8, 8)),
            cube((2, 2, 2), 2, shape=(9, 9, 9)),
            registration_run_id=None,
        )
    with pytest.raises(LongitudinalAnalysisError):
        run_pair(
            tmp_path / "phi",
            cube((2, 2, 2), 2),
            cube((2, 2, 2), 2),
            case_id="patient-name",
        )


def test_quality_gates_export_checksums_overwrite_and_review_arrays(tmp_path: Path) -> None:
    payload = run_pair(tmp_path, cube((3, 3, 3), 2), cube((3, 3, 3), 2))
    output_dir = tmp_path / "out" / "longitudinal-case-001-left-test-longitudinal-v1"
    manifest = validate_longitudinal_analysis(output_dir)
    inspected = inspect_longitudinal_analysis(output_dir)
    previous_review = np.load(output_dir / "review_arrays" / "previous_mid_slice.npy")

    assert manifest["status"] == "PASS"
    assert inspected["engineering_labels"] == ["stable"]
    assert previous_review.ndim == 2
    assert (output_dir / "longitudinal_report.md").exists()
    with pytest.raises(FileExistsError):
        analyse_longitudinal_pair(
            previous_mask_path=write_mask(tmp_path / "prev2.npy", cube((3, 3, 3), 2)),
            current_mask_path=write_mask(tmp_path / "curr2.npy", cube((3, 3, 3), 2)),
            previous_spacing_mm=(2.0, 1.0, 1.0),
            current_spacing_mm=(2.0, 1.0, 1.0),
            case_id="case-001",
            research_subject_id="subject-001",
            side="left",
            previous_timepoint="previous",
            current_timepoint="current",
            output_root=tmp_path / "out",
            config=lng_config(tmp_path),
            registration_run_id="registration-pass",
            overwrite=False,
        )
    (output_dir / "longitudinal_report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(LongitudinalOutputError):
        validate_longitudinal_analysis(output_dir)
    assert payload["checksums"]["longitudinal_report"]
    with pytest.raises(LongitudinalOutputError):
        validate_longitudinal_analysis(tmp_path / "missing-analysis")


def test_quality_gate_rejects_temporal_and_side_flags(tmp_path: Path) -> None:
    payload = run_pair(tmp_path, cube((3, 3, 3), 2), cube((3, 3, 3), 2))
    manifest = validate_longitudinal_analysis(
        tmp_path / "out" / "longitudinal-case-001-left-test-longitudinal-v1"
    )
    status, findings = evaluate_quality(
        config=lng_config(tmp_path),
        pair=type(
            "Pair",
            (),
            {
                "source_checksums": {"a": "b"},
                "upstream_quality_statuses": {"registration": "PASS", "segmentation": "PASS"},
            },
        )(),
        previous_measurements=[],
        current_measurements=[],
        changes=[],
        output_paths=[tmp_path / "missing"],
        side_consistent=False,
        temporal_valid=False,
        geometry_compatible=False,
    )

    assert payload["status"] == "PASS"
    assert manifest["summary"]["engineering_labels"] == ["stable"]
    assert status == "REJECTED"
    assert {"LNG-QC-TIME-001", "LNG-QC-SIDE-001", "LNG-QC-CHK-001"} <= {
        finding.rule_id for finding in findings
    }


def test_cli_commands_and_synthetic_scenarios(tmp_path: Path) -> None:
    dataset = synthetic_dataset(tmp_path)
    output_root = tmp_path / "cli-out"

    assert (
        main(
            [
                "analyse-longitudinal-pair",
                "--previous-mask",
                str(dataset / "synthetic-case-0002" / "previous_lesion_mask.npy"),
                "--current-mask",
                str(dataset / "synthetic-case-0002" / "current_lesion_mask.npy"),
                "--previous-spacing",
                "2.5",
                "2.5",
                "2.5",
                "--current-spacing",
                "2.5",
                "2.5",
                "2.5",
                "--case-id",
                "synthetic-case-0002",
                "--research-subject-id",
                "research-subject-0002",
                "--side",
                "right",
                "--registration-run-id",
                "registration-pass",
                "--output-dir",
                str(output_root),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    analysis_dir = output_root / "longitudinal-synthetic-case-0002-right-m10-longitudinal-v1"
    assert main(["inspect-longitudinal-analysis", str(analysis_dir), "--json"]) == 0
    assert main(["validate-longitudinal-analysis", str(analysis_dir)]) == 0
    assert json.loads((analysis_dir / "longitudinal_summary.json").read_text())[
        "engineering_labels"
    ] == ["increased"]
    assert main(["validate-longitudinal-analysis", str(tmp_path / "missing")]) == 4
    assert (
        main(
            [
                "analyse-longitudinal-pair",
                "--previous-mask",
                str(dataset / "synthetic-case-0001" / "previous_lesion_mask.npy"),
                "--current-mask",
                str(dataset / "synthetic-case-0001" / "current_lesion_mask.npy"),
                "--previous-spacing",
                "2.5",
                "2.5",
                "2.5",
                "--current-spacing",
                "2.5",
                "2.5",
                "2.5",
                "--case-id",
                "synthetic-case-0001",
                "--research-subject-id",
                "research-subject-0001",
                "--side",
                "left",
                "--previous-timepoint",
                "current",
                "--current-timepoint",
                "previous",
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
        == 3
    )
