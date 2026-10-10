import json
from pathlib import Path

import numpy as np
import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.ingestion.fixtures import generate_dicom_fixture_series
from medical_imaging_platform.preprocessing.export import validate_preprocessed_volume
from medical_imaging_platform.preprocessing.models import IntensityProfile, PreprocessingConfig
from medical_imaging_platform.preprocessing.pipeline import preprocess_dicom_series
from medical_imaging_platform.preprocessing_baseline.metrics import compute_image_content_metrics
from medical_imaging_platform.preprocessing_baseline.models import (
    DenoiseConfig,
    EnhancementConfig,
    MetricSettings,
    PreprocessingBaselineConfig,
)
from medical_imaging_platform.preprocessing_baseline.runner import run_preprocessing_baseline
from medical_imaging_platform.preprocessing_baseline.transforms import apply_baseline_transforms
from medical_imaging_platform.quality_control.models import QualityControlConfig
from medical_imaging_platform.utils.config import load_preprocessing_baseline_config


def preprocessing_config(tmp_path: Path) -> PreprocessingConfig:
    return PreprocessingConfig(
        policy_version="test-preprocessing-v0.5",
        axis_order="zyx",
        require_quality_report=True,
        allow_quality_override=True,
        pixel_dtype="float32",
        missing_rescale_slope_default=1.0,
        missing_rescale_intercept_default=0.0,
        intensity_profiles={"none": IntensityProfile(lower=None, upper=None)},
        default_intensity_profile="none",
        normalisation_mode="none",
        crop_mode="none",
        centre_crop_shape=(2, 6, 6),
        fixed_crop_bounds=None,
        minimum_output_shape=None,
        padding_value=0.0,
        output_format="npy",
        output_directory=tmp_path / "processed",
        overwrite=False,
        background_threshold=0.0,
        spacing_irregularity_tolerance_mm=0.01,
    )


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


def baseline_config(tmp_path: Path, **updates: object) -> PreprocessingBaselineConfig:
    data = {
        "policy_version": "test-m19-baseline",
        "baseline_name": "test-baseline",
        "denoise": {"enabled": True, "method": "mean", "kernel_size": 3},
        "enhancement": {
            "enabled": True,
            "method": "percentile_stretch",
            "lower_percentile": 1.0,
            "upper_percentile": 99.0,
            "histogram_bins": 64,
            "clahe_tiles": (2, 2),
            "clahe_clip_limit": 0.05,
        },
        "metrics": {
            "percentiles": (1.0, 50.0, 99.0),
            "histogram_bins": 64,
            "edge_percentile": 90.0,
            "local_contrast_kernel_size": 3,
        },
        "thresholds": {
            "minimum_sharpness": None,
            "minimum_edge_density": None,
            "maximum_noise_estimate": None,
        },
        "preview": {"enabled": True, "axial_slice_index": None},
        "output_directory": tmp_path / "baseline",
        "overwrite": False,
    }
    data.update(updates)
    return PreprocessingBaselineConfig.model_validate(data)


def preprocessed_output(tmp_path: Path) -> Path:
    source = tmp_path / "dicom"
    generate_dicom_fixture_series(source, slice_count=3)
    result = preprocess_dicom_series(
        source,
        output_root=tmp_path / "processed",
        preprocessing_config=preprocessing_config(tmp_path),
        qc_config=qc_config(tmp_path),
        dicom_max_files=20,
        max_file_size_bytes=1_000_000,
        overwrite=True,
    )
    return Path(result.output_paths.output_dir)


def test_metrics_cover_constant_noisy_blurred_edge_and_invalid_inputs() -> None:
    settings = MetricSettings(percentiles=(0.0, 50.0, 100.0), histogram_bins=32)
    constant = np.ones((8, 8), dtype=np.float32)
    edge_rich = np.indices((8, 8)).sum(axis=0).astype(np.float32) % 2
    rng = np.random.default_rng(7)
    noisy = edge_rich + rng.normal(0.0, 0.2, size=edge_rich.shape).astype(np.float32)
    blurred = np.full((8, 8), 0.5, dtype=np.float32)
    blurred[2:6, 2:6] = 0.6

    constant_metrics = compute_image_content_metrics(constant, settings)
    first = compute_image_content_metrics(noisy, settings)
    second = compute_image_content_metrics(noisy, settings)
    edge_metrics = compute_image_content_metrics(edge_rich, settings)
    blurred_metrics = compute_image_content_metrics(blurred, settings)

    assert constant_metrics.entropy == 0.0
    assert first == second
    assert (
        first.noise_high_frequency_residual_std > blurred_metrics.noise_high_frequency_residual_std
    )
    assert edge_metrics.sharpness_laplacian_variance > blurred_metrics.sharpness_laplacian_variance
    assert edge_metrics.edge_density > 0.0
    with pytest.raises(ValueError, match="finite"):
        compute_image_content_metrics(np.array([[np.nan]], dtype=np.float32), settings)
    with pytest.raises(ValueError, match="2D slice or 3D volume"):
        compute_image_content_metrics(np.zeros((1, 1, 1, 1), dtype=np.float32), settings)


def test_transforms_noop_denoise_enhancement_dtype_and_config_validation(tmp_path: Path) -> None:
    volume = np.zeros((3, 8, 8), dtype=np.float32)
    volume[:, 2:6, 2:6] = 4.0
    volume[:, 4, 4] = 10.0
    noop, chain, _ = apply_baseline_transforms(
        volume,
        baseline_config(
            tmp_path,
            denoise=DenoiseConfig(enabled=False),
            enhancement=EnhancementConfig(enabled=False),
        ),
    )
    denoised, denoise_chain, _ = apply_baseline_transforms(
        volume,
        baseline_config(tmp_path, enhancement=EnhancementConfig(enabled=False)),
    )
    enhanced, enhance_chain, _ = apply_baseline_transforms(
        volume,
        baseline_config(
            tmp_path,
            denoise=DenoiseConfig(enabled=False),
            enhancement=EnhancementConfig(enabled=True, method="histogram_equalization"),
        ),
    )
    clahe, clahe_chain, _ = apply_baseline_transforms(
        volume,
        baseline_config(
            tmp_path,
            denoise=DenoiseConfig(enabled=False),
            enhancement=EnhancementConfig(
                enabled=True, method="clahe", clahe_tiles=(2, 2), histogram_bins=32
            ),
        ),
    )

    assert np.array_equal(noop, volume)
    assert chain[0].name == "noop"
    assert denoised.dtype == np.float32
    assert denoised[1, 4, 4] < volume[1, 4, 4]
    assert denoise_chain[-1].name == "mean_denoise"
    assert enhanced.min() >= volume.min()
    assert enhanced.max() <= volume.max()
    assert enhance_chain[-1].name == "histogram_equalization"
    assert clahe.shape == volume.shape
    assert clahe_chain[-1].name == "clahe"
    with pytest.raises(ValueError, match="odd"):
        DenoiseConfig(enabled=True, kernel_size=4)
    with pytest.raises(ValueError, match="less than"):
        EnhancementConfig(enabled=True, lower_percentile=90.0, upper_percentile=10.0)


def test_runner_validates_sources_preserves_source_and_writes_evidence(tmp_path: Path) -> None:
    source_dir = preprocessed_output(tmp_path)
    source_before = (source_dir / "volume.npy").read_bytes()
    config = baseline_config(tmp_path)

    result = run_preprocessing_baseline(
        [source_dir], output_root=tmp_path / "baseline", config=config
    )
    output_dir = Path(result.output_paths.output_dir)
    summary = json.loads((output_dir / "before_after_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "evidence" / "checksum_manifest.json").read_text())

    assert (source_dir / "volume.npy").read_bytes() == source_before
    assert (
        validate_preprocessed_volume(source_dir).run_id
        == result.sources[0].source_preprocessing_run_id
    )
    assert result.sources[0].source_metadata_checksum
    assert (output_dir / "preprocessing_baseline_metrics.json").exists()
    assert (output_dir / "preprocessing_baseline_report.md").exists()
    assert Path(result.sources[0].transformed_volume).exists()
    assert Path(result.sources[0].preview_before or "").exists()
    assert "report" in manifest
    assert summary["monai_comparison"]["label"] == "model-time augmentation baseline"

    with pytest.raises(Exception, match="already exists"):
        run_preprocessing_baseline([source_dir], output_root=tmp_path / "baseline", config=config)
    repeated = run_preprocessing_baseline(
        [source_dir], output_root=tmp_path / "baseline", config=config, overwrite=True
    )
    assert repeated.model_dump(mode="json") == result.model_dump(mode="json")
    with pytest.raises(Exception, match="missing required artefacts"):
        run_preprocessing_baseline(
            [tmp_path / "missing"], output_root=tmp_path / "bad", config=config
        )


def test_cli_success_json_invalid_source_overwrite_and_preprocessing_regression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source_dir = preprocessed_output(tmp_path)
    assert main(["validate-preprocessed-volume", str(source_dir)]) == 0
    capsys.readouterr()

    assert (
        main(
            [
                "run-preprocessing-baseline",
                str(source_dir),
                "--output-dir",
                str(tmp_path / "cli-baseline"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_count"] == 1
    assert payload["sources"][0]["source_preprocessing_run_id"]

    assert (
        main(
            [
                "run-preprocessing-baseline",
                str(source_dir),
                "--output-dir",
                str(tmp_path / "cli-baseline"),
            ]
        )
        == 4
    )
    assert "already exists" in capsys.readouterr().out
    assert (
        main(
            [
                "run-preprocessing-baseline",
                str(tmp_path / "missing"),
                "--output-dir",
                str(tmp_path / "invalid"),
            ]
        )
        == 4
    )
    assert "missing required artefacts" in capsys.readouterr().out


def test_default_config_loader() -> None:
    config = load_preprocessing_baseline_config(Path("config/preprocessing_baseline.yaml"))
    assert config.policy_version == "m19-preprocessing-baseline-v1"
    assert config.denoise.kernel_size == 3
