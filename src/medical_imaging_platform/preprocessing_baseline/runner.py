"""Runner for M19 preprocessing baseline evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from medical_imaging_platform.preprocessing.export import validate_preprocessed_volume
from medical_imaging_platform.preprocessing_baseline.export import (
    build_output_paths,
    preview_slice,
    write_preprocessing_baseline_outputs,
)
from medical_imaging_platform.preprocessing_baseline.metrics import (
    compute_image_content_metrics,
    metric_deltas,
)
from medical_imaging_platform.preprocessing_baseline.models import (
    BaselineSourceEvidence,
    ImageContentMetrics,
    PreprocessingBaselineConfig,
    PreprocessingBaselineResult,
)
from medical_imaging_platform.preprocessing_baseline.transforms import apply_baseline_transforms
from medical_imaging_platform.synthetic.manifest import sha256_file


def run_preprocessing_baseline(
    source_dirs: list[Path],
    *,
    output_root: Path,
    config: PreprocessingBaselineConfig,
    overwrite: bool | None = None,
) -> PreprocessingBaselineResult:
    """Run deterministic classical baseline evidence on validated preprocessing outputs."""
    if not source_dirs:
        raise ValueError("At least one preprocessed output directory is required.")
    validated_sources = [validate_preprocessed_volume(path) for path in source_dirs]
    baseline_run_id = _baseline_run_id(
        [source.run_id for source in validated_sources], config.policy_version
    )
    output_paths = build_output_paths(output_root, baseline_run_id)
    evidence_dir = Path(output_paths.output_dir) / "evidence"
    source_evidence: list[BaselineSourceEvidence] = []
    transformed_volumes: dict[str, np.ndarray] = {}
    preview_arrays: dict[str, np.ndarray] = {}
    run_warnings: list[str] = []

    for source in validated_sources:
        source_dir = Path(source.output_paths.output_dir)
        volume = np.load(source_dir / "volume.npy")
        before = compute_image_content_metrics(volume, config.metrics)
        transformed, transform_chain, warnings = apply_baseline_transforms(volume, config)
        after = compute_image_content_metrics(transformed, config.metrics)
        source_warnings = warnings + _threshold_warnings(config, after)
        run_warnings.extend(source_warnings)
        source_key = source.run_id
        transformed_path = evidence_dir / f"{source_key}_baseline_volume.npy"
        preview_before_path = (
            evidence_dir / f"{source_key}_before_mid_axial.png" if config.preview.enabled else None
        )
        preview_after_path = (
            evidence_dir / f"{source_key}_after_mid_axial.png" if config.preview.enabled else None
        )
        transformed_volumes[f"{source_key}_transformed_volume"] = transformed
        if (
            config.preview.enabled
            and preview_before_path is not None
            and preview_after_path is not None
        ):
            preview_arrays[f"{source_key}_preview_before"] = preview_slice(
                volume, config.preview.axial_slice_index
            )
            preview_arrays[f"{source_key}_preview_after"] = preview_slice(
                transformed, config.preview.axial_slice_index
            )
        source_evidence.append(
            BaselineSourceEvidence(
                source_output_dir=str(source_dir),
                source_preprocessing_run_id=source.run_id,
                source_metadata_checksum=sha256_file(source_dir / "metadata.json"),
                source_volume_checksum=source.checksums.get("volume"),
                source_report_checksum=source.checksums.get("report"),
                volume_shape=source.volume_shape,
                spacing_mm=source.spacing_mm,
                before_metrics=before,
                after_metrics=after,
                metric_deltas=metric_deltas(before, after),
                transform_chain=transform_chain,
                transformed_volume=str(transformed_path),
                preview_before=str(preview_before_path)
                if preview_before_path is not None
                else None,
                preview_after=str(preview_after_path) if preview_after_path is not None else None,
                warnings=sorted(set(source_warnings)),
            )
        )

    result = PreprocessingBaselineResult(
        baseline_run_id=baseline_run_id,
        policy_version=config.policy_version,
        baseline_name=config.baseline_name,
        generated_at="2026-01-01T00:00:00+00:00",
        source_count=len(source_evidence),
        sources=source_evidence,
        monai_comparison=_monai_comparison(),
        output_paths=output_paths,
        checksums={},
        warnings=sorted(set(run_warnings)),
        limitations=_limitations(),
    )
    return write_preprocessing_baseline_outputs(
        result,
        transformed_volumes,
        preview_arrays,
        overwrite=config.overwrite if overwrite is None else overwrite,
    )


def _baseline_run_id(source_run_ids: list[str], policy_version: str) -> str:
    digest = hashlib.sha256(
        "|".join([policy_version, *sorted(source_run_ids)]).encode("utf-8")
    ).hexdigest()[:16]
    return f"preprocessing-baseline-{digest}"


def _threshold_warnings(
    config: PreprocessingBaselineConfig, metrics: ImageContentMetrics
) -> list[str]:
    warnings: list[str] = []
    if (
        config.thresholds.minimum_sharpness is not None
        and metrics.sharpness_laplacian_variance < config.thresholds.minimum_sharpness
    ):
        warnings.append("Sharpness is below configured engineering threshold.")
    if (
        config.thresholds.minimum_edge_density is not None
        and metrics.edge_density < config.thresholds.minimum_edge_density
    ):
        warnings.append("Edge density is below configured engineering threshold.")
    if (
        config.thresholds.maximum_noise_estimate is not None
        and metrics.noise_high_frequency_residual_std > config.thresholds.maximum_noise_estimate
    ):
        warnings.append("Noise estimate is above configured engineering threshold.")
    return warnings


def _monai_comparison() -> dict[str, object]:
    return {
        "label": "model-time augmentation baseline",
        "segmentation_transforms": ["EnsureBinaryMaskd", "RandFlipd", "RandGaussianNoised"],
        "classification_transforms": ["EnsureFloatd", "RandFlipd", "RandGaussianNoised"],
        "relationship": (
            "M19 classical baseline operates on existing preprocessed NumPy volumes. Current MONAI "
            "transforms are training-time augmentation and are not equivalent preprocessing paths."
        ),
        "model_training_invoked": False,
    }


def _limitations() -> list[str]:
    return [
        "Engineering-only image-content metrics; not diagnostic image-quality assessment.",
        "Operates only on existing validated preprocessing outputs and never changes canonical "
        "volumes.",
        "Classical transforms are deterministic baselines, not clinical reconstruction or "
        "enhancement.",
        "MONAI comparison is descriptive only and does not invoke model training.",
    ]
