"""Registration pipeline orchestration."""

from __future__ import annotations

import hashlib
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.affine import run_affine_registration
from medical_imaging_platform.registration.centre_of_mass import centre_of_mass_transform
from medical_imaging_platform.registration.conversion import numpy_to_sitk, validate_round_trip
from medical_imaging_platform.registration.export import (
    build_output_paths,
    write_registration_outputs,
)
from medical_imaging_platform.registration.metrics import compute_metrics
from medical_imaging_platform.registration.models import (
    OptimiserSummary,
    RegistrationConfig,
    RegistrationResult,
    TransformSummary,
)
from medical_imaging_platform.registration.preconditions import (
    has_rejection,
    role_metadata,
    validate_registration_inputs,
)
from medical_imaging_platform.registration.quality import evaluate_registration_quality
from medical_imaging_platform.registration.resampling import resample_moving_to_fixed
from medical_imaging_platform.registration.rigid import run_rigid_registration
from medical_imaging_platform.registration.visualisation import mid_axial_slices


def register_preprocessed_volumes(
    fixed_dir: Path,
    moving_dir: Path,
    *,
    output_root: Path,
    config: RegistrationConfig,
    mode: str | None = None,
    fixed_temporal_label: str | None = None,
    moving_temporal_label: str | None = None,
    overwrite: bool | None = None,
) -> RegistrationResult:
    """Run deterministic registration between explicit fixed and moving inputs."""
    start = time.perf_counter()
    selected_mode = mode or config.default_mode
    fixed_volume, fixed_meta, moving_volume, moving_meta, findings = validate_registration_inputs(
        fixed_dir,
        moving_dir,
        allow_constant_volume=config.allow_constant_volume,
        fixed_temporal_label=fixed_temporal_label,
        moving_temporal_label=moving_temporal_label,
    )
    fixed_role = role_metadata("fixed", fixed_dir, fixed_meta, temporal_label=fixed_temporal_label)
    moving_role = role_metadata(
        "moving", moving_dir, moving_meta, temporal_label=moving_temporal_label
    )
    if has_rejection(findings):
        raise ValueError("Registration input rejected before optimisation.")
    validate_round_trip(fixed_volume, fixed_meta)
    validate_round_trip(moving_volume, moving_meta)
    fixed_image = numpy_to_sitk(fixed_volume, fixed_meta)
    moving_image = numpy_to_sitk(moving_volume, moving_meta)
    metrics_before = compute_metrics(
        fixed_volume,
        moving_volume,
        spacing_zyx=fixed_meta.spacing_mm,
        foreground_threshold=config.foreground_threshold,
    )

    initial_transform, moving_to_fixed_mm, moving_to_fixed_voxels = centre_of_mass_transform(
        fixed_volume,
        moving_volume,
        spacing_zyx=fixed_meta.spacing_mm,
        threshold=config.foreground_threshold,
    )
    optimiser_summaries: list[OptimiserSummary] = []
    final_transform: sitk.Transform = initial_transform
    if selected_mode == "centre_of_mass":
        optimiser_summaries.append(
            OptimiserSummary(
                stage="centre_of_mass",
                attempted=True,
                succeeded=True,
                stop_condition="deterministic centre-of-mass translation",
                iterations=0,
                metric_value=metrics_before.mean_squared_error,
            )
        )
    elif selected_mode == "rigid":
        final_transform, rigid_summary = run_rigid_registration(
            fixed_image, moving_image, initial_transform, config
        )
        optimiser_summaries.append(rigid_summary)
    elif selected_mode == "rigid_then_affine":
        rigid_transform, rigid_summary = run_rigid_registration(
            fixed_image, moving_image, initial_transform, config
        )
        optimiser_summaries.append(rigid_summary)
        final_transform, affine_summary = run_affine_registration(
            fixed_image, moving_image, rigid_transform, config
        )
        optimiser_summaries.append(affine_summary)
    else:
        raise ValueError(f"Unsupported registration mode: {selected_mode}")

    registered = resample_moving_to_fixed(
        moving_image,
        fixed_image,
        final_transform,
        interpolator=config.interpolator,
        default_pixel_value=config.default_pixel_value,
    )
    metrics_after = compute_metrics(
        fixed_volume,
        registered,
        spacing_zyx=fixed_meta.spacing_mm,
        foreground_threshold=config.foreground_threshold,
    )
    transform_summary = summarise_transform(
        final_transform,
        spacing_zyx=fixed_meta.spacing_mm,
        fallback_translation_mm=moving_to_fixed_mm,
        fallback_translation_voxels=moving_to_fixed_voxels,
    )
    status, findings = evaluate_registration_quality(
        transform_summary,
        metrics_before,
        metrics_after,
        registered,
        default_pixel_value=config.default_pixel_value,
        config=config,
        precondition_findings=findings,
    )
    run_id = _run_id(fixed_meta.run_id, moving_meta.run_id, selected_mode, config.policy_version)
    paths = build_output_paths(output_root, run_id)
    duration = round(time.perf_counter() - start, 6)
    result = RegistrationResult(
        registration_run_id=run_id,
        fixed=fixed_role,
        moving=moving_role,
        registration_direction="moving_volume_transformed_into_fixed_volume_space",
        mode=selected_mode,  # type: ignore[arg-type]
        initialisation_method=config.initialisation_method,
        optimiser_configuration=_optimiser_configuration(config),
        optimiser_summaries=optimiser_summaries,
        transform=transform_summary,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        findings=findings,
        status=status,
        warnings=sorted({finding.message for finding in findings if finding.severity == "WARNING"}),
        output_paths=paths,
        checksums={},
        policy_version=config.policy_version,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        processing_duration_seconds=duration,
        recommended_next_action=_recommended_next_action(status),
    )
    return write_registration_outputs(
        registered,
        mid_axial_slices(fixed_volume, moving_volume, registered),
        result,
        overwrite=config.overwrite if overwrite is None else overwrite,
    )


def summarise_transform(
    transform: sitk.Transform,
    *,
    spacing_zyx: tuple[float, float, float],
    fallback_translation_mm: tuple[float, float, float],
    fallback_translation_voxels: tuple[float, float, float],
) -> TransformSummary:
    """Summarise a SimpleITK transform for deterministic JSON output."""
    parameters = [float(item) for item in transform.GetParameters()]
    fixed_parameters = [float(item) for item in transform.GetFixedParameters()]
    name = transform.GetName()
    translation_mm = fallback_translation_mm
    rotation = (0.0, 0.0, 0.0)
    affine_matrix = None
    affine_scale = None
    affine_shear = None
    if "Euler3DTransform" in name and len(parameters) >= 6:
        rotation = (
            math.degrees(parameters[0]),
            math.degrees(parameters[1]),
            math.degrees(parameters[2]),
        )
        offset = parameters[3:6]
        translation_mm = (-float(offset[0]), -float(offset[1]), -float(offset[2]))
    elif "TranslationTransform" in name and len(parameters) >= 3:
        translation_mm = (-float(parameters[0]), -float(parameters[1]), -float(parameters[2]))
    elif "AffineTransform" in name and len(parameters) >= 12:
        matrix = np.array(parameters[:9], dtype=float).reshape(3, 3)
        translation_mm = (
            -float(parameters[9]),
            -float(parameters[10]),
            -float(parameters[11]),
        )
        affine_matrix = [float(item) for item in matrix.reshape(-1)]
        affine_scale = (
            float(np.linalg.norm(matrix[:, 0])),
            float(np.linalg.norm(matrix[:, 1])),
            float(np.linalg.norm(matrix[:, 2])),
        )
        off_diag = matrix - np.diag(np.diag(matrix))
        affine_shear = float(np.max(np.abs(off_diag)))
    translation_voxels = (
        translation_mm[2] / spacing_zyx[0],
        translation_mm[1] / spacing_zyx[1],
        translation_mm[0] / spacing_zyx[2],
    )
    if all(item == 0 for item in translation_voxels):
        translation_voxels = fallback_translation_voxels
    return TransformSummary(
        transform_type=name,
        parameters=parameters,
        fixed_parameters=fixed_parameters,
        translation_mm_xyz=translation_mm,
        translation_voxels_zyx=translation_voxels,
        rotation_degrees=rotation,
        affine_matrix=affine_matrix,
        affine_scale=affine_scale,
        affine_shear=affine_shear,
    )


def _run_id(fixed_run_id: str, moving_run_id: str, mode: str, policy_version: str) -> str:
    digest = hashlib.sha256(
        "|".join([fixed_run_id, moving_run_id, mode, policy_version]).encode("utf-8")
    ).hexdigest()[:16]
    return f"registration-{digest}"


def _optimiser_configuration(config: RegistrationConfig) -> dict[str, object]:
    return {
        "metric": config.metric,
        "metric_bins": config.metric_bins,
        "metric_sampling_strategy": config.metric_sampling_strategy,
        "metric_sampling_percentage": config.metric_sampling_percentage,
        "random_seed": config.random_seed,
        "optimiser": config.optimiser,
        "learning_rate": config.learning_rate,
        "minimum_step": config.minimum_step,
        "maximum_iterations": config.maximum_iterations,
        "convergence_window_size": config.convergence_window_size,
        "shrink_factors": config.shrink_factors,
        "smoothing_sigmas": config.smoothing_sigmas,
        "interpolator": config.interpolator,
        "default_pixel_value": config.default_pixel_value,
    }


def _recommended_next_action(status: str) -> str:
    if status == "PASS":
        return "Proceed only to research review; this is not clinical evidence."
    if status == "PASS_WITH_WARNINGS":
        return "Review warnings before using this engineering artefact downstream."
    return "Do not use this registration output downstream without remediation."
