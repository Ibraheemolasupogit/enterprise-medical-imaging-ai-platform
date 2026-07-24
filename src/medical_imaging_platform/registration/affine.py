"""Affine SimpleITK registration."""

from __future__ import annotations

import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.models import OptimiserSummary, RegistrationConfig
from medical_imaging_platform.registration.rigid import _registration_method


def run_affine_registration(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    rigid_transform: sitk.Transform,
    config: RegistrationConfig,
) -> tuple[sitk.Transform, OptimiserSummary]:
    """Run affine registration initialised from the rigid result."""
    affine = sitk.AffineTransform(3)
    affine.SetCenter(_image_center(fixed_image))
    affine.SetTranslation(rigid_transform.GetParameters()[-3:])
    registration = _registration_method(config)
    registration.SetInitialTransform(affine, inPlace=False)
    try:
        final_transform = registration.Execute(fixed_image, moving_image)
        summary = OptimiserSummary(
            stage="affine",
            attempted=True,
            succeeded=True,
            stop_condition=registration.GetOptimizerStopConditionDescription(),
            iterations=int(registration.GetOptimizerIteration()),
            metric_value=float(registration.GetMetricValue()),
        )
        return final_transform, summary
    except Exception as exc:
        summary = OptimiserSummary(
            stage="affine",
            attempted=True,
            succeeded=False,
            stop_condition=str(exc),
            iterations=0,
            metric_value=0.0,
            rejected=True,
            warnings=["Affine registration failed; preserving rigid result."],
        )
        return rigid_transform, summary


def _image_center(image: sitk.Image) -> tuple[float, float, float]:
    size = image.GetSize()
    point = image.TransformContinuousIndexToPhysicalPoint(
        ((size[0] - 1) / 2, (size[1] - 1) / 2, (size[2] - 1) / 2)
    )
    return (float(point[0]), float(point[1]), float(point[2]))
