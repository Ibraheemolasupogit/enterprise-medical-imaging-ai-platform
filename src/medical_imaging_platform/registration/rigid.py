"""Rigid SimpleITK registration."""

from __future__ import annotations

import SimpleITK as sitk  # noqa: N813

from medical_imaging_platform.registration.models import OptimiserSummary, RegistrationConfig


def run_rigid_registration(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    initial_transform: sitk.Transform,
    config: RegistrationConfig,
) -> tuple[sitk.Transform, OptimiserSummary]:
    """Run Euler 3D rigid registration with explicit settings."""
    euler = sitk.Euler3DTransform()
    euler.SetCenter(_image_center(fixed_image))
    euler.SetTranslation(initial_transform.GetParameters()[:3])
    registration = _registration_method(config)
    registration.SetInitialTransform(euler, inPlace=False)
    final_transform = registration.Execute(fixed_image, moving_image)
    summary = OptimiserSummary(
        stage="rigid",
        attempted=True,
        succeeded=True,
        stop_condition=registration.GetOptimizerStopConditionDescription(),
        iterations=int(registration.GetOptimizerIteration()),
        metric_value=float(registration.GetMetricValue()),
    )
    return final_transform, summary


def _registration_method(config: RegistrationConfig) -> sitk.ImageRegistrationMethod:
    registration = sitk.ImageRegistrationMethod()
    if config.metric == "normalised_correlation":
        registration.SetMetricAsCorrelation()
    else:
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=int(config.metric_bins)
        )
    if config.metric_sampling_strategy == "regular":
        registration.SetMetricSamplingStrategy(registration.REGULAR)
        registration.SetMetricSamplingPercentage(
            config.metric_sampling_percentage, seed=config.random_seed
        )
    elif config.metric_sampling_strategy == "random":
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(
            config.metric_sampling_percentage, seed=config.random_seed
        )
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=config.learning_rate,
        numberOfIterations=config.maximum_iterations,
        convergenceMinimumValue=config.minimum_step,
        convergenceWindowSize=config.convergence_window_size,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel(config.shrink_factors)
    registration.SetSmoothingSigmasPerLevel(config.smoothing_sigmas)
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    return registration


def _image_center(image: sitk.Image) -> tuple[float, float, float]:
    size = image.GetSize()
    point = image.TransformContinuousIndexToPhysicalPoint(
        ((size[0] - 1) / 2, (size[1] - 1) / 2, (size[2] - 1) / 2)
    )
    return (float(point[0]), float(point[1]), float(point[2]))
