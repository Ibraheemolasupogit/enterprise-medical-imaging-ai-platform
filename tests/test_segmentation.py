from pathlib import Path

import numpy as np
import pytest
import torch

from medical_imaging_platform.cli import main
from medical_imaging_platform.segmentation.checkpoint import CheckpointError, load_state_dict
from medical_imaging_platform.segmentation.dataset import (
    SegmentationDataError,
    SegmentationTorchDataset,
    prepare_segmentation_dataset,
    validate_segmentation_dataset,
)
from medical_imaging_platform.segmentation.export import (
    SegmentationOutputError,
    inspect_segmentation_experiment,
    validate_segmentation_experiment,
)
from medical_imaging_platform.segmentation.inference import predict_probability, segment_volume
from medical_imaging_platform.segmentation.losses import dice_loss, segmentation_loss
from medical_imaging_platform.segmentation.metrics import (
    aggregate_metrics,
    compute_segmentation_metrics,
)
from medical_imaging_platform.segmentation.model_factory import build_unet, parameter_count
from medical_imaging_platform.segmentation.models import SegmentationConfig
from medical_imaging_platform.segmentation.pipeline import (
    configuration_checksum,
    dependency_versions,
    resolve_device,
    set_reproducibility,
)
from medical_imaging_platform.segmentation.postprocessing import postprocess_probability_map
from medical_imaging_platform.segmentation.quality import evaluate_quality
from medical_imaging_platform.segmentation.trainer import train_segmentation_experiment
from medical_imaging_platform.segmentation.transforms import build_transforms
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset
from medical_imaging_platform.utils.config import load_segmentation_config


def seg_config(tmp_path: Path, **updates: object) -> SegmentationConfig:
    data = {
        "policy_version": "test-segmentation-v0.8",
        "random_seed": 7,
        "device": "cpu",
        "input_shape": (32, 32, 32),
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "loss": "dice_bce",
        "dice_weight": 0.5,
        "bce_weight": 0.5,
        "channels": (4, 8, 16),
        "strides": (2, 2),
        "num_res_units": 1,
        "augmentation": {"enabled": True, "flip_probability": 0.5, "noise_std": 0.0},
        "early_stopping_patience": 1,
        "threshold": 0.95,
        "minimum_component_voxels": 2,
        "keep_largest_component": False,
        "fill_holes": True,
        "minimum_validation_dice": 0.0,
        "minimum_test_recall": 0.0,
        "maximum_false_positive_voxels": 32768,
        "maximum_relative_volume_error": 100.0,
        "dataset_output_directory": tmp_path / "seg-data",
        "output_directory": tmp_path / "seg-exp",
        "overwrite": False,
    }
    data.update(updates)
    return SegmentationConfig.model_validate(data)


def synthetic_dataset(tmp_path: Path) -> Path:
    config = load_synthetic_config(Path("config/data.yaml")).model_copy(
        update={"output_root": tmp_path / "synthetic", "dataset_size": 6}
    )
    generate_dataset(config, tmp_path / "synthetic", overwrite=True)
    return tmp_path / "synthetic"


def prepared_dataset(tmp_path: Path, config: SegmentationConfig | None = None) -> Path:
    cfg = config or seg_config(tmp_path)
    manifest = prepare_segmentation_dataset(
        synthetic_dataset(tmp_path),
        output_root=tmp_path / "prepared",
        config=cfg,
        overwrite=True,
    )
    return tmp_path / "prepared" / manifest.dataset_id


def test_segmentation_config_loader_and_validation(tmp_path: Path) -> None:
    config = load_segmentation_config(Path("config/segmentation.yaml"))

    assert config.device == "cpu"
    assert config.threshold == pytest.approx(0.95)
    with pytest.raises(ValueError):
        seg_config(tmp_path, channels=(4, 8, 16), strides=(2,))
    with pytest.raises(ValueError):
        seg_config(tmp_path, input_shape=(4, 32, 32))
    with pytest.raises(ValueError):
        seg_config(tmp_path, loss="dice_bce", dice_weight=0.0, bce_weight=0.0)


def test_prepare_segmentation_dataset_manifest_and_leakage(tmp_path: Path) -> None:
    dataset_dir = prepared_dataset(tmp_path)
    manifest = validate_segmentation_dataset(dataset_dir)

    assert len(manifest.samples) == 12
    assert manifest.split_counts == {"train": 6, "validation": 2, "test": 4}
    assert [sample.sample_id for sample in manifest.samples] == sorted(
        sample.sample_id for sample in manifest.samples
    )
    assert any(sample.lesion_volume_voxels == 0 for sample in manifest.samples)
    assert any(sample.lesion_volume_voxels > 0 for sample in manifest.samples)
    train_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "train"
    }
    test_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "test"
    }
    assert not train_subjects & test_subjects


def test_dataset_arrays_are_valid_and_checksum_protected(tmp_path: Path) -> None:
    dataset_dir = prepared_dataset(tmp_path)
    manifest = validate_segmentation_dataset(dataset_dir)
    sample = manifest.samples[0]
    image = np.load(dataset_dir / sample.image)
    mask = np.load(dataset_dir / sample.lesion_mask)

    assert image.shape == mask.shape == (32, 32, 32)
    assert image.dtype == np.float32
    assert set(np.unique(mask)) <= {0, 1}
    assert np.all(np.isfinite(image))
    with (dataset_dir / sample.image).open("wb") as handle:
        np.save(handle, np.zeros((32, 32, 32), dtype=np.float32))
    with pytest.raises(SegmentationDataError):
        validate_segmentation_dataset(dataset_dir)


def test_torch_dataset_and_transforms_are_shape_safe(tmp_path: Path) -> None:
    config = seg_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    manifest = validate_segmentation_dataset(dataset_dir)
    dataset = SegmentationTorchDataset(
        dataset_dir, manifest, "train", transform=build_transforms(config, "train")
    )
    first = dataset[0]
    validation = build_transforms(config, "validation")(first)

    assert first["image"].shape == torch.Size([1, 32, 32, 32])
    assert first["mask"].shape == torch.Size([1, 32, 32, 32])
    assert set(torch.unique(first["mask"]).tolist()) <= {0.0, 1.0}
    assert validation["mask"].shape == first["mask"].shape
    with pytest.raises(SegmentationDataError):
        SegmentationTorchDataset(dataset_dir, manifest, "missing")


def test_model_construction_forward_and_reproducibility(tmp_path: Path) -> None:
    config = seg_config(tmp_path)
    set_reproducibility(config.random_seed)
    first = build_unet(config)
    set_reproducibility(config.random_seed)
    second = build_unet(config)
    x = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)

    assert first(x).shape == x.shape
    assert parameter_count(first) > 0
    assert torch.allclose(next(first.parameters()), next(second.parameters()))
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        resolve_device("cuda")


def test_losses_cover_perfect_empty_and_combined(tmp_path: Path) -> None:
    config = seg_config(tmp_path)
    target = torch.zeros((1, 1, 4, 4, 4), dtype=torch.float32)
    logits_empty = torch.full_like(target, -20.0)
    logits_full = torch.full_like(target, 20.0)

    assert dice_loss(logits_empty, target) < 0.01
    assert segmentation_loss(logits_empty, target, config).isfinite()
    assert segmentation_loss(logits_full, target, seg_config(tmp_path, loss="bce")).item() > 1.0
    assert segmentation_loss(logits_empty, target, seg_config(tmp_path, loss="dice")).item() < 0.01


def test_metrics_empty_perfect_partial_and_surface() -> None:
    empty = np.zeros((8, 8, 8), dtype=np.uint8)
    cube = empty.copy()
    cube[2:5, 2:5, 2:5] = 1
    partial = empty.copy()
    partial[3:6, 2:5, 2:5] = 1

    both_empty = compute_segmentation_metrics(empty, empty, spacing_mm=(2.0, 1.0, 1.0))
    perfect = compute_segmentation_metrics(cube, cube, spacing_mm=(2.0, 1.0, 1.0))
    no_overlap = compute_segmentation_metrics(empty, cube, spacing_mm=(2.0, 1.0, 1.0))
    partial_metrics = compute_segmentation_metrics(partial, cube, spacing_mm=(2.0, 1.0, 1.0))

    assert both_empty.dice == 1.0
    assert perfect.dice == 1.0
    assert perfect.hausdorff95_mm == 0.0
    assert no_overlap.recall == 0.0
    assert partial_metrics.dice is not None and 0.0 < partial_metrics.dice < 1.0
    assert aggregate_metrics([perfect])["dice"]["mean"] == 1.0
    with pytest.raises(ValueError):
        compute_segmentation_metrics(cube, cube[0], spacing_mm=(1.0, 1.0, 1.0))


def test_postprocessing_components_holes_and_empty_warning(tmp_path: Path) -> None:
    probability = np.zeros((8, 8, 8), dtype=np.float32)
    probability[1, 1, 1] = 1.0
    probability[3:6, 3:6, 3:6] = 1.0
    probability[4, 4, 4] = 0.0

    mask, warnings, counts = postprocess_probability_map(
        probability,
        config=seg_config(
            tmp_path, minimum_component_voxels=2, keep_largest_component=True, threshold=0.5
        ),
    )
    assert mask.dtype == np.uint8
    assert counts["voxels_after"] > 1
    assert not warnings
    empty, empty_warnings, _ = postprocess_probability_map(
        probability, config=seg_config(tmp_path, minimum_component_voxels=100, threshold=0.5)
    )
    assert np.count_nonzero(empty) == 0
    assert empty_warnings


def test_training_exports_and_inference_are_valid(tmp_path: Path) -> None:
    config = seg_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    payload = train_segmentation_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=config,
        overwrite=True,
    )
    experiment_dir = tmp_path / "experiments" / payload["experiment_id"]
    manifest = validate_segmentation_experiment(experiment_dir)
    sample = validate_segmentation_dataset(dataset_dir).samples[0]
    inference = segment_volume(
        dataset_dir / sample.image,
        checkpoint_path=experiment_dir / "best_model.pt",
        output_dir=tmp_path / "inference",
        config=config,
        overwrite=True,
    )

    assert manifest["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert inspect_segmentation_experiment(experiment_dir)["status"] == manifest["status"]
    assert Path(experiment_dir / "best_model.pt").exists()
    assert (tmp_path / "inference" / "probability_map.npy").exists()
    assert inference["threshold"] == pytest.approx(config.threshold)
    probability, _ = predict_probability(
        np.load(dataset_dir / sample.image), experiment_dir / "best_model.pt", config
    )
    assert probability.shape == (32, 32, 32)


def test_checkpoint_validation_and_experiment_checksum_failure(tmp_path: Path) -> None:
    config = seg_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    payload = train_segmentation_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=config,
        overwrite=True,
    )
    experiment_dir = tmp_path / "experiments" / payload["experiment_id"]

    assert load_state_dict(experiment_dir / "best_model.pt")
    with pytest.raises(CheckpointError):
        load_state_dict(tmp_path / "missing.pt")
    (experiment_dir / "segmentation_report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(SegmentationOutputError):
        validate_segmentation_experiment(experiment_dir)


def test_quality_gates_pass_fail_and_reject(tmp_path: Path) -> None:
    config = seg_config(tmp_path, minimum_validation_dice=0.5)
    evaluation = {
        "validation": {"aggregate": {"dice": {"mean": 0.1}}},
        "test": {
            "aggregate": {"recall": {"mean": 0.0}},
            "false_positive_voxels_max": 99999,
            "relative_volume_error_max": 200.0,
        },
    }
    status, findings = evaluate_quality(
        config=config,
        training_history=[{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
        evaluation=evaluation,
        checkpoint_paths=[tmp_path / "missing.pt"],
        evidence_paths=[tmp_path / "missing.json"],
    )
    rejected, rejected_findings = evaluate_quality(
        config=seg_config(tmp_path),
        training_history=[{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
        evaluation={
            "validation": {"aggregate": {"dice": {"mean": 1.0}}},
            "test": {
                "aggregate": {"recall": {"mean": 1.0}},
                "false_positive_voxels_max": 0,
                "relative_volume_error_max": 0.0,
            },
        },
        checkpoint_paths=[],
        evidence_paths=[],
        leakage_detected=True,
    )

    assert status == "FAIL"
    assert {"SEG-QC-DICE-001", "SEG-QC-FP-001", "SEG-QC-CHK-001"} <= {
        finding.rule_id for finding in findings
    }
    assert rejected == "REJECTED"
    assert any(finding.rule_id == "SEG-QC-SPLIT-001" for finding in rejected_findings)


def test_cli_prepare_train_validate_infer_and_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    synthetic_dir = synthetic_dataset(tmp_path)
    dataset_root = tmp_path / "cli-data"
    experiment_root = tmp_path / "cli-exp"

    assert (
        main(
            [
                "prepare-segmentation-data",
                "--synthetic-dataset-dir",
                str(synthetic_dir),
                "--output-dir",
                str(dataset_root),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    dataset_dir = next(dataset_root.iterdir())
    assert (
        main(
            [
                "train-segmentation",
                str(dataset_dir),
                "--output-dir",
                str(experiment_root),
                "--epochs",
                "1",
                "--overwrite",
            ]
        )
        == 0
    )
    experiment_dir = next(experiment_root.iterdir())
    assert main(["inspect-segmentation-experiment", str(experiment_dir), "--json"]) == 0
    assert main(["validate-segmentation-experiment", str(experiment_dir)]) == 0
    sample = validate_segmentation_dataset(dataset_dir).samples[0]
    assert (
        main(
            [
                "segment-volume",
                str(dataset_dir / sample.image),
                "--checkpoint",
                str(experiment_dir / "best_model.pt"),
                "--output-dir",
                str(tmp_path / "cli-inference"),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    assert "Validated" not in capsys.readouterr().err
    assert main(["validate-segmentation-experiment", str(tmp_path / "missing")]) == 4
    assert (
        main(
            [
                "segment-volume",
                str(dataset_dir / sample.image),
                "--checkpoint",
                "missing.pt",
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
        == 3
    )


def test_reproducibility_helpers_are_stable(tmp_path: Path) -> None:
    config = seg_config(tmp_path)

    assert configuration_checksum(config) == configuration_checksum(config)
    versions = dependency_versions()
    assert {"python_version", "torch_version", "monai_version"} <= set(versions)
