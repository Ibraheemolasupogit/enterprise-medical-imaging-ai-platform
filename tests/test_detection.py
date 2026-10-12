from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from medical_imaging_platform.cli import main
from medical_imaging_platform.detection.benchmark import run_detection_benchmark
from medical_imaging_platform.detection.checkpoint import load_state_dict
from medical_imaging_platform.detection.dataset import (
    DetectionDataError,
    DetectionTorchDataset,
    mask_to_boxes_xyxy,
    prepare_detection_dataset,
    validate_detection_dataset,
)
from medical_imaging_platform.detection.export import validate_detection_experiment
from medical_imaging_platform.detection.inference import detect_slice
from medical_imaging_platform.detection.metrics import box_iou_xyxy, compute_detection_metrics
from medical_imaging_platform.detection.model_factory import (
    architecture_metadata,
    build_detector,
    parameter_count,
)
from medical_imaging_platform.detection.models import DetectionConfig
from medical_imaging_platform.detection.trainer import train_detection_experiment
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset
from medical_imaging_platform.utils.config import load_detection_config


def det_config(tmp_path: Path, **updates: object) -> DetectionConfig:
    data = {
        "policy_version": "test-detection-v1",
        "random_seed": 11,
        "device": "cpu",
        "model_family": "faster_rcnn",
        "image_shape": (32, 32),
        "batch_size": 2,
        "num_workers": 0,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "max_positive_slices_per_volume": 1,
        "max_negative_slices_per_volume": 1,
        "minimum_lesion_pixels": 1,
        "score_threshold": 0.01,
        "nms_iou_threshold": 0.5,
        "evaluation_iou_thresholds": (0.5, 0.75),
        "faster_rcnn_anchor_sizes": (8, 16),
        "faster_rcnn_aspect_ratios": (0.5, 1.0, 2.0),
        "hidden_channels": 4,
        "embedding_dim": 8,
        "transformer_depth": 1,
        "num_heads": 2,
        "num_queries": 3,
        "mlp_ratio": 2.0,
        "no_object_weight": 0.1,
        "box_loss_weight": 1.0,
        "iou_loss_weight": 0.1,
        "minimum_validation_map": 0.0,
        "maximum_false_positives_per_image": 100.0,
        "dataset_output_directory": tmp_path / "det-data",
        "output_directory": tmp_path / "det-exp",
        "overwrite": False,
    }
    data.update(updates)
    return DetectionConfig.model_validate(data)


def synthetic_dataset(tmp_path: Path) -> Path:
    config = load_synthetic_config(Path("config/data.yaml")).model_copy(
        update={"output_root": tmp_path / "synthetic", "dataset_size": 6}
    )
    generate_dataset(config, tmp_path / "synthetic", overwrite=True)
    return tmp_path / "synthetic"


def prepared_dataset(tmp_path: Path, config: DetectionConfig | None = None) -> Path:
    cfg = config or det_config(tmp_path)
    manifest = prepare_detection_dataset(
        synthetic_dataset(tmp_path),
        output_root=tmp_path / "prepared",
        config=cfg,
        overwrite=True,
    )
    return tmp_path / "prepared" / manifest.dataset_id


def test_mask_to_box_positive_empty_bounds_and_config_validation(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 3:7] = 1

    assert mask_to_boxes_xyxy(mask).tolist() == [[3.0, 2.0, 7.0, 5.0]]
    assert mask_to_boxes_xyxy(np.zeros((8, 8), dtype=np.uint8)).shape == (0, 4)
    assert load_detection_config(Path("config/detection.yaml")).model_family == "faster_rcnn"
    with pytest.raises(ValueError):
        det_config(tmp_path, embedding_dim=7, num_heads=2)


def test_prepare_validate_manifest_checksums_and_split_isolation(tmp_path: Path) -> None:
    dataset_dir = prepared_dataset(tmp_path)
    manifest = validate_detection_dataset(dataset_dir)

    assert manifest.positive_image_count > 0
    assert manifest.negative_image_count > 0
    assert manifest.box_count == manifest.positive_image_count
    assert [sample.sample_id for sample in manifest.samples] == sorted(
        sample.sample_id for sample in manifest.samples
    )
    train_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "train"
    }
    test_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "test"
    }
    assert not train_subjects & test_subjects
    sample = manifest.samples[0]
    image_path = dataset_dir / sample.image
    with image_path.open("wb") as handle:
        np.save(handle, np.zeros((32, 32), dtype=np.float32))
    with pytest.raises(DetectionDataError):
        validate_detection_dataset(dataset_dir)


def test_dataset_overwrite_protection_and_torch_contract(tmp_path: Path) -> None:
    synthetic_dir = synthetic_dataset(tmp_path)
    config = det_config(tmp_path)
    first = prepare_detection_dataset(
        synthetic_dir, output_root=tmp_path / "prepared", config=config, overwrite=True
    )
    with pytest.raises(DetectionDataError):
        prepare_detection_dataset(
            synthetic_dir, output_root=tmp_path / "prepared", config=config, overwrite=False
        )
    dataset = DetectionTorchDataset(tmp_path / "prepared" / first.dataset_id, first, "train")
    item = dataset[0]
    assert item["image"].shape == torch.Size([1, 32, 32])
    assert item["target"]["boxes"].shape[1] == 4


def test_detection_metrics_cover_edge_cases() -> None:
    gt = [{"boxes": [[1, 1, 5, 5]], "labels": [1]}]
    perfect = [{"boxes": [[1, 1, 5, 5]], "labels": [1], "scores": [0.9]}]
    duplicate = [{"boxes": [[1, 1, 5, 5], [1, 1, 5, 5]], "labels": [1, 1], "scores": [0.9, 0.8]}]
    partial = [{"boxes": [[3, 3, 7, 7]], "labels": [1], "scores": [0.9]}]
    empty = [{"boxes": [], "labels": [], "scores": []}]

    assert box_iou_xyxy(np.asarray([[1, 1, 5, 5]]), np.asarray([[1, 1, 5, 5]])).item() == 1.0
    assert compute_detection_metrics(perfect, gt)["ap50"] == 1.0
    assert compute_detection_metrics(duplicate, gt)["false_positives_per_image"] == 1.0
    assert compute_detection_metrics(partial, gt)["recall"] == 0.0
    assert compute_detection_metrics(empty, gt)["recall"] == 0.0
    assert compute_detection_metrics(perfect, empty)["false_positives_per_image"] == 1.0
    assert compute_detection_metrics(empty, empty)["precision"] == 1.0


@pytest.mark.parametrize("family", ["faster_rcnn", "tiny_detr"])
def test_model_forward_parameter_count_and_training_smoke(tmp_path: Path, family: str) -> None:
    config = det_config(tmp_path, model_family=family)
    model = build_detector(config)
    image = torch.zeros((1, 32, 32), dtype=torch.float32)
    target = {"boxes": torch.tensor([[8.0, 8.0, 14.0, 14.0]]), "labels": torch.tensor([1])}

    model.train()
    losses = model.loss([image], [target])
    assert sum(losses.values()).isfinite()
    model.eval()
    prediction = model.predict([image])[0]
    assert {"boxes", "labels", "scores"} <= set(prediction)
    assert parameter_count(model) > 0
    assert architecture_metadata(config, model)["model_family"] == family
    if family == "faster_rcnn":
        from torchvision.models.detection import FasterRCNN

        assert isinstance(model.model, FasterRCNN)


def test_train_infer_validate_and_benchmark_cli(tmp_path: Path) -> None:
    config = det_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    faster = train_detection_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=config,
        overwrite=True,
    )
    faster_dir = tmp_path / "experiments" / faster["experiment_id"]
    assert validate_detection_experiment(faster_dir)["model_family"] == "faster_rcnn"
    assert load_state_dict(faster_dir / "best_model.pt")
    manifest = validate_detection_dataset(dataset_dir)
    sample = manifest.samples[0]
    inference = detect_slice(
        dataset_dir / sample.image,
        checkpoint_path=faster_dir / "best_model.pt",
        output_dir=tmp_path / "infer",
        config=config,
        overwrite=True,
    )
    assert inference["status"] == "completed"

    vit_config = config.model_copy(update={"model_family": "tiny_detr"})
    vit = train_detection_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=vit_config,
        overwrite=True,
    )
    assert vit["model_family"] == "tiny_detr"

    benchmark = run_detection_benchmark(
        dataset_dir,
        output_root=tmp_path / "benchmark",
        config=config,
        overwrite=True,
    )
    assert (
        benchmark["fairness_attestation"]["detection_manifest_checksum"]
        == faster["dataset_manifest_checksum"]
    )
    assert {"faster_rcnn", "tiny_detr"} == set(benchmark["models"])

    config_path = tmp_path / "detection.yaml"
    config_path.write_text(
        yaml.safe_dump({"settings": {"detection": config.model_dump(mode="json")}}, sort_keys=True),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "prepare-detection-data",
                "--synthetic-dataset-dir",
                str(synthetic_dataset(tmp_path / "cli-synthetic")),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "cli-prepared"),
                "--overwrite",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train-detection",
                str(dataset_dir),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "cli-train"),
                "--model-family",
                "tiny_detr",
                "--epochs",
                "1",
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    assert main(["evaluate-detection", str(faster_dir), "--json"]) == 0
    assert main(["validate-detection-experiment", str(faster_dir)]) == 0
    assert (
        main(
            [
                "detect-slice",
                str(dataset_dir / sample.image),
                "--checkpoint",
                str(faster_dir / "best_model.pt"),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "cli-infer"),
                "--model-family",
                "faster_rcnn",
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run-detection-benchmark",
                str(dataset_dir),
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "cli-benchmark"),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
