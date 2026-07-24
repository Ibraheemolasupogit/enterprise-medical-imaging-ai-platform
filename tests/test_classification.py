import json
from pathlib import Path

import numpy as np
import pytest
import torch

from medical_imaging_platform.classification.calibration import apply_calibration, fit_calibration
from medical_imaging_platform.classification.checkpoint import (
    ClassificationCheckpointError,
    load_state_dict,
)
from medical_imaging_platform.classification.dataset import (
    ClassificationDataError,
    ClassificationTorchDataset,
    prepare_classification_dataset,
    validate_classification_dataset,
)
from medical_imaging_platform.classification.export import (
    ClassificationOutputError,
    inspect_classification_experiment,
    validate_classification_experiment,
)
from medical_imaging_platform.classification.inference import apply_abstention, classify_volume
from medical_imaging_platform.classification.losses import classification_loss
from medical_imaging_platform.classification.metrics import compute_classification_metrics
from medical_imaging_platform.classification.model_factory import build_classifier, parameter_count
from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.classification.pipeline import (
    configuration_checksum,
    dependency_versions,
    resolve_device,
    set_reproducibility,
)
from medical_imaging_platform.classification.quality import evaluate_quality
from medical_imaging_platform.classification.thresholds import select_threshold
from medical_imaging_platform.classification.trainer import train_classification_experiment
from medical_imaging_platform.classification.transforms import build_transforms
from medical_imaging_platform.cli import main
from medical_imaging_platform.synthetic.generator import load_synthetic_config
from medical_imaging_platform.synthetic.io import generate_dataset
from medical_imaging_platform.utils.config import load_classification_config


def cls_config(tmp_path: Path, **updates: object) -> ClassificationConfig:
    data = {
        "policy_version": "test-classification-v0.9",
        "random_seed": 11,
        "device": "cpu",
        "input_shape": (16, 16, 16),
        "batch_size": 4,
        "num_workers": 0,
        "epochs": 1,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "channels": (4, 8),
        "dropout": 0.0,
        "positive_class_weight": None,
        "early_stopping_patience": 1,
        "calibration_method": "platt",
        "threshold_method": "fixed",
        "fixed_threshold": 0.5,
        "minimum_sensitivity": 0.0,
        "maximum_false_positives": 32768,
        "abstention_lower": 0.45,
        "abstention_upper": 0.55,
        "minimum_validation_auroc": 0.0,
        "minimum_validation_auprc": 0.0,
        "minimum_validation_recall": 0.0,
        "maximum_false_negatives": 32768,
        "maximum_brier_score": 1.0,
        "dataset_output_directory": tmp_path / "cls-data",
        "output_directory": tmp_path / "cls-exp",
        "overwrite": False,
    }
    data.update(updates)
    return ClassificationConfig.model_validate(data)


def synthetic_dataset(tmp_path: Path, cases: int = 12) -> Path:
    config = load_synthetic_config(Path("config/data.yaml")).model_copy(
        update={"output_root": tmp_path / "synthetic", "dataset_size": cases}
    )
    generate_dataset(config, tmp_path / "synthetic", overwrite=True)
    return tmp_path / "synthetic"


def prepared_dataset(tmp_path: Path, config: ClassificationConfig | None = None) -> Path:
    cfg = config or cls_config(tmp_path)
    manifest = prepare_classification_dataset(
        synthetic_dataset(tmp_path),
        output_root=tmp_path / "prepared",
        config=cfg,
        overwrite=True,
    )
    return tmp_path / "prepared" / manifest.dataset_id


def test_classification_config_loader_and_validation(tmp_path: Path) -> None:
    config = load_classification_config(Path("config/classification.yaml"))

    assert config.device == "cpu"
    assert config.fixed_threshold == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cls_config(tmp_path, input_shape=(4, 16, 16))
    with pytest.raises(ValueError):
        cls_config(tmp_path, channels=())
    with pytest.raises(ValueError):
        cls_config(tmp_path, abstention_lower=0.8, abstention_upper=0.2)


def test_prepare_classification_dataset_manifest_and_split_isolation(tmp_path: Path) -> None:
    dataset_dir = prepared_dataset(tmp_path)
    manifest = validate_classification_dataset(dataset_dir)

    assert len(manifest.samples) == 24
    assert set(manifest.split_counts) == {"train", "validation", "test"}
    assert [sample.sample_id for sample in manifest.samples] == sorted(
        sample.sample_id for sample in manifest.samples
    )
    assert manifest.class_counts["synthetic_lesion_present"] > 0
    assert manifest.class_counts["no_visible_synthetic_lesion"] > 0
    for split in ("train", "validation", "test"):
        assert {sample.label for sample in manifest.samples if sample.split == split} == {0, 1}
    train_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "train"
    }
    test_subjects = {
        sample.research_subject_id for sample in manifest.samples if sample.split == "test"
    }
    assert not train_subjects & test_subjects
    assert all(sample.localisation_run_id is None for sample in manifest.samples)
    assert all(sample.segmentation_run_id is None for sample in manifest.samples)


def test_dataset_arrays_checksums_and_small_invalid_split(tmp_path: Path) -> None:
    dataset_dir = prepared_dataset(tmp_path)
    manifest = validate_classification_dataset(dataset_dir)
    sample = manifest.samples[0]
    image = np.load(dataset_dir / sample.image)

    assert image.shape == (16, 16, 16)
    assert image.dtype == np.float32
    assert np.all(np.isfinite(image))
    with (dataset_dir / sample.image).open("wb") as handle:
        np.save(handle, np.zeros((16, 16, 16), dtype=np.float32))
    with pytest.raises(ClassificationDataError):
        validate_classification_dataset(dataset_dir)
    with pytest.raises(ClassificationDataError):
        prepare_classification_dataset(
            synthetic_dataset(tmp_path / "too-small", cases=6),
            output_root=tmp_path / "too-small-prepared",
            config=cls_config(tmp_path),
            overwrite=True,
        )


def test_torch_dataset_transforms_and_model_reproducibility(tmp_path: Path) -> None:
    config = cls_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    manifest = validate_classification_dataset(dataset_dir)
    dataset = ClassificationTorchDataset(
        dataset_dir, manifest, "train", transform=build_transforms(config, "train")
    )
    item = dataset[0]

    assert item["image"].shape == torch.Size([1, 16, 16, 16])
    assert item["label"].shape == torch.Size([])
    set_reproducibility(config.random_seed)
    first = build_classifier(config)
    set_reproducibility(config.random_seed)
    second = build_classifier(config)
    assert first(torch.zeros((1, 1, 16, 16, 16), dtype=torch.float32)).shape == torch.Size([1])
    assert parameter_count(first) > 0
    assert torch.allclose(next(first.parameters()), next(second.parameters()))
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError):
        resolve_device("cuda")
    with pytest.raises(ClassificationDataError):
        ClassificationTorchDataset(dataset_dir, manifest, "missing")


def test_loss_metrics_calibration_thresholds_and_abstention(tmp_path: Path) -> None:
    config = cls_config(tmp_path)
    logits = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    labels = torch.tensor([0.0, 1.0], dtype=torch.float32)
    metrics = compute_classification_metrics([0, 1], [0.1, 0.9], threshold=0.5)
    calibration = fit_calibration([-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1], config)
    threshold = select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        cls_config(tmp_path, threshold_method="youden"),
    )

    assert classification_loss(logits, labels, config, positive_weight=2.0).isfinite()
    assert metrics["auroc"] == pytest.approx(1.0)
    assert 0.0 <= apply_calibration(0.0, calibration) <= 1.0
    assert threshold["source_split"] == "validation"
    assert apply_abstention(0.5, 0.5, config)["final_label"] == "indeterminate"
    assert apply_abstention(0.9, 0.5, config)["final_label"] == "synthetic_lesion_present"


def test_calibration_threshold_policy_and_failure_branches(tmp_path: Path) -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    probabilities = [0.05, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.95]
    logits = [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0]
    none = fit_calibration(logits, labels, cls_config(tmp_path, calibration_method="none"))
    isotonic = fit_calibration(logits, labels, cls_config(tmp_path, calibration_method="isotonic"))

    assert none["status"] == "fallback_insufficient_validation_data"
    assert 0.0 <= apply_calibration(0.25, isotonic) <= 1.0
    assert fit_calibration([0.1, 0.2], [1, 1], cls_config(tmp_path))["method"] == "none"
    assert (
        select_threshold(
            labels,
            probabilities,
            cls_config(tmp_path, threshold_method="minimum_sensitivity", minimum_sensitivity=1.0),
        )["selected_threshold"]
        <= 0.65
    )
    assert (
        select_threshold(
            labels,
            probabilities,
            cls_config(tmp_path, threshold_method="minimum_npv"),
        )["selected_threshold"]
        >= 0.0
    )
    assert (
        select_threshold(
            labels,
            probabilities,
            cls_config(
                tmp_path, threshold_method="maximum_false_positives", maximum_false_positives=0
            ),
        )["selected_threshold"]
        >= 0.0
    )
    with pytest.raises(ValueError):
        apply_calibration(0.0, {"method": "platt", "parameters": {"coef": []}})
    with pytest.raises(ValueError):
        apply_abstention(-0.1, 0.5, cls_config(tmp_path))


def test_training_exports_inference_and_validation_are_valid(tmp_path: Path) -> None:
    config = cls_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    payload = train_classification_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=config,
        overwrite=True,
    )
    experiment_dir = tmp_path / "experiments" / payload["experiment_id"]
    manifest = validate_classification_experiment(experiment_dir)
    sample = validate_classification_dataset(dataset_dir).samples[0]
    inference = classify_volume(
        dataset_dir / sample.image,
        checkpoint_path=experiment_dir / "best_model.pt",
        calibration_path=experiment_dir / "calibration.json",
        threshold_policy_path=experiment_dir / "threshold_policy.json",
        output_dir=tmp_path / "inference",
        config=config,
        overwrite=True,
    )

    assert manifest["status"] in {"PASS", "PASS_WITH_WARNINGS"}
    assert inspect_classification_experiment(experiment_dir)["status"] == manifest["status"]
    assert Path(experiment_dir / "best_model.pt").exists()
    assert 0.0 <= inference["prediction"]["calibrated_probability"] <= 1.0
    assert (tmp_path / "inference" / "prediction.json").exists()


def test_checkpoint_quality_and_checksum_failures(tmp_path: Path) -> None:
    config = cls_config(tmp_path)
    dataset_dir = prepared_dataset(tmp_path, config)
    payload = train_classification_experiment(
        dataset_dir,
        output_root=tmp_path / "experiments",
        config=config,
        overwrite=True,
    )
    experiment_dir = tmp_path / "experiments" / payload["experiment_id"]
    evaluation = {
        "validation": {"metrics": {"auroc": 0.1, "auprc": 0.1, "recall": 0.0, "brier_score": 1.0}},
        "test": {"metrics": {"false_negative_count": 99999}},
        "calibration_present": False,
        "threshold_present": False,
    }
    status, findings = evaluate_quality(
        config=cls_config(tmp_path, minimum_validation_auroc=0.5, maximum_false_negatives=0),
        training_history=[{"epoch": 1, "train_loss": 1.0, "validation_loss": 1.0}],
        evaluation=evaluation,
        checkpoint_paths=[tmp_path / "missing.pt"],
        evidence_paths=[tmp_path / "missing.json"],
    )

    assert load_state_dict(experiment_dir / "best_model.pt")
    with pytest.raises(ClassificationCheckpointError):
        load_state_dict(tmp_path / "missing.pt")
    assert status == "FAIL"
    assert {"CLS-QC-AUC-001", "CLS-QC-FN-001", "CLS-QC-CHK-001"} <= {
        finding.rule_id for finding in findings
    }
    (experiment_dir / "classification_report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ClassificationOutputError):
        validate_classification_experiment(experiment_dir)


def test_cli_prepare_train_validate_classify_and_failures(tmp_path: Path) -> None:
    synthetic_dir = synthetic_dataset(tmp_path)
    dataset_root = tmp_path / "cli-data"
    experiment_root = tmp_path / "cli-exp"

    assert (
        main(
            [
                "prepare-classification-data",
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
                "train-classification",
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
    sample = validate_classification_dataset(dataset_dir).samples[0]

    assert main(["inspect-classification-experiment", str(experiment_dir), "--json"]) == 0
    assert main(["validate-classification-experiment", str(experiment_dir)]) == 0
    assert (
        main(
            [
                "classify-volume",
                str(dataset_dir / sample.image),
                "--checkpoint",
                str(experiment_dir / "best_model.pt"),
                "--calibration",
                str(experiment_dir / "calibration.json"),
                "--threshold-policy",
                str(experiment_dir / "threshold_policy.json"),
                "--output-dir",
                str(tmp_path / "cli-inference"),
                "--overwrite",
                "--json",
            ]
        )
        == 0
    )
    prediction = json.loads((tmp_path / "cli-inference" / "prediction.json").read_text())
    assert prediction["final_label"]
    assert main(["validate-classification-experiment", str(tmp_path / "missing")]) == 4
    assert (
        main(
            [
                "classify-volume",
                str(dataset_dir / sample.image),
                "--checkpoint",
                "missing.pt",
                "--calibration",
                str(experiment_dir / "calibration.json"),
                "--threshold-policy",
                str(experiment_dir / "threshold_policy.json"),
                "--output-dir",
                str(tmp_path / "bad"),
            ]
        )
        == 3
    )


def test_reproducibility_helpers_are_stable(tmp_path: Path) -> None:
    config = cls_config(tmp_path)

    assert configuration_checksum(config) == configuration_checksum(config)
    versions = dependency_versions()
    assert {"python_version", "torch_version", "monai_version"} <= set(versions)
