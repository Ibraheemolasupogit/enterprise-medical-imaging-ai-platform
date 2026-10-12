import numpy as np
import pytest
import torch
from pydantic import ValidationError

from medical_imaging_platform.classification.model_factory import (
    architecture_metadata,
    build_classifier,
    parameter_count,
)
from medical_imaging_platform.classification.models import ClassificationConfig
from medical_imaging_platform.embeddings.config import EmbeddingConfig
from medical_imaging_platform.embeddings.core import EmbeddingIndex, l2_normalize
from medical_imaging_platform.embeddings.metrics import retrieval_metrics


def cls_config(family="small_3d_cnn"):
    return ClassificationConfig(
        policy_version="test",
        random_seed=7,
        input_shape=(8, 8, 8),
        batch_size=1,
        num_workers=0,
        epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        model_family=family,
        channels=(2, 4),
        dropout=0.0,
        patch_size=(2, 2, 2),
        embedding_dim=8,
        transformer_depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        transformer_dropout=0.0,
        positive_class_weight=1.0,
        early_stopping_patience=1,
        calibration_method="none",
        threshold_method="fixed",
        fixed_threshold=0.5,
        minimum_sensitivity=0.0,
        maximum_false_positives=10,
        abstention_lower=0.0,
        abstention_upper=1.0,
        minimum_validation_auroc=0.0,
        minimum_validation_auprc=0.0,
        minimum_validation_recall=0.0,
        maximum_false_negatives=10,
        maximum_brier_score=1.0,
        dataset_output_directory="data",
        output_directory="out",
        overwrite=False,
    )


def record(eid, sid, scenario="a", label=1):
    return {"embedding_id": eid, "source_sample_id": sid, "scenario": scenario, "label": label}


def test_l2_normalize_and_reject_invalid():
    out = l2_normalize(np.array([[3.0, 4.0]], dtype=np.float32))
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    with pytest.raises(ValueError):
        l2_normalize(np.zeros((1, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        l2_normalize(np.array([[np.nan, 1.0]], dtype=np.float32))


def test_embedding_config_is_strict():
    cfg = EmbeddingConfig(
        policy_version="M22",
        gallery_splits=["train"],
        query_splits=["test"],
        top_k=[1, 5],
        output_directory="out",
    )
    assert cfg.normalization == "l2"
    with pytest.raises(ValidationError):
        EmbeddingConfig(
            policy_version="M22",
            gallery_splits=["train"],
            query_splits=["test"],
            top_k=[5, 1],
            output_directory="out",
        )


def test_models_extract_deterministic_features_and_preserve_logits():
    image = torch.ones((2, 1, 8, 8, 8))
    for family in ("small_3d_cnn", "tiny_3d_vit"):
        model = build_classifier(cls_config(family)).eval()
        with torch.no_grad():
            features_a = model.extract_features(image)
            features_b = model.extract_features(image)
            logits = model(image)
        assert features_a.shape == (2, model.feature_dim)
        assert torch.equal(features_a, features_b)
        assert logits.shape == (2,)
        assert architecture_metadata(cls_config(family))["embedding_dim"] == model.feature_dim
        assert parameter_count(model) > model.feature_dim


def test_exact_search_is_stable_and_excludes_self():
    matrix = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    manifest = {
        "embedding_dimension": 2,
        "records": [record("b", "b"), record("a", "a"), record("c", "c")],
        "encoder_family": "small_3d_cnn",
    }
    index = EmbeddingIndex(matrix, manifest)
    results = index.search(
        np.array([1.0, 0.0], dtype=np.float32), top_k=3, exclude_embedding_id="b"
    )
    assert [item["embedding_id"] for item in results] == ["a", "c"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_retrieval_metrics_cover_ranked_results():
    results = [
        {"source_sample_id": "x", "scenario": "s", "label": 1},
        {"source_sample_id": "y", "scenario": "t", "label": 0},
    ]
    metrics = retrieval_metrics(results, {"sample_ids": {"y"}, "scenarios": {"t"}}, ks=(1, 2))
    assert metrics["recall_at_1"] == 0.0
    assert metrics["recall_at_2"] == 1.0
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["scenario_consistency_at_2"] == 0.5
