"""Behavior coverage for M22 artifacts and the real module command dispatcher."""

import json
import runpy
import sys
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import yaml
from fastapi.testclient import TestClient
from test_api import _write_api_config
from test_embeddings import cls_config

from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.classification.checkpoint import save_state_dict
from medical_imaging_platform.classification.model_factory import build_classifier
from medical_imaging_platform.embeddings.benchmark import run_benchmark
from medical_imaging_platform.embeddings.cli import _parser
from medical_imaging_platform.embeddings.cli import main as embeddings_main
from medical_imaging_platform.embeddings.config import load_embedding_config
from medical_imaging_platform.embeddings.core import EmbeddingIndex, sha256_file, validate_index
from medical_imaging_platform.embeddings.extractor import generate_embeddings
from medical_imaging_platform.embeddings.metrics import retrieval_metrics
from medical_imaging_platform.reviewer_ui.pages.embeddings import render as render_embeddings


def _module_cli(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["medical-imaging-platform", *map(str, args)])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("medical_imaging_platform", run_name="__main__")
    assert exit_info.value.code == 0


def _index(directory, *, family="small_3d_cnn", query=True):
    directory.mkdir(parents=True)
    records = [
        {
            "sample_id": sample_id,
            "source_sample_id": sample_id,
            "embedding_id": f"{family}:{sample_id}",
            "split": split,
            "scenario": "lesion" if sample_id != "b" else "negative",
            "label": int(sample_id != "b"),
        }
        for sample_id, split in (("a", "train"), ("b", "validation"), ("q", "test"))
        if query or split != "test"
    ]
    matrix = np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32)[: len(records)]
    matrix_path = directory / "embedding_matrix.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    manifest = {
        "embedding_dimension": 2,
        "encoder_family": family,
        "encoder_checkpoint_checksum": sha256_file(matrix_path),
        "matrix_checksum": sha256_file(matrix_path),
        "normalization": "l2",
        "config_checksum": sha256_file(matrix_path),
        "source_dataset_manifest_checksum": sha256_file(matrix_path),
        "records": records,
    }
    (directory / "embedding_manifest.json").write_text(json.dumps(manifest))
    (directory / "evidence").mkdir()
    (directory / "evidence" / "checksum_manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    name: sha256_file(directory / name)
                    for name in ("embedding_matrix.npy", "embedding_manifest.json")
                }
            }
        )
    )
    return directory


@pytest.fixture
def generation_inputs(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    samples = []
    # Deliberately reverse sample order and exercise both supported image ranks.
    for sample_id, shape, split in (("z", (1, 8, 8, 8), "test"), ("a", (8, 8, 8), "train")):
        image = f"{sample_id}.npy"
        np.save(dataset / image, np.ones(shape, dtype=np.float32), allow_pickle=False)
        samples.append(
            {"sample_id": sample_id, "image": image, "split": split, "label_name": "lesion"}
        )
    (dataset / "classification_dataset_manifest.json").write_text(
        json.dumps({"dataset_id": "synthetic-test", "samples": samples})
    )
    inputs = {}
    for family in ("small_3d_cnn", "tiny_3d_vit"):
        config = cls_config(family)
        config_path = tmp_path / f"{family}.yaml"
        config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
        checkpoint = tmp_path / f"{family}.pt"
        with torch.random.fork_rng():
            torch.manual_seed(7)
            save_state_dict(checkpoint, build_classifier(config))
        inputs[family] = (dataset, config_path, checkpoint)
    return inputs


@pytest.mark.parametrize("family", ["small_3d_cnn", "tiny_3d_vit"])
def test_generation_checkpoint_roundtrip_and_cli(
    generation_inputs, tmp_path, monkeypatch, capsys, family
):
    dataset, config_path, checkpoint = generation_inputs[family]
    output = tmp_path / "index"
    args = [
        "generate-embeddings",
        dataset,
        "--config",
        config_path,
        "--checkpoint",
        checkpoint,
        "--encoder-family",
        family,
        "--output-dir",
        output,
        "--json",
    ]
    _module_cli(monkeypatch, args)
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["encoder_family"] == family
    assert manifest["dataset_id"] == "synthetic-test"
    assert [r["source_sample_id"] for r in manifest["records"]] == ["a", "z"]
    assert manifest["encoder_checkpoint_checksum"] == sha256_file(checkpoint)
    assert manifest["source_dataset_manifest_checksum"] == sha256_file(
        dataset / "classification_dataset_manifest.json"
    )
    index = EmbeddingIndex.load(output)
    assert index.dimension == (4 if family == "small_3d_cnn" else 8)
    assert index.matrix.dtype == np.float32
    assert np.allclose(np.linalg.norm(index.matrix, axis=1), 1.0)
    evidence = json.loads((output / "evidence" / "checksum_manifest.json").read_text())
    assert evidence["files"] == {
        name: sha256_file(output / name)
        for name in ("embedding_matrix.npy", "embedding_manifest.json", "evidence_envelope.json")
    }
    original_matrix = index.matrix.copy()
    original_manifest = (output / "embedding_manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        _module_cli(monkeypatch, args)
    assert (output / "embedding_manifest.json").read_bytes() == original_manifest
    _module_cli(monkeypatch, [*args, "--overwrite"])
    capsys.readouterr()
    assert np.array_equal(EmbeddingIndex.load(output).matrix, original_matrix)
    _module_cli(monkeypatch, ["validate-embedding-index", output, "--json"])
    assert json.loads(capsys.readouterr().out) == {
        "valid": True,
        "rows": 2,
        "dimension": index.dimension,
        "dtype": "float32",
        "normalized": True,
    }


@pytest.mark.parametrize("failure", ["missing_manifest", "family", "checkpoint", "nonfinite"])
def test_generation_rejects_invalid_inputs(generation_inputs, tmp_path, failure):
    dataset, config, checkpoint = generation_inputs["small_3d_cnn"]
    family = "small_3d_cnn"
    expected, message = ValueError, ""
    if failure == "missing_manifest":
        (dataset / "classification_dataset_manifest.json").unlink()
        expected, message = FileNotFoundError, "classification_dataset_manifest"
    elif failure == "family":
        family, message = "tiny_3d_vit", "family/config mismatch"
    elif failure == "checkpoint":
        checkpoint = generation_inputs["tiny_3d_vit"][2]
        expected, message = RuntimeError, "state_dict"
    else:
        np.save(dataset / "a.npy", np.full((8, 8, 8), np.nan, dtype=np.float32))
        message = "non-finite source image"
    output = tmp_path / "rejected-index"
    with pytest.raises(expected, match=message):
        generate_embeddings(dataset, config, checkpoint, family, output)
    assert not output.exists()


def test_benchmark_artifacts_membership_and_overwrite(tmp_path):
    cnn = _index(tmp_path / "cnn")
    vit = _index(tmp_path / "vit", family="tiny_3d_vit")
    output = tmp_path / "benchmark"
    policy = load_embedding_config("config/embeddings.yaml")
    comparison = run_benchmark(cnn, vit, output, top_k=policy.top_k)
    assert comparison["fairness_attestation"]["identical_ordered_samples"] is True
    assert comparison["fairness_attestation"]["gallery_sample_count"] == 2
    assert comparison["fairness_attestation"]["query_sample_count"] == 1
    for model in comparison["models"].values():
        assert model["query_count"] == 1
        assert model["embedding_dimension"] == 2
        assert model["matrix_storage_bytes"] == 24
        assert model["metrics"]["recall_at_5"] == 1.0
        assert model["query_latency_seconds"] >= 0
    assert json.loads((output / "embedding_benchmark_comparison.json").read_text()) == comparison
    manifest_path = output / "embedding_benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["gallery_sample_ids"] == ["a", "b"]
    assert manifest["query_sample_ids"] == ["q"]
    assert manifest["top_k"] == list(policy.top_k)
    assert manifest["cnn_index_checksum"] == sha256_file(cnn / "embedding_manifest.json")
    assert manifest["vit_index_checksum"] == sha256_file(vit / "embedding_manifest.json")
    assert "not a clinical" in (output / "embedding_benchmark_report.md").read_text()
    with pytest.raises(FileExistsError, match="non-empty"):
        run_benchmark(cnn, vit, output)
    assert json.loads(manifest_path.read_text()) == manifest
    assert run_benchmark(cnn, vit, output, top_k=(1, 5), overwrite=True)["models"]
    vit_manifest = vit / "embedding_manifest.json"
    changed = json.loads(vit_manifest.read_text())
    changed["records"].reverse()
    vit_manifest.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="row ordering"):
        run_benchmark(cnn, vit, tmp_path / "mismatched")
    assert not (tmp_path / "mismatched" / "embedding_benchmark_comparison.json").exists()


def test_query_and_evaluation_cli(tmp_path, monkeypatch, capsys):
    directory = _index(tmp_path / "index")
    _module_cli(monkeypatch, ["query-similar-studies", directory, "--sample-id", "q", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"]["research_only"] is True
    assert [r["source_sample_id"] for r in payload["results"]] == ["a", "b"]
    assert [r["rank"] for r in payload["results"]] == [1, 2]
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "medical-imaging-platform",
            "query-similar-studies",
            str(directory),
            "--sample-id",
            "missing",
        ],
    )
    with pytest.raises(SystemExit, match="Unknown sample ID: missing"):
        runpy.run_module("medical_imaging_platform", run_name="__main__")
    _module_cli(monkeypatch, ["evaluate-retrieval", directory, "--top-k", "1", "2", "--json"])
    metrics = json.loads(capsys.readouterr().out)
    assert metrics["query_count"] == 1
    assert metrics["recall_at_1"] == 1.0
    assert metrics["map_at_2"] == 1.0
    assert metrics["precision_at_2"] == 0.5
    assert metrics["label_consistency_at_2"] == 0.5
    with pytest.raises(SystemExit) as invalid:
        embeddings_main(["evaluate-retrieval", str(directory), "--top-k", "-1"])
    assert invalid.value.code == 2
    no_queries = _index(tmp_path / "no-queries", query=False)
    _module_cli(monkeypatch, ["evaluate-retrieval", no_queries, "--json"])
    empty = json.loads(capsys.readouterr().out)
    assert empty["query_count"] == 0
    assert empty["top_k"] == [1, 5]


def test_benchmark_cli_real_backend_and_overwrite_failure(tmp_path, monkeypatch, capsys):
    cnn = _index(tmp_path / "cnn", query=False)
    vit = _index(tmp_path / "vit", family="tiny_3d_vit", query=False)
    output = tmp_path / "benchmark"
    args = [
        "run-embedding-benchmark",
        "--cnn-index",
        cnn,
        "--vit-index",
        vit,
        "--output-dir",
        output,
        "--json",
    ]
    _module_cli(monkeypatch, args)
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["fairness_attestation"]["query_sample_count"] == 0
    assert all(model["metrics"] == {} for model in comparison["models"].values())
    assert comparison["fairness_attestation"]["top_k"] == [1, 5]
    with pytest.raises(FileExistsError, match="non-empty"):
        _module_cli(monkeypatch, args)


def test_index_failures_and_search_boundaries(tmp_path, monkeypatch):
    directory = _index(tmp_path / "index")
    index = EmbeddingIndex.load(directory)
    # Equal cosine scores sort by embedding ID, independently of storage order.
    assert [r["source_sample_id"] for r in index.search([1, 0], top_k=10)] == ["a", "q", "b"]
    assert [
        r["source_sample_id"] for r in index.search([1, 0], top_k=10, exclude_source_sample_id="a")
    ] == ["q", "b"]
    assert index.search([1, 0], top_k=0) == []
    with pytest.raises(ValueError, match="dimension"):
        index.search([1, 0, 0])
    with pytest.raises(ValueError, match="finite"):
        index.search([np.inf, 0])
    with pytest.raises(ValueError, match="rows"):
        EmbeddingIndex(index.matrix[:1], index.manifest)
    with pytest.raises(ValueError, match="finite"):
        EmbeddingIndex(np.full((3, 2), np.nan), index.manifest)
    (directory / "evidence" / "checksum_manifest.json").unlink()
    with pytest.raises(ValueError, match="Missing evidence"):
        _module_cli(monkeypatch, ["validate-embedding-index", directory])
    np.save(directory / "embedding_matrix.npy", np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_index(directory)


def test_empty_relevance_and_policy_configuration(tmp_path):
    metrics = retrieval_metrics([], {"sample_ids": set()}, ks=(1, 5))
    assert all(value == 0 for value in metrics.values())
    config_path = tmp_path / "embeddings.yaml"
    policy = {
        "policy_version": "M22",
        "gallery_splits": ["train"],
        "query_splits": ["test"],
        "top_k": [1, 5],
        "output_directory": "out",
    }
    config_path.write_text(yaml.safe_dump(policy))
    assert load_embedding_config(config_path).top_k == (1, 5)
    for invalid_k in ([True], [1.0], ["1"], [], [0], [5, 1], [1, 1]):
        config_path.write_text(yaml.safe_dump({**policy, "top_k": invalid_k}))
        with pytest.raises(ValueError):
            load_embedding_config(config_path)
    policy["query_splits"] = [""]
    config_path.write_text(yaml.safe_dump(policy))
    with pytest.raises(ValueError, match="non-empty"):
        load_embedding_config(config_path)


def test_api_embedding_search_returns_ranking_and_provenance(tmp_path):
    config_path = _write_api_config(tmp_path)
    directory = _index(tmp_path / "inputs" / "index")
    with TestClient(create_app(config_path)) as client:
        response = client.post(
            "/v1/embeddings/search",
            json={"index_directory": str(directory), "sample_id": "q", "top_k": 100},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"]["source_sample_id"] == "q"
    assert [r["source_sample_id"] for r in payload["results"]] == ["a", "b"]
    assert payload["encoder_family"] == "small_3d_cnn"
    assert payload["encoder_checkpoint_checksum"] == sha256_file(directory / "embedding_matrix.npy")
    assert payload["index_checksum"] == sha256_file(directory / "embedding_matrix.npy")
    assert payload["latency_ms"] >= 0
    assert "not diagnostic similarity" in payload["warnings"][0]


@pytest.mark.parametrize(
    "failure", ["zero_k", "large_k", "unknown", "tampered", "traversal", "sibling"]
)
def test_api_embedding_search_rejects_invalid_requests(tmp_path, failure):
    config_path = _write_api_config(tmp_path)
    directory = _index(tmp_path / "inputs" / "index")
    request = {"index_directory": str(directory), "sample_id": "q", "top_k": 5}
    expected, detail = 422, "Request validation failed"
    if failure in {"zero_k", "large_k"}:
        request["top_k"] = 0 if failure == "zero_k" else 101
    elif failure == "unknown":
        request["sample_id"] = "missing"
        expected, detail = 400, "query sample_id is not present"
    elif failure == "tampered":
        np.save(directory / "embedding_matrix.npy", np.ones((3, 2), dtype=np.float32))
        expected, detail = 400, "checksum mismatch"
    else:
        outside = _index(tmp_path / "inputs-sibling" / "index")
        request["index_directory"] = (
            str(tmp_path / "inputs" / ".." / "inputs-sibling" / "index")
            if failure == "traversal"
            else str(outside)
        )
        expected, detail = 403, "outside allowed input roots"
    with TestClient(create_app(config_path)) as client:
        response = client.post("/v1/embeddings/search", json=request)
    assert response.status_code == expected
    assert detail in response.text


def test_generated_indexes_real_cli_benchmark(generation_inputs, tmp_path, monkeypatch, capsys):
    dataset = generation_inputs["small_3d_cnn"][0]
    (dataset / "classification_dataset_manifest.json").rename(
        dataset / "classification_manifest.json"
    )
    indexes = {}
    for family, (_, config, checkpoint) in generation_inputs.items():
        # Use the filename and config wrapper produced by existing workflows.
        config.write_text(
            yaml.safe_dump({"settings": {"classification": yaml.safe_load(config.read_text())}})
        )
        index_dir = tmp_path / "inputs" / family
        _module_cli(
            monkeypatch,
            [
                "generate-embeddings",
                dataset,
                "--config",
                config,
                "--checkpoint",
                checkpoint,
                "--encoder-family",
                family,
                "--output-dir",
                index_dir,
                "--json",
            ],
        )
        generated = json.loads(capsys.readouterr().out)
        assert generated["source_dataset_manifest_checksum"] == sha256_file(
            dataset / "classification_manifest.json"
        )
        assert EmbeddingIndex.load(index_dir).sample_index == {"a": 0, "z": 1}
        assert (
            _parser()
            .parse_args(
                ["query-similar-studies", str(index_dir), "--sample-id", "z", "--top-k", "1"]
            )
            .top_k
            == 1
        )
        _module_cli(
            monkeypatch,
            ["query-similar-studies", index_dir, "--sample-id", "z", "--top-k", "1", "--json"],
        )
        assert [r["source_sample_id"] for r in json.loads(capsys.readouterr().out)["results"]] == [
            "a"
        ]
        _module_cli(monkeypatch, ["evaluate-retrieval", index_dir, "--json"])
        evaluation = json.loads(capsys.readouterr().out)
        assert evaluation["top_k"] == [1, 5]
        assert evaluation["recall_at_1"] == 1.0
        indexes[family] = index_dir
    config_path = _write_api_config(tmp_path)
    with TestClient(create_app(config_path)) as client:
        response = client.post(
            "/v1/embeddings/search",
            json={"index_directory": str(indexes["tiny_3d_vit"]), "sample_id": "z", "top_k": 1},
        )
    assert response.status_code == 200
    retrieval = response.json()
    assert retrieval["results"][0]["source_sample_id"] == "a"
    assert isinstance(retrieval["results"][0]["similarity"], float)
    st = Mock()
    render_embeddings(st, retrieval)
    st.warning.assert_called_once_with("Embedding-space similarity is not diagnostic similarity.")
    st.write.assert_called_once_with(
        {
            "query": retrieval["query"],
            "encoder_family": "tiny_3d_vit",
            "checkpoint_checksum": retrieval["encoder_checkpoint_checksum"],
            "index_checksum": retrieval["index_checksum"],
            "latency_ms": retrieval["latency_ms"],
        }
    )
    st.dataframe.assert_called_once_with(retrieval["results"], use_container_width=True)
    captions = [call.args[0] for call in st.caption.call_args_list]
    assert "Research and engineering use only; human interpretation is required." in captions
    assert all(warning in captions for warning in retrieval["warnings"])
    output = tmp_path / "benchmark"
    args = [
        "run-embedding-benchmark",
        "--cnn-index",
        indexes["small_3d_cnn"],
        "--vit-index",
        indexes["tiny_3d_vit"],
        "--output-dir",
        output,
        "--top-k",
        "1",
        "3",
        "--json",
    ]
    _module_cli(monkeypatch, args)
    comparison = json.loads(capsys.readouterr().out)
    assert comparison["fairness_attestation"]["top_k"] == [1, 3]
    for model in comparison["models"].values():
        assert model["query_count"] == 1
        assert model["metrics"]["gallery_size"] == 1
        assert model["metrics"]["recall_at_1"] == 1.0
        assert "recall_at_3" in model["metrics"]
        assert "recall_at_5" not in model["metrics"]
    manifest = json.loads((output / "embedding_benchmark_manifest.json").read_text())
    assert manifest["top_k"] == [1, 3]
    assert manifest["gallery_sample_ids"] == ["a"]
    assert manifest["query_sample_ids"] == ["z"]
    evidence = json.loads((output / "evidence" / "checksum_manifest.json").read_text())
    assert len(evidence["files"]) == 3
    assert all(
        sha256_file(output / name) == checksum for name, checksum in evidence["files"].items()
    )
    _module_cli(monkeypatch, [*args, "--overwrite"])
    repeated = json.loads(capsys.readouterr().out)
    assert repeated["fairness_attestation"] == comparison["fairness_attestation"]
    assert {name: model["metrics"] for name, model in repeated["models"].items()} == {
        name: model["metrics"] for name, model in comparison["models"].items()
    }


@pytest.mark.parametrize("ks", [(), (0,), (-1,), (5, 1), (1, 1), (True,), (1.5,), 5])
def test_benchmark_rejects_invalid_k_before_writing(tmp_path, ks):
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="unique, sorted, positive integers"):
        run_benchmark(tmp_path / "cnn", tmp_path / "vit", output, top_k=ks)
    assert not output.exists()


@pytest.mark.parametrize(
    "command", ["evaluate-retrieval", "run-embedding-benchmark", "query-similar-studies"]
)
def test_cli_rejects_invalid_k(tmp_path, capsys, command):
    if command == "run-embedding-benchmark":
        args = [command, "--cnn-index", "cnn", "--vit-index", "vit", "--output-dir", str(tmp_path)]
    elif command == "query-similar-studies":
        args = [command, "index", "--sample-id", "query"]
    else:
        args = [command, "index"]
    for invalid in (["0"], ["-1"], ["text"]):
        with pytest.raises(SystemExit) as rejected:
            embeddings_main([*args, "--top-k", *invalid])
        assert rejected.value.code == 2
        assert "top-k" in capsys.readouterr().err or invalid == ["text"]
    if command != "query-similar-studies":
        for invalid in (["5", "1"], ["1", "1"]):
            with pytest.raises(SystemExit) as rejected:
                embeddings_main([*args, "--top-k", *invalid])
            assert rejected.value.code == 2
            assert "unique, sorted, positive integers" in capsys.readouterr().err


def test_benchmark_rejects_different_gallery_membership(tmp_path):
    cnn = _index(tmp_path / "cnn")
    vit = _index(tmp_path / "vit", family="tiny_3d_vit")
    path = vit / "embedding_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["records"][0]["split"] = "test"
    path.write_text(json.dumps(manifest))
    evidence_path = vit / "evidence" / "checksum_manifest.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["files"]["embedding_manifest.json"] = sha256_file(path)
    evidence_path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="source dataset and membership"):
        run_benchmark(cnn, vit, tmp_path / "benchmark")
