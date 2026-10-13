"""M23 integration contracts, real CPU demo and governed reviewer regressions."""

import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from test_api import _write_api_config
from test_embeddings_coverage import _index
from test_reviewer_ui import _config, _FakeStreamlit, _Upload

from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.cli import build_parser, main
from medical_imaging_platform.demonstration import run_platform_demo, validate_demo
from medical_imaging_platform.embeddings.benchmark import run_benchmark
from medical_imaging_platform.embeddings.cli import COMMANDS
from medical_imaging_platform.embeddings.config import load_embedding_config
from medical_imaging_platform.embeddings.core import EmbeddingIndex, index_policy, query_index
from medical_imaging_platform.embeddings.metrics import evaluate_index
from medical_imaging_platform.evidence import (
    ArtifactBinding,
    EvidenceEnvelope,
    array_checksum,
    evidence_checksum,
    sha256_file,
    validate_envelope,
    verify_artifact,
    write_envelope,
)
from medical_imaging_platform.portfolio import evidence as portfolio
from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components.visualisation import (
    render_segmentation_overlay,
)
from medical_imaging_platform.reviewer_ui.export import export_review_session
from medical_imaging_platform.reviewer_ui.pages import embeddings, governance, segmentation
from medical_imaging_platform.reviewer_ui.review import create_review_decision


def _rehash(directory):
    path = directory / "embedding_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["matrix_checksum"] = sha256_file(directory / "embedding_matrix.npy")
    path.write_text(json.dumps(manifest))
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


def test_installed_module_and_legacy_command_registry_parity():
    entry = next(
        e
        for e in importlib.metadata.distribution(
            "enterprise-medical-imaging-ai-platform"
        ).entry_points
        if e.name == "medical-imaging-platform"
    )
    assert entry.load() is main
    console = shutil.which("medical-imaging-platform") or str(
        Path(sysconfig.get_path("scripts")) / "medical-imaging-platform"
    )
    assert Path(console).is_file()
    installed = subprocess.run(
        [console, "--help"], capture_output=True, text=True, check=True
    ).stdout
    module = subprocess.run(
        [sys.executable, "-m", "medical_imaging_platform", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert installed == module
    for command in COMMANDS | {"platform-demo", "train-segmentation", "run-detection-benchmark"}:
        assert command in installed
    assert (
        build_parser()
        .parse_args(["query-similar-studies", "index", "--sample-id", "q", "--top-k", "3"])
        .top_k
        == 3
    )


@pytest.mark.parametrize(
    "failure",
    [
        "norm",
        "dtype",
        "dimension",
        "order",
        "duplicate",
        "metadata",
        "record_encoder",
        "evidence",
        "shape",
        "nonfinite",
        "symlink",
        "size",
    ],
)
def test_persisted_index_validation_is_strict_without_mutation(tmp_path, failure):
    directory = _index(tmp_path / "index")
    matrix_path = directory / "embedding_matrix.npy"
    manifest_path = directory / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure in {"norm", "dtype", "shape", "nonfinite"}:
        matrix = np.load(matrix_path)
        if failure == "norm":
            matrix[0] *= 2
        elif failure == "dtype":
            matrix = matrix.astype(np.float64)
        elif failure == "shape":
            matrix = matrix.ravel()
        else:
            matrix[0, 0] = np.nan
        np.save(matrix_path, matrix)
    elif failure == "dimension":
        manifest["embedding_dimension"] = 3
    elif failure == "order":
        manifest["records"].reverse()
    elif failure == "duplicate":
        manifest["records"][1]["embedding_id"] = manifest["records"][0]["embedding_id"]
    elif failure == "metadata":
        manifest.pop("encoder_checkpoint_checksum")
    elif failure == "record_encoder":
        manifest["records"][0]["encoder_family"] = "tiny_3d_vit"
    manifest_path.write_text(json.dumps(manifest))
    _rehash(directory)
    if failure == "evidence":
        (directory / "evidence" / "checksum_manifest.json").write_text('{"files": {}}')
    elif failure == "symlink":
        outside = tmp_path / "outside.npy"
        shutil.copyfile(matrix_path, outside)
        matrix_path.unlink()
        matrix_path.symlink_to(outside)
    before = matrix_path.read_bytes()
    with pytest.raises(ValueError):
        EmbeddingIndex.load(directory, maximum_bytes=1 if failure == "size" else None)
    assert matrix_path.read_bytes() == before


def test_loaded_vectors_are_exactly_persisted_and_provenance_is_complete(tmp_path):
    directory = _index(tmp_path / "index")
    before = (directory / "embedding_matrix.npy").read_bytes()
    stored = np.load(directory / "embedding_matrix.npy")
    index = EmbeddingIndex.load(directory)
    assert np.array_equal(index.matrix, stored)
    assert (directory / "embedding_matrix.npy").read_bytes() == before
    hit = query_index(index, "q", 1)[0]
    assert hit["encoder_family"] == index.manifest["encoder_family"]
    assert hit["checkpoint_checksum"] == index.manifest["encoder_checkpoint_checksum"]


def test_policy_controls_relevance_membership_k_and_exclusions(tmp_path, capsys):
    directory = _index(tmp_path / "index")
    index = EmbeddingIndex.load(directory)
    base = index_policy(index)
    # Validation-only gallery makes the negative sample relevant under label policy.
    index.records[2]["label"] = 0
    policy = base.model_copy(
        update={
            "gallery_splits": ["validation"],
            "query_splits": ["test"],
            "relevance": "label",
            "top_k": (1,),
        }
    )
    label = evaluate_index(index, policy)
    scenario = evaluate_index(index, policy.model_copy(update={"relevance": "scenario"}))
    assert label["recall_at_1"] == 1
    assert scenario["recall_at_1"] == 0
    assert label["top_k"] == [1]
    assert (
        evaluate_index(index, policy.model_copy(update={"query_splits": ["train"]}))["query_count"]
        == 1
    )
    self_policy = base.model_copy(update={"gallery_splits": ["test"], "exclude_self": False})
    assert query_index(index, "q", 5, self_policy)[0]["source_sample_id"] == "q"
    assert query_index(index, "q", 5, self_policy.model_copy(update={"exclude_self": True})) == []
    assert (
        query_index(index, "q", 5, self_policy.model_copy(update={"exclude_same_source": True}))
        == []
    )
    config = tmp_path / "policy.yaml"
    config.write_text(yaml.safe_dump(policy.model_dump(mode="json")))
    assert main(["evaluate-retrieval", str(directory), "--config", str(config), "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["policy"]["gallery_splits"] == ["validation"]
    assert output["policy"]["relevance"] == "label"
    assert load_embedding_config(config).query_splits == ["test"]


def test_benchmark_rejects_incompatible_persisted_policies(tmp_path):
    cnn = _index(tmp_path / "cnn")
    vit = _index(tmp_path / "vit", family="tiny_3d_vit")
    manifest_path = vit / "embedding_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["retrieval_policy"] = (
        index_policy(EmbeddingIndex.load(vit))
        .model_copy(update={"relevance": "label"})
        .model_dump(mode="json")
    )
    manifest_path.write_text(json.dumps(manifest))
    _rehash(vit)
    with pytest.raises(ValueError, match="policies"):
        run_benchmark(cnn, vit, tmp_path / "benchmark")


def test_envelope_and_review_binding_detect_tampering(tmp_path):
    artifact = tmp_path / "mask.npy"
    mask = np.zeros((2, 3, 3), dtype=np.uint8)
    np.save(artifact, mask)
    envelope = EvidenceEnvelope(
        artifact_type="segmentation_mask",
        run_id="run-1",
        artifact_reference="mask.npy",
        artifact_checksum=sha256_file(artifact),
        quality_status="PASS",
    )
    path = tmp_path / "envelope.json"
    payload = write_envelope(path, envelope)
    assert validate_envelope(path, artifact) == envelope
    binding = ArtifactBinding(
        artifact_type="segmentation_mask",
        artifact_reference="prediction:run-1:mask",
        artifact_checksum=array_checksum(mask),
        checksum_kind="sha256-array",
        run_id="run-1",
    )
    verify_artifact(binding, artifact)
    args = {
        "request_id": "run-1",
        "evidence_type": "segmentation",
        "evidence_id": "evidence-1",
        "model_engineering_label": "synthetic",
        "quality_status": "PASS",
        "reviewer_decision": "needs_secondary_review",
    }
    decision = create_review_decision(**args, artifact=binding)
    export_review_session(
        decision=decision,
        evidence_summary={
            "artifact": binding.model_dump(mode="json"),
            "mask_preview": {"values": [1]},
        },
        output_root=tmp_path / "reviews",
    )
    summary = json.loads(
        (tmp_path / "reviews" / decision.review_id / "reviewed_evidence_summary.json").read_text()
    )
    assert "mask_preview" not in summary
    changed = binding.model_copy(update={"artifact_checksum": "1" * 64})
    assert create_review_decision(**args, artifact=changed).review_id != decision.review_id
    with pytest.raises(ValueError, match="mismatch"):
        export_review_session(
            decision=decision,
            evidence_summary={"artifact": changed.model_dump(mode="json")},
            output_root=tmp_path / "rejected",
        )
    assert not (tmp_path / "rejected").exists()
    np.save(artifact, np.ones_like(mask))
    with pytest.raises(ValueError, match="mismatch"):
        verify_artifact(binding, artifact)
    file_binding = binding.model_copy(
        update={"checksum_kind": "sha256-file", "artifact_checksum": sha256_file(artifact)}
    )
    verify_artifact(file_binding, artifact)
    with pytest.raises(ValueError, match="artifact checksum"):
        validate_envelope(path, artifact)
    payload["quality_status"] = "FAIL"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="envelope checksum"):
        validate_envelope(path, artifact)


class ReviewerSurface(_FakeStreamlit):
    def title(self, value):
        self.rendered.append(value)

    def dataframe(self, value, **kwargs):
        self.rendered.append(value)


def test_segmentation_overlay_and_review_provenance(tmp_path):
    source = np.zeros((4, 4, 4), dtype=np.float32)
    mask = np.eye(4, dtype=np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, source)
    upload = _Upload("source.npy", buffer.getvalue())
    surface = ReviewerSurface(uploads={"Upload bounded .npy volume": upload})
    response = {
        "source_checksum": array_checksum(source),
        "checkpoint_checksum": "a" * 64,
        "evidence_reference": "prediction:run:mask",
        "quality_findings": [],
        "mask_preview": {"slice_index": 2, "shape": [4, 4], "values": mask.ravel().tolist()},
    }
    client = Mock()
    client.predict_segmentation.return_value = response
    segmentation.render(surface, _config(tmp_path), client)
    assert any("prediction:run:mask" in str(value) for value in surface.rendered)
    assert any("a" * 64 in str(value) for value in surface.rendered)
    assert any(hasattr(value, "axes") for value in surface.rendered)
    assert np.array_equal(source, np.zeros_like(source))
    assert client.predict_segmentation.call_args.args[0]["preview_slice_index"] == 2
    response["source_checksum"] = "b" * 64
    segmentation.render(surface, _config(tmp_path), client)
    assert any("Source checksum differs" in str(value) for value in surface.rendered)
    with pytest.raises(ValueError, match="binary"):
        render_segmentation_overlay(surface, source, mask * 2, slice_index=2)
    with pytest.raises(ValueError, match="opacity"):
        render_segmentation_overlay(surface, source, mask, slice_index=2, opacity=2)


def test_embeddings_navigation_and_real_api_results(tmp_path, monkeypatch):
    from medical_imaging_platform.reviewer_ui import app

    api_path = _write_api_config(tmp_path)
    directory = _index(tmp_path / "inputs" / "index")
    raw = yaml.safe_load(api_path.read_text())
    raw["settings"]["api"]["embedding_indexes"] = [str(directory)]
    api_path.write_text(yaml.safe_dump(raw))
    with TestClient(create_app(api_path)) as transport:
        client = ReviewerAPIClient(_config(tmp_path), client=transport)
        assert client.embedding_indexes()["indexes"][0]["status"] == "available"
        surface = ReviewerSurface(selected_radio="Embeddings / Similarity Retrieval")
        monkeypatch.setitem(sys.modules, "streamlit", surface)
        monkeypatch.setattr(app, "load_reviewer_ui_config", lambda _: _config(tmp_path))
        monkeypatch.setattr(app, "ReviewerAPIClient", lambda _: client)
        app.run_reviewer_app()
        response = surface.session_state["last_response"]
        assert [r["source_sample_id"] for r in response["results"]] == ["a", "b"]
        assert response["results"][0]["similarity"] == 1
        assert any("not diagnostic similarity" in str(value) for value in surface.rendered)
        assert any("Provenance" in str(value) for value in surface.rendered)
        assert response["artifact"]["artifact_checksum"] == sha256_file(
            directory / "embedding_manifest.json"
        )
        # Reviewer export binds to the response even when the legacy selector has another default.
        surface.false_buttons = {"Clear reviewer session"}
        governance.render(surface, _config(tmp_path))
        assert surface.session_state["review_decision"]["artifact"] == response["artifact"]
        ready = transport.get("/ready")
        assert ready.json()["capabilities"]["segmentation"] == "unavailable"
        assert ready.json()["capabilities"]["embeddings"] == "available"
        np.save(directory / "embedding_matrix.npy", np.zeros((3, 2), dtype=np.float32))
        assert transport.get("/ready").status_code == 503
        assert client.embedding_indexes()["indexes"][0]["status"] == "unavailable"


def test_unavailable_embedding_ui_is_clear(tmp_path):
    client = Mock()
    client.embedding_indexes.return_value = {"indexes": []}
    surface = ReviewerSurface()
    embeddings.render_search(surface, _config(tmp_path), client)
    assert any("No available index" in str(value) for value in surface.rendered)
    client.search_embeddings.assert_not_called()


@pytest.fixture(scope="module")
def canonical_demo(tmp_path_factory):
    root = tmp_path_factory.mktemp("canonical-platform-demo")
    return root, run_platform_demo(root)


def test_real_demo_records_runs_checksums_roles_and_model_context(canonical_demo):
    root, summary = canonical_demo
    assert validate_demo(root) == summary
    assert summary["status"] == "PASS"
    assert summary["context"]["cases"] == 12
    assert summary["context"]["epochs"] == 1
    records = [json.loads((root / entry["envelope"]).read_text()) for entry in summary["artifacts"]]
    assert {r["artifact_type"] for r in records} >= {
        "preprocessing",
        "preprocessing_baseline",
        "localisation",
        "segmentation",
        "classification_benchmark",
        "detection_benchmark",
        "embedding_index",
        "retrieval_benchmark",
        "longitudinal",
    }
    masks = [r for r in records if r["artifact_type"] == "segmentation_prediction"]
    assert all(r["data_role"] == "predicted" for r in masks)
    pair_path = next(
        root / r["artifact_reference"] for r in records if r["artifact_type"] == "longitudinal"
    )
    pair = json.loads(pair_path.read_text())
    assert pair["source_checksums"]["previous_mask"] == masks[0]["artifact_checksum"]
    assert pair["source_checksums"]["current_mask"] == masks[1]["artifact_checksum"]
    source = json.loads((root / "synthetic/manifest.json").read_text())
    case = next(r for r in source["records"] if r["case_id"] == pair["case_id"])
    assert pair["side"] == case["lesion_side"]
    for mask in masks:
        metadata = json.loads(
            (root / mask["artifact_reference"]).with_name("inference_metadata.json").read_text()
        )
        assert mask["warnings"] == metadata["warnings"]
    comparison = json.loads((root / "engineering_comparison.json").read_text())
    assert set(comparison["classification"]) == {"small_3d_cnn", "tiny_3d_vit"}
    assert set(comparison["detection"]) == {"faster_rcnn", "tiny_detr"}
    assert all(r["matrix_storage_bytes"] > 0 for r in comparison["retrieval"].values())
    assert summary["capability_status"]["aws"] == "target state"
    assert "independent" in summary["workflow"]["models"]
    with pytest.raises(FileExistsError):
        run_platform_demo(root)


def test_demo_index_and_file_tampering_fail(canonical_demo):
    root, summary = canonical_demo
    path = root / "engineering_comparison.md"
    original = path.read_bytes()
    path.write_text("tampered")
    try:
        with pytest.raises(ValueError, match="file checksum"):
            validate_demo(root)
    finally:
        path.write_bytes(original)
    index = root / "evidence_index.json"
    original = index.read_bytes()
    changed = {**summary, "status": "FAIL"}
    index.write_text(json.dumps(changed))
    try:
        with pytest.raises(ValueError, match="index checksum"):
            validate_demo(root)
    finally:
        index.write_bytes(original)
    # Even correctly rehashed summary metadata cannot substitute another source/run reference.
    changed = json.loads(original)
    changed["artifacts"][0]["run_id"] = "wrong-run"
    changed["evidence_checksum"] = evidence_checksum(changed)
    index.write_text(json.dumps(changed))
    try:
        with pytest.raises(ValueError, match="run reference"):
            validate_demo(root)
    finally:
        index.write_bytes(original)


def test_readiness_tracks_current_scope_and_rejects_missing_or_duplicate_milestones():
    entries = portfolio.milestone_completion_matrix()
    assert portfolio._milestones_current(entries)
    assert not portfolio._milestones_current(entries[:-1])
    assert not portfolio._milestones_current(entries + [entries[-1]])
    assert {entry.milestone for entry in entries} >= {19, 20, 21, 22, 23}
    assert "image_embeddings_exact_retrieval" in {
        row["capability"] for row in portfolio.build_capability_inventory()
    }
