import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.api.security import safe_output_child
from medical_imaging_platform.classification.checkpoint import (
    save_state_dict as save_classifier_state,
)
from medical_imaging_platform.classification.model_factory import build_classifier
from medical_imaging_platform.cli import main
from medical_imaging_platform.segmentation.checkpoint import save_state_dict as save_unet_state
from medical_imaging_platform.segmentation.model_factory import build_unet
from medical_imaging_platform.utils.config import (
    load_classification_config,
    load_segmentation_config,
)


def _write_api_config(
    tmp_path: Path,
    *,
    maximum_request_bytes: int = 1_000_000,
    maximum_array_bytes: int = 2_000_000,
    segmentation_checkpoint: Path | None = None,
    classification_checkpoint: Path | None = None,
    calibration: Path | None = None,
    threshold_policy: Path | None = None,
) -> Path:
    inputs = tmp_path / "inputs"
    evidence = tmp_path / "evidence"
    output = tmp_path / "outputs"
    for path in (inputs, evidence, output):
        path.mkdir(parents=True, exist_ok=True)
    payload = {
        "settings": {
            "api": {
                "policy_version": "test-api-v1",
                "service_name": "test-medical-imaging-api",
                "service_version": "0.1.0",
                "environment": "test",
                "host": "127.0.0.1",
                "port": 8001,
                "log_level": "INFO",
                "allowed_input_roots": [str(inputs), str(output)],
                "allowed_evidence_roots": [str(evidence), str(output)],
                "maximum_request_bytes": maximum_request_bytes,
                "maximum_array_bytes": maximum_array_bytes,
                "maximum_batch_size": 1,
                "request_timeout_seconds": 30,
                "enable_docs": False,
                "enable_openapi": False,
                "require_model_checksums": True,
                "require_quality_pass": True,
                "allow_degraded_review": True,
                "allow_external_bind": False,
                "allow_threshold_override": False,
                "segmentation_checkpoint": (
                    str(segmentation_checkpoint) if segmentation_checkpoint else None
                ),
                "classification_checkpoint": (
                    str(classification_checkpoint) if classification_checkpoint else None
                ),
                "classification_calibration": str(calibration) if calibration else None,
                "classification_threshold_policy": (
                    str(threshold_policy) if threshold_policy else None
                ),
                "longitudinal_config": "config/longitudinal.yaml",
                "output_directory": str(output),
            }
        }
    }
    path = tmp_path / "api.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return path


def _api_artifacts(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    seg_config = load_segmentation_config(Path("config/segmentation.yaml"))
    cls_config = load_classification_config(Path("config/classification.yaml"))
    seg_checkpoint = tmp_path / "segmentation.pt"
    cls_checkpoint = tmp_path / "classification.pt"
    save_unet_state(seg_checkpoint, build_unet(seg_config))
    save_classifier_state(cls_checkpoint, build_classifier(cls_config))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "method": "platt",
                "status": "fitted",
                "parameters": {"coef": 0.0, "intercept": 0.0},
                "diagnostics": {},
            }
        ),
        encoding="utf-8",
    )
    threshold_policy = tmp_path / "threshold_policy.json"
    threshold_policy.write_text(
        json.dumps({"method": "fixed", "selected_threshold": 0.5}),
        encoding="utf-8",
    )
    return {
        "segmentation": seg_checkpoint,
        "classification": cls_checkpoint,
        "calibration": calibration,
        "threshold_policy": threshold_policy,
    }


@pytest.fixture()
def api_config_path(tmp_path: Path) -> Path:
    artifacts = _api_artifacts(tmp_path)
    return _write_api_config(
        tmp_path,
        segmentation_checkpoint=artifacts["segmentation"],
        classification_checkpoint=artifacts["classification"],
        calibration=artifacts["calibration"],
        threshold_policy=artifacts["threshold_policy"],
    )


@pytest.fixture()
def client(api_config_path: Path) -> TestClient:
    return TestClient(create_app(api_config_path))


def _input_root(config_path: Path) -> Path:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return Path(payload["settings"]["api"]["allowed_input_roots"][0])


def test_health_version_readiness_and_security_headers(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "api-test-request"})

    assert response.status_code == 200
    assert response.json()["request_id"] == "api-test-request"
    assert response.headers["X-Request-ID"] == "api-test-request"
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    assert client.get("/version").json()["api_version"] == "0.1.0"
    assert client.get("/ready").json()["status"] == "ready"
    assert client.get("/openapi.json").status_code == 404


def test_segmentation_prediction_from_allowed_npy(
    client: TestClient, api_config_path: Path
) -> None:
    input_path = _input_root(api_config_path) / "segmentation.npy"
    np.save(input_path, np.zeros((32, 32, 32), dtype=np.float32))

    response = client.post(
        "/v1/segmentation/predict",
        json={
            "input_path": str(input_path),
            "spacing_mm": [1.0, 1.0, 1.0],
            "persist_output": True,
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "completed"
    assert "checkpoint_checksum" in payload
    assert "probability_map" not in payload
    assert not payload["output_paths"]["predicted_mask"].startswith("/")


def test_classification_prediction_from_array_payload(client: TestClient) -> None:
    values = np.zeros((16, 16, 16), dtype=np.float32).ravel().tolist()

    response = client.post(
        "/v1/classification/predict",
        json={"array": {"shape": [16, 16, 16], "values": values}},
    )

    payload = response.json()
    assert response.status_code == 200
    assert payload["engineering_label"] == "indeterminate"
    assert payload["abstained"] is True
    assert "calibration_checksum" in payload


def test_longitudinal_analysis_and_review_are_read_only(client: TestClient) -> None:
    previous = np.zeros((16, 16, 16), dtype=np.uint8)
    current = np.zeros((16, 16, 16), dtype=np.uint8)
    previous[4:7, 4:7, 4:7] = 1
    current[4:8, 4:8, 4:8] = 1

    response = client.post(
        "/v1/longitudinal/analyse",
        json={
            "previous_array": {"shape": [16, 16, 16], "values": previous.ravel().tolist()},
            "current_array": {"shape": [16, 16, 16], "values": current.ravel().tolist()},
            "previous_spacing_mm": [1.0, 1.0, 1.0],
            "current_spacing_mm": [1.0, 1.0, 1.0],
            "case_id": "case-api-001",
            "research_subject_id": "research-subject-api-001",
            "side": "left",
        },
    )

    payload = response.json()
    assert response.status_code == 200
    assert not payload["evidence_path"].startswith("/")
    review = client.get(f"/v1/review/longitudinal/{payload['evidence_path']}")
    assert review.status_code == 200
    assert review.json()["evidence_type"] == "longitudinal"
    assert "/" not in json.dumps(review.json()["provenance"])


def test_review_rejects_path_traversal_and_detects_tampering(client: TestClient) -> None:
    previous = np.zeros((16, 16, 16), dtype=np.uint8)
    current = np.zeros((16, 16, 16), dtype=np.uint8)
    previous[4:7, 4:7, 4:7] = 1
    current[4:8, 4:8, 4:8] = 1
    response = client.post(
        "/v1/longitudinal/analyse",
        json={
            "previous_array": {"shape": [16, 16, 16], "values": previous.ravel().tolist()},
            "current_array": {"shape": [16, 16, 16], "values": current.ravel().tolist()},
            "previous_spacing_mm": [1.0, 1.0, 1.0],
            "current_spacing_mm": [1.0, 1.0, 1.0],
            "case_id": "case-api-002",
            "research_subject_id": "research-subject-api-002",
            "side": "right",
        },
    )
    evidence_id = response.json()["evidence_path"]

    traversal = client.get("/v1/review/longitudinal/%2e%2e%2fsecret")
    assert traversal.status_code in {403, 404}

    app_config = client.app.state.api_config
    (app_config.output_directory / evidence_id / "longitudinal_summary.json").write_text(
        "{}\n", encoding="utf-8"
    )
    tampered = client.get(f"/v1/review/longitudinal/{evidence_id}")
    assert tampered.status_code == 409
    assert tampered.json()["error_code"] == "API-INTEGRITY-409"


def test_path_array_size_and_validation_failures(
    client: TestClient, api_config_path: Path, tmp_path: Path
) -> None:
    input_root = _input_root(api_config_path)
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((32, 32, 32), dtype=np.float32))
    assert (
        client.post(
            "/v1/segmentation/predict",
            json={"input_path": str(outside), "spacing_mm": [1.0, 1.0, 1.0]},
        ).status_code
        == 403
    )

    text_file = input_root / "input.txt"
    text_file.write_text("not an array", encoding="utf-8")
    unsupported = client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(text_file), "spacing_mm": [1.0, 1.0, 1.0]},
    )
    assert unsupported.status_code == 422

    invalid_array = input_root / "invalid.npy"
    np.save(invalid_array, np.full((32, 32, 32), np.inf, dtype=np.float32))
    invalid = client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(invalid_array), "spacing_mm": [1.0, 1.0, 1.0]},
    )
    assert invalid.status_code == 422

    symlink = input_root / "linked.npy"
    if hasattr(symlink, "symlink_to"):
        symlink.symlink_to(outside)
        blocked = client.post(
            "/v1/segmentation/predict",
            json={"input_path": str(symlink), "spacing_mm": [1.0, 1.0, 1.0]},
        )
        assert blocked.status_code == 403

    mismatch = client.post(
        "/v1/classification/predict",
        json={"array": {"shape": [16, 16, 16], "values": [0.0]}},
    )
    assert mismatch.status_code == 422

    bad_spacing = client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(input_root / "missing.npy"), "spacing_mm": [0.0, 1.0, 1.0]},
    )
    assert bad_spacing.status_code == 422


def test_oversized_request_and_sanitized_not_ready_error(tmp_path: Path) -> None:
    config_path = _write_api_config(tmp_path, maximum_request_bytes=10)
    oversized_client = TestClient(create_app(config_path))

    response = oversized_client.post(
        "/v1/classification/predict",
        content=b"x" * 20,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error_code"] == "API-SIZE-413"

    not_ready_client = TestClient(create_app(_write_api_config(tmp_path / "not-ready")))
    not_ready = not_ready_client.post(
        "/v1/classification/predict",
        json={"array": {"shape": [16, 16, 16], "values": [0.0] * 4096}},
    )
    assert not_ready.status_code == 503
    assert not_ready.json()["error_code"] == "API-NOTREADY-503"
    assert "/" not in not_ready.json()["message"]


def test_invalid_timepoint_and_cli_api_commands(
    client: TestClient, api_config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    zeros = [0] * 4096
    invalid_timepoint = client.post(
        "/v1/longitudinal/analyse",
        json={
            "previous_array": {"shape": [16, 16, 16], "values": zeros},
            "current_array": {"shape": [16, 16, 16], "values": zeros},
            "previous_spacing_mm": [1.0, 1.0, 1.0],
            "current_spacing_mm": [1.0, 1.0, 1.0],
            "case_id": "case-api-003",
            "research_subject_id": "research-subject-api-003",
            "side": "left",
            "previous_timepoint": "current",
            "current_timepoint": "previous",
        },
    )
    assert invalid_timepoint.status_code == 422

    assert main(["validate-api-config", "--config", str(api_config_path)]) == 0
    assert "Validated API configuration test-api-v1." in capsys.readouterr().out
    assert main(["inspect-api-readiness", "--config", str(api_config_path)]) == 0
    assert "API readiness status=ready." in capsys.readouterr().out


def test_api_edge_controls_and_sanitized_unhandled_errors(
    tmp_path: Path, api_config_path: Path
) -> None:
    artifacts = _api_artifacts(tmp_path / "edge")
    missing_config = _write_api_config(tmp_path / "missing")
    missing_client = TestClient(create_app(missing_config))

    not_ready = missing_client.post(
        "/v1/segmentation/predict",
        json={"array": {"shape": [32, 32, 32], "values": [0.0] * 32768}, "spacing_mm": [1, 1, 1]},
    )
    assert not_ready.status_code == 503

    configured = create_app(api_config_path)

    @configured.get("/boom")
    def boom() -> None:
        raise RuntimeError("absolute /secret/path must not leak")

    sanitized_client = TestClient(configured, raise_server_exceptions=False)
    internal = sanitized_client.get("/boom")
    assert internal.status_code == 500
    assert internal.json()["message"] == "Internal API error."

    input_root = _input_root(api_config_path)
    remote = sanitized_client.post(
        "/v1/segmentation/predict",
        json={"input_path": "https://example.test/input.npy", "spacing_mm": [1, 1, 1]},
    )
    assert remote.status_code == 403

    missing_path = sanitized_client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(input_root / "missing.npy"), "spacing_mm": [1, 1, 1]},
    )
    assert missing_path.status_code == 404

    object_array = input_root / "object.npy"
    np.save(object_array, np.asarray([{"unsafe": "pickle"}], dtype=object))
    invalid_numpy = sanitized_client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(object_array), "spacing_mm": [1, 1, 1]},
    )
    assert invalid_numpy.status_code == 422

    two_dimensional = input_root / "two_dimensional.npy"
    np.save(two_dimensional, np.zeros((2, 2), dtype=np.float32))
    invalid_shape = sanitized_client.post(
        "/v1/segmentation/predict",
        json={"input_path": str(two_dimensional), "spacing_mm": [1, 1, 1]},
    )
    assert invalid_shape.status_code == 422

    threshold_override = sanitized_client.post(
        "/v1/segmentation/predict",
        json={
            "array": {"shape": [32, 32, 32], "values": [0.0] * 32768},
            "spacing_mm": [1, 1, 1],
            "threshold": 0.5,
        },
    )
    assert threshold_override.status_code == 403

    wrong_classifier_config = _write_api_config(
        tmp_path / "wrong-classifier",
        classification_checkpoint=artifacts["segmentation"],
        calibration=artifacts["calibration"],
        threshold_policy=artifacts["threshold_policy"],
    )
    wrong_classifier = TestClient(create_app(wrong_classifier_config)).post(
        "/v1/classification/predict",
        json={"array": {"shape": [16, 16, 16], "values": [0.0] * 4096}},
    )
    assert wrong_classifier.status_code == 409

    wrong_segmentation_config = _write_api_config(
        tmp_path / "wrong-segmentation",
        segmentation_checkpoint=artifacts["classification"],
    )
    wrong_segmentation = TestClient(create_app(wrong_segmentation_config)).post(
        "/v1/segmentation/predict",
        json={"array": {"shape": [32, 32, 32], "values": [0.0] * 32768}, "spacing_mm": [1, 1, 1]},
    )
    assert wrong_segmentation.status_code == 409

    previous = input_root / "previous.npy"
    current = input_root / "current.npy"
    np.save(previous, np.zeros((16, 16, 16), dtype=np.uint8))
    np.save(current, np.zeros((16, 16, 16), dtype=np.uint8))
    path_based_longitudinal = sanitized_client.post(
        "/v1/longitudinal/analyse",
        json={
            "previous_mask_path": str(previous),
            "current_mask_path": str(current),
            "previous_spacing_mm": [1.0, 1.0, 1.0],
            "current_spacing_mm": [1.0, 1.0, 1.0],
            "case_id": "case-api-004",
            "research_subject_id": "research-subject-api-004",
            "side": "left",
        },
    )
    assert path_based_longitudinal.status_code == 200

    assert sanitized_client.get("/v1/review/segmentation/missing-experiment").status_code == 404
    assert sanitized_client.get("/v1/review/classification/missing-experiment").status_code == 404

    api_config = APIConfig.model_validate(
        {
            "policy_version": "test",
            "service_name": "test",
            "service_version": "0",
            "environment": "test",
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "INFO",
            "allowed_input_roots": [str(tmp_path / "input")],
            "allowed_evidence_roots": [str(tmp_path / "evidence")],
            "maximum_request_bytes": 1,
            "maximum_array_bytes": 1,
            "maximum_batch_size": 1,
            "request_timeout_seconds": 1,
        }
    )
    with pytest.raises(Exception, match="escapes configured root"):
        safe_output_child(tmp_path / "root", "../outside")
    assert api_config.host == "127.0.0.1"
    with pytest.raises(ValueError, match="0.0.0.0"):
        APIConfig.model_validate(
            {
                **api_config.model_dump(),
                "host": "0.0.0.0",
                "allow_external_bind": False,
            }
        )
