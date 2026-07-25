import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from medical_imaging_platform.cli import main
from medical_imaging_platform.reviewer_ui.api_client import ReviewerAPIClient
from medical_imaging_platform.reviewer_ui.components import (
    disclaimers,
    provenance,
    quality,
    status,
    visualisation,
)
from medical_imaging_platform.reviewer_ui.config import load_reviewer_ui_config
from medical_imaging_platform.reviewer_ui.export import export_review_session
from medical_imaging_platform.reviewer_ui.formatting import (
    format_api_error,
    format_classification_response,
    format_evidence_response,
    format_health_status,
    format_longitudinal_response,
    format_segmentation_response,
)
from medical_imaging_platform.reviewer_ui.models import (
    ReviewerAPIError,
    ReviewerUIConfig,
    create_timestamp,
    disclaimer,
    without_sensitive_values,
)
from medical_imaging_platform.reviewer_ui.pages import (
    classification,
    evidence,
    governance,
    longitudinal,
    overview,
    segmentation,
)
from medical_imaging_platform.reviewer_ui.review import create_review_decision
from medical_imaging_platform.reviewer_ui.security import (
    ReviewerUISecurityError,
    array_payload,
    enforce_loopback_url,
    load_uploaded_npy,
    safe_review_output_dir,
    validate_evidence_id,
    validate_upload_metadata,
)
from medical_imaging_platform.reviewer_ui.state import (
    initialise_state,
    remember_response,
    reset_state,
)
from medical_imaging_platform.utils.config import ConfigError


def _config(tmp_path: Path, **updates: object) -> ReviewerUIConfig:
    payload = {
        "policy_version": "test-reviewer-ui-v1",
        "page_title": "Reviewer",
        "page_icon": "R",
        "layout": "wide",
        "api_base_url": "http://127.0.0.1:8000",
        "request_timeout_seconds": 5,
        "allowed_upload_extensions": [".npy", ".json"],
        "maximum_upload_bytes": 100_000,
        "maximum_review_items": 5,
        "enable_segmentation_page": True,
        "enable_classification_page": True,
        "enable_longitudinal_page": True,
        "enable_evidence_page": True,
        "enable_governance_page": True,
        "allow_local_array_upload": True,
        "allow_evidence_export": True,
        "allow_remote_api": False,
        "allow_remote_bind": False,
        "host": "127.0.0.1",
        "port": 8502,
        "review_output_directory": tmp_path / "reviewer-sessions",
        "show_debug_details": False,
    }
    payload.update(updates)
    return ReviewerUIConfig.model_validate(payload)


def _write_config(tmp_path: Path, **updates: object) -> Path:
    config = _config(tmp_path, **updates).model_dump(mode="json")
    path = tmp_path / "reviewer_ui.yaml"
    path.write_text(
        yaml.safe_dump({"settings": {"reviewer_ui": config}}, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _client(config: ReviewerUIConfig) -> ReviewerAPIClient:
    def handler(request: httpx.Request) -> httpx.Response:
        rid = request.headers.get("X-Request-ID", "missing")
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "healthy"}, headers={"X-Request-ID": rid})
        if request.url.path == "/ready":
            return httpx.Response(200, json={"status": "ready"}, headers={"X-Request-ID": rid})
        if request.url.path == "/version":
            return httpx.Response(200, json={"api_version": "0.1.0"})
        if request.url.path == "/v1/segmentation/predict":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "predicted_voxel_count": 4,
                    "predicted_volume_mm3": 10.0,
                    "probability_summary": {"min": 0.1, "mean": 0.2, "max": 0.9},
                    "quality_status": "PASS",
                    "quality_findings": [],
                    "checkpoint_checksum": "abc",
                    "output_paths": {"predicted_mask": "predicted_mask.npy"},
                    "duration_ms": 1.5,
                    "disclaimer": "research-only",
                },
            )
        if request.url.path == "/v1/classification/predict":
            return httpx.Response(
                200,
                json={
                    "raw_probability": 0.5,
                    "calibrated_probability": 0.5,
                    "threshold": 0.5,
                    "engineering_label": "indeterminate",
                    "abstained": True,
                    "abstention_reason": "inside interval",
                    "quality_status": "PASS_WITH_WARNINGS",
                    "checkpoint_checksum": "ckpt",
                    "calibration_checksum": "cal",
                    "threshold_policy_checksum": "thr",
                    "duration_ms": 2.0,
                    "disclaimer": "research-only",
                },
            )
        if request.url.path == "/v1/longitudinal/analyse":
            return httpx.Response(
                200,
                json={
                    "measurements": {"previous": [], "current": []},
                    "match_summary": [{"match_confidence": 0.3}],
                    "change_metrics": [{"volume_change_mm3": None, "diameter_change_mm": None}],
                    "engineering_label": ["indeterminate"],
                    "upstream_quality_propagation": {"classification_abstention": "ABSTAINED"},
                    "quality_findings": [{"rule_id": "UI-QC-QUALITY-001", "passed": False}],
                    "evidence_path": "longitudinal-case",
                    "duration_ms": 3.0,
                    "disclaimer": "research-only",
                },
            )
        if request.url.path == "/v1/review/longitudinal/analysis-1":
            return httpx.Response(
                200,
                json={
                    "evidence_type": "longitudinal",
                    "status": "PASS",
                    "summary": {"analysis_id": "analysis-1"},
                    "quality_findings": [],
                    "provenance": {"source_checksums": {"mask": "123"}},
                },
            )
        return httpx.Response(
            409,
            json={
                "error_code": "API-INTEGRITY-409",
                "message": "Evidence integrity validation failed.",
                "request_id": rid,
                "details": {},
            },
        )

    return ReviewerAPIClient(
        config,
        client=httpx.Client(
            base_url=config.api_base_url,
            transport=httpx.MockTransport(handler),
            timeout=float(config.request_timeout_seconds),
        ),
    )


class _Upload:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


class _FakeStreamlit:
    def __init__(
        self,
        uploads: dict[str, _Upload] | None = None,
        false_buttons: set[str] | None = None,
        selected_radio: str | None = None,
    ) -> None:
        self.uploads = uploads or {}
        self.false_buttons = false_buttons or set()
        self.selected_radio = selected_radio
        self.session_state: dict[str, object] = {}
        self.rendered: list[object] = []
        self.sidebar = self

    def set_page_config(self, **kwargs: object) -> None:
        self.rendered.append({"page_config": kwargs})

    def header(self, value: object) -> None:
        self.rendered.append(value)

    def subheader(self, value: object) -> None:
        self.rendered.append(value)

    def caption(self, value: object) -> None:
        self.rendered.append(value)

    def warning(self, value: object) -> None:
        self.rendered.append(value)

    def error(self, value: object) -> None:
        self.rendered.append(value)

    def info(self, value: object) -> None:
        self.rendered.append(value)

    def success(self, value: object) -> None:
        self.rendered.append(value)

    def write(self, value: object) -> None:
        self.rendered.append(value)

    def json(self, value: object) -> None:
        self.rendered.append(value)

    def pyplot(self, value: object) -> None:
        self.rendered.append(value)

    def file_uploader(self, label: str, **kwargs: object) -> _Upload | None:
        key = str(kwargs.get("key", label))
        return self.uploads.get(key) or self.uploads.get(label)

    def text_input(self, label: str, value: str = "") -> str:
        if label == "Experiment or analysis ID":
            return "analysis-1"
        return value

    def text_area(self, label: str, **kwargs: object) -> str:
        return "Engineering review note."

    def button(self, label: str) -> bool:
        return label not in self.false_buttons

    def selectbox(self, label: str, options: list[str]) -> str:
        if label == "Evidence type":
            return "longitudinal"
        if label == "Reviewer decision":
            return "needs_secondary_review"
        return options[0]

    def radio(self, label: str, options: list[str]) -> str:
        return self.selected_radio or options[0]


def test_reviewer_ui_config_loading_and_validation(tmp_path: Path) -> None:
    path = _write_config(tmp_path)

    loaded = load_reviewer_ui_config(path)

    assert loaded.policy_version == "test-reviewer-ui-v1"
    assert loaded.api_base_url == "http://127.0.0.1:8000"
    assert loaded.show_debug_details is False
    with pytest.raises(ValueError, match="Remote API"):
        _config(tmp_path, api_base_url="https://example.com")
    with pytest.raises(ValueError, match="limited"):
        _config(tmp_path, allowed_upload_extensions=[".npy", ".exe"])
    with pytest.raises(ValueError, match="specific local"):
        _config(tmp_path, review_output_directory=Path("/"))
    with pytest.raises(ValueError, match="0.0.0.0"):
        _config(tmp_path, host="0.0.0.0")


def test_api_client_success_error_and_request_id(tmp_path: Path) -> None:
    client = _client(_config(tmp_path))

    health = client.health("rid-1")
    segmentation = client.predict_segmentation({"array": {"shape": [1, 1, 1], "values": [0]}})
    classification = client.predict_classification({"array": {"shape": [1, 1, 1], "values": [0]}})
    longitudinal = client.analyse_longitudinal({"previous_array": {}, "current_array": {}})
    evidence = client.review_longitudinal("analysis-1")

    assert health["request_id"] == "rid-1"
    assert format_health_status(health, client.ready())["operational"] == "yes"
    assert format_segmentation_response(segmentation)["predicted_voxel_count"] == 4
    assert format_classification_response(classification)["abstained"] is True
    assert format_classification_response(classification)["engineering_label"] == "indeterminate"
    assert format_longitudinal_response(longitudinal)["indeterminate"] is True
    assert format_evidence_response(evidence)["provenance"] == {"source_checksums": {"mask": "123"}}
    with pytest.raises(ReviewerAPIError) as exc:
        client.review_segmentation("bad")
    assert exc.value.status_code == 409
    assert exc.value.error_code == "API-INTEGRITY-409"


def test_api_client_unavailable_and_sanitised_path_error(tmp_path: Path) -> None:
    config = _config(tmp_path)

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "error_code": "API-VALID-422",
                "message": "bad /tmp/private/path",
                "request_id": "rid",
                "details": {},
            },
        )

    client = ReviewerAPIClient(
        config,
        client=httpx.Client(
            base_url=config.api_base_url, transport=httpx.MockTransport(failing_handler)
        ),
    )
    with pytest.raises(ReviewerAPIError, match="Review configured inputs"):
        client.health()

    unavailable = ReviewerAPIClient(config, client=httpx.Client(base_url="http://127.0.0.1:9"))
    with pytest.raises(ReviewerAPIError, match="unavailable"):
        unavailable.health()


def test_api_client_bad_response_shapes_and_display_helpers(tmp_path: Path) -> None:
    config = _config(tmp_path)

    invalid_json_client = ReviewerAPIClient(
        config,
        client=httpx.Client(
            base_url=config.api_base_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"not-json")),
        ),
    )
    with pytest.raises(ReviewerAPIError, match="invalid response"):
        invalid_json_client.health()

    list_client = ReviewerAPIClient(
        config,
        client=httpx.Client(
            base_url=config.api_base_url,
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=[])),
        ),
    )
    with pytest.raises(ReviewerAPIError, match="non-object"):
        list_client.health()

    error = ReviewerAPIError(422, "API-VALID-422", "Validation failed.", request_id="rid")
    assert format_api_error(error)["request_id"] == "rid"
    assert disclaimer().startswith("This platform is a research")
    assert "T" in create_timestamp()
    assert without_sensitive_values({"array": [1], "model_weights": "x", "safe": "ok"}) == {
        "safe": "ok"
    }


def test_upload_and_path_security(tmp_path: Path) -> None:
    config = _config(tmp_path, maximum_upload_bytes=5000)
    array = np.zeros((2, 2, 2), dtype=np.float32)
    buffer = tmp_path / "array.npy"
    np.save(buffer, array)
    content = buffer.read_bytes()

    validate_upload_metadata("array.npy", len(content), config)
    loaded = load_uploaded_npy("array.npy", content, config)

    assert loaded.shape == (2, 2, 2)
    assert array_payload(loaded)["shape"] == [2, 2, 2]
    assert (
        enforce_loopback_url("http://localhost:8000", allow_remote=False) == "http://localhost:8000"
    )
    assert validate_evidence_id("analysis-1") == "analysis-1"
    with pytest.raises(ReviewerUISecurityError, match="extension"):
        validate_upload_metadata("array.txt", 1, config)
    with pytest.raises(ReviewerUISecurityError, match="size"):
        validate_upload_metadata("array.npy", 999999, config)
    with pytest.raises(ReviewerUISecurityError, match="Remote"):
        enforce_loopback_url("https://example.com", allow_remote=False)
    with pytest.raises(ReviewerUISecurityError, match="path traversal"):
        validate_evidence_id("../secret")
    with pytest.raises(ReviewerUISecurityError, match="not safe"):
        safe_review_output_dir(tmp_path / "root", "../escape")


def test_numpy_pickle_invalid_shape_and_non_finite_rejection(tmp_path: Path) -> None:
    config = _config(tmp_path, maximum_upload_bytes=100_000)
    object_path = tmp_path / "object.npy"
    np.save(object_path, np.asarray([{"unsafe": "pickle"}], dtype=object))
    flat_path = tmp_path / "flat.npy"
    np.save(flat_path, np.zeros((4,), dtype=np.float32))
    nonfinite_path = tmp_path / "nonfinite.npy"
    np.save(nonfinite_path, np.full((2, 2, 2), np.inf, dtype=np.float32))

    with pytest.raises(ReviewerUISecurityError, match="Invalid NumPy"):
        load_uploaded_npy("object.npy", object_path.read_bytes(), config)
    with pytest.raises(ReviewerUISecurityError, match="shape"):
        load_uploaded_npy("flat.npy", flat_path.read_bytes(), config)
    with pytest.raises(ReviewerUISecurityError, match="non-finite"):
        load_uploaded_npy("nonfinite.npy", nonfinite_path.read_bytes(), config)


def test_reviewer_decision_validation_and_model_decision_separation() -> None:
    decision = create_review_decision(
        request_id="request-1",
        evidence_type="classification",
        evidence_id="classification-1",
        model_engineering_label="indeterminate",
        quality_status="PASS_WITH_WARNINGS",
        reviewer_decision="needs_secondary_review",
        review_notes="Needs another engineering review.",
    )

    assert decision.model_engineering_label == "indeterminate"
    assert decision.reviewer_decision == "needs_secondary_review"
    assert decision.model_engineering_label != decision.reviewer_decision
    with pytest.raises(ValueError, match="at most 500"):
        create_review_decision(
            request_id="request-1",
            evidence_type="classification",
            evidence_id="classification-1",
            model_engineering_label="indeterminate",
            quality_status="PASS",
            reviewer_decision="insufficient_information",
            review_notes="x" * 501,
        )
    with pytest.raises(ValueError, match="identifiers"):
        create_review_decision(
            request_id="request-1",
            evidence_type="classification",
            evidence_id="classification-1",
            model_engineering_label="indeterminate",
            quality_status="PASS",
            reviewer_decision="insufficient_information",
            review_notes="patient name: example",
        )


def test_review_export_checksums_overwrite_and_no_weight_exposure(tmp_path: Path) -> None:
    decision = create_review_decision(
        request_id="request-2",
        evidence_type="longitudinal",
        evidence_id="analysis-1",
        model_engineering_label="indeterminate",
        quality_status="FAIL",
        reviewer_decision="rejected_due_to_quality",
    )

    result = export_review_session(
        decision=decision,
        evidence_summary={"summary": {"analysis_id": "analysis-1"}, "model_weights": "blocked"},
        output_root=tmp_path / "reviewer-sessions",
    )
    output_dir = tmp_path / "reviewer-sessions" / result.output_directory
    summary = json.loads(
        (output_dir / "reviewed_evidence_summary.json").read_text(encoding="utf-8")
    )

    assert set(result.files) == {"review_decision", "reviewed_evidence_summary", "review_report"}
    assert set(result.checksums) == set(result.files)
    assert "model_weights" not in summary
    assert (output_dir / "review_report.md").exists()
    with pytest.raises(FileExistsError):
        export_review_session(
            decision=decision, evidence_summary={}, output_root=tmp_path / "reviewer-sessions"
        )


def test_session_state_initialisation_memory_and_reset() -> None:
    state: dict[str, object] = {}

    initialise_state(state)
    remember_response(state, {"page": "classification", "values": [1, 2, 3]}, {"result": "ok"})

    assert "values" not in state["last_api_request"]
    assert state["last_response"] == {"result": "ok"}
    reset_state(state)
    assert state["last_response"] is None
    assert state["review_decision"] == "insufficient_information"


def test_cli_reviewer_ui_validation_and_readiness(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path)

    assert main(["validate-reviewer-ui-config", "--config", str(config_path)]) == 0
    assert "Validated reviewer UI configuration test-reviewer-ui-v1." in capsys.readouterr().out
    assert main(["inspect-reviewer-ui-readiness", "--config", str(config_path)]) == 3
    assert "Reviewer UI readiness status=not_ready." in capsys.readouterr().out

    remote_path = tmp_path / "remote.yaml"
    remote_payload = _config(
        tmp_path, allow_remote_api=True, api_base_url="https://example.com"
    ).model_dump(mode="json")
    remote_payload["allow_remote_api"] = False
    remote_path.write_text(
        yaml.safe_dump({"settings": {"reviewer_ui": remote_payload}}, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_reviewer_ui_config(remote_path)


def test_streamlit_pages_render_with_fake_streamlit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    client = _client(config)
    array_path = tmp_path / "upload.npy"
    np.save(array_path, np.zeros((16, 16, 16), dtype=np.float32))
    upload = _Upload("upload.npy", array_path.read_bytes())

    overview_st = _FakeStreamlit()
    overview.render(overview_st, config, client)
    assert any("Configured API endpoint" in str(item) for item in overview_st.rendered)

    segmentation_st = _FakeStreamlit({"Upload bounded .npy volume": upload})
    segmentation.render(segmentation_st, config, client)
    assert segmentation_st.session_state["last_response"] is not None

    classification_st = _FakeStreamlit({"Upload bounded ROI .npy array": upload})
    classification.render(classification_st, config, client)
    assert any("Indeterminate" in str(item) for item in classification_st.rendered)

    longitudinal_st = _FakeStreamlit({"previous_mask": upload, "current_mask": upload})
    longitudinal.render(longitudinal_st, config, client)
    assert any("Indeterminate" in str(item) for item in longitudinal_st.rendered)

    evidence_st = _FakeStreamlit()
    evidence.render(evidence_st, config, client)
    assert evidence_st.session_state["last_response"] is not None

    governance_st = _FakeStreamlit(false_buttons={"Clear reviewer session"})
    governance.render(governance_st, config)
    assert governance_st.session_state["review_decision"] != "insufficient_information"


def test_components_and_visualisation_render_with_fake_streamlit() -> None:
    fake = _FakeStreamlit()

    disclaimers.render_disclaimer(fake)
    status.render_status(fake, "Status", {"status": "ready"})
    quality.render_quality(fake, [{"rule_id": "UI-QC-QUALITY-001"}])
    quality.render_quality(fake, [])
    provenance.render_provenance(fake, {"checksum": "abc"})
    visualisation.render_axial_slice(fake, np.zeros((3, 4, 4), dtype=np.float32), "Axial")

    assert visualisation.axial_slice(np.zeros((3, 4, 4), dtype=np.float32)).shape == (4, 4)
    with pytest.raises(ValueError, match="3D"):
        visualisation.axial_slice(np.zeros((4,), dtype=np.float32))


def test_reviewer_app_entrypoint_routes_without_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from medical_imaging_platform.reviewer_ui import app as reviewer_app

    config = _config(tmp_path)
    client = _client(config)
    monkeypatch.setattr(reviewer_app, "load_reviewer_ui_config", lambda path: config)
    monkeypatch.setattr(reviewer_app, "ReviewerAPIClient", lambda loaded_config: client)

    for page in (
        "Overview",
        "Segmentation",
        "Classification",
        "Longitudinal",
        "Evidence",
        "Governance",
    ):
        fake = _FakeStreamlit(selected_radio=page, false_buttons={"Export review decision"})
        monkeypatch.setitem(sys.modules, "streamlit", fake)
        reviewer_app.run_reviewer_app(Path("unused.yaml"))
        assert any("page_config" in str(item) for item in fake.rendered)
