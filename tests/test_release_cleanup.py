"""Final-release dependency, configuration and truthful-evidence regressions."""

import json
import os
import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock

import pytest

from medical_imaging_platform.api.app import create_app
from medical_imaging_platform.api.models import APIConfig
from medical_imaging_platform.cli import main
from medical_imaging_platform.kubernetes import assurance
from medical_imaging_platform.portfolio import evidence
from medical_imaging_platform.release.models import ReleaseManifest
from medical_imaging_platform.utils.config import (
    ConfigError,
    load_api_config,
    load_reviewer_ui_config,
)
from scripts import validate_docs


def test_runtime_requirements_are_complete_and_containers_resolve_project_dependencies():
    project = tomllib.loads(Path("pyproject.toml").read_text())
    required = project["project"]["dependencies"]
    for name in ("httpx", "matplotlib", "torchvision", "numpy"):
        assert any(r.startswith(name) for r in required)
    runtime = Path("requirements/container-runtime.txt").read_text()
    assert "numpy==2.4.6" in required and "numpy==2.4.6" in runtime
    assert "torchvision==0.28.0+cpu" in runtime and "torch==2.13.0+cpu" in runtime
    for dockerfile in ("docker/api/Dockerfile", "docker/reviewer-ui/Dockerfile"):
        text = Path(dockerfile).read_text()
        assert "python -m pip check" in text
        assert "--no-deps enterprise-medical" not in text
    import httpx
    import matplotlib.pyplot as pyplot
    from torchvision.models.detection import FasterRCNN

    assert httpx.Client and pyplot and FasterRCNN


@pytest.mark.parametrize(
    "field",
    [
        "require_model_checksums",
        "require_quality_pass",
        "request_timeout_seconds",
        "maximum_batch_size",
    ],
)
def test_dead_api_configuration_is_rejected_explicitly(field):
    config = load_api_config(Path("config/api.yaml"))
    assert field not in APIConfig.model_fields
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        APIConfig.model_validate(config.model_dump() | {field: 1})


def test_environment_wiring_preserves_typed_validation_and_explicit_api_path(monkeypatch):
    monkeypatch.setenv("MEDICAL_IMAGING_API_CONFIG", "config/container/api.yaml")
    monkeypatch.setenv("PLATFORM_ENVIRONMENT", "release-test")
    monkeypatch.setenv("PLATFORM_LOG_LEVEL", "DEBUG")
    app = create_app()
    assert app.state.api_config.environment == "release-test"
    assert app.state.api_config.log_level == "DEBUG"
    assert app.state.api_config.segmentation_config == Path("/app/config/segmentation.yaml")
    server = Mock()
    monkeypatch.setattr("uvicorn.run", server)
    assert main(["serve-api"]) == 0
    assert server.call_args.kwargs["log_level"] == "debug"
    assert server.call_args.kwargs["host"] == "0.0.0.0"
    assert server.call_args.kwargs["access_log"] is False
    assert create_app(Path("config/api.yaml")).state.api_config.host == "127.0.0.1"
    monkeypatch.setenv("REVIEWER_API_BASE_URL", "http://release-api:8000")
    assert (
        load_reviewer_ui_config(Path("config/container/reviewer_ui.yaml")).api_base_url
        == "http://release-api:8000"
    )
    with pytest.raises(ConfigError, match="Remote API URLs"):
        load_reviewer_ui_config(Path("config/reviewer_ui.yaml"))
    monkeypatch.setenv("PLATFORM_ENVIRONMENT", "")
    with pytest.raises(ConfigError):
        create_app()


def test_container_configs_and_helm_service_url_match_runtime_consumption():
    config = load_api_config(Path("config/container/api.yaml"))
    for family in ("segmentation", "classification", "detection"):
        assert getattr(config, family + "_checkpoint") is None
        assert getattr(config, family + "_config") == Path(f"/app/config/{family}.yaml")
        compose = Path("docker-compose.yml").read_text()
        assert f"target: /app/config/{family}.yaml" in compose
    values = assurance.load_values()
    rendered = assurance._render_objects(values)
    env = next(r["data"] for r in rendered if r["kind"] == "ConfigMap")
    assert env["MEDICAL_IMAGING_API_CONFIG"] == "/app/config/api.yaml"
    assert env["REVIEWER_API_BASE_URL"] == "http://medical-imaging-platform-api:8000"
    assert "REVIEWER_ENVIRONMENT" not in env and "PLATFORM_SYNTHETIC_ONLY" not in env
    values["reviewerUi"]["config"]["apiBaseUrl"] = "http://alternate-api:8000"
    env = next(r["data"] for r in assurance._render_objects(values) if r["kind"] == "ConfigMap")
    assert env["REVIEWER_API_BASE_URL"] == "http://alternate-api:8000"


def test_fast_demo_plan_and_real_stage_results_cannot_manufacture_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(evidence, "PORTFOLIO_EVIDENCE_DIR", tmp_path)
    monkeypatch.setattr(evidence, "DEMO_FAST_PATH", tmp_path / "demo.json")
    runner = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr(evidence.subprocess, "run", runner)
    plan = evidence.run_demo_fast()
    assert not plan["executed"] and plan["overall_status"] == "UNAVAILABLE"
    runner.assert_not_called()
    assert main(["run-demo-fast", "--execute"]) == 0
    assert runner.call_count == 8
    assert json.loads(evidence.DEMO_FAST_PATH.read_text())["overall_status"] == "PASS"
    runner.reset_mock()
    runner.side_effect = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 2)]
    failed = evidence.run_demo_fast(execute=True)
    assert failed["overall_status"] == "FAIL" and runner.call_count == 2
    assert [r["status"] for r in failed["stages"]][:3] == ["PASS", "FAIL", "UNAVAILABLE"]
    runner.side_effect = subprocess.TimeoutExpired(["make"], 300)
    failed = evidence.run_demo_fast(execute=True)
    assert failed["overall_status"] == "FAIL" and failed["stages"][0]["status"] == "ERROR"
    monkeypatch.setattr(evidence.shutil, "which", lambda _: None)
    missing = evidence.run_demo_fast(execute=True)
    assert not missing["executed"] and missing["stages"][0]["status"] == "ERROR"


def _release(root, name, *, status="PASS", executed=True):
    directory = root / name
    directory.mkdir()
    inputs = evidence.checksum_paths(
        [
            Path("docker/api/Dockerfile"),
            Path("docker/reviewer-ui/Dockerfile"),
            Path("requirements/container-runtime.txt"),
            Path("pyproject.toml"),
            Path("docker-compose.yml"),
        ]
    )
    manifest = ReleaseManifest(
        release_id=name,
        git_revision="test",
        git_dirty=True,
        build_timestamp="2026-10-13T00:00:00Z",
        python_version="3.12",
        release_status=status,
        dependency_strategy={},
        dependency_versions={},
        images={
            "api": {"digest": "sha256:" + "1" * 64},
            "reviewer_ui": {"digest": "sha256:" + "2" * 64},
        },
        dockerfile_checksums=inputs,
        configuration_checksums={},
        source_checksums=evidence.checksum_paths(
            list(Path("src/medical_imaging_platform").rglob("*.py"))
        ),
        test_results=[],
        scan_results=[],
        smoke_test_results={
            "status": status,
            "executed": executed,
            "steps": [
                {"check_id": c, "status": status, "message": "Observed exit result"}
                for c in ("SMOKE-COMPOSE-BUILD", "SMOKE-API-HEALTH", "SMOKE-REVIEWER-HEALTH")
            ],
        },
    )
    path = directory / "release_manifest.json"
    path.write_text(manifest.model_dump_json())
    _release_hash(path)
    return path


def _release_hash(path):
    (path.parent / "checksum_manifest.json").write_text(json.dumps(evidence.checksum_paths([path])))


def test_container_observation_requires_current_integrity_and_actual_runtime_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(evidence, "CONTAINER_EVIDENCE_DIR", tmp_path)
    assert evidence.container_observation()["status"] == "UNAVAILABLE"
    assert all(
        c["claim_classification"] != "locally executed"
        for c in evidence.build_capability_inventory()
    )
    path = _release(tmp_path, "success")
    observed = evidence.container_observation()
    assert observed["status"] == "PASS" and observed["executed"]
    failed = _release(tmp_path, "failure", status="FAIL")
    os.utime(path, ns=(1, 1))
    assert evidence.container_observation()["status"] == "FAIL"
    payload = json.loads(failed.read_text())
    payload["release_status"] = "PASS"
    payload["smoke_test_results"]["executed"] = False
    failed.write_text(json.dumps(payload))
    _release_hash(failed)
    assert evidence.container_observation()["status"] == "ERROR"
    payload["smoke_test_results"]["executed"] = True
    payload["images"] = {}
    failed.write_text(json.dumps(payload))
    _release_hash(failed)
    assert evidence.container_observation()["status"] == "ERROR"
    payload["release_status"] = "FAIL"
    payload["dockerfile_checksums"]["pyproject.toml"] = "0" * 64
    failed.write_text(json.dumps(payload))
    _release_hash(failed)
    observed = evidence.container_observation()
    assert observed["status"] == "UNAVAILABLE" and observed["historical_release_status"] == "FAIL"
    failed.write_text("tampered")
    assert evidence.container_observation()["status"] == "ERROR"


def test_public_docs_validate_each_claim_and_current_scaffolds():
    path = Path("SECURITY.md")
    assert validate_docs.validate_claims(path, "Not a clinical decision support system.") == []
    for separator in (". It is a ", ", but it is a ", "\nIt is a "):
        assert validate_docs.validate_claims(
            path,
            "Not a clinical decision support system"
            + separator
            + "clinical decision support system.",
        )
    assert validate_docs.validate_current_state(
        Path("api/README.md"), "# API\nPlanned - not yet implemented."
    )
    assert validate_docs.validate_current_state(Path("docs/roadmap.md"), "18. Complete.")
    assert (
        validate_docs.validate_current_state(
            Path("docs/roadmap.md"), Path("docs/roadmap.md").read_text()
        )
        == []
    )
    assert {Path("docs/detection_design.md"), Path("docs/embeddings.md"), path} <= set(
        validate_docs.REQUIRED_MARKDOWN_FILES
    )
    assert validate_docs.main() == 0
