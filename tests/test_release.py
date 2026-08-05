import json
import tomllib
from pathlib import Path

import pytest

from medical_imaging_platform.cli import main
from medical_imaging_platform.release.compose import inspect_compose
from medical_imaging_platform.release.config import load_container_release_config
from medical_imaging_platform.release.dependencies import (
    dependency_strategy,
    inspect_container_dependency_policy,
)
from medical_imaging_platform.release.dockerfiles import inspect_dockerfile
from medical_imaging_platform.release.evidence import (
    CANONICAL_TOOL_KEYS,
    load_canonical_tool_evidence,
    load_smoke_evidence,
    write_smoke_evidence,
    write_tool_evidence,
)
from medical_imaging_platform.release.export import (
    export_release_evidence,
    validate_release_evidence,
)
from medical_imaging_platform.release.image_policy import (
    detect_prohibited_files,
    inspect_dockerignore,
)
from medical_imaging_platform.release.manifest import (
    _docker_inspect,
    _parse_int,
    build_release_manifest,
    dependency_versions,
    deterministic_release_id,
    git_revision,
)
from medical_imaging_platform.release.models import (
    ContainerReleaseConfig,
    ReleaseCheckResult,
    SmokeTestResult,
    ToolResult,
)
from medical_imaging_platform.release.scanners import (
    _pip_audit_command,
    generate_context_sbom,
    lint_dockerfiles,
    parse_json_findings,
    run_tool,
    sanitize_output,
    scan_dependencies,
    scan_image,
    scan_repository_secrets,
    severity_gate,
    tool_version,
)
from medical_imaging_platform.release.smoke import _run_command, run_container_smoke_tests
from medical_imaging_platform.release.status import aggregate_release_status
from medical_imaging_platform.utils.config import ConfigError, load_container_config


def test_container_config_validation() -> None:
    config = load_container_config(Path("config/container.yaml"))

    assert config.policy_version == "m13-container-release-v1"
    assert config.api_port != config.reviewer_ui_port
    assert config.required_uid > 0
    assert config.output_mount.as_posix().startswith("/app/")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("required_uid", 0, "greater than 0"),
        ("api_port", 8501, "ports must be distinct"),
        ("output_mount", "/", "specific absolute"),
        ("api_health_path", "https://example.test/health", "local absolute"),
        ("read_only_root_filesystem", False, "Read-only root"),
        ("drop_all_capabilities", False, "capabilities"),
    ],
)
def test_container_config_rejects_unsafe_values(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    base = Path("config/container.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "container.yaml"
    config_path.write_text(
        base.replace(
            f"    {field}: {json.dumps(_original(field))}", f"    {field}: {json.dumps(value)}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match=message):
        load_container_config(config_path)


def test_dockerfile_static_controls_and_failures(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))
    checks = inspect_dockerfile(Path("docker/api/Dockerfile"), "medical-imaging-api", config)

    assert all(check.status == "PASS" for check in checks)

    bad = tmp_path / "Dockerfile"
    bad.write_text(
        "FROM python:latest\n"
        "WORKDIR /app\n"
        "LABEL org.opencontainers.image.title=x\n"
        "CMD python app.py\n",
        encoding="utf-8",
    )
    bad_checks = inspect_dockerfile(bad, "medical-imaging-api", config)

    assert any(
        check.check_id == "DOCKERFILE-NO-LATEST" and check.status == "FAIL" for check in bad_checks
    )
    assert any(
        check.check_id == "DOCKERFILE-NONROOT" and check.status == "FAIL" for check in bad_checks
    )
    assert any(
        check.check_id == "DOCKERFILE-EXEC-CMD" and check.status == "FAIL" for check in bad_checks
    )

    missing = inspect_dockerfile(tmp_path / "missing.Dockerfile", "medical-imaging-api", config)
    assert missing[0].status == "FAIL"


def test_cpu_only_container_dependency_policy(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))
    checks = inspect_container_dependency_policy(config)

    assert all(check.status == "PASS" for check in checks)
    assert any(check.check_id == "CONTAINER-DEPS-PYTORCH-CPU-PIN" for check in checks)
    monai_check = next(
        check for check in checks if check.check_id == "CONTAINER-DEPS-MONAI-PATCHED-PIN"
    )
    assert monai_check.status == "PASS"
    assert monai_check.details["pinned_version"] == "1.6.0"

    bad_requirements = tmp_path / "container-runtime.txt"
    bad_requirements.write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch==2.13.0+cpu\n"
        "monai==1.6.0\n"
        "nvidia-cublas-cu12==12.1.3.1\n",
        encoding="utf-8",
    )
    bad_config = config.model_copy(update={"container_runtime_requirements": bad_requirements})
    bad_checks = inspect_container_dependency_policy(bad_config)

    assert any(
        check.check_id == "CONTAINER-DEPS-NO-CUDA-NVIDIA" and check.status == "FAIL"
        for check in bad_checks
    )


def test_vulnerable_monai_container_pin_is_rejected(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))
    vulnerable_requirements = tmp_path / "container-runtime.txt"
    vulnerable_requirements.write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\ntorch==2.13.0+cpu\nmonai==1.5.2\n",
        encoding="utf-8",
    )
    vulnerable_config = config.model_copy(
        update={"container_runtime_requirements": vulnerable_requirements}
    )

    checks = inspect_container_dependency_policy(vulnerable_config)
    monai_check = next(
        check for check in checks if check.check_id == "CONTAINER-DEPS-MONAI-PATCHED-PIN"
    )

    assert monai_check.status == "FAIL"
    assert monai_check.details == {
        "minimum_safe_version": "1.6.0",
        "pinned_version": "1.5.2",
    }
    assert (
        aggregate_release_status(
            checks,
            [
                ToolResult(
                    tool="pip-audit", available=True, status="PASS", details={"mandatory": True}
                )
            ],
            SmokeTestResult(status="PASS", executed=True, steps=[]),
        )
        == "FAIL"
    )


def test_pyproject_monai_lower_bound_excludes_vulnerable_range() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    monai_requirement = next(item for item in dependencies if item.startswith("monai"))

    assert monai_requirement == "monai>=1.6,<2"
    assert "==1.6" not in monai_requirement


def test_compose_security_controls() -> None:
    config = load_container_config(Path("config/container.yaml"))
    checks = inspect_compose(config)

    assert all(check.status == "PASS" for check in checks)
    assert any(check.check_id == "COMPOSE-api-CAPDROP" for check in checks)
    assert any(check.check_id == "COMPOSE-reviewer-ui-NO-CHECKPOINTS" for check in checks)


def test_compose_rejects_privileged_socket_and_writable_root(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))
    compose = tmp_path / "docker-compose.yml"
    compose.write_text(
        """
services:
  api:
    privileged: true
    read_only: false
    cap_drop: []
    security_opt: []
    user: "0:0"
    ports: ["8000:8000"]
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
  reviewer-ui:
    read_only: true
networks:
  medical-imaging-local: {}
""",
        encoding="utf-8",
    )

    checks = inspect_compose(config, compose)

    assert any(
        check.check_id == "COMPOSE-api-NO-PRIVILEGED" and check.status == "FAIL" for check in checks
    )
    assert any(
        check.check_id == "COMPOSE-api-READONLY" and check.status == "FAIL" for check in checks
    )
    assert any(
        check.check_id == "COMPOSE-api-NO-DOCKER-SOCKET" and check.status == "FAIL"
        for check in checks
    )


def test_compose_missing_parse_and_service_errors(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))

    assert inspect_compose(config, tmp_path / "missing.yml")[0].status == "FAIL"

    invalid = tmp_path / "invalid.yml"
    invalid.write_text("[not-a-compose-list]\n", encoding="utf-8")
    assert inspect_compose(config, invalid)[0].status == "FAIL"

    bad_services = tmp_path / "bad-services.yml"
    bad_services.write_text("services: []\n", encoding="utf-8")
    service_checks = inspect_compose(config, bad_services)
    assert service_checks[0].check_id == "COMPOSE-SERVICES"
    assert service_checks[0].status == "FAIL"

    missing_service = tmp_path / "missing-service.yml"
    missing_service.write_text("services:\n  api: {}\n", encoding="utf-8")
    checks = inspect_compose(config, missing_service)
    assert any(check.check_id == "COMPOSE-reviewer-ui-SERVICE" for check in checks)


def test_image_content_policy() -> None:
    ignore_checks = inspect_dockerignore()
    assert all(check.status == "PASS" for check in ignore_checks)

    clean = detect_prohibited_files(["/app/src/medical_imaging_platform/api/app.py"])
    dirty = detect_prohibited_files(["/app/.env", "/app/ml/experiments/model.pt"])

    assert all(check.status == "PASS" for check in clean)
    assert any(check.status == "FAIL" for check in dirty)


def test_config_and_model_edge_cases(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml"))
    invalid = tmp_path / "container.yaml"
    invalid.write_text("settings: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Missing settings.container"):
        load_container_release_config(invalid)

    invalid_schema = tmp_path / "invalid-container.yaml"
    invalid_schema.write_text(
        "settings:\n  container:\n    policy_version: bad\n", encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="Invalid container configuration"):
        load_container_release_config(invalid_schema)

    bad_data = config.model_dump(mode="python")
    bad_data["required_gid"] = 0
    with pytest.raises(ValueError, match="greater than 0"):
        ContainerReleaseConfig.model_validate(bad_data)

    bad_data = config.model_dump(mode="python")
    bad_data["run_as_non_root"] = False
    with pytest.raises(ValueError, match="non-root"):
        ContainerReleaseConfig.model_validate(bad_data)

    bad_data = config.model_dump(mode="python")
    bad_data["no_new_privileges"] = False
    with pytest.raises(ValueError, match="no-new-privileges"):
        ContainerReleaseConfig.model_validate(bad_data)

    bad_data = config.model_dump(mode="python")
    bad_data["vulnerability_fail_severities"] = ["UNKNOWN"]
    with pytest.raises(ValueError, match="Unsupported vulnerability"):
        ContainerReleaseConfig.model_validate(bad_data)


def test_release_id_manifest_and_evidence_export(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml")).model_copy(
        update={"release_output_directory": tmp_path}
    )
    static_check = ReleaseCheckResult(
        check_id="STATIC",
        status="PASS",
        message="static checks passed",
    )
    smoke = SmokeTestResult(status="SKIPPED", executed=False, steps=[])
    scanner = ToolResult(tool="syft", available=False, status="UNAVAILABLE")
    manifest = build_release_manifest(config, [static_check], [scanner], smoke)

    assert deterministic_release_id(config, manifest.git_revision) == manifest.release_id
    assert isinstance(manifest.git_dirty, bool)
    assert manifest.release_status == "INCOMPLETE"
    assert manifest.dependency_strategy["pytorch_version"] == "2.13.0+cpu"
    assert manifest.dependency_strategy["monai_version"] == "1.6.0"
    assert manifest.dependency_strategy["minimum_safe_monai_version"] == "1.6.0"
    assert "size_bytes" in manifest.images["api"]

    output_dir = export_release_evidence(config, manifest)
    checks = validate_release_evidence(output_dir)

    assert (output_dir / "release_manifest.json").is_file()
    assert (output_dir / "checksum_manifest.json").is_file()
    assert any(check.check_id == "RELEASE-EVIDENCE-OVERALL-STATUS" for check in checks)
    assert any(check.status == "INCOMPLETE" for check in checks)
    with pytest.raises(FileExistsError):
        export_release_evidence(config, manifest)


def test_release_status_aggregation_no_false_pass() -> None:
    passing_check = ReleaseCheckResult(check_id="STATIC", status="PASS", message="ok")
    passing_scan = ToolResult(
        tool="pip-audit",
        available=True,
        status="PASS",
        details={"mandatory": True},
    )
    smoke_pass = SmokeTestResult(status="PASS", executed=True, steps=[])

    assert aggregate_release_status([passing_check], [passing_scan], smoke_pass) == "PASS"
    assert (
        aggregate_release_status(
            [passing_check],
            [ToolResult(tool="syft", available=False, status="UNAVAILABLE")],
            smoke_pass,
        )
        == "PASS"
    )
    assert (
        aggregate_release_status(
            [passing_check],
            [
                ToolResult(
                    tool="syft",
                    available=False,
                    status="UNAVAILABLE",
                    command=["syft", "image:example"],
                    details={"mandatory": True},
                )
            ],
            smoke_pass,
        )
        == "INCOMPLETE"
    )
    assert (
        aggregate_release_status(
            [passing_check],
            [passing_scan],
            SmokeTestResult(status="FAIL", executed=True, steps=[]),
        )
        == "FAIL"
    )


def _passing_tool(tool: str, target: str | None = None) -> ToolResult:
    return ToolResult(
        tool=tool,
        available=True,
        status="PASS",
        output="ok",
        details={"target": target} if target else {},
    )


def _write_passing_canonical_tool_evidence(config: ContainerReleaseConfig) -> None:
    write_tool_evidence(config, "gitleaks", _passing_tool("gitleaks"))
    write_tool_evidence(config, "pip-audit", _passing_tool("pip-audit"))
    write_tool_evidence(config, "syft-api", _passing_tool("syft", "api"))
    write_tool_evidence(config, "syft-reviewer-ui", _passing_tool("syft", "reviewer-ui"))
    write_tool_evidence(config, "trivy-api", _passing_tool("trivy", "api"))
    write_tool_evidence(config, "trivy-reviewer-ui", _passing_tool("trivy", "reviewer-ui"))
    write_tool_evidence(
        config,
        "hadolint",
        ToolResult(
            tool="hadolint",
            available=False,
            status="UNAVAILABLE",
            output="Hadolint not installed.",
            details={"optional_reason": "Internal Dockerfile validator is mandatory."},
        ),
    )


def test_canonical_tool_evidence_prevents_duplicates_and_uses_latest(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml")).model_copy(
        update={"release_output_directory": tmp_path}
    )
    _write_passing_canonical_tool_evidence(config)

    write_tool_evidence(
        config,
        "pip-audit",
        ToolResult(
            tool="pip-audit",
            available=True,
            status="ERROR",
            output="stale error",
            details={"run_id": "stale"},
        ),
    )
    write_tool_evidence(
        config,
        "pip-audit",
        ToolResult(
            tool="pip-audit",
            available=True,
            status="PASS",
            output="latest pass",
            details={"run_id": "latest"},
        ),
    )

    results = load_canonical_tool_evidence(config)
    keys = [result.details["canonical_key"] for result in results]
    pip_audit_results = [
        result for result in results if result.details["canonical_key"] == "pip-audit"
    ]

    assert keys == list(CANONICAL_TOOL_KEYS)
    assert len(pip_audit_results) == 1
    assert pip_audit_results[0].status == "PASS"
    assert pip_audit_results[0].output == "latest pass"
    assert pip_audit_results[0].details["mandatory"] is True


def test_hadolint_optional_but_missing_mandatory_evidence_blocks_release(
    tmp_path: Path,
) -> None:
    config = load_container_config(Path("config/container.yaml")).model_copy(
        update={"release_output_directory": tmp_path}
    )
    write_tool_evidence(config, "gitleaks", _passing_tool("gitleaks"))
    write_tool_evidence(
        config,
        "hadolint",
        ToolResult(tool="hadolint", available=False, status="UNAVAILABLE"),
    )
    results = load_canonical_tool_evidence(config)
    smoke = SmokeTestResult(status="PASS", executed=True, steps=[])
    static_check = ReleaseCheckResult(check_id="STATIC", status="PASS", message="ok")

    assert (
        next(result for result in results if result.tool == "hadolint").details["mandatory"]
        is False
    )
    assert aggregate_release_status([static_check], results, smoke) == "INCOMPLETE"


def test_smoke_evidence_retains_passed_steps_and_cleanup(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml")).model_copy(
        update={"release_output_directory": tmp_path}
    )
    write_smoke_evidence(config, SmokeTestResult(status="SKIPPED", executed=False, steps=[]))
    write_smoke_evidence(
        config,
        SmokeTestResult(
            status="PASS",
            executed=True,
            steps=[
                ReleaseCheckResult(check_id="SMOKE-API-HEALTH", status="PASS", message="ok"),
                ReleaseCheckResult(
                    check_id="SMOKE-COMPOSE-CLEANUP",
                    status="PASS",
                    message="compose cleanup completed",
                ),
            ],
        ),
    )

    loaded = load_smoke_evidence(config)

    assert loaded.status == "PASS"
    assert loaded.executed is True
    assert [step.check_id for step in loaded.steps] == [
        "SMOKE-API-HEALTH",
        "SMOKE-COMPOSE-CLEANUP",
    ]


def test_manifest_uses_latest_evidence_without_stale_false_fail(tmp_path: Path) -> None:
    config = load_container_config(Path("config/container.yaml")).model_copy(
        update={"release_output_directory": tmp_path}
    )
    stale_manifest_dir = tmp_path / "20200101T000000Z-stale"
    stale_manifest_dir.mkdir(parents=True)
    (stale_manifest_dir / "pip-audit.json").write_text(
        ToolResult(tool="pip-audit", available=True, status="FAIL").model_dump_json(),
        encoding="utf-8",
    )
    _write_passing_canonical_tool_evidence(config)
    write_smoke_evidence(
        config,
        SmokeTestResult(
            status="PASS",
            executed=True,
            steps=[
                ReleaseCheckResult(
                    check_id="SMOKE-COMPOSE-CLEANUP",
                    status="PASS",
                    message="compose cleanup completed",
                )
            ],
        ),
    )
    static_check = ReleaseCheckResult(check_id="STATIC", status="PASS", message="ok")

    manifest = build_release_manifest(
        config,
        [static_check],
        load_canonical_tool_evidence(config),
        load_smoke_evidence(config),
    )

    assert manifest.release_status == "PASS"
    assert manifest.smoke_test_results.status == "PASS"
    assert manifest.smoke_test_results.executed is True
    assert [scan.details["canonical_key"] for scan in manifest.scan_results] == list(
        CANONICAL_TOOL_KEYS
    )


def test_manifest_git_and_dependency_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata

    class Completed:
        returncode = 1
        stdout = ""
        stderr = "no git"

    monkeypatch.setattr(
        "medical_imaging_platform.release.manifest.subprocess.run",
        lambda *_, **__: Completed(),
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.manifest.importlib.metadata.version",
        lambda _: (_ for _ in ()).throw(importlib.metadata.PackageNotFoundError()),
    )

    assert git_revision() == "unknown"
    versions = dependency_versions()
    assert versions["fastapi"] == "not-installed"


def test_dependency_strategy_records_patched_monai_container_pin() -> None:
    config = load_container_config(Path("config/container.yaml"))
    strategy = dependency_strategy(config)

    assert strategy["monai_version"] == "1.6.0"
    assert strategy["minimum_safe_monai_version"] == "1.6.0"


def test_manifest_image_metadata_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0
        stdout = '{"Id":"sha256:abc","Size":123,"Labels":{"x":"y"}}'
        stderr = ""

    monkeypatch.setattr(
        "medical_imaging_platform.release.manifest.subprocess.run",
        lambda *_, **__: Completed(),
    )

    assert _docker_inspect("example:local")["Size"] == "123"
    assert _parse_int("123") == 123
    assert _parse_int("not-int") is None

    class ListCompleted:
        returncode = 0
        stdout = "[]"
        stderr = ""

    monkeypatch.setattr(
        "medical_imaging_platform.release.manifest.subprocess.run",
        lambda *_, **__: ListCompleted(),
    )
    assert _docker_inspect("example:local") == {}


def test_scanner_helpers_and_unavailable_tool() -> None:
    assert sanitize_output("token=abc\nok") == "[REDACTED]\nok"
    assert parse_json_findings("") == []
    assert parse_json_findings("not-json") == []
    assert parse_json_findings('[{"Severity": "LOW"}, "skip"]') == [{"Severity": "LOW"}]
    findings = parse_json_findings('{"Results": [{"Vulnerabilities": []}]}')
    assert findings == [{"Vulnerabilities": []}]
    assert severity_gate([{"Severity": "LOW"}], ["HIGH", "CRITICAL"])
    assert not severity_gate([{"severity": "CRITICAL"}], ["HIGH", "CRITICAL"])

    result = tool_version("definitely-not-a-real-m13-tool")
    assert result.status == "UNAVAILABLE"
    assert not result.available


def test_smoke_tests_report_docker_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_container_config(Path("config/container.yaml"))
    monkeypatch.setattr("medical_imaging_platform.release.smoke.shutil.which", lambda _: None)

    result = run_container_smoke_tests(config)

    assert result.status == "UNAVAILABLE"
    assert result.executed is False


def test_smoke_tests_execute_success_and_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_container_config(Path("config/container.yaml"))
    calls: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stdout = "ok"
            self.stderr = ""

    def fake_run(command: list[str], **_: object) -> Completed:
        calls.append(command)
        return Completed(0)

    monkeypatch.setattr(
        "medical_imaging_platform.release.smoke.shutil.which",
        lambda _: "/bin/docker",
    )
    monkeypatch.setattr("medical_imaging_platform.release.smoke.subprocess.run", fake_run)

    success = run_container_smoke_tests(config)

    assert success.status == "PASS"
    assert success.executed is True
    assert any(command[:3] == ["docker", "compose", "logs"] for command in calls)

    calls.clear()

    def failing_run(command: list[str], **_: object) -> Completed:
        calls.append(command)
        return Completed(1 if command[:3] == ["docker", "compose", "up"] else 0)

    monkeypatch.setattr("medical_imaging_platform.release.smoke.subprocess.run", failing_run)

    failure = run_container_smoke_tests(config)

    assert failure.status == "FAIL"
    failed_step = next(step for step in failure.steps if step.status == "FAIL")
    assert failed_step.details["command"][:3] == ["docker", "compose", "up"]
    assert failed_step.details["exit_code"] == 1
    assert "stderr" in failed_step.details
    assert any(command[:3] == ["docker", "compose", "down"] for command in calls)


def test_smoke_tests_expected_degraded_readiness_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_container_config(Path("config/container.yaml"))
    calls: list[list[str]] = []

    class Completed:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stdout = "expected readiness degradation"
            self.stderr = ""

    def fake_run(command: list[str], **_: object) -> Completed:
        calls.append(command)
        is_ready_probe = "http://127.0.0.1:8000/ready" in " ".join(command)
        return Completed(10 if is_ready_probe else 0)

    monkeypatch.setattr(
        "medical_imaging_platform.release.smoke.shutil.which",
        lambda _: "/bin/docker",
    )
    monkeypatch.setattr("medical_imaging_platform.release.smoke.subprocess.run", fake_run)

    result = run_container_smoke_tests(config)

    assert result.status == "INCOMPLETE"
    assert any(
        step.check_id == "SMOKE-API-READY"
        and step.status == "WARN"
        and step.details["expected_degraded_readiness"] is True
        for step in result.steps
    )
    assert any(command[:3] == ["docker", "compose", "down"] for command in calls)


def test_scanner_wrapper_unavailable_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("medical_imaging_platform.release.scanners.shutil.which", lambda _: None)

    assert run_tool(["missing"], "missing", 1).status == "UNAVAILABLE"
    assert scan_repository_secrets(1).status == "UNAVAILABLE"
    assert scan_dependencies(1).status == "UNAVAILABLE"
    assert scan_image("example:local", 1).status == "UNAVAILABLE"
    assert generate_context_sbom("example", tmp_path / "sbom.json", 1).status == "UNAVAILABLE"
    assert all(result.status == "UNAVAILABLE" for result in lint_dockerfiles(1))


def test_scanner_run_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fake_run(*_: object, **__: object) -> object:
        raise subprocess.TimeoutExpired(["tool"], 1, output=b"partial", stderr=b"late")

    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.shutil.which",
        lambda _: "/bin/tool",
    )
    monkeypatch.setattr("medical_imaging_platform.release.scanners.subprocess.run", fake_run)

    result = run_tool(["tool"], "tool", 1)

    assert result.status == "ERROR"
    assert "partial" in result.output


def test_scanner_status_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.shutil.which",
        lambda tool: "/bin/tool" if tool == "pip-audit" else None,
    )
    assert _pip_audit_command()[0] == "pip-audit"

    result = run_tool(["tool"], "tool", 1)
    assert result.status == "UNAVAILABLE"

    class Completed:
        returncode = 1
        stdout = '{"Results":[]}'
        stderr = "failed to download vulnerability database"

    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.shutil.which",
        lambda _: "/bin/tool",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.subprocess.run",
        lambda *_, **__: Completed(),
    )
    assert run_tool(["tool"], "tool", 1).status == "ERROR"


def test_smoke_run_command_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    def fake_run(*_: object, **__: object) -> object:
        raise subprocess.TimeoutExpired(["docker"], 1, output=b"partial", stderr=b"late")

    monkeypatch.setattr("medical_imaging_platform.release.smoke.subprocess.run", fake_run)

    result = _run_command(["docker"], 1)

    assert result.returncode == 124
    assert "partial" in result.output
    assert "late" in result.stderr


def test_container_smoke_cli_prints_failed_step_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config_path = _write_temp_container_config(tmp_path)

    class Completed:
        def __init__(self, returncode: int, stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr

    def fake_run(command: list[str], **_: object) -> Completed:
        if command[:3] == ["docker", "compose", "build"]:
            return Completed(1, "build failed clearly")
        return Completed(0)

    monkeypatch.setattr(
        "medical_imaging_platform.release.smoke.shutil.which",
        lambda _: "/bin/docker",
    )
    monkeypatch.setattr("medical_imaging_platform.release.smoke.subprocess.run", fake_run)

    assert main(["run-container-smoke-tests", "--config", str(config_path)]) == 2
    output = capsys.readouterr().out

    assert "Container smoke-test status=FAIL" in output
    assert "Failed smoke steps:" in output
    assert "SMOKE-COMPOSE-BUILD" in output
    assert "exit_code=1" in output
    assert "build failed clearly" in output


def test_release_scanner_cli_branches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config_path = _write_temp_container_config(tmp_path)
    unavailable = ToolResult(tool="stub", available=False, status="UNAVAILABLE", output="missing")
    pass_result = ToolResult(tool="stub", available=True, status="PASS", findings=[])

    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.scan_repository_secrets",
        lambda _: unavailable,
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.scan_dependencies",
        lambda *_: pass_result,
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.generate_context_sbom",
        lambda *_: unavailable,
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.scanners.scan_image",
        lambda *_: pass_result.model_copy(),
    )
    monkeypatch.setattr(
        "medical_imaging_platform.release.smoke.shutil.which",
        lambda _: "/bin/docker",
    )

    assert main(["scan-release-secrets", "--config", str(config_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "UNAVAILABLE"

    assert main(["scan-release-dependencies", "--config", str(config_path)]) == 0
    assert "Dependency scan status=PASS" in capsys.readouterr().out

    assert (
        main(["generate-release-sbom", "--config", str(config_path), "--output-dir", str(tmp_path)])
        == 2
    )
    assert "SBOM generation statuses" in capsys.readouterr().out

    assert main(["scan-release-images", "--config", str(config_path), "--json"]) == 0
    assert {item["status"] for item in json.loads(capsys.readouterr().out)} == {"PASS"}

    assert main(["run-container-smoke-tests", "--config", str(config_path), "--no-execute"]) == 0
    assert "Container smoke-test status=SKIPPED" in capsys.readouterr().out


def test_release_cli_commands(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    config_path = _write_temp_container_config(tmp_path)

    assert main(["validate-container-config", "--config", str(config_path)]) == 0
    assert "Validated container configuration" in capsys.readouterr().out

    assert main(["inspect-container-security", "--config", str(config_path)]) == 0
    assert "Container security inspection status=PASS" in capsys.readouterr().out

    assert (
        main(
            [
                "build-release-manifest",
                "--config",
                str(config_path),
                "--overwrite",
                "--no-smoke-execute",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["release_dir"].startswith(tmp_path.as_posix())

    assert main(["validate-release-evidence", "--release-dir", payload["release_dir"]]) == 2
    validation_output = capsys.readouterr().out
    assert (
        "Release evidence validation status=INCOMPLETE" in validation_output
        or "Release evidence validation status=FAIL" in validation_output
    )


def test_load_container_release_config_direct() -> None:
    config = load_container_release_config(Path("config/container.yaml"))

    assert config.api_image_name == "medical-imaging-api"


def _original(field: str) -> object:
    values: dict[str, object] = {
        "required_uid": 10001,
        "api_port": 8000,
        "output_mount": "/app/outputs",
        "api_health_path": "/health",
        "read_only_root_filesystem": True,
        "drop_all_capabilities": True,
    }
    return values[field]


def _write_temp_container_config(tmp_path: Path) -> Path:
    config_text = Path("config/container.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "container.yaml"
    config_path.write_text(
        config_text.replace(
            '    release_output_directory: "reports/generated/releases"',
            f'    release_output_directory: "{tmp_path.as_posix()}"',
        ),
        encoding="utf-8",
    )
    return config_path
