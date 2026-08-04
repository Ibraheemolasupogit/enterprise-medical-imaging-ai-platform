import json
import subprocess
from pathlib import Path

import yaml

import medical_imaging_platform.kubernetes.assurance as kubernetes_assurance
from medical_imaging_platform.cli import main
from medical_imaging_platform.kubernetes.assurance import (
    KIND_CLUSTER_NAME,
    KIND_CONTEXT,
    RENDERED_MANIFEST_PATH,
    CommandResult,
    _write_runtime_state,
    build_kubernetes_evidence,
    build_security_summary,
    build_workload_inventory,
    clean_local_kubernetes,
    deploy_local_kubernetes,
    kubernetes_smoke,
    load_values,
    render_kubernetes_manifests,
    validate_helm_chart,
    validate_kubernetes_evidence,
    validate_kubernetes_policy,
    validate_values_schema,
)
from medical_imaging_platform.kubernetes.models import KubernetesCheckResult, KubernetesSmokeResult


def _redirect_kubernetes_evidence(monkeypatch, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    evidence_dir = tmp_path / "kubernetes-evidence"
    path_names = [
        "RENDERED_MANIFEST_PATH",
        "HELM_LINT_PATH",
        "SCHEMA_RESULT_PATH",
        "POLICY_RESULT_PATH",
        "SMOKE_RESULT_PATH",
        "WORKLOAD_INVENTORY_PATH",
        "SECURITY_SUMMARY_PATH",
        "EVIDENCE_MANIFEST_PATH",
        "EVIDENCE_REPORT_PATH",
        "CHECKSUMS_PATH",
        "RUNTIME_STATE_PATH",
    ]
    monkeypatch.setattr(kubernetes_assurance, "EVIDENCE_DIR", evidence_dir)
    for name in path_names:
        current = getattr(kubernetes_assurance, name)
        monkeypatch.setattr(kubernetes_assurance, name, evidence_dir / current.name)
    return evidence_dir


def test_helm_values_schema_and_defaults_are_secure() -> None:
    values = load_values()
    schema_check = validate_values_schema()

    assert schema_check.status == "PASS"
    assert values["global"]["imageTag"] != "latest"
    assert values["api"]["service"]["type"] == "ClusterIP"
    assert values["reviewerUi"]["service"]["type"] == "ClusterIP"
    assert values["ingress"]["enabled"] is False
    assert values["podSecurityContext"]["runAsUser"] == 10001
    assert values["containerSecurityContext"]["readOnlyRootFilesystem"] is True
    assert values["containerSecurityContext"]["capabilities"]["drop"] == ["ALL"]


def test_deterministic_rendering_and_workload_inventory(tmp_path: Path) -> None:
    first_path = tmp_path / "rendered-1.yaml"
    second_path = tmp_path / "rendered-2.yaml"
    first = render_kubernetes_manifests(first_path)
    second = render_kubernetes_manifests(second_path)

    assert first_path.read_text(encoding="utf-8") == second_path.read_text(encoding="utf-8")
    assert len(first) == len(second) == 14
    assert {item["kind"] for item in first} >= {
        "Deployment",
        "Service",
        "NetworkPolicy",
        "HorizontalPodAutoscaler",
        "PodDisruptionBudget",
        "ServiceAccount",
        "ConfigMap",
    }

    inventory = build_workload_inventory(first_path)
    assert {
        "kind": "Deployment",
        "name": "medical-imaging-platform-api",
        "component": "api",
    } in inventory


def test_rendered_security_contexts_probes_resources_and_network_policy(tmp_path: Path) -> None:
    manifest_path = tmp_path / "rendered.yaml"
    manifests = render_kubernetes_manifests(manifest_path)
    deployments = [item for item in manifests if item["kind"] == "Deployment"]

    for deployment in deployments:
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        assert pod_spec["automountServiceAccountToken"] is False
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["runAsUser"] == 10001
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["privileged"] is False
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert "livenessProbe" in container
        assert "readinessProbe" in container
        assert container["resources"]["requests"]["cpu"]
        assert container["resources"]["limits"]["memory"]
        assert all("emptyDir" in volume for volume in pod_spec["volumes"])

    services = [item for item in manifests if item["kind"] == "Service"]
    policies = [item for item in manifests if item["kind"] == "NetworkPolicy"]
    hpas = [item for item in manifests if item["kind"] == "HorizontalPodAutoscaler"]
    pdbs = [item for item in manifests if item["kind"] == "PodDisruptionBudget"]

    assert all(service["spec"]["type"] == "ClusterIP" for service in services)
    assert len(policies) == 4
    assert any(policy["metadata"]["name"].endswith("reviewer-to-api") for policy in policies)
    assert any(policy["metadata"]["name"].endswith("reviewer-egress-api") for policy in policies)
    assert len(hpas) == 2
    assert len(pdbs) == 2
    assert all(item["kind"] != "Ingress" for item in manifests)


def test_policy_validation_passes_and_rejects_unsafe_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "rendered.yaml"
    render_kubernetes_manifests(manifest_path)
    checks = validate_kubernetes_policy(manifest_path)

    assert all(check.status == "PASS" for check in checks)
    assert build_security_summary(checks)["read_only_root_filesystem"] is True

    manifests = list(yaml.safe_load_all(manifest_path.read_text(encoding="utf-8")))
    deployment = next(item for item in manifests if item and item["kind"] == "Deployment")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]
    container["image"] = "medical-imaging-api:latest"
    container["securityContext"]["readOnlyRootFilesystem"] = False
    container["securityContext"]["capabilities"]["drop"] = []
    pod_spec["volumes"].append({"name": "bad", "hostPath": {"path": "/var/run"}})
    container["resources"]["limits"].pop("memory")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump_all(manifests, sort_keys=False), encoding="utf-8")

    bad_checks = validate_kubernetes_policy(bad_path)
    assert any("NO-LATEST" in check.check_id and check.status == "FAIL" for check in bad_checks)
    assert any("READONLY" in check.check_id and check.status == "FAIL" for check in bad_checks)
    assert any("DROP-CAPS" in check.check_id and check.status == "FAIL" for check in bad_checks)
    assert any("NO-HOSTPATH" in check.check_id and check.status == "FAIL" for check in bad_checks)
    assert any("RESOURCES" in check.check_id and check.status == "FAIL" for check in bad_checks)


def test_runtime_unavailable_and_no_false_pass(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_kubernetes_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which", lambda _: None
    )

    smoke = kubernetes_smoke()
    manifest = build_kubernetes_evidence()

    assert smoke.status == "UNAVAILABLE"
    assert smoke.executed is False
    assert manifest.overall_status == "INCOMPLETE"
    assert manifest.smoke_result.status == "UNAVAILABLE"


def test_evidence_validation_and_cli_commands(capsys, tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    evidence_dir = _redirect_kubernetes_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which", lambda _: None
    )

    assert main(["validate-helm"]) == 0
    assert "Helm validation" in capsys.readouterr().out

    assert main(["render-kubernetes"]) == 0
    assert "Rendered" in capsys.readouterr().out

    assert main(["validate-kubernetes-policy"]) == 0
    assert "status=PASS" in capsys.readouterr().out

    assert main(["kubernetes-smoke"]) == 0
    assert main(["build-kubernetes-evidence"]) == 0
    assert main(["validate-kubernetes-evidence"]) == 0

    evidence_checks = validate_kubernetes_evidence()
    assert all(check.status == "PASS" for check in evidence_checks)
    assert (evidence_dir / RENDERED_MANIFEST_PATH.name).exists()


def test_schema_rejects_latest_public_exposure_and_missing_resources(tmp_path: Path) -> None:
    values = load_values()
    values["global"]["imageTag"] = "latest"
    values["api"]["service"]["type"] = "LoadBalancer"
    values["ingress"]["enabled"] = True
    values["reviewerUi"]["resources"]["limits"]["memory"] = ""
    values_path = tmp_path / "values.yaml"
    values_path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")

    check = validate_values_schema(values_path)

    assert check.status == "FAIL"
    assert any("latest" in failure for failure in check.details["failures"])
    assert any("ClusterIP" in failure for failure in check.details["failures"])
    assert any("ingress.enabled" in failure for failure in check.details["failures"])
    assert any("resources.limits.memory" in failure for failure in check.details["failures"])


def test_evidence_model_rejects_false_pass() -> None:
    smoke = KubernetesSmokeResult(status="UNAVAILABLE", executed=False, message="missing")
    payload = smoke.model_dump(mode="json")

    assert json.loads(json.dumps(payload))["status"] == "UNAVAILABLE"

    checks = validate_helm_chart()
    assert checks.status in {"PASS", "UNAVAILABLE"}


def test_kind_cluster_creation_image_loading_and_helm_deploy(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    commands: list[list[str]] = []

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "smoke.json",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", tmp_path / "state.json"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: f"/usr/local/bin/{tool}",
    )

    def fake_run(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        commands.append(args)
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, "", "")
        return CommandResult(args, 0, "ok", "")

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", fake_run)

    result = deploy_local_kubernetes()

    assert result.status == "PASS"
    assert result.executed is True
    assert result.details["created_cluster"] is True
    assert ["kind", "create", "cluster", "--name", KIND_CLUSTER_NAME, "--wait", "120s"] in commands
    assert any(command[:3] == ["kind", "load", "docker-image"] for command in commands)
    assert any(command[:3] == ["helm", "upgrade", "--install"] for command in commands)
    assert all(
        KIND_CONTEXT in command
        for command in commands
        if command and command[0] in {"kubectl", "helm"} and command[1] != "lint"
    )


def test_kind_cluster_reuse_does_not_create_cluster(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    commands: list[list[str]] = []

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "smoke.json",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", tmp_path / "state.json"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: f"/usr/local/bin/{tool}",
    )

    def fake_run(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        commands.append(args)
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, f"{KIND_CLUSTER_NAME}\n", "")
        return CommandResult(args, 0, "ok", "")

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", fake_run)

    result = deploy_local_kubernetes()

    assert result.status == "PASS"
    assert result.details["created_cluster"] is False
    assert not any(command[:4] == ["kind", "create", "cluster", "--name"] for command in commands)


def test_runtime_state_preserves_workflow_created_cluster(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", state_path
    )

    _write_runtime_state(True, "kind")
    _write_runtime_state(False, "kind")

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["created_cluster"] is True
    assert payload["cluster"] == KIND_CLUSTER_NAME
    assert payload["context"] == KIND_CONTEXT


def test_deployment_timeout_records_error_and_diagnostics(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "smoke.json",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", tmp_path / "state.json"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: f"/usr/local/bin/{tool}",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._deployment_diagnostics",
        lambda: {"pods": {"stdout": "pending"}},
    )

    def fake_run(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        if args[:3] == ["helm", "upgrade", "--install"]:
            raise subprocess.TimeoutExpired(args, timeout_seconds)
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, f"{KIND_CLUSTER_NAME}\n", "")
        return CommandResult(args, 0, "ok", "")

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", fake_run)

    result = deploy_local_kubernetes()

    assert result.status == "FAIL"
    assert any(step.status == "ERROR" for step in result.steps)
    assert result.details["diagnostics"]["pods"]["stdout"] == "pending"


def test_smoke_executes_runtime_checks_and_no_false_pass(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "smoke.json",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", tmp_path / "state.json"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: f"/usr/local/bin/{tool}",
    )

    def fake_run(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, f"{KIND_CLUSTER_NAME}\n", "")
        return CommandResult(args, 0, "ok", "")

    def fake_get_json(args: list[str], timeout_seconds: int = 180) -> dict[str, object]:
        if "pods" in args:
            return {
                "items": [
                    {
                        "metadata": {
                            "name": "api-pod",
                            "labels": {"app.kubernetes.io/component": "api"},
                        },
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    },
                    {
                        "metadata": {
                            "name": "ui-pod",
                            "labels": {"app.kubernetes.io/component": "reviewer-ui"},
                        },
                        "status": {"conditions": [{"type": "Ready", "status": "True"}]},
                    },
                ]
            }
        if "services" in args:
            return {
                "items": [
                    {"metadata": {"name": "api"}, "spec": {"type": "ClusterIP"}},
                    {"metadata": {"name": "ui"}, "spec": {"type": "ClusterIP"}},
                ]
            }
        if "networkpolicy" in args:
            return {"items": [{}, {}, {}, {}]}
        if "ingress" in args:
            return {"items": []}
        return {"items": []}

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", fake_run)
    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._get_json", fake_get_json)
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._http_runtime_checks",
        lambda: [
            KubernetesCheckResult(
                check_id="K8S-SMOKE-HTTP",
                status="PASS",
                message="HTTP checks passed.",
            )
        ],
    )

    result = kubernetes_smoke()

    assert result.status == "PASS"
    assert result.executed is True
    assert result.details["context"] == KIND_CONTEXT
    assert any(step.check_id == "K8S-SMOKE-HTTP" for step in result.steps)


def test_cleanup_uninstalls_namespace_and_created_cluster(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    commands: list[list[str]] = []
    smoke_path = tmp_path / "smoke.json"
    state_path = tmp_path / "state.json"
    smoke_path.write_text(
        KubernetesSmokeResult(
            status="PASS",
            executed=True,
            runtime="kind",
            message="smoke passed",
            steps=[
                KubernetesCheckResult(
                    check_id="K8S-SMOKE-PASS",
                    status="PASS",
                    message="passed",
                )
            ],
        ).model_dump_json(),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({"runtime": "kind", "created_cluster": True, "cluster": KIND_CLUSTER_NAME}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH", smoke_path
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", state_path
    )

    def fake_run(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        commands.append(args)
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, "", "")
        return CommandResult(args, 0, "ok", "")

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", fake_run)

    result = clean_local_kubernetes()

    assert result.status == "PASS"
    assert result.cleanup_status == "PASS"
    assert any(command[:2] == ["helm", "uninstall"] for command in commands)
    assert ["kind", "delete", "cluster", "--name", KIND_CLUSTER_NAME] in commands


def test_background_process_cleanup() -> None:
    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> None:
            if not self.killed:
                raise subprocess.TimeoutExpired(["kubectl"], timeout)

        def kill(self) -> None:
            self.killed = True

    from medical_imaging_platform.kubernetes.assurance import _terminate_process

    process = FakeProcess()
    _terminate_process(process)  # type: ignore[arg-type]

    assert process.terminated is True
    assert process.killed is True


def test_runtime_helper_error_branches(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from medical_imaging_platform.kubernetes.assurance import (
        _aggregate_status,
        _available_runtime,
        _get_json,
        _http_check,
        _load_runtime_state,
        _load_smoke_result,
        _result_check,
        _run_command,
    )

    bad_values = tmp_path / "values.yaml"
    bad_values.write_text("- not-a-map\n", encoding="utf-8")
    try:
        load_values(bad_values)
    except ValueError as exc:
        assert "mapping" in str(exc)

    assert validate_helm_chart(tmp_path / "missing-chart").status == "FAIL"
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: None if tool == "helm" else f"/usr/local/bin/{tool}",
    )
    assert validate_helm_chart().status == "UNAVAILABLE"

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH",
        tmp_path / "missing-state.json",
    )
    assert _load_runtime_state()["context"] == KIND_CONTEXT

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "missing-smoke.json",
    )
    assert _load_smoke_result().status == "UNAVAILABLE"

    pass_check = KubernetesCheckResult(check_id="pass", status="PASS", message="pass")
    fail_check = KubernetesCheckResult(check_id="fail", status="FAIL", message="fail")
    smoke = KubernetesSmokeResult(status="PASS", executed=True, message="pass")
    assert _aggregate_status(pass_check, pass_check, [pass_check], smoke) == "PASS"
    assert _aggregate_status(fail_check, pass_check, [pass_check], smoke) == "FAIL"
    assert (
        _aggregate_status(
            pass_check,
            pass_check,
            [pass_check],
            KubernetesSmokeResult(status="UNAVAILABLE", executed=False, message="missing"),
        )
        == "INCOMPLETE"
    )

    command_result = _run_command(["python3", "-c", "print('ok')"], timeout_seconds=10)
    assert command_result.returncode == 0
    assert _result_check("CHECK", command_result, "ok").status == "PASS"
    assert _http_check("HTTP-FAIL", "http://127.0.0.1:9").status == "FAIL"

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 1, "[]", "bad"),
    )
    try:
        _get_json(["kubectl", "bad"])
    except ValueError as exc:
        assert "Command failed" in str(exc)

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: "/usr/local/bin/minikube" if tool == "minikube" else None,
    )
    assert _available_runtime() == "minikube"


def test_deploy_tooling_image_and_cluster_failure_branches(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH",
        tmp_path / "smoke.json",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH",
        tmp_path / "state.json",
    )

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which", lambda _: None
    )
    assert deploy_local_kubernetes().status == "UNAVAILABLE"

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.shutil.which",
        lambda tool: f"/usr/local/bin/{tool}",
    )

    def image_failure(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        if args[:3] == ["docker", "image", "inspect"]:
            return CommandResult(args, 1, "[]", "missing")
        return CommandResult(args, 0, "", "")

    monkeypatch.setattr("medical_imaging_platform.kubernetes.assurance._run_command", image_failure)
    image_result = deploy_local_kubernetes()
    assert image_result.status == "FAIL"
    assert "image verification failed" in image_result.message

    def cluster_failure(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        if args == ["kind", "get", "clusters"]:
            return CommandResult(args, 0, "", "")
        if args[:4] == ["kind", "create", "cluster", "--name"]:
            return CommandResult(args, 1, "", "create failed")
        return CommandResult(args, 0, "ok", "")

    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._run_command", cluster_failure
    )
    cluster_result = deploy_local_kubernetes()
    assert cluster_result.status == "FAIL"
    assert any(step.check_id == "K8S-DEPLOY-KIND-CREATE" for step in cluster_result.steps)


def test_cleanup_reused_cluster_and_namespace_absence(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    smoke_path = tmp_path / "smoke.json"
    state_path = tmp_path / "state.json"
    smoke_path.write_text(
        KubernetesSmokeResult(
            status="PASS",
            executed=True,
            runtime="kind",
            message="smoke passed",
            steps=[KubernetesCheckResult(check_id="K8S-SMOKE-PASS", status="PASS", message="p")],
        ).model_dump_json(),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({"runtime": "kind", "created_cluster": False}), encoding="utf-8"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.SMOKE_RESULT_PATH", smoke_path
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance.RUNTIME_STATE_PATH", state_path
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(
            args,
            1 if args[:4] == ["kubectl", "--context", KIND_CONTEXT, "get"] else 0,
            "ok",
            "not found",
        ),
    )

    result = clean_local_kubernetes()

    assert result.status == "PASS"
    assert result.cleanup_status == "PASS"
    assert any(step.check_id == "K8S-CLEAN-KIND-CLUSTER-REUSED" for step in result.steps)


def test_http_runtime_checks_success_path(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from medical_imaging_platform.kubernetes.assurance import _http_runtime_checks

    class FakeProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int) -> int:
            return 0

    processes = [FakeProcess(), FakeProcess()]
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._start_port_forward",
        lambda resource, local_port, remote_port: processes.pop(0),
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._http_check",
        lambda check_id, url: KubernetesCheckResult(
            check_id=check_id,
            status="PASS",
            message=f"{url} passed.",
        ),
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._component_pod_name",
        lambda component: "reviewer-pod",
    )
    monkeypatch.setattr(
        "medical_imaging_platform.kubernetes.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 0, "200\n", ""),
    )

    checks = _http_runtime_checks()

    assert all(check.status == "PASS" for check in checks)
    assert {check.check_id for check in checks} >= {
        "K8S-SMOKE-API-HEALTH",
        "K8S-SMOKE-UI-TO-API",
    }
