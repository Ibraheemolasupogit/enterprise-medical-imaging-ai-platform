"""Checkout-safe Kubernetes and Helm assurance for Milestone 15."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess  # nosec B404 - optional helm lint uses shell=False and bounded timeout.
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from medical_imaging_platform.kubernetes.models import (
    KUBERNETES_DISCLAIMER,
    KubernetesCheckResult,
    KubernetesEvidenceManifest,
    KubernetesSmokeResult,
    KubernetesStatus,
)
from medical_imaging_platform.release.checksums import checksum_paths

CHART_DIR = Path("helm/medical-imaging-platform")
VALUES_PATH = CHART_DIR / "values.yaml"
SCHEMA_PATH = CHART_DIR / "values.schema.json"
EVIDENCE_DIR = Path("reports/generated/kubernetes")
RENDERED_MANIFEST_PATH = EVIDENCE_DIR / "rendered-manifests.yaml"
HELM_LINT_PATH = EVIDENCE_DIR / "helm_lint_result.json"
SCHEMA_RESULT_PATH = EVIDENCE_DIR / "schema_validation_result.json"
POLICY_RESULT_PATH = EVIDENCE_DIR / "policy_validation_result.json"
SMOKE_RESULT_PATH = EVIDENCE_DIR / "deployment_smoke_result.json"
WORKLOAD_INVENTORY_PATH = EVIDENCE_DIR / "workload_inventory.json"
SECURITY_SUMMARY_PATH = EVIDENCE_DIR / "security_control_summary.json"
EVIDENCE_MANIFEST_PATH = EVIDENCE_DIR / "kubernetes_evidence_manifest.json"
EVIDENCE_REPORT_PATH = EVIDENCE_DIR / "kubernetes_evidence_report.md"
CHECKSUMS_PATH = EVIDENCE_DIR / "checksum_manifest.json"
RUNTIME_STATE_PATH = EVIDENCE_DIR / "kubernetes_runtime_state.json"

RELEASE_NAME = "medical-imaging-platform"
KIND_CLUSTER_NAME = "medical-imaging-platform"
KIND_CONTEXT = f"kind-{KIND_CLUSTER_NAME}"
NAMESPACE = "medical-imaging-platform"
LOCAL_IMAGE_TAG = "0.1.0-m13-local"
API_IMAGE = f"medical-imaging-api:{LOCAL_IMAGE_TAG}"
REVIEWER_UI_IMAGE = f"medical-imaging-reviewer-ui:{LOCAL_IMAGE_TAG}"
COMMAND_TIMEOUT_SECONDS = 180
APP_LABELS = {
    "app.kubernetes.io/name": "medical-imaging-platform",
    "app.kubernetes.io/instance": RELEASE_NAME,
    "app.kubernetes.io/version": "0.1.0",
    "app.kubernetes.io/managed-by": "Helm",
    "medical-imaging-platform.openai.com/scope": "research-engineering-demonstrator",
}


@dataclass(frozen=True)
class CommandResult:
    """Bounded command result for Kubernetes runtime execution."""

    args: list[str]
    returncode: int
    stdout: str
    stderr: str


def load_values(path: Path = VALUES_PATH) -> dict[str, Any]:
    """Load Helm values from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Helm values must be a mapping.")
    return payload


def validate_helm_chart(chart_dir: Path = CHART_DIR) -> KubernetesCheckResult:
    """Validate chart structure and run helm lint when available."""
    required = [
        chart_dir / "Chart.yaml",
        chart_dir / "values.yaml",
        chart_dir / "values.schema.json",
        chart_dir / "templates",
    ]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        return KubernetesCheckResult(
            check_id="HELM-CHART-STRUCTURE",
            status="FAIL",
            message="Required Helm chart files are missing.",
            details={"missing": missing},
        )
    helm = shutil.which("helm")
    if helm is None:
        return KubernetesCheckResult(
            check_id="HELM-LINT",
            status="UNAVAILABLE",
            message="helm is not installed; internal chart structure validation passed.",
            details={
                "mandatory": False,
                "bootstrap": "Install Helm locally, then run `make validate-helm`.",
            },
        )
    result = subprocess.run(  # nosec B603 - command list is fixed internal helm lint invocation.
        [helm, "lint", chart_dir.as_posix()],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return KubernetesCheckResult(
        check_id="HELM-LINT",
        status="PASS" if result.returncode == 0 else "FAIL",
        message="helm lint completed.",
        details={"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
    )


def validate_values_schema(
    values_path: Path = VALUES_PATH, schema_path: Path = SCHEMA_PATH
) -> KubernetesCheckResult:
    """Run a small built-in schema validation for critical values."""
    values = load_values(values_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if schema.get("type") != "object":
        failures.append("schema root must be an object")
    if values["global"]["imageTag"] == "latest":
        failures.append("global.imageTag must not be latest")
    for workload in ("api", "reviewerUi"):
        service_type = values[workload]["service"]["type"]
        if service_type != "ClusterIP":
            failures.append(f"{workload}.service.type must remain ClusterIP by default")
        resources = values[workload]["resources"]
        for section in ("requests", "limits"):
            for resource in ("cpu", "memory"):
                if not resources.get(section, {}).get(resource):
                    failures.append(f"{workload}.resources.{section}.{resource} is required")
    if values["ingress"]["enabled"]:
        failures.append("ingress.enabled must be false by default")
    status: KubernetesStatus = "PASS" if not failures else "FAIL"
    return KubernetesCheckResult(
        check_id="HELM-VALUES-SCHEMA",
        status=status,
        message="Helm values schema validation completed.",
        details={"failures": failures, "schema": schema_path.as_posix()},
    )


def render_kubernetes_manifests(
    output_path: Path = RENDERED_MANIFEST_PATH,
    values_path: Path = VALUES_PATH,
) -> list[dict[str, Any]]:
    """Render deterministic Kubernetes manifests from chart values."""
    values = load_values(values_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifests = _render_objects(values)
    output_path.write_text(
        "\n---\n".join(yaml.safe_dump(item, sort_keys=False) for item in manifests),
        encoding="utf-8",
    )
    return manifests


def load_rendered_manifests(path: Path = RENDERED_MANIFEST_PATH) -> list[dict[str, Any]]:
    """Load rendered manifest documents."""
    if not path.exists():
        render_kubernetes_manifests(path)
    return [item for item in yaml.safe_load_all(path.read_text(encoding="utf-8")) if item]


def validate_kubernetes_policy(
    manifest_path: Path = RENDERED_MANIFEST_PATH,
) -> list[KubernetesCheckResult]:
    """Reject unsafe Kubernetes patterns in rendered manifests."""
    manifests = load_rendered_manifests(manifest_path)
    checks: list[KubernetesCheckResult] = []
    deployments = [item for item in manifests if item.get("kind") == "Deployment"]
    services = [item for item in manifests if item.get("kind") == "Service"]
    network_policies = [item for item in manifests if item.get("kind") == "NetworkPolicy"]
    hpas = [item for item in manifests if item.get("kind") == "HorizontalPodAutoscaler"]
    pdbs = [item for item in manifests if item.get("kind") == "PodDisruptionBudget"]

    for deployment in deployments:
        name = deployment["metadata"]["name"]
        pod_spec = deployment["spec"]["template"]["spec"]
        pod_security = pod_spec.get("securityContext", {})
        containers = pod_spec.get("containers", [])
        checks.extend(
            [
                _check(
                    f"K8S-{name}-POD-NONROOT",
                    pod_security.get("runAsNonRoot") is True
                    and pod_security.get("runAsUser") == 10001
                    and pod_security.get("runAsGroup") == 10001,
                    "Pods run as non-root UID/GID 10001.",
                    {"securityContext": pod_security},
                ),
                _check(
                    f"K8S-{name}-SECCOMP",
                    pod_security.get("seccompProfile", {}).get("type") == "RuntimeDefault",
                    "Pods use RuntimeDefault seccomp.",
                ),
                _check(
                    f"K8S-{name}-NO-HOST-NS",
                    not any(pod_spec.get(field) for field in ("hostNetwork", "hostPID", "hostIPC")),
                    "Pods do not use host namespaces.",
                ),
                _check(
                    f"K8S-{name}-SA-TOKEN",
                    pod_spec.get("automountServiceAccountToken") is False,
                    "Service account token automount is disabled.",
                ),
                _check(
                    f"K8S-{name}-TERMINATION",
                    0 < int(pod_spec.get("terminationGracePeriodSeconds", 0)) <= 60,
                    "Graceful termination is bounded.",
                ),
                _check(
                    f"K8S-{name}-NO-HOSTPATH",
                    all("hostPath" not in volume for volume in pod_spec.get("volumes", [])),
                    "No hostPath volumes are used.",
                ),
                _check(
                    f"K8S-{name}-WRITABLE-EMPTYDIR",
                    all("emptyDir" in volume for volume in pod_spec.get("volumes", [])),
                    "Writable mounts are emptyDir only.",
                ),
            ]
        )
        for container in containers:
            cname = container["name"]
            image = str(container.get("image", ""))
            security = container.get("securityContext", {})
            resources = container.get("resources", {})
            checks.extend(
                [
                    _check(
                        f"K8S-{name}-{cname}-NO-LATEST",
                        ":latest" not in image and not image.endswith(":latest"),
                        "Images do not use a floating latest tag.",
                        {"image": image},
                    ),
                    _check(
                        f"K8S-{name}-{cname}-READONLY",
                        security.get("readOnlyRootFilesystem") is True,
                        "Container root filesystem is read-only.",
                    ),
                    _check(
                        f"K8S-{name}-{cname}-NO-PRIVILEGE",
                        security.get("allowPrivilegeEscalation") is False
                        and security.get("privileged") is False,
                        "Container is not privileged and cannot escalate privileges.",
                    ),
                    _check(
                        f"K8S-{name}-{cname}-DROP-CAPS",
                        security.get("capabilities", {}).get("drop") == ["ALL"],
                        "All Linux capabilities are dropped.",
                    ),
                    _check(
                        f"K8S-{name}-{cname}-RESOURCES",
                        _has_resources(resources),
                        "CPU and memory requests and limits are set.",
                        {"resources": resources},
                    ),
                    _check(
                        f"K8S-{name}-{cname}-PROBES",
                        "livenessProbe" in container and "readinessProbe" in container,
                        "Liveness and readiness probes are configured.",
                    ),
                ]
            )
    checks.extend(
        [
            _check(
                "K8S-SERVICES-INTERNAL",
                all(service.get("spec", {}).get("type") == "ClusterIP" for service in services),
                "Services are ClusterIP by default.",
            ),
            _check(
                "K8S-INGRESS-DISABLED",
                all(item.get("kind") != "Ingress" for item in manifests),
                "Ingress is disabled by default.",
            ),
            _check(
                "K8S-NETWORKPOLICY-PRESENT",
                len(network_policies) >= 3,
                "Default-deny, reviewer/API, and DNS NetworkPolicies are rendered.",
            ),
            _check("K8S-HPA-PRESENT", len(hpas) == 2, "API and reviewer UI HPAs are rendered."),
            _check("K8S-PDB-PRESENT", len(pdbs) == 2, "API and reviewer UI PDBs are rendered."),
        ]
    )
    return checks


def build_workload_inventory(manifest_path: Path = RENDERED_MANIFEST_PATH) -> list[dict[str, Any]]:
    """Summarise rendered workloads without sensitive values."""
    inventory: list[dict[str, Any]] = []
    for item in load_rendered_manifests(manifest_path):
        kind = item.get("kind")
        if kind not in {"Deployment", "Service", "HorizontalPodAutoscaler", "PodDisruptionBudget"}:
            continue
        metadata = item.get("metadata", {})
        inventory.append(
            {
                "kind": kind,
                "name": metadata.get("name"),
                "component": metadata.get("labels", {}).get("app.kubernetes.io/component"),
            }
        )
    return inventory


def build_security_summary(policy_checks: list[KubernetesCheckResult]) -> dict[str, Any]:
    """Summarise policy controls for evidence."""
    return {
        "run_as_non_root_uid_gid": 10001,
        "read_only_root_filesystem": _all_pass(policy_checks, "READONLY"),
        "capabilities_drop_all": _all_pass(policy_checks, "DROP-CAPS"),
        "no_privileged_containers": _all_pass(policy_checks, "NO-PRIVILEGE"),
        "no_host_namespaces": _all_pass(policy_checks, "NO-HOST-NS"),
        "no_hostpath": _all_pass(policy_checks, "NO-HOSTPATH"),
        "resource_limits_required": _all_pass(policy_checks, "RESOURCES"),
        "probes_required": _all_pass(policy_checks, "PROBES"),
        "network_policy_required": _all_pass(policy_checks, "NETWORKPOLICY"),
    }


def deploy_local_kubernetes() -> KubernetesSmokeResult:
    """Create or reuse the dedicated kind cluster and deploy the chart with Helm."""
    steps: list[KubernetesCheckResult] = []
    runtime = "kind" if shutil.which("kind") is not None else None
    if runtime is None or shutil.which("kubectl") is None or shutil.which("helm") is None:
        result = KubernetesSmokeResult(
            status="UNAVAILABLE",
            executed=False,
            runtime=runtime,
            cleanup_status="UNAVAILABLE",
            message="kind, kubectl and helm are required for automatic local runtime execution.",
            steps=[
                KubernetesCheckResult(
                    check_id="K8S-DEPLOY-TOOLING",
                    status="UNAVAILABLE",
                    message="Required local Kubernetes tooling is unavailable.",
                    details={
                        "kind": shutil.which("kind"),
                        "kubectl": shutil.which("kubectl"),
                        "helm": shutil.which("helm"),
                    },
                )
            ],
        )
        _write_json(SMOKE_RESULT_PATH, result.model_dump(mode="json"))
        return result

    created_cluster = False
    try:
        steps.append(
            _command_check("K8S-DEPLOY-DOCKER", ["docker", "info"], "Docker is available.")
        )
        for image in (API_IMAGE, REVIEWER_UI_IMAGE):
            steps.append(
                _command_check(
                    f"K8S-DEPLOY-IMAGE-{image}",
                    ["docker", "image", "inspect", image],
                    f"Local image {image} exists.",
                )
            )
        if any(step.status != "PASS" for step in steps):
            return _persist_runtime_result(
                "FAIL",
                True,
                runtime,
                "Docker or local image verification failed.",
                steps,
                {"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
            )

        clusters = _run_command(["kind", "get", "clusters"])
        cluster_exists = KIND_CLUSTER_NAME in clusters.stdout.splitlines()
        if cluster_exists:
            steps.append(
                KubernetesCheckResult(
                    check_id="K8S-DEPLOY-KIND-REUSE",
                    status="PASS",
                    message="Dedicated kind cluster already exists and will be reused.",
                    details={"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
                )
            )
        else:
            create_result = _run_command(
                ["kind", "create", "cluster", "--name", KIND_CLUSTER_NAME, "--wait", "120s"],
                timeout_seconds=180,
            )
            created_cluster = create_result.returncode == 0
            steps.append(
                _result_check(
                    "K8S-DEPLOY-KIND-CREATE",
                    create_result,
                    "Dedicated kind cluster created.",
                    {"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
                )
            )
            if not created_cluster:
                _write_runtime_state(False, "kind")
                return _persist_runtime_result(
                    "FAIL",
                    True,
                    runtime,
                    "Dedicated kind cluster creation failed.",
                    steps,
                    {"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
                )

        _write_runtime_state(created_cluster, "kind")
        for image in (API_IMAGE, REVIEWER_UI_IMAGE):
            load_result = _run_command(
                ["kind", "load", "docker-image", image, "--name", KIND_CLUSTER_NAME],
                timeout_seconds=180,
            )
            steps.append(
                _result_check(
                    f"K8S-DEPLOY-LOAD-{image}",
                    load_result,
                    f"Loaded {image} into kind.",
                )
            )

        namespace_result = _run_command(
            ["kubectl", "--context", KIND_CONTEXT, "create", "namespace", NAMESPACE],
            timeout_seconds=60,
        )
        namespace_pass = (
            namespace_result.returncode == 0 or "AlreadyExists" in namespace_result.stderr
        )
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-DEPLOY-NAMESPACE",
                status="PASS" if namespace_pass else "FAIL",
                message="Dedicated namespace is present.",
                details=_command_details(namespace_result),
            )
        )
        helm_result = _run_command(
            [
                "helm",
                "upgrade",
                "--install",
                RELEASE_NAME,
                CHART_DIR.as_posix(),
                "--namespace",
                NAMESPACE,
                "--kube-context",
                KIND_CONTEXT,
                "--set",
                f"global.imageTag={LOCAL_IMAGE_TAG}",
                "--set",
                "global.imagePullPolicy=IfNotPresent",
                "--set",
                "api.image.pullPolicy=IfNotPresent",
                "--set",
                "reviewerUi.image.pullPolicy=IfNotPresent",
                "--wait",
                "--timeout",
                "180s",
            ],
            timeout_seconds=240,
        )
        steps.append(
            _result_check("K8S-DEPLOY-HELM", helm_result, "Helm install/upgrade completed.")
        )
        for deployment in ("medical-imaging-platform-api", "medical-imaging-platform-reviewer-ui"):
            wait_result = _run_command(
                [
                    "kubectl",
                    "--context",
                    KIND_CONTEXT,
                    "-n",
                    NAMESPACE,
                    "wait",
                    f"deployment/{deployment}",
                    "--for=condition=Available",
                    "--timeout=180s",
                ],
                timeout_seconds=210,
            )
            steps.append(
                _result_check(
                    f"K8S-DEPLOY-READY-{deployment}",
                    wait_result,
                    f"{deployment} became Available.",
                )
            )
    except (subprocess.TimeoutExpired, OSError) as exc:
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-DEPLOY-ERROR",
                status="ERROR",
                message=f"Deployment command failed: {exc}",
            )
        )

    status: KubernetesStatus = "PASS" if all(step.status == "PASS" for step in steps) else "FAIL"
    diagnostics = _deployment_diagnostics() if status != "PASS" else {}
    runtime_state = _load_runtime_state()
    return _persist_runtime_result(
        status,
        True,
        runtime,
        "Local Kubernetes deployment completed."
        if status == "PASS"
        else "Local deployment failed.",
        steps,
        {
            "cluster": KIND_CLUSTER_NAME,
            "context": KIND_CONTEXT,
            "namespace": NAMESPACE,
            "created_cluster": bool(runtime_state.get("created_cluster")),
            "diagnostics": diagnostics,
        },
    )


def kubernetes_smoke() -> KubernetesSmokeResult:
    """Run deterministic runtime smoke checks against the dedicated kind context."""
    steps: list[KubernetesCheckResult] = []
    runtime = "kind" if shutil.which("kind") is not None else None
    if runtime is None or shutil.which("kubectl") is None:
        return _persist_runtime_result(
            "UNAVAILABLE",
            False,
            runtime,
            "kubectl and kind are required for live Kubernetes smoke testing.",
            [
                KubernetesCheckResult(
                    check_id="K8S-SMOKE-TOOLING",
                    status="UNAVAILABLE",
                    message="Local Kubernetes smoke tooling is unavailable.",
                )
            ],
            {"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
        )

    try:
        cluster_check = _run_command(["kind", "get", "clusters"])
        dedicated_cluster = KIND_CLUSTER_NAME in cluster_check.stdout.splitlines()
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-SMOKE-DEDICATED-CONTEXT",
                status="PASS" if dedicated_cluster else "FAIL",
                message="Smoke uses only the dedicated kind context.",
                details={"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
            )
        )
        for deployment in ("medical-imaging-platform-api", "medical-imaging-platform-reviewer-ui"):
            steps.append(
                _command_check(
                    f"K8S-SMOKE-DEPLOYMENT-{deployment}",
                    [
                        "kubectl",
                        "--context",
                        KIND_CONTEXT,
                        "-n",
                        NAMESPACE,
                        "wait",
                        f"deployment/{deployment}",
                        "--for=condition=Available",
                        "--timeout=120s",
                    ],
                    f"{deployment} is Available.",
                    timeout_seconds=150,
                )
            )
        pods = _get_json(
            ["kubectl", "--context", KIND_CONTEXT, "-n", NAMESPACE, "get", "pods", "-o", "json"]
        )
        pod_items = pods.get("items", [])
        steps.extend(_pod_readiness_checks(pod_items))
        manifests = load_rendered_manifests()
        steps.extend(validate_kubernetes_policy())
        steps.extend(_service_runtime_checks())
        steps.extend(_runtime_manifest_checks(manifests))
        steps.extend(_http_runtime_checks())
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-SMOKE-ERROR",
                status="ERROR",
                message=f"Runtime smoke failed: {exc}",
                details={"diagnostics": _deployment_diagnostics()},
            )
        )

    status: KubernetesStatus = "PASS" if all(step.status == "PASS" for step in steps) else "FAIL"
    return _persist_runtime_result(
        status,
        True,
        runtime,
        "Kubernetes smoke completed." if status == "PASS" else "Kubernetes smoke failed.",
        steps,
        {
            "cluster": KIND_CLUSTER_NAME,
            "context": KIND_CONTEXT,
            "namespace": NAMESPACE,
            "diagnostics": {} if status == "PASS" else _deployment_diagnostics(),
        },
    )


def clean_local_kubernetes() -> KubernetesSmokeResult:
    """Uninstall the release, remove namespace, and delete workflow-created kind cluster."""
    previous = _load_smoke_result()
    state = _load_runtime_state()
    steps = list(previous.steps)
    runtime = state.get("runtime", "kind") if state else "kind"
    try:
        uninstall = _run_command(
            [
                "helm",
                "uninstall",
                RELEASE_NAME,
                "--namespace",
                NAMESPACE,
                "--kube-context",
                KIND_CONTEXT,
            ],
            timeout_seconds=120,
        )
        uninstall_pass = uninstall.returncode == 0 or "not found" in uninstall.stderr.lower()
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-CLEAN-HELM-UNINSTALL",
                status="PASS" if uninstall_pass else "FAIL",
                message="Helm release uninstalled or already absent.",
                details=_command_details(uninstall),
            )
        )
        delete_namespace = _run_command(
            [
                "kubectl",
                "--context",
                KIND_CONTEXT,
                "delete",
                "namespace",
                NAMESPACE,
                "--ignore-not-found=true",
                "--wait=true",
                "--timeout=120s",
            ],
            timeout_seconds=150,
        )
        steps.append(
            _result_check(
                "K8S-CLEAN-NAMESPACE",
                delete_namespace,
                "Dedicated namespace deleted or already absent.",
            )
        )
        if bool(state.get("created_cluster")):
            delete_cluster = _run_command(
                ["kind", "delete", "cluster", "--name", KIND_CLUSTER_NAME],
                timeout_seconds=180,
            )
            steps.append(
                _result_check(
                    "K8S-CLEAN-KIND-CLUSTER",
                    delete_cluster,
                    "Workflow-created kind cluster deleted.",
                )
            )
        else:
            steps.append(
                KubernetesCheckResult(
                    check_id="K8S-CLEAN-KIND-CLUSTER-REUSED",
                    status="PASS",
                    message="Dedicated kind cluster was reused and was not deleted.",
                    details={"cluster": KIND_CLUSTER_NAME, "context": KIND_CONTEXT},
                )
            )
        remaining = _confirm_cleanup(bool(state.get("created_cluster")))
        steps.extend(remaining)
    except (subprocess.TimeoutExpired, OSError) as exc:
        steps.append(
            KubernetesCheckResult(
                check_id="K8S-CLEAN-ERROR",
                status="ERROR",
                message=f"Cleanup command failed: {exc}",
                details={"diagnostics": _deployment_diagnostics()},
            )
        )
    cleanup_checks = [step for step in steps if step.check_id.startswith("K8S-CLEAN")]
    cleanup_status: KubernetesStatus = (
        "PASS" if all(step.status == "PASS" for step in cleanup_checks) else "FAIL"
    )
    prior_smoke_steps = [step for step in steps if not step.check_id.startswith("K8S-CLEAN")]
    prior_smoke_pass = prior_smoke_steps and all(
        step.status == "PASS" for step in prior_smoke_steps
    )
    runtime_status: KubernetesStatus = (
        "PASS" if prior_smoke_pass and cleanup_status == "PASS" else "FAIL"
    )
    cleanup_message = (
        "Local Kubernetes cleanup completed."
        if cleanup_status == "PASS"
        else "Local Kubernetes cleanup failed."
    )
    result = KubernetesSmokeResult(
        status=runtime_status,
        executed=previous.executed,
        runtime=runtime,
        cleanup_status=cleanup_status,
        message=cleanup_message,
        steps=steps,
        details={
            "cluster": KIND_CLUSTER_NAME,
            "context": KIND_CONTEXT,
            "namespace": NAMESPACE,
            "created_cluster": bool(state.get("created_cluster")),
        },
    )
    _write_json(SMOKE_RESULT_PATH, result.model_dump(mode="json"))
    return result


def build_kubernetes_evidence() -> KubernetesEvidenceManifest:
    """Build deterministic ignored Kubernetes evidence."""
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    helm_lint = validate_helm_chart()
    schema_validation = validate_values_schema()
    render_kubernetes_manifests(RENDERED_MANIFEST_PATH)
    policy_validation = validate_kubernetes_policy(RENDERED_MANIFEST_PATH)
    smoke_result = _load_smoke_result()
    inventory = build_workload_inventory(RENDERED_MANIFEST_PATH)
    security = build_security_summary(policy_validation)
    _write_json(HELM_LINT_PATH, helm_lint.model_dump(mode="json"))
    _write_json(SCHEMA_RESULT_PATH, schema_validation.model_dump(mode="json"))
    _write_json(POLICY_RESULT_PATH, [check.model_dump(mode="json") for check in policy_validation])
    _write_json(WORKLOAD_INVENTORY_PATH, inventory)
    _write_json(SECURITY_SUMMARY_PATH, security)
    checksums = checksum_paths(
        [
            RENDERED_MANIFEST_PATH,
            HELM_LINT_PATH,
            SCHEMA_RESULT_PATH,
            POLICY_RESULT_PATH,
            SMOKE_RESULT_PATH,
            WORKLOAD_INVENTORY_PATH,
            SECURITY_SUMMARY_PATH,
        ]
    )
    manifest = KubernetesEvidenceManifest(
        evidence_id="m15-kubernetes-evidence-v1",
        generated_timestamp="2026-01-01T00:30:00Z",
        overall_status=_aggregate_status(
            helm_lint, schema_validation, policy_validation, smoke_result
        ),
        helm_chart=CHART_DIR.as_posix(),
        rendered_manifest=RENDERED_MANIFEST_PATH.as_posix(),
        helm_lint=helm_lint,
        schema_validation=schema_validation,
        policy_validation=policy_validation,
        smoke_result=smoke_result,
        workload_inventory=inventory,
        security_controls=security,
        checksums=checksums,
    )
    _write_json(EVIDENCE_MANIFEST_PATH, manifest.model_dump(mode="json"))
    checksums = checksum_paths(
        [
            RENDERED_MANIFEST_PATH,
            HELM_LINT_PATH,
            SCHEMA_RESULT_PATH,
            POLICY_RESULT_PATH,
            SMOKE_RESULT_PATH,
            WORKLOAD_INVENTORY_PATH,
            SECURITY_SUMMARY_PATH,
            EVIDENCE_MANIFEST_PATH,
        ]
    )
    _write_json(CHECKSUMS_PATH, checksums)
    _write_report(manifest)
    return manifest.model_copy(update={"checksums": checksums})


def validate_kubernetes_evidence() -> list[KubernetesCheckResult]:
    """Validate that Kubernetes evidence is present, consistent, and honest."""
    required_paths = [
        RENDERED_MANIFEST_PATH,
        HELM_LINT_PATH,
        SCHEMA_RESULT_PATH,
        POLICY_RESULT_PATH,
        SMOKE_RESULT_PATH,
        WORKLOAD_INVENTORY_PATH,
        SECURITY_SUMMARY_PATH,
        EVIDENCE_MANIFEST_PATH,
        EVIDENCE_REPORT_PATH,
        CHECKSUMS_PATH,
    ]
    checks = [
        KubernetesCheckResult(
            check_id=f"K8S-EVIDENCE-{path.name}",
            status="PASS" if path.exists() else "FAIL",
            message=f"{path.as_posix()} exists."
            if path.exists()
            else f"{path.as_posix()} is missing.",
        )
        for path in required_paths
    ]
    if EVIDENCE_MANIFEST_PATH.exists():
        manifest = KubernetesEvidenceManifest.model_validate_json(
            EVIDENCE_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        no_false_pass = manifest.overall_status != "PASS" or manifest.smoke_result.status == "PASS"
        checks.append(
            KubernetesCheckResult(
                check_id="K8S-EVIDENCE-NO-FALSE-PASS",
                status="PASS" if no_false_pass else "FAIL",
                message="Overall PASS is only allowed when runtime smoke is PASS.",
                details={
                    "overall_status": manifest.overall_status,
                    "smoke": manifest.smoke_result.status,
                },
            )
        )
    return checks


def _run_command(args: list[str], timeout_seconds: int = COMMAND_TIMEOUT_SECONDS) -> CommandResult:
    completed = subprocess.run(  # nosec B603 - command lists are fixed internal invocations.
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return CommandResult(
        args=args,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _command_details(result: CommandResult) -> dict[str, Any]:
    return {
        "command": result.args,
        "returncode": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
    }


def _result_check(
    check_id: str,
    result: CommandResult,
    message: str,
    details: dict[str, Any] | None = None,
) -> KubernetesCheckResult:
    payload = _command_details(result)
    if details:
        payload.update(details)
    return KubernetesCheckResult(
        check_id=check_id,
        status="PASS" if result.returncode == 0 else "FAIL",
        message=message if result.returncode == 0 else f"{message} Command failed.",
        details=payload,
    )


def _command_check(
    check_id: str,
    args: list[str],
    message: str,
    timeout_seconds: int = COMMAND_TIMEOUT_SECONDS,
) -> KubernetesCheckResult:
    return _result_check(check_id, _run_command(args, timeout_seconds), message)


def _write_runtime_state(created_cluster: bool, runtime: str) -> None:
    previous_state = _load_runtime_state()
    created_by_workflow = created_cluster or bool(previous_state.get("created_cluster"))
    _write_json(
        RUNTIME_STATE_PATH,
        {
            "runtime": runtime,
            "cluster": KIND_CLUSTER_NAME,
            "context": KIND_CONTEXT,
            "namespace": NAMESPACE,
            "created_cluster": created_by_workflow,
        },
    )


def _load_runtime_state() -> dict[str, Any]:
    if not RUNTIME_STATE_PATH.exists():
        return {
            "runtime": "kind",
            "cluster": KIND_CLUSTER_NAME,
            "context": KIND_CONTEXT,
            "namespace": NAMESPACE,
            "created_cluster": False,
        }
    payload = json.loads(RUNTIME_STATE_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _persist_runtime_result(
    status: KubernetesStatus,
    executed: bool,
    runtime: str | None,
    message: str,
    steps: list[KubernetesCheckResult],
    details: dict[str, Any],
) -> KubernetesSmokeResult:
    previous = _load_smoke_result()
    cleanup_status = previous.cleanup_status if previous.executed else "UNAVAILABLE"
    result = KubernetesSmokeResult(
        status=status,
        executed=executed,
        runtime=runtime,
        cleanup_status=cleanup_status,
        message=message,
        steps=steps,
        details=details,
    )
    _write_json(SMOKE_RESULT_PATH, result.model_dump(mode="json"))
    return result


def _get_json(args: list[str], timeout_seconds: int = COMMAND_TIMEOUT_SECONDS) -> dict[str, Any]:
    result = _run_command(args, timeout_seconds)
    if result.returncode != 0:
        raise ValueError(f"Command failed: {' '.join(args)}\n{result.stderr}")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from kubectl.")
    return payload


def _deployment_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {}
    commands = {
        "pods": [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "pods",
            "-o",
            "wide",
        ],
        "events": [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "events",
            "--sort-by=.lastTimestamp",
        ],
        "helm_status": [
            "helm",
            "status",
            RELEASE_NAME,
            "--namespace",
            NAMESPACE,
            "--kube-context",
            KIND_CONTEXT,
        ],
    }
    for name, command in commands.items():
        try:
            diagnostics[name] = _command_details(_run_command(command, timeout_seconds=30))
        except (subprocess.TimeoutExpired, OSError) as exc:
            diagnostics[name] = {"error": str(exc)}
    return diagnostics


def _pod_readiness_checks(pods: list[dict[str, Any]]) -> list[KubernetesCheckResult]:
    checks: list[KubernetesCheckResult] = []
    for component in ("api", "reviewer-ui"):
        matching = [
            pod
            for pod in pods
            if pod.get("metadata", {}).get("labels", {}).get("app.kubernetes.io/component")
            == component
        ]
        ready = bool(matching) and all(_pod_ready(pod) for pod in matching)
        checks.append(
            KubernetesCheckResult(
                check_id=f"K8S-SMOKE-POD-READY-{component}",
                status="PASS" if ready else "FAIL",
                message=f"{component} pod is Ready.",
                details={"pods": [pod.get("metadata", {}).get("name") for pod in matching]},
            )
        )
    return checks


def _pod_ready(pod: dict[str, Any]) -> bool:
    conditions = pod.get("status", {}).get("conditions", [])
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in conditions
    )


def _service_runtime_checks() -> list[KubernetesCheckResult]:
    services = _get_json(
        [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "services",
            "-o",
            "json",
        ]
    ).get("items", [])
    policies = _get_json(
        [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "networkpolicy",
            "-o",
            "json",
        ]
    ).get("items", [])
    ingress = _get_json(
        [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "ingress",
            "-o",
            "json",
        ]
    ).get("items", [])
    services_are_internal = bool(services) and all(
        service.get("spec", {}).get("type") == "ClusterIP" for service in services
    )
    return [
        KubernetesCheckResult(
            check_id="K8S-SMOKE-SERVICES-CLUSTERIP",
            status="PASS" if services_are_internal else "FAIL",
            message="Runtime services are ClusterIP only.",
            details={"services": [service.get("metadata", {}).get("name") for service in services]},
        ),
        KubernetesCheckResult(
            check_id="K8S-SMOKE-NETWORKPOLICY",
            status="PASS" if len(policies) >= 3 else "FAIL",
            message="Runtime NetworkPolicies are present.",
            details={"count": len(policies)},
        ),
        KubernetesCheckResult(
            check_id="K8S-SMOKE-INGRESS-ABSENT",
            status="PASS" if not ingress else "FAIL",
            message="Ingress is absent by default.",
            details={"count": len(ingress)},
        ),
    ]


def _runtime_manifest_checks(manifests: list[dict[str, Any]]) -> list[KubernetesCheckResult]:
    deployments = [item for item in manifests if item.get("kind") == "Deployment"]
    checks: list[KubernetesCheckResult] = []
    for deployment in deployments:
        component = deployment["metadata"]["labels"]["app.kubernetes.io/component"]
        pod_spec = deployment["spec"]["template"]["spec"]
        container = pod_spec["containers"][0]
        security = container["securityContext"]
        resources = container["resources"]
        pod_security = pod_spec["securityContext"]
        checks.extend(
            [
                KubernetesCheckResult(
                    check_id=f"K8S-SMOKE-UID-GID-{component}",
                    status="PASS"
                    if pod_security.get("runAsUser") == 10001
                    and pod_security.get("runAsGroup") == 10001
                    else "FAIL",
                    message=f"{component} runs as UID/GID 10001.",
                    details={"securityContext": pod_security},
                ),
                KubernetesCheckResult(
                    check_id=f"K8S-SMOKE-SECURITY-{component}",
                    status="PASS"
                    if pod_security.get("runAsNonRoot") is True
                    and security.get("readOnlyRootFilesystem") is True
                    and security.get("allowPrivilegeEscalation") is False
                    and security.get("capabilities", {}).get("drop") == ["ALL"]
                    else "FAIL",
                    message=f"{component} non-root, read-only and capability controls are present.",
                ),
                KubernetesCheckResult(
                    check_id=f"K8S-SMOKE-PROBES-{component}",
                    status="PASS"
                    if "livenessProbe" in container and "readinessProbe" in container
                    else "FAIL",
                    message=f"{component} probes are present.",
                ),
                KubernetesCheckResult(
                    check_id=f"K8S-SMOKE-RESOURCES-{component}",
                    status="PASS" if _has_resources(resources) else "FAIL",
                    message=f"{component} CPU and memory requests/limits are present.",
                    details={"resources": resources},
                ),
            ]
        )
    return checks


def _http_runtime_checks() -> list[KubernetesCheckResult]:
    checks: list[KubernetesCheckResult] = []
    port_forwards: list[subprocess.Popen[str]] = []
    try:
        api_forward = _start_port_forward("service/medical-imaging-platform-api", 18080, 8000)
        ui_forward = _start_port_forward(
            "service/medical-imaging-platform-reviewer-ui", 18501, 8501
        )
        port_forwards.extend([api_forward, ui_forward])
        checks.extend(
            [
                _http_check("K8S-SMOKE-API-HEALTH", "http://127.0.0.1:18080/health"),
                _http_check("K8S-SMOKE-API-READY", "http://127.0.0.1:18080/ready"),
                _http_check("K8S-SMOKE-API-VERSION", "http://127.0.0.1:18080/version"),
                _http_check("K8S-SMOKE-UI-HEALTH", "http://127.0.0.1:18501/_stcore/health"),
            ]
        )
        reviewer_pod = _component_pod_name("reviewer-ui")
        exec_result = _run_command(
            [
                "kubectl",
                "--context",
                KIND_CONTEXT,
                "-n",
                NAMESPACE,
                "exec",
                reviewer_pod,
                "--",
                "python",
                "-c",
                (
                    "import urllib.request; "
                    "print(urllib.request.urlopen("
                    "'http://medical-imaging-platform-api:8000/health', timeout=5).status)"
                ),
            ],
            timeout_seconds=30,
        )
        checks.append(
            KubernetesCheckResult(
                check_id="K8S-SMOKE-UI-TO-API",
                status="PASS"
                if exec_result.returncode == 0 and "200" in exec_result.stdout
                else "FAIL",
                message="Reviewer UI pod can reach API service.",
                details=_command_details(exec_result),
            )
        )
    finally:
        for process in port_forwards:
            _terminate_process(process)
    return checks


def _start_port_forward(resource: str, local_port: int, remote_port: int) -> subprocess.Popen[str]:
    args = [
        "kubectl",
        "--context",
        KIND_CONTEXT,
        "-n",
        NAMESPACE,
        "port-forward",
        resource,
        f"{local_port}:{remote_port}",
    ]
    process = subprocess.Popen(  # nosec B603 - fixed kubectl port-forward command list.
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if process.poll() is not None:
            _, stderr = process.communicate(timeout=1)
            raise ValueError(f"port-forward failed for {resource}: {stderr}")
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                pass
            return process
        except OSError:
            time.sleep(0.25)
    _terminate_process(process)
    raise TimeoutError(f"Timed out waiting for port-forward {resource}.")


def _http_check(check_id: str, url: str) -> KubernetesCheckResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1":
        return KubernetesCheckResult(
            check_id=check_id,
            status="FAIL",
            message="Smoke HTTP probe rejected a non-local URL.",
            details={"url": url},
        )
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # nosec B310
            status_code = response.status
            body = response.read(512).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        return KubernetesCheckResult(
            check_id=check_id,
            status="FAIL",
            message=f"{url} request failed.",
            details={"error": str(exc)},
        )
    return KubernetesCheckResult(
        check_id=check_id,
        status="PASS" if 200 <= status_code < 300 else "FAIL",
        message=f"{url} returned HTTP {status_code}.",
        details={"status_code": status_code, "body": body},
    )


def _component_pod_name(component: str) -> str:
    pods = _get_json(
        [
            "kubectl",
            "--context",
            KIND_CONTEXT,
            "-n",
            NAMESPACE,
            "get",
            "pods",
            "-l",
            f"app.kubernetes.io/component={component}",
            "-o",
            "json",
        ]
    )
    items = pods.get("items", [])
    if not items:
        raise ValueError(f"No pod found for component {component}.")
    return str(items[0]["metadata"]["name"])


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _confirm_cleanup(cluster_created: bool) -> list[KubernetesCheckResult]:
    checks: list[KubernetesCheckResult] = []
    if cluster_created:
        clusters = _run_command(["kind", "get", "clusters"], timeout_seconds=60)
        cluster_absent = KIND_CLUSTER_NAME not in clusters.stdout.splitlines()
        checks.append(
            KubernetesCheckResult(
                check_id="K8S-CLEAN-CONFIRM-CLUSTER-ABSENT",
                status="PASS" if cluster_absent else "FAIL",
                message="Workflow-created kind cluster is absent.",
                details=_command_details(clusters),
            )
        )
        return checks
    namespace = _run_command(
        ["kubectl", "--context", KIND_CONTEXT, "get", "namespace", NAMESPACE],
        timeout_seconds=60,
    )
    checks.append(
        KubernetesCheckResult(
            check_id="K8S-CLEAN-CONFIRM-NAMESPACE-ABSENT",
            status="PASS" if namespace.returncode != 0 else "FAIL",
            message="Dedicated namespace is absent.",
            details=_command_details(namespace),
        )
    )
    return checks


def _render_objects(values: dict[str, Any]) -> list[dict[str, Any]]:
    fullname = RELEASE_NAME
    labels = APP_LABELS
    service_account = values["global"]["serviceAccount"]["name"] or fullname
    objects: list[dict[str, Any]] = [
        {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {"name": service_account, "labels": labels},
            "automountServiceAccountToken": False,
        },
        {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {"name": f"{fullname}-config", "labels": labels},
            "data": {
                "API_CONFIG_PATH": values["api"]["config"]["path"],
                "PLATFORM_ENVIRONMENT": values["api"]["config"]["environment"],
                "PLATFORM_LOG_LEVEL": values["api"]["config"]["logLevel"],
                "PLATFORM_SYNTHETIC_ONLY": str(values["global"]["syntheticOnly"]).lower(),
                "REVIEWER_API_BASE_URL": values["reviewerUi"]["config"]["apiBaseUrl"],
                "REVIEWER_ENVIRONMENT": values["reviewerUi"]["config"]["environment"],
                "REVIEWER_LOG_LEVEL": values["reviewerUi"]["config"]["logLevel"],
            },
        },
    ]
    objects.extend(
        [
            _deployment("api", values["api"], values, f"{fullname}-api", service_account),
            _deployment(
                "reviewer-ui",
                values["reviewerUi"],
                values,
                f"{fullname}-reviewer-ui",
                service_account,
            ),
            _service("api", values["api"], f"{fullname}-api"),
            _service("reviewer-ui", values["reviewerUi"], f"{fullname}-reviewer-ui"),
            _hpa("api", values["api"], f"{fullname}-api"),
            _hpa("reviewer-ui", values["reviewerUi"], f"{fullname}-reviewer-ui"),
            _pdb("api", values["api"], f"{fullname}-api"),
            _pdb("reviewer-ui", values["reviewerUi"], f"{fullname}-reviewer-ui"),
        ]
    )
    if values["networkPolicy"]["enabled"]:
        objects.extend(_network_policies(values, fullname))
    return objects


def _deployment(
    component: str, cfg: dict[str, Any], values: dict[str, Any], name: str, service_account: str
) -> dict[str, Any]:
    image_tag = cfg["image"].get("tag") or values["global"]["imageTag"]
    image_digest = cfg["image"].get("digest") or values["global"].get("imageDigest")
    image = (
        f"{cfg['image']['repository']}@{image_digest}"
        if image_digest
        else f"{cfg['image']['repository']}:{image_tag}"
    )
    probes = cfg["probes"]
    container: dict[str, Any] = {
        "name": component,
        "image": image,
        "imagePullPolicy": cfg["image"].get("pullPolicy") or values["global"]["imagePullPolicy"],
        "ports": [{"name": "http", "containerPort": cfg["service"]["port"]}],
        "envFrom": [{"configMapRef": {"name": f"{RELEASE_NAME}-config"}}]
        + [{"secretRef": {"name": name}} for name in cfg.get("secretRefs", [])],
        "securityContext": values["containerSecurityContext"],
        "resources": cfg["resources"],
        "livenessProbe": _http_probe(probes["liveness"]),
        "readinessProbe": _http_probe(probes["readiness"]),
        "volumeMounts": [
            {"name": mount["name"], "mountPath": mount["mountPath"]}
            for mount in cfg["writableMounts"]
        ],
    }
    if "startup" in probes:
        container["startupProbe"] = _http_probe(probes["startup"])
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "labels": {**APP_LABELS, "app.kubernetes.io/component": component},
        },
        "spec": {
            "replicas": cfg["replicaCount"],
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/instance": RELEASE_NAME,
                    "app.kubernetes.io/component": component,
                }
            },
            "template": {
                "metadata": {"labels": {**APP_LABELS, "app.kubernetes.io/component": component}},
                "spec": {
                    "serviceAccountName": service_account,
                    "automountServiceAccountToken": False,
                    "securityContext": values["podSecurityContext"],
                    "terminationGracePeriodSeconds": values["terminationGracePeriodSeconds"],
                    "containers": [container],
                    "volumes": [
                        {"name": mount["name"], "emptyDir": {"sizeLimit": mount["sizeLimit"]}}
                        for mount in cfg["writableMounts"]
                    ],
                },
            },
        },
    }


def _http_probe(probe: dict[str, Any]) -> dict[str, Any]:
    return {
        "httpGet": {"path": probe["path"], "port": "http"},
        **{key: value for key, value in probe.items() if key != "path"},
    }


def _service(component: str, cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": name,
            "labels": {**APP_LABELS, "app.kubernetes.io/component": component},
        },
        "spec": {
            "type": cfg["service"]["type"],
            "selector": {
                "app.kubernetes.io/instance": RELEASE_NAME,
                "app.kubernetes.io/component": component,
            },
            "ports": [{"name": "http", "port": cfg["service"]["port"], "targetPort": "http"}],
        },
    }


def _hpa(component: str, cfg: dict[str, Any], name: str) -> dict[str, Any]:
    metrics = [
        {
            "type": "Resource",
            "resource": {
                "name": "cpu",
                "target": {
                    "type": "Utilization",
                    "averageUtilization": cfg["autoscaling"]["targetCPUUtilizationPercentage"],
                },
            },
        }
    ]
    memory_target = cfg["autoscaling"].get("targetMemoryUtilizationPercentage")
    if memory_target is not None:
        metrics.append(
            {
                "type": "Resource",
                "resource": {
                    "name": "memory",
                    "target": {"type": "Utilization", "averageUtilization": memory_target},
                },
            }
        )
    return {
        "apiVersion": "autoscaling/v2",
        "kind": "HorizontalPodAutoscaler",
        "metadata": {
            "name": name,
            "labels": {**APP_LABELS, "app.kubernetes.io/component": component},
        },
        "spec": {
            "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": name},
            "minReplicas": cfg["autoscaling"]["minReplicas"],
            "maxReplicas": cfg["autoscaling"]["maxReplicas"],
            "metrics": metrics,
        },
    }


def _pdb(component: str, cfg: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "apiVersion": "policy/v1",
        "kind": "PodDisruptionBudget",
        "metadata": {
            "name": name,
            "labels": {**APP_LABELS, "app.kubernetes.io/component": component},
        },
        "spec": {
            "minAvailable": cfg["podDisruptionBudget"]["minAvailable"],
            "selector": {
                "matchLabels": {
                    "app.kubernetes.io/instance": RELEASE_NAME,
                    "app.kubernetes.io/component": component,
                }
            },
        },
    }


def _network_policies(values: dict[str, Any], fullname: str) -> list[dict[str, Any]]:
    return [
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": f"{fullname}-default-deny", "labels": APP_LABELS},
            "spec": {"podSelector": {}, "policyTypes": ["Ingress", "Egress"]},
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": f"{fullname}-reviewer-to-api", "labels": APP_LABELS},
            "spec": {
                "podSelector": {"matchLabels": {"app.kubernetes.io/component": "api"}},
                "policyTypes": ["Ingress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "podSelector": {
                                    "matchLabels": {"app.kubernetes.io/component": "reviewer-ui"}
                                }
                            }
                        ],
                        "ports": [{"protocol": "TCP", "port": values["api"]["service"]["port"]}],
                    }
                ],
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": f"{fullname}-reviewer-egress-api", "labels": APP_LABELS},
            "spec": {
                "podSelector": {"matchLabels": {"app.kubernetes.io/component": "reviewer-ui"}},
                "policyTypes": ["Egress"],
                "egress": [
                    {
                        "to": [
                            {"podSelector": {"matchLabels": {"app.kubernetes.io/component": "api"}}}
                        ],
                        "ports": [{"protocol": "TCP", "port": values["api"]["service"]["port"]}],
                    }
                ],
            },
        },
        {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {"name": f"{fullname}-dns-egress", "labels": APP_LABELS},
            "spec": {
                "podSelector": {},
                "policyTypes": ["Egress"],
                "egress": [
                    {
                        "to": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                                }
                            }
                        ],
                        "ports": [{"protocol": "UDP", "port": 53}, {"protocol": "TCP", "port": 53}],
                    }
                ],
            },
        },
    ]


def _check(
    check_id: str, passed: bool, message: str, details: dict[str, Any] | None = None
) -> KubernetesCheckResult:
    return KubernetesCheckResult(
        check_id=check_id,
        status="PASS" if passed else "FAIL",
        message=message,
        details=details or {},
    )


def _has_resources(resources: dict[str, Any]) -> bool:
    return all(
        resources.get(section, {}).get(resource)
        for section in ("requests", "limits")
        for resource in ("cpu", "memory")
    )


def _all_pass(checks: list[KubernetesCheckResult], needle: str) -> bool:
    selected = [check for check in checks if needle in check.check_id]
    return bool(selected) and all(check.status == "PASS" for check in selected)


def _available_runtime() -> str | None:
    if shutil.which("kind") is not None:
        return "kind"
    if shutil.which("minikube") is not None:
        return "minikube"
    return None


def _load_smoke_result() -> KubernetesSmokeResult:
    if SMOKE_RESULT_PATH.exists():
        return KubernetesSmokeResult.model_validate_json(
            SMOKE_RESULT_PATH.read_text(encoding="utf-8")
        )
    return KubernetesSmokeResult(
        status="UNAVAILABLE",
        executed=False,
        cleanup_status="UNAVAILABLE",
        message="Local Kubernetes smoke has not been executed.",
        steps=[
            KubernetesCheckResult(
                check_id="K8S-SMOKE-NOT-RUN",
                status="UNAVAILABLE",
                message="Run `make kubernetes-smoke` with kind or minikube available.",
            )
        ],
    )


def _aggregate_status(
    helm_lint: KubernetesCheckResult,
    schema_validation: KubernetesCheckResult,
    policy_validation: list[KubernetesCheckResult],
    smoke_result: KubernetesSmokeResult,
) -> KubernetesStatus:
    mandatory_statuses = [schema_validation.status, *(check.status for check in policy_validation)]
    if helm_lint.status in {"FAIL", "ERROR"}:
        mandatory_statuses.append(helm_lint.status)
    if smoke_result.status == "PASS":
        mandatory_statuses.append(smoke_result.status)
    else:
        mandatory_statuses.append("INCOMPLETE")
    if any(status in {"FAIL", "ERROR"} for status in mandatory_statuses):
        return "FAIL"
    if any(status in {"INCOMPLETE", "UNAVAILABLE"} for status in mandatory_statuses):
        return "INCOMPLETE"
    return "PASS"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(manifest: KubernetesEvidenceManifest) -> None:
    policy_pass = sum(1 for check in manifest.policy_validation if check.status == "PASS")
    policy_total = len(manifest.policy_validation)
    EVIDENCE_REPORT_PATH.write_text(
        "\n".join(
            [
                "# Kubernetes Evidence Report",
                "",
                KUBERNETES_DISCLAIMER,
                "",
                f"- Evidence ID: `{manifest.evidence_id}`",
                f"- Overall status: `{manifest.overall_status}`",
                f"- Helm lint: `{manifest.helm_lint.status}`",
                f"- Schema validation: `{manifest.schema_validation.status}`",
                f"- Policy validation: `{policy_pass}/{policy_total}` checks PASS",
                (
                    f"- Runtime smoke: `{manifest.smoke_result.status}` "
                    f"executed=`{manifest.smoke_result.executed}`"
                ),
                "",
                "Runtime deployment evidence is required before overall status can become PASS.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
