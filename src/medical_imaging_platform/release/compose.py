"""Docker Compose static assurance checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from medical_imaging_platform.release.models import ContainerReleaseConfig, ReleaseCheckResult


def load_compose(path: Path = Path("docker-compose.yml")) -> dict[str, Any]:
    """Load docker-compose.yml as a mapping."""
    parsed = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("Compose file must contain a mapping.")
    return parsed


def inspect_compose(
    config: ContainerReleaseConfig, path: Path = Path("docker-compose.yml")
) -> list[ReleaseCheckResult]:
    """Inspect Compose security controls."""
    if not path.exists():
        return [_result("COMPOSE-EXISTS", False, "docker-compose.yml exists.")]
    try:
        compose = load_compose(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [_result("COMPOSE-PARSE", False, f"Compose parse failed: {exc}")]

    services = compose.get("services", {})
    if not isinstance(services, dict):
        return [_result("COMPOSE-SERVICES", False, "Compose services mapping is present.")]
    checks = [
        _result(
            "COMPOSE-SERVICES",
            {"api", "reviewer-ui"}.issubset(services),
            "API and reviewer UI services are defined.",
        )
    ]
    for name in ("api", "reviewer-ui"):
        service = services.get(name)
        if not isinstance(service, dict):
            checks.append(
                _result(f"COMPOSE-{name}-SERVICE", False, f"{name} service is a mapping.")
            )
            continue
        checks.extend(_service_checks(name, service, config))
    networks = compose.get("networks", {})
    checks.append(
        _result(
            "COMPOSE-NETWORK",
            isinstance(networks, dict) and "medical-imaging-local" in networks,
            "Named internal Compose network is defined.",
        )
    )
    return checks


def _service_checks(
    name: str, service: dict[str, Any], config: ContainerReleaseConfig
) -> list[ReleaseCheckResult]:
    volumes = service.get("volumes", [])
    volume_text = repr(volumes)
    port_text = repr(service.get("ports", []))
    expected_user = f"{config.required_uid}:{config.required_gid}"
    checks = [
        _result(
            f"COMPOSE-{name}-NO-PRIVILEGED",
            service.get("privileged") is not True,
            f"{name} is not privileged.",
        ),
        _result(
            f"COMPOSE-{name}-READONLY",
            service.get("read_only") is True,
            f"{name} root filesystem is read-only.",
        ),
        _result(
            f"COMPOSE-{name}-CAPDROP",
            service.get("cap_drop") == ["ALL"],
            f"{name} drops all Linux capabilities.",
        ),
        _result(
            f"COMPOSE-{name}-NO-NEW-PRIVS",
            "no-new-privileges:true" in service.get("security_opt", []),
            f"{name} sets no-new-privileges.",
        ),
        _result(
            f"COMPOSE-{name}-USER",
            service.get("user") == expected_user,
            f"{name} runs as non-root UID/GID.",
        ),
        _result(
            f"COMPOSE-{name}-TMPFS",
            bool(service.get("tmpfs")),
            f"{name} has tmpfs for temporary paths.",
        ),
        _result(
            f"COMPOSE-{name}-HEALTHCHECK",
            isinstance(service.get("healthcheck"), dict),
            f"{name} has a bounded health check.",
        ),
        _result(
            f"COMPOSE-{name}-LOCAL-PORT",
            "127.0.0.1:" in port_text,
            f"{name} publishes local-only ports.",
        ),
        _result(
            f"COMPOSE-{name}-NO-HOST-NET",
            service.get("network_mode") != "host",
            f"{name} does not use host networking.",
        ),
        _result(
            f"COMPOSE-{name}-NO-DOCKER-SOCKET",
            "/var/run/docker.sock" not in volume_text,
            f"{name} does not mount the Docker socket.",
        ),
        _result(
            f"COMPOSE-{name}-OUTPUT-NAMED-VOLUME",
            _has_named_output_volume(volumes, config),
            f"{name} uses a named writable output volume.",
        ),
        _result(
            f"COMPOSE-{name}-RESOURCE-LIMITS",
            "mem_limit" in service and "cpus" in service,
            f"{name} has CPU and memory limits.",
        ),
    ]
    if name == "reviewer-ui":
        checks.append(
            _result(
                "COMPOSE-reviewer-ui-DEPENDS-HEALTH",
                "condition" in repr(service.get("depends_on", {})),
                "Reviewer UI depends on API health.",
            )
        )
        checks.append(
            _result(
                "COMPOSE-reviewer-ui-NO-CHECKPOINTS",
                config.checkpoint_mount.as_posix() not in volume_text,
                "Reviewer UI does not mount checkpoints.",
            )
        )
    return checks


def _has_named_output_volume(volumes: Any, config: ContainerReleaseConfig) -> bool:
    if not isinstance(volumes, list):
        return False
    for volume in volumes:
        if not isinstance(volume, dict):
            continue
        if volume.get("target") != config.output_mount.as_posix():
            continue
        return volume.get("type") == "volume" and bool(volume.get("source"))
    return False


def _result(check_id: str, passed: bool, message: str) -> ReleaseCheckResult:
    return ReleaseCheckResult(
        check_id=check_id, status="PASS" if passed else "FAIL", message=message
    )
