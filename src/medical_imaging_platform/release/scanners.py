"""Optional scanner wrappers for local release assurance."""

from __future__ import annotations

import json
import shutil
import subprocess  # nosec B404 - scanner wrappers use shell=False and bounded timeouts.
import sys
from pathlib import Path
from typing import Any

from medical_imaging_platform.release.models import Status, ToolResult

SECRET_KEYS = ("password", "token", "secret", "apikey", "api_key", "authorization")


def sanitize_output(output: str, limit: int = 4000) -> str:
    """Redact likely secret-bearing lines and bound command output."""
    safe_lines = []
    for line in output.splitlines():
        lowered = line.lower()
        if any(key in lowered for key in SECRET_KEYS):
            safe_lines.append("[REDACTED]")
        else:
            safe_lines.append(line)
    return "\n".join(safe_lines)[:limit]


def tool_version(tool: str, timeout_seconds: int = 15) -> ToolResult:
    """Return scanner version or unavailable status."""
    if shutil.which(tool) is None:
        return ToolResult(
            tool=tool,
            available=False,
            status="UNAVAILABLE",
            output=f"{tool} is not installed.",
        )
    command = [tool, "--version"]
    completed = _run(command, timeout_seconds)
    return ToolResult(
        tool=tool,
        available=completed.returncode == 0,
        status="PASS" if completed.returncode == 0 else "FAIL",
        version=sanitize_output(completed.stdout or completed.stderr).splitlines()[0]
        if (completed.stdout or completed.stderr).strip()
        else None,
        command=command,
        output=sanitize_output(completed.stdout + completed.stderr),
    )


def run_tool(
    command: list[str], tool: str, timeout_seconds: int, *, tool_label: str | None = None
) -> ToolResult:
    """Run a bounded scanner command without shell expansion."""
    if shutil.which(tool) is None:
        return ToolResult(
            tool=tool_label or tool,
            available=False,
            status="UNAVAILABLE",
            command=command,
            output=f"{tool_label or tool} is not installed.",
        )
    completed = _run(command, timeout_seconds)
    output = sanitize_output(completed.stdout + completed.stderr)
    status = _scanner_status(completed.returncode, completed.stdout, output)
    return ToolResult(
        tool=tool_label or tool,
        available=True,
        status=status,
        command=command,
        output=output,
        findings=parse_json_findings(completed.stdout),
    )


def parse_json_findings(output: str) -> list[dict[str, Any]]:
    """Best-effort JSON scanner finding parsing."""
    if not output.strip():
        return []
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        for key in ("Results", "vulnerabilities", "findings", "dependencies"):
            value = parsed.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [parsed]
    return []


def severity_gate(findings: list[dict[str, Any]], fail_severities: list[str]) -> bool:
    """Return True when no finding meets configured fail severities."""
    fail = {severity.upper() for severity in fail_severities}
    for finding in findings:
        severity = str(finding.get("Severity", finding.get("severity", ""))).upper()
        if severity in fail:
            return False
    return True


def scanner_inventory(timeout_seconds: int = 15) -> list[ToolResult]:
    """Record versions or unavailable status for preferred scanner tools."""
    tools = [tool_version(tool, timeout_seconds) for tool in ("hadolint", "gitleaks")]
    pip_audit_command = _pip_audit_command()
    pip_audit_version_command = (
        [sys.executable, "-m", "pip_audit", "--version"]
        if pip_audit_command[0] == sys.executable
        else ["pip-audit", "--version"]
    )
    pip_audit_version = run_tool(
        pip_audit_version_command,
        pip_audit_command[0],
        timeout_seconds,
        tool_label="pip-audit",
    )
    tools.append(pip_audit_version)
    tools.extend(tool_version(tool, timeout_seconds) for tool in ("trivy", "syft", "docker"))
    return tools


def scan_repository_secrets(timeout_seconds: int) -> ToolResult:
    return run_tool(
        [
            "gitleaks",
            "detect",
            "--no-git",
            "--redact",
            "--config",
            ".gitleaks.toml",
            "--source",
            ".",
        ],
        "gitleaks",
        timeout_seconds,
    )


def scan_dependencies(timeout_seconds: int, requirements_path: Path | None = None) -> ToolResult:
    command = _pip_audit_command()
    if requirements_path is not None:
        command = [*command, "--requirement", requirements_path.as_posix()]
    return run_tool(command, command[0], timeout_seconds, tool_label="pip-audit")


def lint_dockerfiles(timeout_seconds: int) -> list[ToolResult]:
    return [
        run_tool(["hadolint", "docker/api/Dockerfile"], "hadolint", timeout_seconds),
        run_tool(["hadolint", "docker/reviewer-ui/Dockerfile"], "hadolint", timeout_seconds),
    ]


def generate_context_sbom(image_name: str, output_path: Path, timeout_seconds: int) -> ToolResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return run_tool(
        [
            "syft",
            f"{image_name}:0.1.0-m13-local" if ":" not in image_name else image_name,
            "-o",
            f"cyclonedx-json={output_path.as_posix()}",
        ],
        "syft",
        timeout_seconds,
    )


def scan_image(image_ref: str, timeout_seconds: int) -> ToolResult:
    return run_tool(
        [
            "trivy",
            "image",
            "--format",
            "json",
            "--cache-dir",
            ".trivy-cache",
            image_ref,
        ],
        "trivy",
        timeout_seconds,
    )


def _pip_audit_command() -> list[str]:
    if shutil.which("pip-audit") is not None:
        return ["pip-audit", "--progress-spinner", "off", "--format", "json"]
    return [sys.executable, "-m", "pip_audit", "--progress-spinner", "off", "--format", "json"]


def _run(command: list[str], timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(  # nosec B603 - command is constructed from fixed scanner args.
            command, capture_output=True, text=True, timeout=timeout_seconds, check=False
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        return subprocess.CompletedProcess(
            command,
            124,
            stdout or "",
            stderr or "Command timed out.",
        )


def _scanner_status(returncode: int, stdout: str, output: str) -> Status:
    if returncode == 0:
        return "PASS"
    if returncode == 124:
        return "ERROR"
    if not stdout.strip():
        return "ERROR"
    lowered = output.lower()
    if "failed to download" in lowered or "database" in lowered or "db" in lowered:
        return "ERROR"
    return "FAIL"
