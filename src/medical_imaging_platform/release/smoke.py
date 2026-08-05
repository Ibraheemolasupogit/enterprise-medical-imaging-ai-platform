"""Local Docker Compose smoke-test orchestration."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - Docker smoke commands use shell=False and fixed args.
from dataclasses import dataclass

from medical_imaging_platform.release.models import (
    ContainerReleaseConfig,
    ReleaseCheckResult,
    SmokeTestResult,
    Status,
)
from medical_imaging_platform.release.scanners import sanitize_output


@dataclass(frozen=True)
class SmokeCommandResult:
    """Bounded smoke command output for evidence and CLI diagnostics."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return sanitize_output(f"{self.stdout}{self.stderr}")


def docker_available() -> bool:
    return shutil.which("docker") is not None


def run_container_smoke_tests(
    config: ContainerReleaseConfig, execute: bool = True
) -> SmokeTestResult:
    """Run bounded local Compose smoke tests when Docker is available."""
    if not docker_available():
        return SmokeTestResult(
            status="UNAVAILABLE",
            executed=False,
            steps=[
                ReleaseCheckResult(
                    check_id="SMOKE-DOCKER-AVAILABLE",
                    status="UNAVAILABLE",
                    message="docker is not installed.",
                )
            ],
        )
    if not execute:
        return SmokeTestResult(
            status="SKIPPED",
            executed=False,
            steps=[
                ReleaseCheckResult(
                    check_id="SMOKE-EXECUTION",
                    status="SKIPPED",
                    message="Smoke-test execution was disabled.",
                )
            ],
        )
    commands = _smoke_commands(config)
    steps: list[ReleaseCheckResult] = []
    overall: Status = "PASS"
    cleanup_ran = False
    try:
        for check_id, command, degraded_ok in commands:
            command_result = _run_command(command, config.smoke_timeout_seconds)
            passed = command_result.returncode == 0
            degraded = degraded_ok and command_result.returncode == 10
            status: Status = "PASS" if passed else "WARN" if degraded else "FAIL"
            steps.append(
                ReleaseCheckResult(
                    check_id=check_id,
                    status=status,
                    message=f"{' '.join(command[:3])} returned {command_result.returncode}.",
                    details={
                        "command": command,
                        "exit_code": command_result.returncode,
                        "stderr": command_result.stderr,
                        "output": command_result.output,
                        "expected_degraded_readiness": degraded,
                    },
                )
            )
            if status == "WARN" and overall == "PASS":
                overall = "INCOMPLETE"
            if status == "FAIL":
                overall = "FAIL"
                break
    finally:
        cleanup_ran = True
        cleanup_result = _run_command(
            ["docker", "compose", "down", "--volumes", "--remove-orphans"], 60
        )
        steps.append(
            ReleaseCheckResult(
                check_id="SMOKE-COMPOSE-CLEANUP",
                status="PASS" if cleanup_result.returncode == 0 else "FAIL",
                message=f"docker compose down returned {cleanup_result.returncode}.",
                details={
                    "command": ["docker", "compose", "down", "--volumes", "--remove-orphans"],
                    "exit_code": cleanup_result.returncode,
                    "stderr": cleanup_result.stderr,
                    "output": cleanup_result.output,
                    "cleanup_executed": cleanup_ran,
                },
            )
        )
        if cleanup_result.returncode != 0:
            overall = "FAIL"
    return SmokeTestResult(status=overall, executed=True, steps=steps)


def _smoke_commands(
    config: ContainerReleaseConfig,
) -> list[tuple[str, list[str], bool]]:
    uid_gid_check = (
        "import os,sys; "
        f"expected='{config.required_uid}:{config.required_gid}'; "
        "actual=f'{os.getuid()}:{os.getgid()}'; "
        "sys.exit(0 if actual == expected else 1)"
    )
    return [
        ("SMOKE-COMPOSE-CONFIG", ["docker", "compose", "config"], False),
        ("SMOKE-COMPOSE-BUILD", ["docker", "compose", "build", "--pull=false"], False),
        ("SMOKE-COMPOSE-UP", ["docker", "compose", "up", "-d"], False),
        ("SMOKE-COMPOSE-PS", ["docker", "compose", "ps"], False),
        ("SMOKE-API-HEALTH", _http_exec("api", "http://127.0.0.1:8000/health"), False),
        ("SMOKE-API-READY", _http_exec("api", "http://127.0.0.1:8000/ready", True), True),
        ("SMOKE-API-VERSION", _http_exec("api", "http://127.0.0.1:8000/version"), False),
        (
            "SMOKE-API-UID-GID",
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "api",
                "python",
                "-c",
                uid_gid_check,
            ],
            False,
        ),
        (
            "SMOKE-REVIEWER-UID-GID",
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "reviewer-ui",
                "python",
                "-c",
                uid_gid_check,
            ],
            False,
        ),
        ("SMOKE-API-FILESYSTEM", _filesystem_exec("api"), False),
        ("SMOKE-REVIEWER-FILESYSTEM", _filesystem_exec("reviewer-ui"), False),
        (
            "SMOKE-REVIEWER-HEALTH",
            _http_exec("reviewer-ui", "http://127.0.0.1:8501/_stcore/health", attempts=30),
            False,
        ),
        (
            "SMOKE-REVIEWER-API-REACHABLE",
            _http_exec("reviewer-ui", "http://api:8000/health", attempts=30),
            False,
        ),
        ("SMOKE-COMPOSE-LOGS", ["docker", "compose", "logs", "--no-color"], False),
    ]


def _http_exec(
    service: str, url: str, allow_degraded: bool = False, attempts: int = 1
) -> list[str]:
    degraded_clause = (
        "except urllib.error.HTTPError as exc:\n"
        "    body=exc.read().decode('utf-8','replace')\n"
        "    print(body)\n"
        "    sys.exit(10 if exc.code == 503 else 1)\n"
        if allow_degraded
        else ""
    )
    code = (
        "import sys, time, urllib.error, urllib.request\n"
        f"last_error=''\n"
        f"for attempt in range({attempts}):\n"
        "    try:\n"
        f"        body=urllib.request.urlopen('{url}', timeout=5).read()\n"
        "        print(body.decode('utf-8','replace'))\n"
        "        sys.exit(0)\n"
        f"{_indent_retry_clause(degraded_clause)}"
        "    except Exception as exc:\n"
        "        last_error=str(exc)\n"
        f"        time.sleep(1 if attempt < {attempts} - 1 else 0)\n"
        "print(last_error)\n"
        "sys.exit(1)\n"
    )
    return ["docker", "compose", "exec", "-T", service, "python", "-c", code]


def _indent_retry_clause(code: str) -> str:
    return "".join(f"    {line}\n" if line else "\n" for line in code.splitlines())


def _filesystem_exec(service: str) -> list[str]:
    code = (
        "from pathlib import Path\n"
        "import sys\n"
        "out=Path('/app/outputs/.smoke-write')\n"
        "out.write_text('ok', encoding='utf-8')\n"
        "out.unlink()\n"
        "try:\n"
        "    Path('/app/.smoke-root').write_text('bad', encoding='utf-8')\n"
        "except OSError:\n"
        "    sys.exit(0)\n"
        "sys.exit(1)\n"
    )
    return ["docker", "compose", "exec", "-T", service, "python", "-c", code]


def _run_command(command: list[str], timeout_seconds: int) -> SmokeCommandResult:
    try:
        completed = subprocess.run(  # nosec B603 - command list comes from fixed smoke steps.
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        return SmokeCommandResult(
            completed.returncode,
            sanitize_output(completed.stdout),
            sanitize_output(completed.stderr),
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
        return SmokeCommandResult(
            124,
            sanitize_output(stdout or ""),
            sanitize_output(stderr or "Command timed out."),
        )
