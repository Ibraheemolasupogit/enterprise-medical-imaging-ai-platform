import json
import subprocess
from pathlib import Path

import medical_imaging_platform.aws.assurance as aws_assurance
from medical_imaging_platform.aws.assurance import (
    TERRAFORM_ROOT,
    CommandResult,
    _aggregate_status,
    _no_broad_iam,
    _no_hardcoded_secrets,
    _no_unrestricted_admin_ingress,
    _optional_scan,
    _terraform_command,
    aws_plan,
    build_architecture_manifest,
    build_aws_evidence,
    build_cost_driver_report,
    build_encryption_summary,
    build_iam_permission_summary,
    build_networking_summary,
    scan_terraform,
    terraform_fmt_check,
    validate_aws_evidence,
    validate_aws_policy,
)
from medical_imaging_platform.aws.models import AwsCheckResult
from medical_imaging_platform.cli import main


def _redirect_aws_evidence(monkeypatch, tmp_path: Path) -> Path:  # type: ignore[no-untyped-def]
    evidence_dir = tmp_path / "aws-evidence"
    path_names = [
        "TERRAFORM_FMT_PATH",
        "TERRAFORM_INIT_PATH",
        "TERRAFORM_VALIDATE_PATH",
        "TFLINT_PATH",
        "CHECKOV_PATH",
        "TRIVY_CONFIG_PATH",
        "POLICY_RESULT_PATH",
        "RESOURCE_INVENTORY_PATH",
        "IAM_SUMMARY_PATH",
        "NETWORKING_SUMMARY_PATH",
        "ENCRYPTION_SUMMARY_PATH",
        "COST_REPORT_PATH",
        "ARCHITECTURE_MANIFEST_PATH",
        "EVIDENCE_MANIFEST_PATH",
        "EVIDENCE_REPORT_PATH",
        "CHECKSUMS_PATH",
        "AWS_PLAN_PATH",
    ]
    monkeypatch.setattr(aws_assurance, "AWS_EVIDENCE_DIR", evidence_dir)
    for name in path_names:
        current = getattr(aws_assurance, name)
        monkeypatch.setattr(aws_assurance, name, evidence_dir / current.name)
    return evidence_dir


def _all_tf_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in TERRAFORM_ROOT.rglob("*.tf"))


def test_terraform_structure_and_module_composition() -> None:
    required = [
        TERRAFORM_ROOT / "versions.tf",
        TERRAFORM_ROOT / "providers.tf",
        TERRAFORM_ROOT / "variables.tf",
        TERRAFORM_ROOT / "outputs.tf",
        TERRAFORM_ROOT / "environments/dev/main.tf",
        TERRAFORM_ROOT / "environments/dev/backend.tf.example",
    ]
    modules = {"networking", "security", "ecr", "storage", "eks", "observability"}

    assert all(path.exists() for path in required)
    assert modules == {
        path.name for path in (TERRAFORM_ROOT / "modules").iterdir() if path.is_dir()
    }
    root = (TERRAFORM_ROOT / "main.tf").read_text(encoding="utf-8")
    assert all(f'module "{module}"' in root for module in modules)


def test_pinned_versions_and_no_real_backend() -> None:
    versions = (TERRAFORM_ROOT / "versions.tf").read_text(encoding="utf-8")
    active_backend_files = [
        path
        for path in TERRAFORM_ROOT.rglob("*.tf")
        if "backend.tf.example" not in path.as_posix()
        and 'backend "' in path.read_text(encoding="utf-8")
    ]

    assert 'required_version = ">= 1.9.0, < 1.16.0"' in versions
    assert 'version = "~> 5.84.0"' in versions
    assert 'version = "~> 4.0.6"' in versions
    assert active_backend_files == []


def test_private_networking_defaults_and_no_public_eks_endpoint() -> None:
    text = _all_tf_text()
    dev = (TERRAFORM_ROOT / "environments/dev/main.tf").read_text(encoding="utf-8")

    assert "enable_public_ingress_subnets = false" in dev
    assert "enable_nat_gateway            = false" in dev
    assert "map_public_ip_on_launch = false" in text
    assert "endpoint_private_access = true" in text
    assert "endpoint_public_access  = false" in text
    assert '"0.0.0.0/0"' not in text or "Kubernetes API from private workload CIDR only" in text


def test_cpu_only_eks_node_group() -> None:
    eks = (TERRAFORM_ROOT / "modules/eks/main.tf").read_text(encoding="utf-8")

    assert 'ami_type       = "AL2023_x86_64_STANDARD"' in eks
    assert 'gpu      = "false"' in eks
    assert 'Accelerator = "none"' in eks
    assert "p3." not in eks and "p4." not in eks and "g5" not in eks


def test_ecr_immutable_scan_on_push_and_encryption() -> None:
    ecr = (TERRAFORM_ROOT / "modules/ecr/main.tf").read_text(encoding="utf-8")

    assert 'image_tag_mutability = "IMMUTABLE"' in ecr
    assert "scan_on_push = true" in ecr
    assert 'encryption_type = "KMS"' in ecr
    assert "medical-imaging-api" in ecr
    assert "medical-imaging-reviewer-ui" in ecr


def test_s3_public_access_encryption_tls_and_versioning() -> None:
    storage = (TERRAFORM_ROOT / "modules/storage/main.tf").read_text(encoding="utf-8")

    assert "aws_s3_bucket_public_access_block" in storage
    assert "block_public_acls       = true" in storage
    assert "restrict_public_buckets = true" in storage
    assert "aws_s3_bucket_server_side_encryption_configuration" in storage
    assert 'sse_algorithm     = "aws:kms"' in storage
    assert "DenyInsecureTransport" in storage
    assert "aws_s3_bucket_versioning" in storage


def test_kms_rotation_and_secret_placeholders_only() -> None:
    security = (TERRAFORM_ROOT / "modules/security/main.tf").read_text(encoding="utf-8")
    variables = (TERRAFORM_ROOT / "variables.tf").read_text(encoding="utf-8")

    assert security.count("enable_key_rotation     = true") >= 4
    assert "secretsmanager:GetSecretValue" in security
    assert "secrets_manager_secret_arns" in variables
    assert "secret_string" not in security


def test_iam_workload_identity_boundaries() -> None:
    summary = build_iam_permission_summary()

    assert summary["reviewer_ui"] == "No direct broad S3 policy is defined."
    assert "s3:GetObject" in summary["api_checkpoint_read"]
    assert "s3:PutObject" in summary["monitoring_evidence_write"]
    assert summary["long_lived_access_keys"] is False


def test_cloudwatch_retention_alarms_and_cloudtrail_separation() -> None:
    observability = (TERRAFORM_ROOT / "modules/observability/main.tf").read_text(encoding="utf-8")

    assert "retention_in_days = var.log_retention_days" in observability
    assert "api_error_rate" in observability
    assert "readiness_failures" in observability
    assert "pod_restart_rate" in observability
    assert "high_latency" in observability
    assert "node_pressure" in observability
    assert "aws_cloudtrail" in observability
    assert 'AuditScope = "aws-control-plane"' in observability


def test_cost_driver_classification_and_architecture_manifest() -> None:
    costs = build_cost_driver_report()
    architecture = build_architecture_manifest()
    networking = build_networking_summary()
    encryption = build_encryption_summary()

    assert {item["classification"] for item in costs} >= {
        "always-on cost",
        "usage-based cost",
        "avoidable in local validation",
        "production-only option",
    }
    assert architecture["compute"]["gpu_nodes"] is False
    assert "SageMaker" in architecture["excluded"]
    assert networking["public_inference_default"] is False
    assert encryption["kms_rotation"] is True


def test_policy_validation_passes_for_repository() -> None:
    checks = validate_aws_policy()

    assert all(check.status == "PASS" for check in checks)
    assert any(check.check_id == "AWS-IAM-NO-BROAD-ADMIN" for check in checks)


def test_policy_validation_rejects_wildcard_iam(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    bad_root = tmp_path / "terraform"
    bad_root.mkdir()
    (bad_root / "bad.tf").write_text(
        'resource "aws_iam_policy" "bad" {\n'
        '  name = "bad"\n'
        '  policy = jsonencode({ Statement = [{ Effect = "Allow", '
        'Action = ["*"], Resource = "*" }] })\n'
        '  tags = { ManagedBy = "terraform" }\n'
        "}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aws_assurance, "TERRAFORM_ROOT", bad_root)
    _redirect_aws_evidence(monkeypatch, tmp_path)

    checks = validate_aws_policy()

    assert any(
        check.check_id == "AWS-IAM-NO-BROAD-ADMIN" and check.status == "FAIL" for check in checks
    )


def test_unavailable_tool_handling_and_absent_aws_credentials(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_aws_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr("medical_imaging_platform.aws.assurance.shutil.which", lambda _: None)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)

    fmt = terraform_fmt_check()
    scans = scan_terraform()
    plan = aws_plan()

    assert fmt.status == "UNAVAILABLE"
    assert all(scan.status == "UNAVAILABLE" and scan.mandatory is False for scan in scans)
    assert plan.status == "INCOMPLETE"
    assert plan.details["apply_executed"] is False


def test_deterministic_evidence_and_no_false_pass(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    evidence_dir = _redirect_aws_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr("medical_imaging_platform.aws.assurance.shutil.which", lambda _: None)

    first = build_aws_evidence()
    second = build_aws_evidence()
    validation = validate_aws_evidence()

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.overall_status == "INCOMPLETE"
    assert all(check.status == "PASS" for check in validation)
    assert (
        json.loads((evidence_dir / "aws_evidence_manifest.json").read_text())["overall_status"]
        != "PASS"
    )


def test_cli_commands_for_aws_assurance(tmp_path: Path, monkeypatch, capsys) -> None:  # type: ignore[no-untyped-def]
    _redirect_aws_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr("medical_imaging_platform.aws.assurance.shutil.which", lambda _: None)

    assert main(["terraform-fmt-check"]) == 0
    assert "AWS-TERRAFORM-FMT status=UNAVAILABLE" in capsys.readouterr().out
    assert main(["validate-aws-policy"]) == 0
    assert "AWS policy validation status=PASS" in capsys.readouterr().out
    assert main(["scan-terraform"]) == 0
    assert "Terraform scanner status=INCOMPLETE" in capsys.readouterr().out
    assert main(["build-aws-evidence"]) == 0
    assert "Built AWS evidence status=INCOMPLETE" in capsys.readouterr().out
    assert main(["validate-aws-evidence"]) == 0
    assert "AWS evidence validation status=PASS" in capsys.readouterr().out
    assert main(["aws-plan"]) == 0
    assert "AWS-TERRAFORM-PLAN status=INCOMPLETE" in capsys.readouterr().out


def test_clean_terraform_removes_only_local_cache(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    root = tmp_path / "terraform"
    cache = root / "module" / ".terraform"
    cache.mkdir(parents=True)
    lock = root / ".terraform.lock.hcl"
    lock.write_text("provider lock", encoding="utf-8")
    keep = root / "main.tf"
    keep.write_text("# keep\n", encoding="utf-8")
    monkeypatch.setattr(aws_assurance, "TERRAFORM_ROOT", root)

    result = aws_assurance.clean_terraform()

    assert result.status == "PASS"
    assert not cache.exists()
    assert not lock.exists()
    assert keep.exists()
    assert len(result.details["removed"]) == 2


def test_terraform_command_success_failure_and_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance.shutil.which", lambda _: "/bin/tool"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 0, "ok", ""),
    )
    passed = _terraform_command("AWS-TEST", ["terraform", "validate"], "validate", output)

    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 1, "", "bad"),
    )
    failed = _terraform_command("AWS-TEST", ["terraform", "validate"], "validate", output)

    def raise_timeout(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        raise subprocess.TimeoutExpired(args, timeout_seconds)

    monkeypatch.setattr("medical_imaging_platform.aws.assurance._run_command", raise_timeout)
    errored = _terraform_command("AWS-TEST", ["terraform", "validate"], "validate", output)

    assert passed.status == "PASS"
    assert failed.status == "FAIL"
    assert errored.status == "ERROR"


def test_optional_scan_success_failure_error(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "scan.json"
    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance.shutil.which", lambda _: "/bin/tool"
    )
    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 0, "clean", ""),
    )
    passed = _optional_scan("AWS-SCAN", "scanner", ["scanner"], output, "scan")

    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance._run_command",
        lambda args, timeout_seconds=180: CommandResult(args, 2, "", "finding"),
    )
    failed = _optional_scan("AWS-SCAN", "scanner", ["scanner"], output, "scan")

    def raise_oserror(args: list[str], timeout_seconds: int = 180) -> CommandResult:
        raise OSError("boom")

    monkeypatch.setattr("medical_imaging_platform.aws.assurance._run_command", raise_oserror)
    errored = _optional_scan("AWS-SCAN", "scanner", ["scanner"], output, "scan")

    assert passed.status == "PASS"
    assert failed.status == "FAIL"
    assert errored.status == "ERROR"
    assert errored.mandatory is False


def test_aws_plan_with_credentials_uses_no_apply(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_aws_evidence(monkeypatch, tmp_path)
    monkeypatch.setattr("medical_imaging_platform.aws.assurance._has_aws_credentials", lambda: True)

    def fake_terraform_command(*args, **kwargs):  # type: ignore[no-untyped-def]
        return AwsCheckResult(
            check_id="AWS-TERRAFORM-PLAN",
            status="PASS",
            message="planned",
            mandatory=False,
            details={"args": args[1]},
        )

    monkeypatch.setattr(
        "medical_imaging_platform.aws.assurance._terraform_command", fake_terraform_command
    )

    result = aws_assurance.aws_plan()

    assert result.status == "PASS"
    assert result.details["apply_executed"] is False
    assert "apply" not in result.details["args"]


def test_evidence_validation_reports_missing_manifest(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _redirect_aws_evidence(monkeypatch, tmp_path)

    checks = validate_aws_evidence()

    assert any(check.status == "FAIL" for check in checks)
    assert all(check.check_id.startswith("AWS-EVIDENCE-") for check in checks)


def test_aggregate_status_and_negative_policy_helpers(tmp_path: Path) -> None:
    pass_check = AwsCheckResult(check_id="PASS", status="PASS", message="pass")
    fail_check = AwsCheckResult(check_id="FAIL", status="FAIL", message="fail")
    unavailable_check = AwsCheckResult(
        check_id="UNAVAILABLE", status="UNAVAILABLE", message="unavailable"
    )
    optional_fail = AwsCheckResult(
        check_id="OPTIONAL", status="FAIL", message="optional", mandatory=False
    )
    bad_sg = tmp_path / "bad-sg.tf"
    bad_sg.write_text(
        'resource "aws_security_group" "bad" {\n'
        "  ingress {\n"
        "    from_port   = 22\n"
        '    cidr_blocks = ["0.0.0.0/0"]\n'
        "  }\n"
        "}\n",
        encoding="utf-8",
    )
    bad_secret = tmp_path / "secret.tf"
    bad_secret.write_text('password = "not-a-real-password"\n', encoding="utf-8")
    broad = tmp_path / "broad.tf"
    broad.write_text('actions = ["iam:*"]\n', encoding="utf-8")

    assert _aggregate_status([pass_check], [pass_check], [optional_fail]) == "PASS"
    assert _aggregate_status([fail_check], [pass_check], []) == "FAIL"
    assert _aggregate_status([unavailable_check], [pass_check], []) == "INCOMPLETE"
    assert _no_unrestricted_admin_ingress([bad_sg]) is False
    assert _no_hardcoded_secrets([bad_secret]) is False
    assert _no_broad_iam([broad]) is False
