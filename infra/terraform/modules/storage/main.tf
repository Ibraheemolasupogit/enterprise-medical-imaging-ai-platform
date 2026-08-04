locals {
  buckets = {
    checkpoints = {
      suffix      = "model-checkpoints"
      lifecycle   = 365
      description = "approved model checkpoint objects only"
    }
    artifacts = {
      suffix      = "deidentified-artifacts"
      lifecycle   = 180
      description = "synthetic or publicly available de-identified artefacts"
    }
    evidence = {
      suffix      = "governed-evidence"
      lifecycle   = 365
      description = "monitoring, registry and application audit evidence"
    }
    audit_logs = {
      suffix      = "control-plane-audit-logs"
      lifecycle   = 2555
      description = "CloudTrail and AWS control-plane audit logs"
    }
  }
}

resource "aws_s3_bucket" "this" {
  for_each = local.buckets

  bucket        = "${var.name_prefix}-${each.value.suffix}"
  force_destroy = var.deletion_protection ? false : true

  tags = merge(var.tags, {
    Name        = "${var.name_prefix}-${each.value.suffix}"
    Boundary    = each.value.description
    PublicData  = "false"
    PatientData = "false"
  })
}

resource "aws_s3_bucket_public_access_block" "this" {
  for_each = aws_s3_bucket.this

  bucket                  = each.value.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "this" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = var.kms_key_arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "this" {
  for_each = local.buckets

  bucket = aws_s3_bucket.this[each.key].id
  rule {
    id     = "retain-current-and-expire-noncurrent"
    status = "Enabled"
    noncurrent_version_expiration {
      noncurrent_days = each.value.lifecycle
    }
  }
}

data "aws_iam_policy_document" "tls_only" {
  for_each = aws_s3_bucket.this

  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    actions   = ["s3:*"]
    resources = [each.value.arn, "${each.value.arn}/*"]
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "tls_only" {
  for_each = aws_s3_bucket.this

  bucket = each.value.id
  policy = data.aws_iam_policy_document.tls_only[each.key].json
}
