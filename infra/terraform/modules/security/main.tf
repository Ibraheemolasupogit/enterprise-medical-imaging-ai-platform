data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

locals {
  checkpoint_bucket_arn = "arn:aws:s3:::${var.name_prefix}-model-checkpoints"
  evidence_bucket_arn   = "arn:aws:s3:::${var.name_prefix}-governed-evidence"
  log_group_arn_prefix  = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/*/${var.name_prefix}*"
}

data "aws_iam_policy_document" "kms_key" {
  statement {
    sid    = "AllowAccountKeyAdministration"
    effect = "Allow"
    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
    actions = [
      "kms:CancelKeyDeletion",
      "kms:CreateAlias",
      "kms:DeleteAlias",
      "kms:DescribeKey",
      "kms:DisableKey",
      "kms:EnableKey",
      "kms:EnableKeyRotation",
      "kms:GetKeyPolicy",
      "kms:GetKeyRotationStatus",
      "kms:ListResourceTags",
      "kms:PutKeyPolicy",
      "kms:ScheduleKeyDeletion",
      "kms:TagResource",
      "kms:UntagResource",
      "kms:UpdateAlias",
      "kms:UpdateKeyDescription",
    ]
    resources = ["*"]
  }

  statement {
    sid    = "AllowServiceUse"
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["logs.amazonaws.com", "s3.amazonaws.com", "eks.amazonaws.com", "ecr.amazonaws.com"]
    }
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
      "kms:GenerateDataKeyWithoutPlaintext",
      "kms:ReEncryptFrom",
      "kms:ReEncryptTo",
    ]
    resources = ["*"]
  }
}

resource "aws_kms_key" "eks" {
  description             = "KMS key for EKS Kubernetes secret encryption."
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_key.json

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-eks-kms"
  })
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.name_prefix}/eks"
  target_key_id = aws_kms_key.eks.key_id
}

resource "aws_kms_key" "storage" {
  description             = "KMS key for governed S3 storage."
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_key.json

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-storage-kms"
  })
}

resource "aws_kms_alias" "storage" {
  name          = "alias/${var.name_prefix}/storage"
  target_key_id = aws_kms_key.storage.key_id
}

resource "aws_kms_key" "ecr" {
  description             = "KMS key for private ECR repositories."
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_key.json

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-ecr-kms"
  })
}

resource "aws_kms_alias" "ecr" {
  name          = "alias/${var.name_prefix}/ecr"
  target_key_id = aws_kms_key.ecr.key_id
}

resource "aws_kms_key" "logs" {
  description             = "KMS key for CloudWatch and CloudTrail logs."
  deletion_window_in_days = 30
  enable_key_rotation     = true
  policy                  = data.aws_iam_policy_document.kms_key.json

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-logs-kms"
  })
}

resource "aws_kms_alias" "logs" {
  name          = "alias/${var.name_prefix}/logs"
  target_key_id = aws_kms_key.logs.key_id
}

data "aws_iam_policy_document" "eks_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["eks.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

data "aws_iam_policy_document" "node_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "eks_cluster" {
  name               = "${var.name_prefix}-eks-cluster"
  assume_role_policy = data.aws_iam_policy_document.eks_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_iam_role" "eks_node" {
  name               = "${var.name_prefix}-eks-node"
  assume_role_policy = data.aws_iam_policy_document.node_assume_role.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "node_worker" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "node_cni" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "node_ecr_read" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

data "aws_iam_policy_document" "vpc_flow_logs_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["vpc-flow-logs.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "vpc_flow_logs" {
  name               = "${var.name_prefix}-vpc-flow-logs"
  assume_role_policy = data.aws_iam_policy_document.vpc_flow_logs_assume_role.json

  tags = var.tags
}

data "aws_iam_policy_document" "vpc_flow_logs" {
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:DescribeLogGroups",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]
    resources = ["${local.log_group_arn_prefix}:*"]
  }
}

resource "aws_iam_policy" "vpc_flow_logs" {
  name   = "${var.name_prefix}-vpc-flow-logs"
  policy = data.aws_iam_policy_document.vpc_flow_logs.json

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "vpc_flow_logs" {
  role       = aws_iam_role.vpc_flow_logs.name
  policy_arn = aws_iam_policy.vpc_flow_logs.arn
}

data "aws_iam_policy_document" "api_checkpoint_read" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket",
    ]
    resources = [
      local.checkpoint_bucket_arn,
      "${local.checkpoint_bucket_arn}/approved/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
    ]
    resources = [aws_kms_key.storage.arn]
  }
}

resource "aws_iam_policy" "api_checkpoint_read" {
  name   = "${var.name_prefix}-api-checkpoint-read"
  policy = data.aws_iam_policy_document.api_checkpoint_read.json

  tags = var.tags
}

data "aws_iam_policy_document" "monitoring_evidence_write" {
  statement {
    effect = "Allow"
    actions = [
      "s3:AbortMultipartUpload",
      "s3:GetObject",
      "s3:ListBucket",
      "s3:PutObject",
    ]
    resources = [
      local.evidence_bucket_arn,
      "${local.evidence_bucket_arn}/monitoring/*",
      "${local.evidence_bucket_arn}/audit/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKey",
    ]
    resources = [aws_kms_key.storage.arn]
  }
}

resource "aws_iam_policy" "monitoring_evidence_write" {
  name   = "${var.name_prefix}-monitoring-evidence-write"
  policy = data.aws_iam_policy_document.monitoring_evidence_write.json

  tags = var.tags
}

data "aws_iam_policy_document" "secrets_read" {
  count = length(var.secrets_manager_secret_arns) > 0 ? 1 : 0

  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:DescribeSecret",
      "secretsmanager:GetSecretValue",
    ]
    resources = var.secrets_manager_secret_arns
  }
}

resource "aws_iam_policy" "secrets_read" {
  count  = length(var.secrets_manager_secret_arns) > 0 ? 1 : 0
  name   = "${var.name_prefix}-named-secrets-read"
  policy = data.aws_iam_policy_document.secrets_read[0].json

  tags = var.tags
}
