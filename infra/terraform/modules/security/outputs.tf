output "eks_kms_key_arn" {
  value = aws_kms_key.eks.arn
}

output "storage_kms_key_arn" {
  value = aws_kms_key.storage.arn
}

output "ecr_kms_key_arn" {
  value = aws_kms_key.ecr.arn
}

output "logs_kms_key_arn" {
  value = aws_kms_key.logs.arn
}

output "eks_cluster_role_arn" {
  value = aws_iam_role.eks_cluster.arn
}

output "eks_node_role_arn" {
  value = aws_iam_role.eks_node.arn
}

output "vpc_flow_log_role_arn" {
  value = aws_iam_role.vpc_flow_logs.arn
}

output "api_checkpoint_read_policy_arn" {
  value = aws_iam_policy.api_checkpoint_read.arn
}

output "monitoring_evidence_write_policy_arn" {
  value = aws_iam_policy.monitoring_evidence_write.arn
}
