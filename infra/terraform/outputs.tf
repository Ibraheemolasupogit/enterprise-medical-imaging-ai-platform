output "aws_region" {
  description = "Configured AWS region."
  value       = var.aws_region
}

output "cluster_name" {
  description = "EKS cluster name for Helm deployment."
  value       = module.eks.cluster_name
}

output "cluster_endpoint" {
  description = "EKS cluster endpoint. Private by default."
  value       = module.eks.cluster_endpoint
  sensitive   = true
}

output "api_ecr_repository_url" {
  description = "Private ECR repository URL for the API image."
  value       = module.ecr.api_repository_url
}

output "reviewer_ui_ecr_repository_url" {
  description = "Private ECR repository URL for the reviewer UI image."
  value       = module.ecr.reviewer_ui_repository_url
}

output "model_checkpoint_bucket_name" {
  description = "Bucket for approved model checkpoints."
  value       = module.storage.model_checkpoint_bucket_name
}

output "evidence_bucket_name" {
  description = "Bucket for governed monitoring and audit evidence."
  value       = module.storage.evidence_bucket_name
}

output "private_subnet_ids" {
  description = "Private workload subnet IDs."
  value       = module.networking.private_subnet_ids
}
