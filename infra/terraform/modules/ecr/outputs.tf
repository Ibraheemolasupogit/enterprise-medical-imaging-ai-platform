output "api_repository_url" {
  value = aws_ecr_repository.this["api"].repository_url
}

output "reviewer_ui_repository_url" {
  value = aws_ecr_repository.this["reviewerui"].repository_url
}
