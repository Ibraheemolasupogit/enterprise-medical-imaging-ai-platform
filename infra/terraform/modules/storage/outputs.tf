output "model_checkpoint_bucket_name" {
  value = aws_s3_bucket.this["checkpoints"].bucket
}

output "artifact_bucket_name" {
  value = aws_s3_bucket.this["artifacts"].bucket
}

output "evidence_bucket_name" {
  value = aws_s3_bucket.this["evidence"].bucket
}

output "audit_log_bucket_name" {
  value = aws_s3_bucket.this["audit_logs"].bucket
}
