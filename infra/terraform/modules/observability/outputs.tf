output "application_log_group_names" {
  value = values(aws_cloudwatch_log_group.application)[*].name
}

output "cloudtrail_name" {
  value = aws_cloudtrail.control_plane.name
}
