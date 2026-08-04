locals {
  alarm_actions = var.alert_sns_topic_arn == "" ? [] : [var.alert_sns_topic_arn]
  log_groups = {
    api         = "/aws/eks/${var.name_prefix}/application/api"
    reviewer_ui = "/aws/eks/${var.name_prefix}/application/reviewer-ui"
    monitoring  = "/aws/eks/${var.name_prefix}/application/monitoring"
    eks_control = "/aws/eks/${var.cluster_name}/cluster"
  }
}

resource "aws_cloudwatch_log_group" "application" {
  for_each = local.log_groups

  name              = each.value
  retention_in_days = var.log_retention_days
  kms_key_id        = var.kms_key_arn

  tags = merge(var.tags, {
    Name = each.value
  })
}

resource "aws_cloudwatch_metric_alarm" "api_error_rate" {
  alarm_name          = "${var.name_prefix}-api-error-rate"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ApiErrorRate"
  namespace           = "MedicalImagingPlatform"
  period              = 300
  statistic           = "Average"
  threshold           = 0.05
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "readiness_failures" {
  alarm_name          = "${var.name_prefix}-readiness-failures"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ReadinessFailures"
  namespace           = "MedicalImagingPlatform"
  period              = 300
  statistic           = "Sum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "pod_restart_rate" {
  alarm_name          = "${var.name_prefix}-pod-restarts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "PodRestartRate"
  namespace           = "MedicalImagingPlatform"
  period              = 300
  statistic           = "Sum"
  threshold           = 3
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.name_prefix}-high-latency"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "P95LatencyMilliseconds"
  namespace           = "MedicalImagingPlatform"
  period              = 300
  statistic           = "Average"
  threshold           = 2000
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  tags                = var.tags
}

resource "aws_cloudwatch_metric_alarm" "node_pressure" {
  alarm_name          = "${var.name_prefix}-node-pressure"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "NodePressure"
  namespace           = "MedicalImagingPlatform"
  period              = 300
  statistic           = "Maximum"
  threshold           = 0
  treat_missing_data  = "notBreaching"
  alarm_actions       = local.alarm_actions
  tags                = var.tags
}

resource "aws_cloudtrail" "control_plane" {
  name                          = "${var.name_prefix}-control-plane"
  s3_bucket_name                = var.audit_bucket_name
  include_global_service_events = true
  is_multi_region_trail         = false
  enable_log_file_validation    = true
  kms_key_id                    = var.kms_key_arn

  event_selector {
    read_write_type           = "All"
    include_management_events = true
  }

  tags = merge(var.tags, {
    Name       = "${var.name_prefix}-control-plane"
    AuditScope = "aws-control-plane"
  })
}
