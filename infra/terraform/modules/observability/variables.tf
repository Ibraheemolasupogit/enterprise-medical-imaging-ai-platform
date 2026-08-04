variable "name_prefix" {
  type = string
}

variable "log_retention_days" {
  type = number
}

variable "alert_sns_topic_arn" {
  type    = string
  default = ""
}

variable "cluster_name" {
  type = string
}

variable "audit_bucket_name" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "tags" {
  type = map(string)
}
