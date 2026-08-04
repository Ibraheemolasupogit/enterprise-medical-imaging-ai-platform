variable "project_name" {
  description = "Project prefix used for AWS resources."
  type        = string
  default     = "medical-imaging-platform"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region for target-state infrastructure."
  type        = string
  default     = "eu-west-2"
}

variable "vpc_cidr" {
  description = "CIDR for the platform VPC."
  type        = string
  default     = "10.42.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones used by private workload subnets and optional public ingress subnets."
  type        = list(string)
  default     = ["eu-west-2a", "eu-west-2b"]
}

variable "enable_public_ingress_subnets" {
  description = "Whether to create public ingress subnets. Services remain internal by default."
  type        = bool
  default     = false
}

variable "enable_nat_gateway" {
  description = "Whether private workload subnets receive NAT egress. Disabled by default for low-cost validation."
  type        = bool
  default     = false
}

variable "enable_vpc_flow_logs" {
  description = "Whether to create VPC flow logs to CloudWatch."
  type        = bool
  default     = true
}

variable "kubernetes_version" {
  description = "Pinned configurable EKS Kubernetes version."
  type        = string
  default     = "1.31"
}

variable "node_instance_types" {
  description = "CPU-only managed node group instance types."
  type        = list(string)
  default     = ["t3.medium"]
}

variable "node_desired_size" {
  description = "Conservative desired capacity for the CPU-only managed node group."
  type        = number
  default     = 1
}

variable "node_min_size" {
  description = "Minimum CPU-only managed node count."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum CPU-only managed node count."
  type        = number
  default     = 2
}

variable "enable_internal_alb" {
  description = "Whether to permit an optional internal ALB security boundary. Disabled by default."
  type        = bool
  default     = false
}

variable "alert_sns_topic_arn" {
  description = "Optional existing SNS topic ARN for CloudWatch alarms. Leave empty in validation."
  type        = string
  default     = ""
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days."
  type        = number
  default     = 30
}

variable "deletion_protection" {
  description = "Protect stateful resources from accidental deletion where supported."
  type        = bool
  default     = true
}

variable "secrets_manager_secret_arns" {
  description = "Explicit Secrets Manager secret ARNs readable by workloads. Empty by default."
  type        = list(string)
  default     = []
}

variable "additional_tags" {
  description = "Additional tags applied to resources."
  type        = map(string)
  default     = {}
}
