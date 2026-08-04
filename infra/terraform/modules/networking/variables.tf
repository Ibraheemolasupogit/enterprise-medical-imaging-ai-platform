variable "name_prefix" {
  type = string
}

variable "vpc_cidr" {
  type = string
}

variable "availability_zones" {
  type = list(string)
}

variable "enable_public_ingress_subnets" {
  type    = bool
  default = false
}

variable "enable_nat_gateway" {
  type    = bool
  default = false
}

variable "enable_vpc_flow_logs" {
  type    = bool
  default = true
}

variable "flow_log_role_arn" {
  type = string
}

variable "tags" {
  type = map(string)
}
