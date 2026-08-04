variable "name_prefix" {
  type = string
}

variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "secrets_manager_secret_arns" {
  type    = list(string)
  default = []
}

variable "tags" {
  type = map(string)
}
