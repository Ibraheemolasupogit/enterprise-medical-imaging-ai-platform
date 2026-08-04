variable "name_prefix" {
  type = string
}

variable "kms_key_arn" {
  type = string
}

variable "deletion_protection" {
  type    = bool
  default = true
}

variable "tags" {
  type = map(string)
}
