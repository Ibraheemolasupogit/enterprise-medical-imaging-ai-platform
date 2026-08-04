locals {
  name_prefix = "${var.project_name}-${var.environment}"

  common_tags = merge(
    {
      Project      = var.project_name
      Environment  = var.environment
      ManagedBy    = "terraform"
      Milestone    = "16"
      DataBoundary = "synthetic-or-deidentified-only"
      ClinicalUse  = "not-for-diagnosis"
      CostProfile  = "validation-safe-defaults"
    },
    var.additional_tags,
  )
}
