module "security" {
  source = "./modules/security"

  name_prefix                 = local.name_prefix
  project_name                = var.project_name
  environment                 = var.environment
  secrets_manager_secret_arns = var.secrets_manager_secret_arns
  tags                        = local.common_tags
}

module "networking" {
  source = "./modules/networking"

  name_prefix                   = local.name_prefix
  vpc_cidr                      = var.vpc_cidr
  availability_zones            = var.availability_zones
  enable_public_ingress_subnets = var.enable_public_ingress_subnets
  enable_nat_gateway            = var.enable_nat_gateway
  enable_vpc_flow_logs          = var.enable_vpc_flow_logs
  flow_log_role_arn             = module.security.vpc_flow_log_role_arn
  tags                          = local.common_tags
}

module "ecr" {
  source = "./modules/ecr"

  name_prefix = local.name_prefix
  kms_key_arn = module.security.ecr_kms_key_arn
  tags        = local.common_tags
}

module "storage" {
  source = "./modules/storage"

  name_prefix         = local.name_prefix
  kms_key_arn         = module.security.storage_kms_key_arn
  deletion_protection = var.deletion_protection
  tags                = local.common_tags
}

module "eks" {
  source = "./modules/eks"

  name_prefix         = local.name_prefix
  kubernetes_version  = var.kubernetes_version
  subnet_ids          = module.networking.private_subnet_ids
  cluster_role_arn    = module.security.eks_cluster_role_arn
  node_role_arn       = module.security.eks_node_role_arn
  kms_key_arn         = module.security.eks_kms_key_arn
  security_group_ids  = [module.networking.eks_control_plane_security_group_id]
  node_instance_types = var.node_instance_types
  node_desired_size   = var.node_desired_size
  node_min_size       = var.node_min_size
  node_max_size       = var.node_max_size
  tags                = local.common_tags
}

module "observability" {
  source = "./modules/observability"

  name_prefix         = local.name_prefix
  log_retention_days  = var.log_retention_days
  alert_sns_topic_arn = var.alert_sns_topic_arn
  cluster_name        = module.eks.cluster_name
  audit_bucket_name   = module.storage.audit_log_bucket_name
  kms_key_arn         = module.security.logs_kms_key_arn
  tags                = local.common_tags
}
