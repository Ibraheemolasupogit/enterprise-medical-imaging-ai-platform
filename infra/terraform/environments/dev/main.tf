module "platform" {
  source = "../.."

  project_name                  = "medical-imaging-platform"
  environment                   = "dev"
  aws_region                    = "eu-west-2"
  availability_zones            = ["eu-west-2a", "eu-west-2b"]
  enable_public_ingress_subnets = false
  enable_nat_gateway            = false
  enable_vpc_flow_logs          = true
  kubernetes_version            = "1.31"
  node_instance_types           = ["t3.medium"]
  node_desired_size             = 1
  node_min_size                 = 1
  node_max_size                 = 2
  enable_internal_alb           = false
  deletion_protection           = true
}
