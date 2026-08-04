output "vpc_id" {
  value = aws_vpc.this.id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "public_ingress_subnet_ids" {
  value = aws_subnet.public_ingress[*].id
}

output "eks_control_plane_security_group_id" {
  value = aws_security_group.eks_control_plane.id
}
