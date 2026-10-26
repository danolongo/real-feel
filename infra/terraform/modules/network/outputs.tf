output "vpc_id" {
  description = "ID of the VPC."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets. The API and worker instances run here."
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets in two Availability Zones. Postgres runs here."
  value       = aws_subnet.private[*].id
}

output "api_sg_id" {
  description = "ID of the API security group."
  value       = aws_security_group.api.id
}

output "worker_sg_id" {
  description = "ID of the worker security group."
  value       = aws_security_group.worker.id
}

output "db_sg_id" {
  description = "ID of the database security group."
  value       = aws_security_group.db.id
}
