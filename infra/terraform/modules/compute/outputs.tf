output "api_public_ip" {
  description = "Elastic IP of the API instance. The API listens on port 80 of this address, from operator_cidr only."
  value       = aws_eip.api.public_ip
}

output "api_instance_id" {
  description = "ID of the API instance. The deploy workflow targets it with tag role=api."
  value       = aws_instance.api.id
}

output "worker_instance_id" {
  description = "ID of the worker instance. The deploy workflow targets it with tag role=worker."
  value       = aws_instance.worker.id
}
