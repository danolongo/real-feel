output "api_instance_id" {
  description = "ID of the API instance. The deploy workflow targets it with tag role=api."
  value       = aws_instance.api.id
}

output "worker_instance_id" {
  description = "ID of the worker instance. The deploy workflow targets it with tag role=worker."
  value       = aws_instance.worker.id
}
