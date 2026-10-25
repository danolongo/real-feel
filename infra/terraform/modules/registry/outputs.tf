output "api_repository_url" {
  description = "URL of the API image repository."
  value       = aws_ecr_repository.api.repository_url
}

output "worker_repository_url" {
  description = "URL of the worker image repository."
  value       = aws_ecr_repository.worker.repository_url
}

output "repository_arns" {
  description = "ARNs of both image repositories."
  value       = [aws_ecr_repository.api.arn, aws_ecr_repository.worker.arn]
}
