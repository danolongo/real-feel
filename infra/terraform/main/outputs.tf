output "alb_dns_name" {
  description = "DNS name of the ALB. The domain name points here."
  value       = module.edge.alb_dns_name
}

output "github_role_arn" {
  description = "Role that GitHub Actions assumes to deploy. Set it as the repo variable AWS_DEPLOY_ROLE_ARN."
  value       = module.iam.github_role_arn
}

output "datadog_role_arn" {
  description = "Role that Datadog assumes to read the SQS, RDS, and EC2 metrics."
  value       = module.iam.datadog_role_arn
}

output "api_instance_id" {
  description = "ID of the API instance. Target it with the tag role=api."
  value       = module.compute.api_instance_id
}

output "worker_instance_id" {
  description = "ID of the worker instance. Target it with the tag role=worker."
  value       = module.compute.worker_instance_id
}
