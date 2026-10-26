output "github_role_arn" {
  description = "ARN of the role that GitHub Actions assumes to deploy."
  value       = aws_iam_role.github_deploy.arn
}

output "datadog_role_arn" {
  description = "ARN of the role that Datadog assumes to read the SQS, RDS, and EC2 metrics."
  value       = aws_iam_role.datadog.arn
}
