output "state_bucket_name" {
  description = "Name of the state bucket. Pass it to the main configuration with terraform init -backend-config."
  value       = aws_s3_bucket.state.bucket
}

output "eval_bucket_name" {
  description = "Name of the eval bucket. Copy the eval set and its manifest here with aws s3 cp."
  value       = aws_s3_bucket.eval.bucket
}

output "zone_id" {
  description = "ID of the Route 53 hosted zone."
  value       = aws_route53_zone.main.zone_id
}

output "zone_name_servers" {
  description = "Name servers of the hosted zone. Set them at your registrar when the registrar is not Route 53."
  value       = aws_route53_zone.main.name_servers
}

output "certificate_arn" {
  description = "ARN of the validated ACM certificate."
  value       = aws_acm_certificate_validation.main.certificate_arn
}
