output "state_bucket_name" {
  description = "Name of the state bucket. Pass it to the main configuration with terraform init -backend-config."
  value       = aws_s3_bucket.state.bucket
}

output "eval_bucket_name" {
  description = "Name of the eval bucket. Copy the eval set and its manifest here with aws s3 cp."
  value       = aws_s3_bucket.eval.bucket
}
