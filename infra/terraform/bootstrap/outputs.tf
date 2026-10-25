output "state_bucket_name" {
  description = "Name of the state bucket. Pass it to the main configuration with terraform init -backend-config."
  value       = aws_s3_bucket.state.bucket
}
