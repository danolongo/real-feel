variable "github_repo" {
  description = "GitHub repository, as owner/name, whose main branch may assume the deploy role."
  type        = string
  default     = "danolongo/real-feel"
}

variable "datadog_external_id" {
  description = "External ID that Datadog sends when it assumes the integration role."
  type        = string
}

variable "datadog_aws_account_id" {
  description = "AWS account ID that Datadog uses to assume the integration role."
  type        = string
  default     = "464622532012"
}
