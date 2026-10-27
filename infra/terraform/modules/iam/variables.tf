variable "github_repo" {
  description = "GitHub repository, as owner/name, whose deploy branches may assume the deploy role."
  type        = string
  default     = "danolongo/real-feel"
}

variable "github_deploy_branches" {
  description = "Branches of the repository that may assume the deploy role. A branch can end in * to match a prefix."
  type        = list(string)
  default     = ["main", "stress-testing", "perf/*"]
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
