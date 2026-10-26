variable "region" {
  description = "AWS region for the state bucket."
  type        = string
  default     = "us-east-1"
}

variable "domain_name" {
  description = "Domain name of the API. The hosted zone and the certificate use it."
  type        = string
}
