variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "us-east-1"
}

variable "operator_cidr" {
  description = "CIDR block that may reach the API on port 80, such as 203.0.113.7/32."
  type        = string

  validation {
    condition     = can(cidrhost(var.operator_cidr, 0)) && var.operator_cidr != "0.0.0.0/0"
    error_message = "operator_cidr must be an IPv4 CIDR block that is not 0.0.0.0/0, such as 203.0.113.7/32."
  }
}

variable "datadog_external_id" {
  description = "External ID that the Datadog integration role requires."
  type        = string
}

variable "datadog_site" {
  description = "Datadog site that the agent sends data to."
  type        = string
  default     = "datadoghq.com"
}
