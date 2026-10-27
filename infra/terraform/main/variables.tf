variable "region" {
  description = "AWS region for every resource."
  type        = string
  default     = "us-east-1"
}

variable "domain_name" {
  description = "Domain name of the API. The bootstrap configuration owns its hosted zone and certificate."
  type        = string
  default     = "api.danielemartinez.com"
}

variable "waf_rate_limit" {
  description = "Requests that one IP address may send in 5 minutes before WAF answers 429."
  type        = number
  default     = 1000
}

variable "waf_rate_limit_enabled" {
  description = "true makes the rate rule block. false makes the rule count requests and block none."
  type        = bool
  default     = true
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
