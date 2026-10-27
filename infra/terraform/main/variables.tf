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
  default     = "us5.datadoghq.com"
}

variable "api_instance_type" {
  description = "EC2 instance type of the api Auto Scaling group."
  type        = string
  default     = "t4g.small"
}

variable "api_min_size" {
  description = "Minimum instance count of the api Auto Scaling group. The group starts with this many instances."
  type        = number
  default     = 1
}

variable "api_max_size" {
  description = "Maximum instance count of the api Auto Scaling group."
  type        = number
  default     = 1
}

variable "scraper_instance_type" {
  description = "EC2 instance type of the scraper Auto Scaling group."
  type        = string
  default     = "t4g.small"
}

variable "scraper_min_size" {
  description = "Minimum instance count of the scraper Auto Scaling group. The group starts with this many instances."
  type        = number
  default     = 1
}

variable "scraper_max_size" {
  description = "Maximum instance count of the scraper Auto Scaling group."
  type        = number
  default     = 1
}

variable "classifier_instance_type" {
  description = "EC2 instance type of the classifier Auto Scaling group."
  type        = string
  default     = "t4g.small"
}

variable "classifier_min_size" {
  description = "Minimum instance count of the classifier Auto Scaling group. The group starts with this many instances."
  type        = number
  default     = 1
}

variable "classifier_max_size" {
  description = "Maximum instance count of the classifier Auto Scaling group."
  type        = number
  default     = 1
}

variable "api_requests_per_target" {
  description = "Requests per minute that one API instance serves before the API group adds an instance."
  type        = number
  default     = 1000
}

variable "scraper_backlog_per_instance" {
  description = "Visible scraper jobs per scraper instance before the scraper group adds an instance."
  type        = number
  default     = 10
}

variable "classifier_backlog_per_instance" {
  description = "Visible classifier jobs per classifier instance before the classifier group adds an instance."
  type        = number
  default     = 20
}

variable "db_instance_class" {
  description = "RDS instance class."
  type        = string
  default     = "db.t4g.micro"
}

variable "config_overrides" {
  description = "Settings that replace or add to the config map of main.tf. Each entry becomes a String parameter at /real-feel/<NAME>. A tfvars file sets SCRAPER, CLASSIFIER, QUERY_QUOTA_ENABLED, SYNTHETIC_*, FAKE_LATENCY_MS, and DB_MAX_CONNECTIONS here."
  type        = map(string)
  default     = {}
}
