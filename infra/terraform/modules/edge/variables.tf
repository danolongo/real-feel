variable "vpc_id" {
  description = "ID of the VPC that holds the target group."
  type        = string
}

variable "public_subnet_ids" {
  description = "Public subnets of the ALB."
  type        = list(string)
}

variable "alb_sg_id" {
  description = "Security group of the ALB."
  type        = string
}

variable "domain_name" {
  description = "Domain name of the API. The alias record points it at the ALB."
  type        = string
}

variable "zone_id" {
  description = "ID of the Route 53 hosted zone of the domain name."
  type        = string
}

variable "certificate_arn" {
  description = "ARN of the ACM certificate for the HTTPS listener."
  type        = string
}

variable "waf_rate_limit" {
  description = "Requests that one IP address may send in 5 minutes before WAF answers 429."
  type        = number
}

variable "waf_rate_limit_enabled" {
  description = "true makes the rate rule block. false makes the rule count requests and block none."
  type        = bool
}
