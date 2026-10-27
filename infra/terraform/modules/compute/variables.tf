variable "public_subnet_ids" {
  description = "Public subnets for the instances. Each Auto Scaling group spreads its instances over them."
  type        = list(string)
}

variable "api_sg_id" {
  description = "Security group of the API instances."
  type        = string
}

variable "worker_sg_id" {
  description = "Security group of the scraper and classifier instances."
  type        = string
}

variable "repository_arns" {
  description = "ARNs of the ECR repositories that the instances pull from."
  type        = list(string)
}

variable "api_repository_url" {
  description = "URL of the API image repository. The restart script reads it from SSM."
  type        = string
}

variable "worker_repository_url" {
  description = "URL of the worker image repository. The restart script reads it from SSM."
  type        = string
}

variable "queue_arns" {
  description = "ARNs of the SQS queues. The API sends jobs to them, and the workers send, receive, and delete jobs."
  type        = list(string)
}

variable "db_secret_arn" {
  description = "ARN of the Secrets Manager secret that holds the database credentials."
  type        = string
}

variable "config" {
  description = "Non-secret settings. Each entry becomes a String parameter at /real-feel/<NAME>."
  type        = map(string)
}

variable "groups" {
  description = "Size of each Auto Scaling group. The keys are api, scraper, and classifier."
  type = map(object({
    instance_type = string
    min_size      = number
    max_size      = number
  }))

  validation {
    condition     = toset(keys(var.groups)) == toset(["api", "scraper", "classifier"])
    error_message = "groups needs exactly the keys api, scraper, and classifier."
  }
}

variable "target_group_arn" {
  description = "ARN of the ALB target group. The API Auto Scaling group registers its instances here."
  type        = string
}

variable "target_group_arn_suffix" {
  description = "ARN suffix of the ALB target group, for the ALBRequestCountPerTarget resource label."
  type        = string
}

variable "alb_arn_suffix" {
  description = "ARN suffix of the ALB, for the ALBRequestCountPerTarget resource label."
  type        = string
}

variable "scraper_queue_name" {
  description = "Name of the scraper-jobs queue. The scraper group scales on its backlog."
  type        = string
}

variable "classifier_queue_name" {
  description = "Name of the classifier-jobs queue. The classifier group scales on its backlog."
  type        = string
}

variable "api_requests_per_target" {
  description = "Requests per minute that one API instance serves before the API group adds an instance."
  type        = number
}

variable "scraper_backlog_per_instance" {
  description = "Visible scraper jobs per scraper instance before the scraper group adds an instance."
  type        = number
}

variable "classifier_backlog_per_instance" {
  description = "Visible classifier jobs per classifier instance before the classifier group adds an instance."
  type        = number
}
