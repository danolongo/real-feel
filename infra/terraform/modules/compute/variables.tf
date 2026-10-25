variable "public_subnet_ids" {
  description = "Public subnets for the API and worker instances. The API takes the first one and the worker takes the second."
  type        = list(string)
}

variable "api_sg_id" {
  description = "Security group of the API instance."
  type        = string
}

variable "worker_sg_id" {
  description = "Security group of the worker instance."
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

variable "instance_type" {
  description = "EC2 instance type for the API and worker instances."
  type        = string
  default     = "t3.small"
}
