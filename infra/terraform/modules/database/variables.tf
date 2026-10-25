variable "private_subnet_ids" {
  description = "IDs of the private subnets. RDS needs subnets in at least two Availability Zones."
  type        = list(string)
}

variable "db_sg_id" {
  description = "ID of the security group that the database instance uses."
  type        = string
}

variable "instance_class" {
  description = "RDS instance class. The smallest class keeps the cost low."
  type        = string
  default     = "db.t4g.micro"
}
