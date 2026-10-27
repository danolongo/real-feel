output "alb_dns_name" {
  description = "DNS name of the ALB."
  value       = aws_lb.api.dns_name
}

output "target_group_arn" {
  description = "ARN of the target group. The API Auto Scaling group registers its instances here."
  value       = aws_lb_target_group.api.arn
}

output "target_group_arn_suffix" {
  description = "ARN suffix of the target group, for the ALBRequestCountPerTarget resource label."
  value       = aws_lb_target_group.api.arn_suffix
}

output "alb_arn_suffix" {
  description = "ARN suffix of the ALB, for the ALBRequestCountPerTarget resource label."
  value       = aws_lb.api.arn_suffix
}
