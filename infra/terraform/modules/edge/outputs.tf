output "alb_dns_name" {
  description = "DNS name of the ALB."
  value       = aws_lb.api.dns_name
}
